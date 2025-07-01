#!/usr/bin/env python3
"""
Galaxy Zoo Hubble Image Downloader

This script downloads HST images for galaxies in the Galaxy Zoo Hubble catalog.
It uses multiple download strategies to maximize success rate:

1. MAST HAPCut service for standardized cutouts
2. astroquery.mast for direct MAST archive queries  
3. Survey-specific HLSP data (COSMOS, GOODS, etc.)
4. Legacy HLA cutout service as fallback

Usage:
    python download_hubble_data.py --sample-size 1000 --output-dir data/hubble
    python download_hubble_data.py --full-catalog --parallel-jobs 8
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path
from urllib.parse import urlencode
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import optional modules for PNG conversion
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, PowerNorm
    from PIL import Image
    PNG_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib/PIL not available. PNG output disabled.")
    PNG_AVAILABLE = False

# Try to import astroquery modules
try:
    from astroquery.mast import Observations
    ASTROQUERY_AVAILABLE = True
    print("astroquery successfully imported!")
except ImportError as e:
    print(f"Warning: astroquery import failed: {e}. Some download methods will be disabled.")
    ASTROQUERY_AVAILABLE = False

@dataclass
class GalaxyInfo:
    """Container for galaxy information"""
    zooniverse_id: str
    survey_id: str
    ra: float
    dec: float
    imaging: str
    z_best: float
    mag_best: float
    correction_type: int
    total_count: int

class HubbleImageDownloader:
    """
    Download HST images for Galaxy Zoo Hubble catalog
    """
    
    def __init__(self, output_dir: str = 'data/hubble', 
                 cutout_size: int = 256, 
                 pixel_scale: float = 0.05,
                 save_png: bool = False,
                 preferred_filter: str = 'F814W'):
        """
        Initialize the downloader
        
        Parameters:
        -----------
        output_dir : str
            Directory to save downloaded images
        cutout_size : int
            Size of image cutouts in pixels (default: 256)
        pixel_scale : float
            Pixel scale in arcsec/pixel (default: 0.05 for HST)
        save_png : bool
            Whether to save PNG versions of images (default: False)
        preferred_filter : str
            Preferred HST filter (default: 'F814W')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cutout_size = cutout_size
        self.pixel_scale = pixel_scale
        self.cutout_radius = (cutout_size * pixel_scale) / 2.0  # arcsec
        self.save_png = save_png and PNG_AVAILABLE
        self.preferred_filter = preferred_filter.upper()
        
        # Create subdirectories
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        (self.output_dir / 'failed').mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'download.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Download statistics
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'converted': 0,  # FITS to PNG conversions
            'methods_used': {
                'hapcut': 0,
                'mast_query': 0,
                'hlsp_cosmos': 0,
                'hlsp_goods': 0,
                'hla_cutout': 0
            }
        }
        
        # Survey-specific information
        self.survey_info = {
            'COSMOS': {
                'hlsp_base': 'https://archive.stsci.edu/pub/hlsp/cosmos/',
                'instruments': ['acs', 'wfpc2', 'nicmos'],
                'preferred_filter': 'I'
            },
            'GOODS-N-FULLDEPTH': {
                'hlsp_base': 'https://archive.stsci.edu/pub/hlsp/goods/',
                'field': 'goods-n',
                'preferred_filter': 'F814W'
            },
            'GOODS-S-FULLDEPTH': {
                'hlsp_base': 'https://archive.stsci.edu/pub/hlsp/goods/',
                'field': 'goods-s', 
                'preferred_filter': 'F814W'
            },
            'GEMS': {
                'part_of': 'GOODS-S-FULLDEPTH',
                'preferred_filter': 'F850LP'
            },
            'AEGIS': {
                'hlsp_base': 'https://archive.stsci.edu/pub/hlsp/aegis/',
                'preferred_filter': 'F814W'
            }
        }
    
    def load_catalog(self, catalog_path: str, sample_size: Optional[int] = None) -> List[GalaxyInfo]:
        """Load Galaxy Zoo Hubble catalog and filter for COSMOS galaxies"""
        self.logger.info(f"Loading catalog from {catalog_path}")
        
        # Read catalog
        df = pd.read_csv(catalog_path)
        self.logger.info(f"Loaded {len(df)} galaxies from catalog")
        
        # Filter for COSMOS only
        cosmos_df = df[df['imaging'] == 'COSMOS'].copy()
        self.logger.info(f"Found {len(cosmos_df)} COSMOS galaxies")
        
        if len(cosmos_df) == 0:
            self.logger.error("No COSMOS galaxies found in catalog!")
            return []
        
        # Sample if requested
        if sample_size and sample_size < len(cosmos_df):
            cosmos_df = cosmos_df.sample(n=sample_size, random_state=42)
            self.logger.info(f"Sampled {sample_size} COSMOS galaxies")
        
        # Convert to GalaxyInfo objects
        galaxies = []
        for _, row in cosmos_df.iterrows():
            # Handle missing values
            z_best = row.get('Z_BEST', 999.0)
            if pd.isna(z_best) or z_best == 999.0:
                z_best = -1.0
                
            mag_best = row.get('MAG_BEST_HI', 99.0) 
            if pd.isna(mag_best):
                mag_best = 99.0
            
            galaxy = GalaxyInfo(
                zooniverse_id=row['zooniverse_id'],
                survey_id=str(row['survey_id']),
                ra=row['RA'],
                dec=row['DEC'],
                imaging=row['imaging'],
                z_best=z_best,
                mag_best=mag_best,
                correction_type=int(row.get('correction_type', 0)),
                total_count=int(row.get('total_count', 0))
            )
            galaxies.append(galaxy)
        
        self.stats['total'] = len(galaxies)
        return galaxies

    def download_cosmos_cutout(self, galaxy: GalaxyInfo) -> Optional[str]:
        """
        Download COSMOS cutout using astroquery.mast
        """
        if not ASTROQUERY_AVAILABLE:
            self.logger.warning(f"astroquery not available for {galaxy.zooniverse_id}")
            return None
            
        try:
            from astroquery.mast import Observations
            
            # Create coordinate
            coord = SkyCoord(ra=galaxy.ra*u.deg, dec=galaxy.dec*u.deg)
            
            # Query for HST observations at this location
            obs_table = Observations.query_region(coord, radius=30*u.arcsec)
            
            # Filter for HST observations
            hst_obs = obs_table[obs_table['obs_collection'] == 'HST']
            
            if len(hst_obs) == 0:
                self.logger.debug(f"No HST observations found for {galaxy.zooniverse_id}")
                return None
            
            # Get products for the first HST observation
            products = Observations.get_product_list(hst_obs[0])
            
            # Look for drizzled images (DRZ) - these are the best processed products
            product_types = np.array(products['productSubGroupDescription'])
            drizzled_mask = product_types == 'DRZ'
            drizzled_products = products[drizzled_mask]
            
            if len(drizzled_products) == 0:
                self.logger.debug(f"No drizzled products found for {galaxy.zooniverse_id}")
                return None
            
            # Filter for preferred filter images specifically
            filenames = np.array(drizzled_products['productFilename'])
            filter_mask = np.array([self.preferred_filter.lower() in fn.lower() for fn in filenames])
            filter_products = drizzled_products[filter_mask]
            
            if len(filter_products) == 0:
                self.logger.debug(f"No {self.preferred_filter} drizzled products found for {galaxy.zooniverse_id}")
                # Fall back to any drizzled product if preferred filter not found
                self.logger.debug(f"Available products: {list(filenames)}")
                filter_products = drizzled_products
            else:
                self.logger.debug(f"Found {len(filter_products)} {self.preferred_filter} products for {galaxy.zooniverse_id}")
                
            # Use preferred filter products (or fallback to any drizzled products)
            selected_products = filter_products
            
            # Download the first drizzled product to temp directory
            temp_dir = self.output_dir / 'temp'
            temp_dir.mkdir(exist_ok=True)
            
            self.logger.debug(f"Downloading {selected_products[0]['productFilename']} for {galaxy.zooniverse_id}")
            
            download_result = Observations.download_products(
                selected_products[0:1],
                download_dir=str(temp_dir)
            )
            
            if len(download_result) == 0:
                self.logger.warning(f"Download failed for {galaxy.zooniverse_id}")
                return None
            
            # Extract cutout from the large drizzled image
            downloaded_file = download_result['Local Path'][0]
            cutout_path = self._extract_cutout_from_drizzled_image(
                downloaded_file, galaxy
            )
            
            # Clean up the large file
            try:
                os.remove(downloaded_file)
                # Try to remove empty directories
                parent_dir = Path(downloaded_file).parent
                if parent_dir.exists() and not any(parent_dir.iterdir()):
                    os.rmdir(parent_dir)
                    # Try to remove grandparent if empty
                    grandparent = parent_dir.parent
                    if grandparent.exists() and not any(grandparent.iterdir()):
                        os.rmdir(grandparent)
            except:
                pass  # Don't fail if cleanup doesn't work
            
            return cutout_path
            
        except Exception as e:
            self.logger.warning(f"astroquery download failed for {galaxy.zooniverse_id}: {e}")
            return None

    def _extract_cutout_from_drizzled_image(self, image_path: str, galaxy: GalaxyInfo) -> Optional[str]:
        """
        Extract a cutout from a large HST drizzled image
        """
        try:
            with fits.open(image_path) as hdul:
                # HST drizzled images typically have science data in HDU 1
                if len(hdul) > 1 and hdul[1].data is not None:
                    data = hdul[1].data
                    header = hdul[1].header
                else:
                    # Fallback to HDU 0
                    data = hdul[0].data
                    header = hdul[0].header
                
                if data is None:
                    self.logger.warning(f"No data found in {image_path}")
                    return None
                
                # Use WCS to find the galaxy position in pixels
                try:
                    from astropy.wcs import WCS
                    wcs = WCS(header)
                    
                    # Convert RA/Dec to pixel coordinates
                    world_coord = SkyCoord(ra=galaxy.ra*u.deg, dec=galaxy.dec*u.deg)
                    pixel_coords = wcs.world_to_pixel(world_coord)
                    
                    x_center = int(pixel_coords[0])
                    y_center = int(pixel_coords[1])
                    
                    self.logger.debug(f"Galaxy {galaxy.zooniverse_id} at pixel ({x_center}, {y_center})")
                    
                except Exception as e:
                    self.logger.warning(f"WCS conversion failed, using image center: {e}")
                    # Fallback to image center
                    y_center, x_center = data.shape[0] // 2, data.shape[1] // 2
                
                # Extract cutout region
                half_size = self.cutout_size // 2
                x_min = max(0, x_center - half_size)
                x_max = min(data.shape[1], x_center + half_size)
                y_min = max(0, y_center - half_size)
                y_max = min(data.shape[0], y_center + half_size)
                
                # Check if we have enough data
                if (x_max - x_min) < self.cutout_size // 2 or (y_max - y_min) < self.cutout_size // 2:
                    self.logger.warning(f"Galaxy {galaxy.zooniverse_id} too close to edge")
                    return None
                
                # Extract cutout
                cutout_data = data[y_min:y_max, x_min:x_max]
                
                # Create new header for cutout
                cutout_header = header.copy()
                
                # Update WCS reference pixel
                if 'CRPIX1' in cutout_header:
                    cutout_header['CRPIX1'] = cutout_header['CRPIX1'] - x_min
                if 'CRPIX2' in cutout_header:
                    cutout_header['CRPIX2'] = cutout_header['CRPIX2'] - y_min
                
                # Add metadata
                cutout_header['GALAXY_ID'] = galaxy.zooniverse_id
                cutout_header['ORIG_RA'] = galaxy.ra
                cutout_header['ORIG_DEC'] = galaxy.dec
                cutout_header['CUTOUT_X'] = x_center
                cutout_header['CUTOUT_Y'] = y_center
                
                # Save cutout
                filename = f"cosmos_{galaxy.zooniverse_id}.fits"
                output_path = self.output_dir / 'images' / filename
                
                fits.writeto(output_path, cutout_data, cutout_header, overwrite=True)
                
                self.logger.debug(f"Saved cutout: {filename} ({cutout_data.shape})")
                return str(output_path)
                
        except Exception as e:
            self.logger.warning(f"Cutout extraction failed for {galaxy.zooniverse_id}: {e}")
            return None

    def _create_png_from_fits(self, fits_path: str, galaxy: GalaxyInfo) -> Optional[str]:
        """
        Create a PNG image from a FITS file for visualization
        """
        if not self.save_png or not PNG_AVAILABLE:
            return None
            
        try:
            # Read FITS data
            with fits.open(fits_path) as hdul:
                data = hdul[0].data
                
            if data is None:
                return None
            
            # Handle NaN values
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize data for visualization
            # Use percentile clipping to handle outliers
            vmin = np.percentile(data, 1)
            vmax = np.percentile(data, 99)
            
            # Clip and normalize to 0-1 range
            data_normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
            
            # Apply gamma correction for better visibility
            gamma = 0.5  # Makes faint features more visible
            data_gamma = np.power(data_normalized, gamma)
            
            # Convert to 8-bit
            data_8bit = (data_gamma * 255).astype(np.uint8)
            
            # Create PNG filename
            png_filename = f"cosmos_{galaxy.zooniverse_id}.png"
            png_path = self.output_dir / 'images' / png_filename
            
            # Save using matplotlib for high quality
            plt.figure(figsize=(8, 8))
            plt.imshow(data_8bit, cmap='gray', origin='lower')
            plt.title(f'{galaxy.zooniverse_id} (RA={galaxy.ra:.4f}, Dec={galaxy.dec:.4f})')
            plt.xlabel('Pixels')
            plt.ylabel('Pixels')
            plt.colorbar(label='Normalized Intensity')
            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.debug(f"Saved PNG: {png_filename}")
            return str(png_path)
            
        except Exception as e:
            self.logger.warning(f"PNG creation failed for {galaxy.zooniverse_id}: {e}")
            return None

    def download_galaxy(self, galaxy: GalaxyInfo) -> bool:
        """Download image for a single COSMOS galaxy"""
        fits_file = self.output_dir / 'images' / f"cosmos_{galaxy.zooniverse_id}.fits"
        png_file = self.output_dir / 'images' / f"cosmos_{galaxy.zooniverse_id}.png"
        
        fits_exists = fits_file.exists()
        png_exists = png_file.exists()
        
        # Determine what action to take based on file existence and PNG request
        if fits_exists and png_exists:
            # Both files exist - skip completely
            self.stats['skipped'] += 1
            return True
        elif fits_exists and not png_exists and self.save_png:
            # FITS exists but PNG doesn't and PNG requested - convert only
            png_path = self._create_png_from_fits(str(fits_file), galaxy)
            if png_path:
                self.stats['converted'] += 1
                self.logger.debug(f"Converted FITS to PNG for {galaxy.zooniverse_id}")
            return True
        elif fits_exists and not self.save_png:
            # FITS exists and PNG not requested - skip
            self.stats['skipped'] += 1
            return True
        elif not fits_exists:
            # FITS doesn't exist - need to download
            result = self.download_cosmos_cutout(galaxy)
            
            if result:
                # Create PNG version if requested
                png_path = None
                if self.save_png:
                    png_path = self._create_png_from_fits(result, galaxy)
                
                # Save metadata
                metadata = {
                    'zooniverse_id': galaxy.zooniverse_id,
                    'survey_id': galaxy.survey_id,
                    'ra': galaxy.ra,
                    'dec': galaxy.dec,
                    'imaging': galaxy.imaging,
                    'z_best': galaxy.z_best,
                    'mag_best': galaxy.mag_best,
                    'file_path': result
                }
                
                # Add PNG path to metadata if available
                if png_path:
                    metadata['png_path'] = png_path
                
                metadata_file = self.output_dir / 'metadata' / f"{galaxy.zooniverse_id}.json"
                import json
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                self.stats['success'] += 1
                return True
            else:
                # Save to failed list
                with open(self.output_dir / 'failed' / 'failed_downloads.txt', 'a') as f:
                    f.write(f"{galaxy.zooniverse_id},{galaxy.ra},{galaxy.dec},{galaxy.imaging}\n")
                
                self.stats['failed'] += 1
                return False
        
        return True

    def download_batch(self, galaxies: List[GalaxyInfo], parallel_jobs: int = 4) -> None:
        """Download images for a batch of galaxies"""
        self.logger.info(f"Starting download of {len(galaxies)} COSMOS galaxies")
        
        # Create progress bar
        progress_bar = tqdm(total=len(galaxies), desc="Downloading COSMOS")
        
        if parallel_jobs == 1:
            # Sequential download
            for galaxy in galaxies:
                self.download_galaxy(galaxy)
                progress_bar.update(1)
                # Removed artificial delay for faster downloads
        else:
            # Parallel download with rate limiting
            with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
                future_to_galaxy = {
                    executor.submit(self.download_galaxy, galaxy): galaxy 
                    for galaxy in galaxies
                }
                
                for future in as_completed(future_to_galaxy):
                    future.result()
                    progress_bar.update(1)
                    # Removed artificial delay for faster downloads
        
        progress_bar.close()
        self.print_statistics()

    def print_statistics(self) -> None:
        """Print download statistics"""
        print("\n" + "="*50)
        print("COSMOS DOWNLOAD STATISTICS")
        print("="*50)
        print(f"Total COSMOS galaxies: {self.stats['total']}")
        print(f"Successfully downloaded: {self.stats['success']}")
        print(f"FITS->PNG conversions: {self.stats['converted']}")
        print(f"Already existed (skipped): {self.stats['skipped']}")
        print(f"Failed downloads: {self.stats['failed']}")
        
        if self.stats['total'] > 0:
            success_rate = (self.stats['success'] + self.stats['skipped'] + self.stats['converted']) / self.stats['total'] * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        print(f"\nImages saved to: {self.output_dir / 'images'}")
        print(f"Metadata saved to: {self.output_dir / 'metadata'}")
        if self.stats['failed'] > 0:
            print(f"Failed downloads logged to: {self.output_dir / 'failed'}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Download HST COSMOS images for Galaxy Zoo Hubble catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download sample of 100 COSMOS galaxies (FITS only)
    python download_hubble_data.py --sample-size 100
    
    # Download with PNG generation
    python download_hubble_data.py --sample-size 100 --save-png
    
    # Convert existing FITS files to PNG (no re-download)
    python download_hubble_data.py --sample-size 100 --save-png
    
    # Download all COSMOS galaxies with 8 parallel jobs
    python download_hubble_data.py --full-catalog --parallel-jobs 8
        """
    )
    
    parser.add_argument(
        '--catalog-path', 
        default='/pscratch/sd/s/sihany/galaxy-sommelier-data/catalogs/gz_hubble_main.csv',
        help='Path to Galaxy Zoo Hubble catalog CSV file'
    )
    
    parser.add_argument(
        '--output-dir',
        default='/pscratch/sd/s/sihany/galaxy-sommelier-data/hubble',
        help='Output directory for downloaded images'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Number of COSMOS galaxies to download'
    )
    
    parser.add_argument(
        '--full-catalog',
        action='store_true',
        help='Download all COSMOS galaxies in catalog'
    )
    
    parser.add_argument(
        '--parallel-jobs',
        type=int,
        default=8,
        help='Number of parallel download jobs (default: 8)'
    )
    
    parser.add_argument(
        '--cutout-size',
        type=int,
        default=256,
        help='Size of image cutouts in pixels (default: 256)'
    )
    
    parser.add_argument(
        '--save-png',
        action='store_true',
        help='Also save PNG versions of images for visualization'
    )
    
    parser.add_argument(
        '--filter',
        default='F814W',
        help='Preferred HST filter for downloads (default: F814W)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.full_catalog and not args.sample_size:
        parser.error("Must specify either --sample-size or --full-catalog")
    
    if not os.path.exists(args.catalog_path):
        parser.error(f"Catalog file not found: {args.catalog_path}")
    
    # Initialize downloader
    downloader = HubbleImageDownloader(
        output_dir=args.output_dir,
        cutout_size=args.cutout_size,
        save_png=args.save_png,
        preferred_filter=args.filter
    )
    
    # Load catalog (will automatically filter for COSMOS)
    sample_size = None if args.full_catalog else args.sample_size
    galaxies = downloader.load_catalog(args.catalog_path, sample_size)
    
    if len(galaxies) == 0:
        print("No COSMOS galaxies found to download!")
        return
    
    # Start downloads
    downloader.download_batch(galaxies, args.parallel_jobs)


if __name__ == '__main__':
    main()
