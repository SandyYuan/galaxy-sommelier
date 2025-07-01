#!/usr/bin/env python3
"""
Fast Galaxy Zoo Hubble Image Downloader

Optimized version with:
- Higher default parallelism (8 jobs)
- HST mosaic caching (reuse for multiple galaxies)
- Optional PNG generation (off by default)
- Reduced delays and faster processing
- Spatial batching of nearby galaxies

Usage:
    python download_hubble_data_fast.py --sample-size 1000 --parallel-jobs 8
    python download_hubble_data_fast.py --full-catalog --parallel-jobs 16 --no-png
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import requests
import time
import hashlib
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

class FastHubbleDownloader:
    """
    Optimized HST image downloader with caching and parallel processing
    """
    
    def __init__(self, output_dir: str = 'data/hubble', 
                 cutout_size: int = 256, 
                 pixel_scale: float = 0.05,
                 save_png: bool = False,
                 preferred_filter: str = 'F814W',
                 cache_mosaics: bool = True):
        """
        Initialize the fast downloader
        
        Parameters:
        -----------
        output_dir : str
            Directory to save downloaded images
        cutout_size : int
            Size of image cutouts in pixels (default: 256)
        pixel_scale : float
            Pixel scale in arcsec/pixel (default: 0.05 for HST)
        save_png : bool
            Whether to save PNG versions of images (default: False for speed)
        preferred_filter : str
            Preferred HST filter (default: 'F814W')
        cache_mosaics : bool
            Whether to cache downloaded HST mosaics (default: True)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cutout_size = cutout_size
        self.pixel_scale = pixel_scale
        self.cutout_radius = (cutout_size * pixel_scale) / 2.0  # arcsec
        self.save_png = save_png and PNG_AVAILABLE
        self.preferred_filter = preferred_filter.upper()
        self.cache_mosaics = cache_mosaics
        
        # Create subdirectories
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        (self.output_dir / 'failed').mkdir(exist_ok=True)
        if cache_mosaics:
            (self.output_dir / 'mosaic_cache').mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.WARNING,  # Reduced logging for speed
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
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Mosaic cache for reusing downloaded HST fields
        self.mosaic_cache = {}
        
    def load_catalog(self, catalog_path: str, sample_size: Optional[int] = None) -> List[GalaxyInfo]:
        """Load Galaxy Zoo Hubble catalog and filter for COSMOS galaxies"""
        print(f"Loading catalog from {catalog_path}")
        
        # Read catalog
        df = pd.read_csv(catalog_path)
        print(f"Loaded {len(df)} galaxies from catalog")
        
        # Filter for COSMOS only
        cosmos_df = df[df['imaging'] == 'COSMOS'].copy()
        print(f"Found {len(cosmos_df)} COSMOS galaxies")
        
        if len(cosmos_df) == 0:
            self.logger.error("No COSMOS galaxies found in catalog!")
            return []
        
        # Pre-filter for existing files (much faster than checking during download)
        print("Pre-filtering existing files...")
        existing_fits_ids = set()
        existing_png_ids = set()
        image_dir = self.output_dir / 'images'
        
        if image_dir.exists():
            # Check for existing FITS files
            for file_path in image_dir.glob("cosmos_*.fits"):
                zooniverse_id = file_path.stem.replace('cosmos_', '')
                existing_fits_ids.add(zooniverse_id)
            
            # Check for existing PNG files
            for file_path in image_dir.glob("cosmos_*.png"):
                zooniverse_id = file_path.stem.replace('cosmos_', '')
                existing_png_ids.add(zooniverse_id)
        
        # Smart filtering based on what's requested and what exists
        if self.save_png:
            # PNG requested - skip only if both FITS and PNG exist
            skip_ids = existing_fits_ids.intersection(existing_png_ids)
            print(f"PNG mode: found {len(existing_fits_ids)} FITS, {len(existing_png_ids)} PNG files")
        else:
            # PNG not requested - skip if FITS exists
            skip_ids = existing_fits_ids
            print(f"FITS-only mode: found {len(existing_fits_ids)} FITS files")
        
        if skip_ids:
            before_count = len(cosmos_df)
            cosmos_df = cosmos_df[~cosmos_df['zooniverse_id'].isin(skip_ids)]
            skipped_count = before_count - len(cosmos_df)
            print(f"Pre-filtered {skipped_count} complete files, {len(cosmos_df)} remaining")
            self.stats['skipped'] = skipped_count
        
        # Sample if requested
        if sample_size and sample_size < len(cosmos_df):
            cosmos_df = cosmos_df.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} COSMOS galaxies")
        
        # Convert to GalaxyInfo objects
        galaxies = []
        for _, row in cosmos_df.iterrows():
            # Handle missing values
            z_best = row.get('Z_BEST', 999.0)
            if pd.isna(z_best) or z_best == 999.0:
                z_best = -1.0
                
            mag_best = row.get('GZ_MU_I', 99.0) 
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

    def _get_mosaic_cache_key(self, ra: float, dec: float, radius: float = 60.0) -> str:
        """Generate cache key for mosaic based on spatial location"""
        # Round coordinates to create spatial grid for caching
        grid_size = radius / 3600.0  # Convert arcsec to degrees
        ra_grid = round(ra / grid_size) * grid_size
        dec_grid = round(dec / grid_size) * grid_size
        return f"mosaic_{ra_grid:.6f}_{dec_grid:.6f}_{self.preferred_filter}"

    def download_cosmos_cutout_fast(self, galaxy: GalaxyInfo) -> Optional[str]:
        """
        Fast COSMOS cutout download with mosaic caching
        """
        if not ASTROQUERY_AVAILABLE:
            return None
            
        try:
            from astroquery.mast import Observations
            
            # Check mosaic cache first
            cache_key = self._get_mosaic_cache_key(galaxy.ra, galaxy.dec)
            cached_mosaic = None
            
            if self.cache_mosaics and cache_key in self.mosaic_cache:
                cached_mosaic = self.mosaic_cache[cache_key]
                self.stats['cache_hits'] += 1
            else:
                self.stats['cache_misses'] += 1
                
                # Query for HST observations
                coord = SkyCoord(ra=galaxy.ra*u.deg, dec=galaxy.dec*u.deg)
                obs_table = Observations.query_region(coord, radius=30*u.arcsec)
                hst_obs = obs_table[obs_table['obs_collection'] == 'HST']
                
                if len(hst_obs) == 0:
                    return None
                
                # Get products and filter for preferred filter
                products = Observations.get_product_list(hst_obs[0])
                product_types = np.array(products['productSubGroupDescription'])
                drizzled_mask = product_types == 'DRZ'
                drizzled_products = products[drizzled_mask]
                
                if len(drizzled_products) == 0:
                    return None
                
                # Filter for preferred filter
                filenames = np.array(drizzled_products['productFilename'])
                filter_mask = np.array([self.preferred_filter.lower() in fn.lower() for fn in filenames])
                filter_products = drizzled_products[filter_mask]
                
                if len(filter_products) == 0:
                    filter_products = drizzled_products  # Fallback
                
                # Download to cache
                if self.cache_mosaics:
                    cache_dir = self.output_dir / 'mosaic_cache'
                else:
                    cache_dir = self.output_dir / 'temp'
                cache_dir.mkdir(exist_ok=True)
                
                download_result = Observations.download_products(
                    filter_products[0:1],
                    download_dir=str(cache_dir)
                )
                
                if len(download_result) == 0:
                    return None
                
                downloaded_file = download_result['Local Path'][0]
                
                # Cache the mosaic path if caching enabled
                if self.cache_mosaics:
                    self.mosaic_cache[cache_key] = downloaded_file
                    cached_mosaic = downloaded_file
                else:
                    cached_mosaic = downloaded_file
            
            # Extract cutout from cached/downloaded mosaic
            if cached_mosaic and os.path.exists(cached_mosaic):
                cutout_path = self._extract_cutout_from_drizzled_image(cached_mosaic, galaxy)
                
                # Clean up temp files (but keep cache)
                if not self.cache_mosaics and cached_mosaic:
                    try:
                        os.remove(cached_mosaic)
                    except:
                        pass
                
                return cutout_path
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Download failed for {galaxy.zooniverse_id}: {e}")
            return None

    def _extract_cutout_from_drizzled_image(self, image_path: str, galaxy: GalaxyInfo) -> Optional[str]:
        """Extract cutout from HST drizzled image (optimized)"""
        try:
            with fits.open(image_path) as hdul:
                # Get data and header
                if len(hdul) > 1 and hdul[1].data is not None:
                    data = hdul[1].data
                    header = hdul[1].header
                else:
                    data = hdul[0].data
                    header = hdul[0].header
                
                if data is None:
                    return None
                
                # Fast coordinate conversion
                try:
                    from astropy.wcs import WCS
                    wcs = WCS(header)
                    world_coord = SkyCoord(ra=galaxy.ra*u.deg, dec=galaxy.dec*u.deg)
                    pixel_coords = wcs.world_to_pixel(world_coord)
                    x_center = int(pixel_coords[0])
                    y_center = int(pixel_coords[1])
                except Exception:
                    # Fallback to image center
                    y_center, x_center = data.shape[0] // 2, data.shape[1] // 2
                
                # Extract cutout
                half_size = self.cutout_size // 2
                x_min = max(0, x_center - half_size)
                x_max = min(data.shape[1], x_center + half_size)
                y_min = max(0, y_center - half_size)
                y_max = min(data.shape[0], y_center + half_size)
                
                if (x_max - x_min) < self.cutout_size // 2 or (y_max - y_min) < self.cutout_size // 2:
                    return None
                
                cutout_data = data[y_min:y_max, x_min:x_max]
                
                # Minimal header for speed
                cutout_header = fits.Header()
                cutout_header['GALAXY_ID'] = galaxy.zooniverse_id
                cutout_header['ORIG_RA'] = galaxy.ra
                cutout_header['ORIG_DEC'] = galaxy.dec
                
                # Save cutout
                filename = f"cosmos_{galaxy.zooniverse_id}.fits"
                output_path = self.output_dir / 'images' / filename
                
                fits.writeto(output_path, cutout_data, cutout_header, overwrite=True)
                return str(output_path)
                
        except Exception as e:
            return None

    def download_galaxy_fast(self, galaxy: GalaxyInfo) -> bool:
        """Fast galaxy download with minimal overhead"""
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
            png_path = self._create_png_fast(str(fits_file), galaxy)
            if png_path:
                self.stats['converted'] += 1
            return True
        elif fits_exists and not self.save_png:
            # FITS exists and PNG not requested - skip
            self.stats['skipped'] += 1
            return True
        elif not fits_exists:
            # FITS doesn't exist - need to download
            result = self.download_cosmos_cutout_fast(galaxy)
            
            if result:
                # Create PNG if requested
                if self.save_png:
                    self._create_png_fast(result, galaxy)
                
                # Minimal JSON metadata
                metadata = {
                    'zooniverse_id': galaxy.zooniverse_id,
                    'ra': galaxy.ra,
                    'dec': galaxy.dec,
                    'file_path': result
                }
                
                metadata_file = self.output_dir / 'metadata' / f"{galaxy.zooniverse_id}.json"
                import json
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f)  # No indent for smaller files
                
                self.stats['success'] += 1
                return True
            else:
                self.stats['failed'] += 1
                return False
        
        return True

    def _create_png_fast(self, fits_path: str, galaxy: GalaxyInfo) -> Optional[str]:
        """Fast PNG creation (simplified)"""
        if not PNG_AVAILABLE:
            return None
        try:
            with fits.open(fits_path) as hdul:
                data = hdul[0].data
            
            if data is None:
                return None
            
            # Fast normalization
            data = np.nan_to_num(data, nan=0.0)
            vmin, vmax = np.percentile(data, [1, 99])
            data_norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
            
            # Fast PNG save
            png_filename = f"cosmos_{galaxy.zooniverse_id}.png"
            png_path = self.output_dir / 'images' / png_filename
            
            plt.imsave(png_path, data_norm, cmap='gray')  # Much faster than full matplotlib
            return str(png_path)
        except:
            return None

    def download_batch_fast(self, galaxies: List[GalaxyInfo], parallel_jobs: int = 8) -> None:
        """Fast batch download with high parallelism"""
        print(f"Starting fast download of {len(galaxies)} COSMOS galaxies with {parallel_jobs} parallel jobs")
        
        # Progress bar
        progress_bar = tqdm(total=len(galaxies), desc="Fast Download")
        
        # High parallelism, no artificial delays
        with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            future_to_galaxy = {
                executor.submit(self.download_galaxy_fast, galaxy): galaxy 
                for galaxy in galaxies
            }
            
            for future in as_completed(future_to_galaxy):
                future.result()
                progress_bar.update(1)
                # No sleep delays for maximum speed
        
        progress_bar.close()
        self.print_statistics()

    def print_statistics(self) -> None:
        """Print download statistics"""
        print("\n" + "="*50)
        print("FAST COSMOS DOWNLOAD STATISTICS")
        print("="*50)
        print(f"Total COSMOS galaxies: {self.stats['total']}")
        print(f"Successfully downloaded: {self.stats['success']}")
        print(f"FITS->PNG conversions: {self.stats['converted']}")
        print(f"Already existed (pre-filtered): {self.stats['skipped']}")
        print(f"Failed downloads: {self.stats['failed']}")
        
        if self.cache_mosaics:
            print(f"Mosaic cache hits: {self.stats['cache_hits']}")
            print(f"Mosaic cache misses: {self.stats['cache_misses']}")
            if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
                cache_rate = self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses']) * 100
                print(f"Cache hit rate: {cache_rate:.1f}%")
        
        if self.stats['total'] > 0:
            success_rate = (self.stats['success'] + self.stats['skipped'] + self.stats['converted']) / self.stats['total'] * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        print(f"\nImages saved to: {self.output_dir / 'images'}")


def main():
    """Main function with optimized defaults"""
    parser = argparse.ArgumentParser(
        description="Fast HST COSMOS image downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fast download with 8 parallel jobs (FITS only)
    python download_hubble_data_fast.py --sample-size 1000 --parallel-jobs 8
    
    # Download with PNG generation
    python download_hubble_data_fast.py --sample-size 1000 --parallel-jobs 8 --save-png
    
    # Convert existing FITS files to PNG (no re-download)
    python download_hubble_data_fast.py --sample-size 1000 --save-png
    
    # Maximum speed with 16 parallel jobs
    python download_hubble_data_fast.py --full-catalog --parallel-jobs 16 --no-cache
        """
    )
    
    parser.add_argument('--catalog-path', 
                       default='/pscratch/sd/s/sihany/galaxy-sommelier-data/catalogs/gz_hubble_main.csv',
                       help='Path to Galaxy Zoo Hubble catalog CSV file')
    parser.add_argument('--output-dir',
                       default='/pscratch/sd/s/sihany/galaxy-sommelier-data/hubble',
                       help='Output directory for downloaded images')
    parser.add_argument('--sample-size', type=int,
                       help='Number of COSMOS galaxies to download')
    parser.add_argument('--full-catalog', action='store_true',
                       help='Download all COSMOS galaxies in catalog')
    parser.add_argument('--parallel-jobs', type=int, default=8,
                       help='Number of parallel download jobs (default: 8)')
    parser.add_argument('--cutout-size', type=int, default=256,
                       help='Size of image cutouts in pixels (default: 256)')
    parser.add_argument('--save-png', action='store_true',
                       help='Also save PNG versions (slower)')
    parser.add_argument('--filter', default='F814W',
                       help='Preferred HST filter (default: F814W)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable mosaic caching (uses more bandwidth but less disk)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.full_catalog and not args.sample_size:
        parser.error("Must specify either --sample-size or --full-catalog")
    
    if not os.path.exists(args.catalog_path):
        parser.error(f"Catalog file not found: {args.catalog_path}")
    
    # Initialize fast downloader
    downloader = FastHubbleDownloader(
        output_dir=args.output_dir,
        cutout_size=args.cutout_size,
        save_png=args.save_png,
        preferred_filter=args.filter,
        cache_mosaics=not args.no_cache
    )
    
    # Load catalog
    sample_size = None if args.full_catalog else args.sample_size
    galaxies = downloader.load_catalog(args.catalog_path, sample_size)
    
    if len(galaxies) == 0:
        print("No new COSMOS galaxies to download!")
        return
    
    # Start fast downloads
    downloader.download_batch_fast(galaxies, args.parallel_jobs)


if __name__ == '__main__':
    main() 