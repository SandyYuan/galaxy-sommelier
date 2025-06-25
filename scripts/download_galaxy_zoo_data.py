#!/usr/bin/env python3
"""
Galaxy Zoo Data Downloader for Galaxy Sommelier Project
Downloads Galaxy Zoo catalogs and SDSS images for training.
"""

import os
import pandas as pd
import requests
import numpy as np
from astropy.io import fits
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from tqdm import tqdm
import time
import logging
from pathlib import Path
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GalaxyZooDownloader:
    """Download Galaxy Zoo catalogs and SDSS images"""
    
    def __init__(self, scratch_dir='/pscratch/sd/s/sihany/galaxy-sommelier-data'):
        self.data_dir = Path(scratch_dir) / 'sdss'
        self.catalog_dir = Path(scratch_dir) / 'catalogs'
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Catalog directory: {self.catalog_dir}")
    
    def download_catalogs(self):
        """Download Galaxy Zoo catalogs"""
        logger.info("Downloading Galaxy Zoo 2 catalogs...")
        
        # Galaxy Zoo 2 main catalog from Hart et al. 2016 (best debiased table)
        gz2_url = "https://gz2hart.s3.amazonaws.com/gz2_hart16.fits.gz"
        gz2_path = self.catalog_dir / 'gz2_hart16.fits.gz'
        
        if not gz2_path.exists():
            logger.info("Downloading Galaxy Zoo 2 catalog (FITS)...")
            try:
                response = requests.get(gz2_url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                
                with open(gz2_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading GZ2 FITS") as pbar:
                        for chunk in response.iter_content(chunk_size=block_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                logger.info(f"Successfully downloaded catalog to {gz2_path}")
            except Exception as e:
                logger.error(f"Error downloading catalog: {e}")
                raise
        else:
            logger.info(f"Catalog already exists at {gz2_path}")
        
        # Load and validate catalog
        try:
            catalog_table = Table.read(gz2_path)
            catalog = catalog_table.to_pandas()
            logger.info(f"Loaded catalog with {len(catalog)} galaxies")
            logger.info(f"Columns: {list(catalog.columns[:10])}...")  # Show first 10 columns
            
            # Basic validation
            required_cols = ['dr7objid', 'ra', 'dec']
            missing_cols = [col for col in required_cols if col not in catalog.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            return catalog
        
        except Exception as e:
            logger.error(f"Error loading catalog: {e}")
            raise
    
    def download_sdss_images(self, catalog, limit=None, start_index=0):
        """Download SDSS images for Galaxy Zoo objects"""
        logger.info(f"Starting SDSS image download...")
        
        if limit:
            end_index = min(start_index + limit, len(catalog))
            catalog_subset = catalog.iloc[start_index:end_index]
            logger.info(f"Downloading {len(catalog_subset)} images (indices {start_index}-{end_index-1})")
        else:
            catalog_subset = catalog.iloc[start_index:]
            logger.info(f"Downloading {len(catalog_subset)} images (from index {start_index})")
        
        successful_downloads = 0
        failed_downloads = 0
        
        for idx, row in tqdm(catalog_subset.iterrows(), total=len(catalog_subset), desc="Downloading SDSS images"):
            ra, dec = row['ra'], row['dec']
            objid = row['dr7objid']
            
            output_path = self.data_dir / f'sdss_{objid}.fits'
            
            # Skip if already exists
            if output_path.exists():
                continue
                
            try:
                # Create SkyCoord object
                coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
                
                # Download SDSS image with timeout
                images = SDSS.get_images(coordinates=coords, radius=30*u.arcsec, 
                                       data_release=8, band='r', timeout=30)
                
                if images and len(images) > 0:
                    # Save the first (usually only) image
                    images[0].writeto(output_path, overwrite=True)
                    successful_downloads += 1
                else:
                    logger.warning(f"No image found for objid {objid}")
                    failed_downloads += 1
                
                # Add small delay to be respectful to servers
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error downloading objid {objid}: {e}")
                failed_downloads += 1
                continue
        
        logger.info(f"Download completed: {successful_downloads} successful, {failed_downloads} failed")
        return successful_downloads, failed_downloads
    
    def create_sample_catalog(self, full_catalog, sample_size=1000, save_path=None):
        """Create a smaller sample catalog for testing"""
        logger.info(f"Creating sample catalog with {sample_size} galaxies...")
        
        # Filter for galaxies with good quality morphology classifications
        # Focus on galaxies with high vote counts for cleaner labels
        filtered = full_catalog.copy()
        
        # Basic quality filters
        if 't01_smooth_or_features_a01_smooth_count' in full_catalog.columns:
            # Calculate total votes for task 1
            vote_cols = [col for col in full_catalog.columns if 't01_' in col and '_count' in col]
            if vote_cols:
                filtered['total_votes_t01'] = full_catalog[vote_cols].sum(axis=1)
                # Keep galaxies with at least 10 votes
                filtered = filtered[filtered['total_votes_t01'] >= 10]
                logger.info(f"After vote filtering: {len(filtered)} galaxies")
        
        # Random sample
        if len(filtered) > sample_size:
            sample_catalog = filtered.sample(n=sample_size, random_state=42).reset_index(drop=True)
        else:
            sample_catalog = filtered.reset_index(drop=True)
            logger.warning(f"Only {len(sample_catalog)} galaxies available (less than requested {sample_size})")
        
        if save_path:
            sample_path = Path(save_path)
            sample_catalog.to_csv(sample_path, index=False)
            logger.info(f"Sample catalog saved to {sample_path}")
        
        return sample_catalog
    
    def validate_downloads(self):
        """Validate downloaded FITS files"""
        logger.info("Validating downloaded FITS files...")
        
        fits_files = list(self.data_dir.glob("*.fits"))
        logger.info(f"Found {len(fits_files)} FITS files")
        
        valid_files = 0
        corrupted_files = []
        
        for fits_path in tqdm(fits_files, desc="Validating"):
            try:
                with fits.open(fits_path) as hdul:
                    # Basic validation - check if we can read the data
                    data = hdul[0].data
                    if data is not None and data.size > 0:
                        valid_files += 1
                    else:
                        corrupted_files.append(fits_path)
            except Exception as e:
                logger.error(f"Corrupted file {fits_path}: {e}")
                corrupted_files.append(fits_path)
        
        logger.info(f"Validation complete: {valid_files} valid, {len(corrupted_files)} corrupted")
        
        # Remove corrupted files
        for corrupted_file in corrupted_files:
            try:
                corrupted_file.unlink()
                logger.info(f"Removed corrupted file: {corrupted_file}")
            except Exception as e:
                logger.error(f"Error removing {corrupted_file}: {e}")
        
        return valid_files, len(corrupted_files)

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Download Galaxy Zoo data")
    parser.add_argument("--scratch-dir", default="/pscratch/sd/s/sihany/galaxy-sommelier-data",
                       help="Scratch directory for data storage")
    parser.add_argument("--download-catalogs", action="store_true",
                       help="Download Galaxy Zoo catalogs")
    parser.add_argument("--download-images", action="store_true",
                       help="Download SDSS images")
    parser.add_argument("--sample-size", type=int, default=1000,
                       help="Number of images to download for testing")
    parser.add_argument("--start-index", type=int, default=0,
                       help="Starting index for image download")
    parser.add_argument("--validate", action="store_true",
                       help="Validate downloaded files")
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = GalaxyZooDownloader(scratch_dir=args.scratch_dir)
    
    # Download catalogs
    if args.download_catalogs:
        catalog = downloader.download_catalogs()
        
        # Create sample catalog
        sample_catalog = downloader.create_sample_catalog(
            catalog, 
            sample_size=args.sample_size,
            save_path=downloader.catalog_dir / f'gz2_sample_{args.sample_size}.csv'
        )
        
        logger.info(f"Sample catalog created with {len(sample_catalog)} galaxies")
    
    # Download images
    if args.download_images:
        # Load catalog
        catalog_path = downloader.catalog_dir / 'gz2_hart16.fits.gz'
        if not catalog_path.exists():
            logger.error("Catalog not found. Run with --download-catalogs first.")
            return
        
        catalog_table = Table.read(catalog_path)
        catalog = catalog_table.to_pandas()
        
        # Download images
        downloader.download_sdss_images(
            catalog, 
            limit=args.sample_size,
            start_index=args.start_index
        )
    
    # Validate downloads
    if args.validate:
        downloader.validate_downloads()

if __name__ == "__main__":
    main() 