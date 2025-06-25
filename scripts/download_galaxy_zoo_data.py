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
from astropy.table import Table
from tqdm import tqdm
import time
import logging
from pathlib import Path
import argparse
import zipfile
import random

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
        
        # Galaxy Zoo 2 main catalog from Hart et al. 2016 (FITS version)
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
            logger.info(f"Columns: {list(catalog.columns[:10])}...")

            required_cols = ['dr7objid', 'ra', 'dec']
            missing_cols = [col for col in required_cols if col not in catalog.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            return catalog
        
        except Exception as e:
            logger.error(f"Error loading catalog: {e}")
            raise
    
    def download_images_from_zip(self):
        """Download and extract Galaxy Zoo 2 images from Zenodo."""
        logger.info("Downloading Galaxy Zoo 2 images from Zenodo...")
        
        zip_url = "https://zenodo.org/records/3565489/files/images_gz2.zip?download=1"
        zip_path = self.catalog_dir / 'images_gz2.zip'
        
        if not zip_path.exists():
            logger.info(f"Downloading image archive from {zip_url}...")
            try:
                response = requests.get(zip_url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                
                with open(zip_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading images_gz2.zip") as pbar:
                        for chunk in response.iter_content(chunk_size=block_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                logger.info(f"Successfully downloaded image archive to {zip_path}")
            except Exception as e:
                logger.error(f"Error downloading image archive: {e}")
                raise
        else:
            logger.info(f"Image archive already exists at {zip_path}")

        logger.info(f"Extracting images to {self.data_dir}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in tqdm(zip_ref.infolist(), desc="Extracting images"):
                    if member.is_dir():
                        continue
                    
                    target_filename = Path(member.filename).name
                    if not target_filename: continue # Skip empty filenames

                    target_path = self.data_dir / target_filename
                    if not target_path.exists():
                        with zip_ref.open(member) as source:
                            with open(target_path, "wb") as target:
                                target.write(source.read())

            logger.info("Image extraction complete.")
        except Exception as e:
            logger.error(f"Error extracting images: {e}")
            raise

    def create_sample_catalog(self, full_catalog, sample_size=1000, save_path=None):
        """Create a smaller sample catalog for testing"""
        logger.info(f"Creating sample catalog with {sample_size} galaxies...")
        
        filtered = full_catalog.copy()
        
        if 't01_smooth_or_features_a01_smooth_count' in full_catalog.columns:
            vote_cols = [col for col in full_catalog.columns if 't01_' in col and '_count' in col]
            if vote_cols:
                filtered['total_votes_t01'] = full_catalog[vote_cols].sum(axis=1)
                filtered = filtered[filtered['total_votes_t01'] >= 10]
                logger.info(f"After vote filtering: {len(filtered)} galaxies")
        
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
    
    def validate_downloads(self, sample_size=None):
        """Validate downloaded JPG files"""
        logger.info("Validating downloaded JPG files...")
        
        try:
            from PIL import Image
        except ImportError:
            logger.error("Pillow library not found. Cannot validate images. Please run 'pip install Pillow'")
            return 0, 0

        all_img_files = list(self.data_dir.glob("*.jpg"))
        
        # Filter out macOS metadata files
        img_files_to_validate = [p for p in all_img_files if not p.name.startswith("._")]
        
        # Clean up metadata files
        meta_files = [p for p in all_img_files if p.name.startswith("._")]
        if meta_files:
            logger.info(f"Found and removing {len(meta_files)} macOS metadata files (e.g., ._*)...")
            for f in meta_files:
                try:
                    f.unlink()
                except OSError as e:
                    logger.error(f"Error removing metadata file {f}: {e}")

        logger.info(f"Found {len(img_files_to_validate)} JPG files to validate.")
        
        if sample_size and len(img_files_to_validate) > sample_size:
            logger.info(f"Validating a random sample of {sample_size} files.")
            img_files_to_validate = random.sample(img_files_to_validate, sample_size)
        
        valid_files = 0
        corrupted_files = []
        
        for img_path in tqdm(img_files_to_validate, desc="Validating"):
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_files += 1
            except Exception as e:
                logger.error(f"Corrupted file {img_path}: {e}")
                corrupted_files.append(img_path)
        
        logger.info(f"Validation complete: {valid_files} valid, {len(corrupted_files)} corrupted")
        
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
                       help="Download SDSS images from Zenodo")
    parser.add_argument("--sample-size", type=int, default=1000,
                       help="Number of galaxies for the sample catalog")
    parser.add_argument("--validate", action="store_true",
                       help="Validate downloaded files")
    parser.add_argument("--validation-sample-size", type=int, default=None,
                       help="Number of images to sample for validation.")
    
    args = parser.parse_args()
    
    downloader = GalaxyZooDownloader(scratch_dir=args.scratch_dir)
    
    if args.download_catalogs:
        catalog = downloader.download_catalogs()
        
        sample_catalog = downloader.create_sample_catalog(
            catalog, 
            sample_size=args.sample_size,
            save_path=downloader.catalog_dir / f'gz2_sample_{args.sample_size}.csv'
        )
        logger.info(f"Sample catalog created with {len(sample_catalog)} galaxies")
    
    if args.download_images:
        downloader.download_images_from_zip()
    
    if args.validate:
        downloader.validate_downloads(sample_size=args.validation_sample_size)

if __name__ == "__main__":
    main() 