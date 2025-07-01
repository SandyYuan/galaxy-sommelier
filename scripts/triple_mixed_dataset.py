#!/usr/bin/env python3
"""
Triple Mixed SDSS + DECaLS + HST Dataset for Galaxy Sommelier
Combines three galaxy surveys using common morphological features.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import logging
from tqdm import tqdm
import random

# Import the feature mapping
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from sdss_decals_feature_mapping import SDSS_TO_DECALS, DECALS_TO_SDSS

logger = logging.getLogger(__name__)

class TripleMixedDataset(Dataset):
    """Dataset class that combines SDSS, DECaLS, and HST Galaxy Zoo data"""
    
    def __init__(self, sdss_catalog_path, decals_catalog_path, hst_catalog_path,
                 sdss_image_dir, decals_image_dir, hst_image_dir,
                 survey_fractions={'sdss': 0.33, 'decals': 0.33, 'hst': 0.34}, 
                 max_galaxies=None, transform=None, feature_set='sdss', 
                 random_seed=42, high_quality=False):
        """
        Args:
            sdss_catalog_path: Path to SDSS catalog CSV
            decals_catalog_path: Path to DECaLS catalog CSV  
            hst_catalog_path: Path to HST catalog CSV
            sdss_image_dir: Directory containing SDSS images
            decals_image_dir: Directory containing DECaLS images
            hst_image_dir: Directory containing HST images
            survey_fractions: Dict with fractions for each survey (should sum to 1.0)
            max_galaxies: Total number of galaxies to use (None = use all available)
            transform: Image transforms to apply
            feature_set: 'sdss' - feature naming convention to use
            random_seed: Random seed for reproducible sampling
            high_quality: If True, select galaxies with highest classification counts
        """
        self.sdss_image_dir = Path(sdss_image_dir)
        self.decals_image_dir = Path(decals_image_dir)
        self.hst_image_dir = Path(hst_image_dir)
        self.transform = transform
        self.feature_set = feature_set
        self.random_seed = random_seed
        self.high_quality = high_quality
        self.survey_fractions = survey_fractions
        
        # Validate survey fractions
        total_fraction = sum(survey_fractions.values())
        if abs(total_fraction - 1.0) > 0.01:
            raise ValueError(f"Survey fractions must sum to 1.0, got {total_fraction}")
        
        # Set random seed for reproducible sampling
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        quality_mode = "high-quality" if high_quality else "random"
        logger.info(f"Loading triple mixed dataset with {survey_fractions} ({quality_mode})")
        
        # Load and prepare catalogs
        self.load_catalogs(sdss_catalog_path, decals_catalog_path, hst_catalog_path)
        
        # Create mixed dataset
        self.create_mixed_dataset(max_galaxies)
        
        logger.info(f"Triple mixed dataset created with {len(self.mixed_catalog)} galaxies")
        for survey in ['sdss', 'decals', 'hst']:
            count = (self.mixed_catalog['survey'] == survey).sum()
            logger.info(f"{survey.upper()}: {count} galaxies ({count/len(self.mixed_catalog)*100:.1f}%)")
        
    def load_catalogs(self, sdss_catalog_path, decals_catalog_path, hst_catalog_path):
        """Load and prepare SDSS, DECaLS, and HST catalogs"""
        
        # Load SDSS catalog
        logger.info(f"Loading SDSS catalog from {sdss_catalog_path}")
        self.sdss_catalog = pd.read_csv(sdss_catalog_path)
        logger.info(f"SDSS catalog: {len(self.sdss_catalog)} galaxies")
        
        # Load DECaLS catalog  
        logger.info(f"Loading DECaLS catalog from {decals_catalog_path}")
        self.decals_catalog = pd.read_csv(decals_catalog_path)
        logger.info(f"DECaLS catalog: {len(self.decals_catalog)} galaxies")
        
        # Load HST catalog
        logger.info(f"Loading HST catalog from {hst_catalog_path}")
        self.hst_catalog = pd.read_csv(hst_catalog_path)
        logger.info(f"HST catalog: {len(self.hst_catalog)} galaxies")
        
        # Add image existence checks and paths
        self.check_image_availability()
        
        # Get overlapping morphological features
        self.get_common_features()
        
    def check_image_availability(self):
        """Check which galaxies have available images"""
        
        logger.info("Checking SDSS image availability...")
        sdss_available = []
        for idx, row in tqdm(self.sdss_catalog.iterrows(), total=len(self.sdss_catalog), desc="SDSS images"):
            objid = row['dr7objid']
            # Try different image naming conventions
            jpg_path = self.sdss_image_dir / f"{row.get('asset_id', objid)}.jpg"
            if not jpg_path.exists():
                jpg_path = self.sdss_image_dir / f"sdss_{objid}.jpg"
            if not jpg_path.exists():
                jpg_path = self.sdss_image_dir / f"{objid}.jpg"
            
            if jpg_path.exists():
                sdss_available.append(idx)
                self.sdss_catalog.loc[idx, 'image_path'] = str(jpg_path)
        
        self.sdss_catalog = self.sdss_catalog.loc[sdss_available].reset_index(drop=True)
        logger.info(f"SDSS galaxies with images: {len(self.sdss_catalog)}")
        
        logger.info("Checking DECaLS image availability...")
        decals_available = []
        for idx, row in tqdm(self.decals_catalog.iterrows(), total=len(self.decals_catalog), desc="DECaLS images"):
            iauname = row['iauname']
            if pd.isna(iauname) or not isinstance(iauname, str) or len(iauname) < 4:
                continue
                
            # DECaLS naming: J103438.28-005109.6 -> J103/J103438.28-005109.6.png
            directory = iauname[:4]  # e.g., 'J103'
            image_path = self.decals_image_dir / directory / f"{iauname}.png"
            
            if image_path.exists():
                decals_available.append(idx)
                self.decals_catalog.loc[idx, 'image_path'] = str(image_path)
        
        self.decals_catalog = self.decals_catalog.loc[decals_available].reset_index(drop=True)
        logger.info(f"DECaLS galaxies with images: {len(self.decals_catalog)}")
        
        logger.info("Checking HST image availability...")
        hst_available = []
        for idx, row in tqdm(self.hst_catalog.iterrows(), total=len(self.hst_catalog), desc="HST images"):
            zooniverse_id = row['zooniverse_id']
            
            # HST naming: cosmos_AHZ2001smd.png
            png_path = self.hst_image_dir / f"cosmos_{zooniverse_id}.png"
            fits_path = self.hst_image_dir / f"cosmos_{zooniverse_id}.fits"
            
            # Prefer PNG, fallback to FITS
            if png_path.exists():
                hst_available.append(idx)
                self.hst_catalog.loc[idx, 'image_path'] = str(png_path)
            elif fits_path.exists():
                hst_available.append(idx)
                self.hst_catalog.loc[idx, 'image_path'] = str(fits_path)
        
        self.hst_catalog = self.hst_catalog.loc[hst_available].reset_index(drop=True)
        logger.info(f"HST galaxies with images: {len(self.hst_catalog)}")
        
    def get_common_features(self):
        """Identify common morphological features across all three surveys"""
        
        # Get morphological feature columns (fractions only for consistency)
        sdss_morphology = [col for col in self.sdss_catalog.columns if '_fraction' in col]
        decals_morphology = [col for col in self.decals_catalog.columns if '_fraction' in col]
        hst_morphology = [col for col in self.hst_catalog.columns if '_fraction' in col]
        
        # Core Galaxy Zoo tasks that exist in all surveys
        core_tasks = {
            't01_smooth_or_features_a01_smooth_fraction': 'smooth',
            't01_smooth_or_features_a02_features_or_disk_fraction': 'features_disk',
            't02_edgeon_a01_yes_fraction': 'edge_on_yes',
            't02_edgeon_a02_no_fraction': 'edge_on_no',
            't03_bar_a01_bar_fraction': 'bar_yes',
            't03_bar_a02_no_bar_fraction': 'bar_no',
            't04_spiral_a01_spiral_fraction': 'spiral_yes',
            't04_spiral_a02_no_spiral_fraction': 'spiral_no',
            't05_bulge_prominence_a01_no_bulge_fraction': 'bulge_none',
            't05_bulge_prominence_a02_just_noticeable_fraction': 'bulge_noticeable',
            't05_bulge_prominence_a03_obvious_fraction': 'bulge_obvious',
            't05_bulge_prominence_a04_dominant_fraction': 'bulge_dominant',
            't06_odd_a01_yes_fraction': 'odd_yes',
            't06_odd_a02_no_fraction': 'odd_no',
        }
        
        # Find features available in all three surveys
        self.common_features_raw = []
        self.common_features_names = []
        
        for raw_name, clean_name in core_tasks.items():
            # Check if feature exists in all surveys
            in_sdss = raw_name in sdss_morphology
            
            # For DECaLS, need to check mapped name
            decals_name = SDSS_TO_DECALS.get(raw_name, raw_name)
            in_decals = decals_name in decals_morphology
            
            # HST uses same naming as SDSS
            in_hst = raw_name in hst_morphology
            
            if in_sdss and in_decals and in_hst:
                self.common_features_raw.append(raw_name)
                self.common_features_names.append(clean_name)
        
        logger.info(f"Found {len(self.common_features_raw)} overlapping morphological features across all surveys")
        logger.info(f"Features: {self.common_features_names}")
        
        # Set the output feature names (use clean names)
        self.output_features = self.common_features_names
        
    def create_mixed_dataset(self, max_galaxies):
        """Create the mixed dataset by sampling from all three surveys"""
        
        # Determine sample sizes
        if max_galaxies is None:
            # Use minimum available across surveys * 3
            min_available = min(len(self.sdss_catalog), len(self.decals_catalog), len(self.hst_catalog))
            max_galaxies = min_available * 3
        
        sdss_count = int(max_galaxies * self.survey_fractions['sdss'])
        decals_count = int(max_galaxies * self.survey_fractions['decals'])
        hst_count = max_galaxies - sdss_count - decals_count  # Ensure exact total
        
        # Ensure we don't exceed available data
        sdss_count = min(sdss_count, len(self.sdss_catalog))
        decals_count = min(decals_count, len(self.decals_catalog))
        hst_count = min(hst_count, len(self.hst_catalog))
        
        logger.info(f"Sampling {sdss_count} SDSS + {decals_count} DECaLS + {hst_count} HST galaxies")
        
        # Sample galaxies
        if self.high_quality:
            # Select galaxies with highest classification counts
            sdss_votes = self.sdss_catalog['total_votes'].fillna(0)
            sdss_sample = self.sdss_catalog.loc[sdss_votes.nlargest(sdss_count).index]
            
            decals_votes = self.decals_catalog['total_classifications'].fillna(0)
            decals_sample = self.decals_catalog.loc[decals_votes.nlargest(decals_count).index]
            
            hst_votes = self.hst_catalog['total_count'].fillna(0)
            hst_sample = self.hst_catalog.loc[hst_votes.nlargest(hst_count).index]
            
            logger.info(f"High-quality selection:")
            logger.info(f"  SDSS votes: {sdss_votes.nlargest(sdss_count).min():.0f} to {sdss_votes.nlargest(sdss_count).max():.0f}")
            logger.info(f"  DECaLS votes: {decals_votes.nlargest(decals_count).min():.0f} to {decals_votes.nlargest(decals_count).max():.0f}")
            logger.info(f"  HST votes: {hst_votes.nlargest(hst_count).min():.0f} to {hst_votes.nlargest(hst_count).max():.0f}")
        else:
            sdss_sample = self.sdss_catalog.sample(n=sdss_count, random_state=self.random_seed)
            decals_sample = self.decals_catalog.sample(n=decals_count, random_state=self.random_seed+1)
            hst_sample = self.hst_catalog.sample(n=hst_count, random_state=self.random_seed+2)
        
        # Create unified catalog
        mixed_data = []
        
        # Add SDSS galaxies
        for idx, row in sdss_sample.iterrows():
            entry = {
                'survey': 'sdss',
                'galaxy_id': row['dr7objid'],
                'ra': row['ra'],
                'dec': row['dec'],
                'image_path': row['image_path'],
                'weight': row.get('total_votes', 1.0)
            }
            
            # Extract morphological features
            for raw_feature, clean_feature in zip(self.common_features_raw, self.common_features_names):
                entry[clean_feature] = row.get(raw_feature, np.nan)
            
            mixed_data.append(entry)
        
        # Add DECaLS galaxies  
        for idx, row in decals_sample.iterrows():
            entry = {
                'survey': 'decals',
                'galaxy_id': row['iauname'],
                'ra': row['ra'],
                'dec': row['dec'],
                'image_path': row['image_path'],
                'weight': row.get('total_classifications', 1.0)
            }
            
            # Extract morphological features (using mapping)
            for raw_feature, clean_feature in zip(self.common_features_raw, self.common_features_names):
                decals_feature = SDSS_TO_DECALS.get(raw_feature, raw_feature)
                entry[clean_feature] = row.get(decals_feature, np.nan)
            
            mixed_data.append(entry)
        
        # Add HST galaxies
        for idx, row in hst_sample.iterrows():
            entry = {
                'survey': 'hst',
                'galaxy_id': row['zooniverse_id'],
                'ra': row['RA'],
                'dec': row['DEC'],
                'image_path': row['image_path'],
                'weight': row.get('total_count', 1.0)
            }
            
            # Extract morphological features (same naming as SDSS)
            for raw_feature, clean_feature in zip(self.common_features_raw, self.common_features_names):
                entry[clean_feature] = row.get(raw_feature, np.nan)
            
            mixed_data.append(entry)
        
        # Create DataFrame and shuffle
        self.mixed_catalog = pd.DataFrame(mixed_data)
        self.mixed_catalog = self.mixed_catalog.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
    def __len__(self):
        return len(self.mixed_catalog)
        
    def __getitem__(self, idx):
        """Get a single galaxy sample"""
        row = self.mixed_catalog.iloc[idx]
        
        # Load image
        image_path = row['image_path']
        try:
            if image_path.endswith('.fits'):
                # Handle FITS files
                from astropy.io import fits
                with fits.open(image_path) as hdul:
                    image_data = hdul[0].data
                # Convert to PIL Image
                image_data = np.nan_to_num(image_data, nan=0.0)
                # Normalize to 0-255
                vmin, vmax = np.percentile(image_data, [1, 99])
                if vmax > vmin:
                    image_data = np.clip((image_data - vmin) / (vmax - vmin) * 255, 0, 255)
                else:
                    image_data = np.zeros_like(image_data)
                image = Image.fromarray(image_data.astype(np.uint8), mode='L')
                # Convert to RGB
                image = image.convert('RGB')
            else:
                # Handle JPG/PNG files
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Create blank image
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Extract morphological labels
        labels = []
        for feature in self.output_features:
            value = row.get(feature, np.nan)
            if pd.isna(value):
                value = 0.0  # Use 0 for missing values
            labels.append(float(value))
        
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return {
            'image': image,
            'labels': labels,
            'survey': row['survey'],
            'galaxy_id': row['galaxy_id'],
            'weight': float(row['weight'])
        }

def create_triple_mixed_data_loaders(config, batch_size=16, num_workers=4):
    """Create data loaders for triple mixed training"""
    
    # Data paths
    sdss_catalog = "/pscratch/sd/s/sihany/galaxy-sommelier-data/catalogs/gz2_master_catalog_corrected.csv"
    decals_catalog = "/pscratch/sd/s/sihany/galaxy-sommelier-data/catalogs/gz_decals_volunteers_5_votes_non_overlap.csv"
    hst_catalog = "/pscratch/sd/s/sihany/galaxy-sommelier-data/catalogs/gz_hubble_main.csv"
    
    sdss_images = "/pscratch/sd/s/sihany/galaxy-sommelier-data/sdss"
    decals_images = "/pscratch/sd/s/sihany/galaxy-sommelier-data/decals"
    hst_images = "/pscratch/sd/s/sihany/galaxy-sommelier-data/hubble/images"
    
    # Image transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    logger.info("Creating triple mixed dataset...")
    full_dataset = TripleMixedDataset(
        sdss_catalog_path=sdss_catalog,
        decals_catalog_path=decals_catalog,
        hst_catalog_path=hst_catalog,
        sdss_image_dir=sdss_images,
        decals_image_dir=decals_images,
        hst_image_dir=hst_images,
        survey_fractions={'sdss': 0.33, 'decals': 0.33, 'hst': 0.34},
        max_galaxies=150000,  # ~50k from each survey
        transform=None,  # Will be set per split
        random_seed=42
    )
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Set transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the dataset
    train_loader, val_loader, test_loader = create_triple_mixed_data_loaders({})
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Test loading a batch
    batch = next(iter(train_loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    print(f"Surveys in batch: {set(batch['survey'])}") 