#!/usr/bin/env python3
"""
Mixed SDSS + DECaLS Dataset for Galaxy Sommelier
Combines SDSS and DECaLS galaxy data using feature mapping.
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

class MixedSDSSDECaLSDataset(Dataset):
    """Dataset class that combines SDSS and DECaLS Galaxy Zoo data"""
    
    def __init__(self, sdss_catalog_path, decals_catalog_path, 
                 sdss_image_dir, decals_image_dir,
                 sdss_fraction=0.5, max_galaxies=None, transform=None, 
                 feature_set='sdss', random_seed=42):
        """
        Args:
            sdss_catalog_path: Path to SDSS catalog CSV
            decals_catalog_path: Path to DECaLS catalog CSV  
            sdss_image_dir: Directory containing SDSS images
            decals_image_dir: Directory containing DECaLS images
            sdss_fraction: Fraction of data to come from SDSS (0.5 = 50/50 split)
            max_galaxies: Total number of galaxies to use (None = use all available)
            transform: Image transforms to apply
            feature_set: 'sdss' or 'decals' - which feature naming convention to use
            random_seed: Random seed for reproducible sampling
        """
        self.sdss_image_dir = Path(sdss_image_dir)
        self.decals_image_dir = Path(decals_image_dir)
        self.transform = transform
        self.feature_set = feature_set
        self.random_seed = random_seed
        
        # Set random seed for reproducible sampling
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        logger.info(f"Loading mixed dataset with {sdss_fraction:.1%} SDSS, {1-sdss_fraction:.1%} DECaLS")
        
        # Load and prepare catalogs
        self.load_catalogs(sdss_catalog_path, decals_catalog_path)
        
        # Create mixed dataset
        self.create_mixed_dataset(sdss_fraction, max_galaxies)
        
        logger.info(f"Mixed dataset created with {len(self.mixed_catalog)} galaxies")
        logger.info(f"SDSS: {(self.mixed_catalog['survey'] == 'sdss').sum()}")
        logger.info(f"DECaLS: {(self.mixed_catalog['survey'] == 'decals').sum()}")
        
    def load_catalogs(self, sdss_catalog_path, decals_catalog_path):
        """Load and prepare SDSS and DECaLS catalogs"""
        
        # Load SDSS catalog
        logger.info(f"Loading SDSS catalog from {sdss_catalog_path}")
        self.sdss_catalog = pd.read_csv(sdss_catalog_path)
        logger.info(f"SDSS catalog: {len(self.sdss_catalog)} galaxies")
        
        # Load DECaLS catalog  
        logger.info(f"Loading DECaLS catalog from {decals_catalog_path}")
        self.decals_catalog = pd.read_csv(decals_catalog_path)
        logger.info(f"DECaLS catalog: {len(self.decals_catalog)} galaxies")
        
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
        
    def get_common_features(self):
        """Identify common morphological features between SDSS and DECaLS"""
        
        # Get morphological feature columns (fractions and debiased)
        sdss_morphology = [col for col in self.sdss_catalog.columns 
                          if '_fraction' in col or '_debiased' in col]
        decals_morphology = [col for col in self.decals_catalog.columns 
                            if '_fraction' in col or '_debiased' in col]
        
        # Find overlapping features using our mapping
        self.common_features_sdss = []
        self.common_features_decals = []
        
        for sdss_col, decals_col in SDSS_TO_DECALS.items():
            if sdss_col in sdss_morphology and decals_col in decals_morphology:
                self.common_features_sdss.append(sdss_col)
                self.common_features_decals.append(decals_col)
        
        logger.info(f"Found {len(self.common_features_sdss)} overlapping morphological features")
        
        # Set the output feature names based on chosen convention
        if self.feature_set == 'sdss':
            self.output_features = self.common_features_sdss
        else:
            self.output_features = self.common_features_decals
            
        logger.info(f"Using {self.feature_set.upper()} feature naming convention")
        
    def create_mixed_dataset(self, sdss_fraction, max_galaxies):
        """Create the mixed dataset by sampling from both surveys"""
        
        # Determine sample sizes
        if max_galaxies is None:
            # Use all available galaxies
            max_galaxies = min(len(self.sdss_catalog), len(self.decals_catalog)) * 2
        
        sdss_count = int(max_galaxies * sdss_fraction)
        decals_count = max_galaxies - sdss_count
        
        # Ensure we don't exceed available data
        sdss_count = min(sdss_count, len(self.sdss_catalog))
        decals_count = min(decals_count, len(self.decals_catalog))
        
        logger.info(f"Sampling {sdss_count} SDSS + {decals_count} DECaLS galaxies")
        
        # Sample galaxies
        sdss_sample = self.sdss_catalog.sample(n=sdss_count, random_state=self.random_seed)
        decals_sample = self.decals_catalog.sample(n=decals_count, random_state=self.random_seed)
        
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
            
            # Add morphological features (using SDSS names)
            for feature in self.common_features_sdss:
                if self.feature_set == 'sdss':
                    entry[feature] = row[feature]
                else:
                    # Convert to DECaLS naming
                    decals_feature = SDSS_TO_DECALS[feature]
                    entry[decals_feature] = row[feature]
            
            mixed_data.append(entry)
        
        # Add DECaLS galaxies
        for idx, row in decals_sample.iterrows():
            entry = {
                'survey': 'decals',
                'galaxy_id': row['iauname'],
                'ra': row['ra'], 
                'dec': row['dec'],
                'image_path': row['image_path'],
                'weight': row.get('smooth-or-featured_total-votes', 1.0)
            }
            
            # Add morphological features
            for i, decals_feature in enumerate(self.common_features_decals):
                if self.feature_set == 'decals':
                    entry[decals_feature] = row[decals_feature]
                else:
                    # Convert to SDSS naming
                    sdss_feature = self.common_features_sdss[i]
                    entry[sdss_feature] = row[decals_feature]
            
            mixed_data.append(entry)
        
        # Create DataFrame and shuffle
        self.mixed_catalog = pd.DataFrame(mixed_data)
        self.mixed_catalog = self.mixed_catalog.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        # Handle NaN values
        for feature in self.output_features:
            self.mixed_catalog[feature] = self.mixed_catalog[feature].fillna(0.0)
            
    def __len__(self):
        return len(self.mixed_catalog)
    
    def __getitem__(self, idx):
        row = self.mixed_catalog.iloc[idx]
        
        # Load image
        image_path = row['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get morphological labels
        labels = torch.tensor([row[feature] for feature in self.output_features], dtype=torch.float32)
        
        # Get weight
        weight = torch.tensor(row['weight'], dtype=torch.float32)
        
        return {
            'image': image,
            'labels': labels,
            'weight': weight
            # Note: Removing string fields (galaxy_id, survey) to avoid collate issues
            # Could add these back with a custom collate function if needed
        }

def create_mixed_data_loaders(config_path, sdss_fraction=0.5, sample_size=None):
    """Create mixed SDSS+DECaLS data loaders"""
    
    import yaml
    from data_processing import get_transforms
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    training_config = config['training']
    
    # Paths
    sdss_catalog = Path(data_config['catalogs_dir']) / 'gz2_master_catalog_corrected.csv'
    decals_catalog = Path(data_config['catalogs_dir']) / 'gz_decals_volunteers_1_and_2.csv'
    
    # Create mixed dataset
    full_dataset = MixedSDSSDECaLSDataset(
        sdss_catalog_path=sdss_catalog,
        decals_catalog_path=decals_catalog,
        sdss_image_dir=data_config['sdss_dir'],
        decals_image_dir=data_config['decals_dir'],
        sdss_fraction=sdss_fraction,
        max_galaxies=sample_size,
        transform=None,  # Will be set per split
        feature_set='sdss'  # Use SDSS feature naming
    )
    
    # Split dataset  
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size) 
    test_size = dataset_size - train_size - val_size
    
    logger.info(f"Dataset splits: train={train_size}, val={val_size}, test={test_size}")
    
    # Create splits
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, dataset_size))
    
    # Create datasets with transforms
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Set transforms for training
    full_dataset.transform = get_transforms('train')
    
    # Create separate dataset instances for validation with different transforms
    val_dataset_with_transform = MixedSDSSDECaLSDataset(
        sdss_catalog_path=sdss_catalog,
        decals_catalog_path=decals_catalog,
        sdss_image_dir=data_config['sdss_dir'],
        decals_image_dir=data_config['decals_dir'],
        sdss_fraction=sdss_fraction,
        max_galaxies=sample_size,
        transform=get_transforms('val'),
        feature_set='sdss',
        random_seed=42  # Same seed for consistent splits
    )
    val_dataset = torch.utils.data.Subset(val_dataset_with_transform, val_indices)
    
    test_dataset_with_transform = MixedSDSSDECaLSDataset(
        sdss_catalog_path=sdss_catalog,
        decals_catalog_path=decals_catalog,
        sdss_image_dir=data_config['sdss_dir'],
        decals_image_dir=data_config['decals_dir'],
        sdss_fraction=sdss_fraction,
        max_galaxies=sample_size,
        transform=get_transforms('val'),
        feature_set='sdss',
        random_seed=42  # Same seed for consistent splits
    )
    test_dataset = torch.utils.data.Subset(test_dataset_with_transform, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )
    
    return train_loader, val_loader, test_loader 