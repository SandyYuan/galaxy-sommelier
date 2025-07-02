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

# Import the standardized feature mapping
from standard_26_features import get_survey_columns, FEATURE_NAMES

logger = logging.getLogger(__name__)

class MixedSDSSDECaLSDataset(Dataset):
    """Dataset class that combines SDSS and DECaLS Galaxy Zoo data"""
    
    def __init__(self, sdss_catalog_path, decals_catalog_path, 
                 sdss_image_dir, decals_image_dir,
                 sdss_fraction=0.5, max_galaxies=None, transform=None, 
                 feature_set='sdss', random_seed=42, high_quality=False,
                 master_dataset=False):
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
            high_quality: If True, select SDSS galaxies with highest classification counts
        """
        self.sdss_image_dir = Path(sdss_image_dir)
        self.decals_image_dir = Path(decals_image_dir)
        self.transform = transform
        self.feature_set = feature_set
        self.random_seed = random_seed
        self.high_quality = high_quality
        self.master_dataset = master_dataset
        
        # Load the standardized 26-feature columns for each survey
        self.sdss_feature_columns = get_survey_columns('sdss')
        self.decals_feature_columns = get_survey_columns('decals')
        
        # Set random seed for reproducible sampling
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        quality_mode = "high-quality" if high_quality else "random"
        logger.info(f"Loading mixed dataset with standardized 26 features")
        logger.info(f"Sampling: {sdss_fraction:.1%} SDSS ({quality_mode}), {1-sdss_fraction:.1%} DECaLS")
        
        # Load and prepare catalogs
        self.load_catalogs(sdss_catalog_path, decals_catalog_path)
        
        # Create mixed dataset
        self.create_mixed_dataset(sdss_fraction, max_galaxies)
        
        if master_dataset:
            # Master dataset keeps all data for splits to use
            self.mixed_catalog_full = self.mixed_catalog.copy()
            logger.info(f"Master dataset created with {len(self.mixed_catalog_full)} galaxies")
            logger.info(f"SDSS: {(self.mixed_catalog_full['survey'] == 'sdss').sum()}")
            logger.info(f"DECaLS: {(self.mixed_catalog_full['survey'] == 'decals').sum()}")
        else:
        logger.info(f"Mixed dataset created with {len(self.mixed_catalog)} galaxies")
        logger.info(f"SDSS: {(self.mixed_catalog['survey'] == 'sdss').sum()}")
        logger.info(f"DECaLS: {(self.mixed_catalog['survey'] == 'decals').sum()}")
        
        logger.info(f"Output features: {len(self.output_features)} standardized features")
        logger.info(f"Feature verification: Each sample will output exactly {len(self.output_features)} features")
        
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
        """Verify standardized 26 features are available in both catalogs"""
        
        # Verify SDSS features
        missing_sdss = [col for col in self.sdss_feature_columns if col not in self.sdss_catalog.columns]
        if missing_sdss:
            raise ValueError(f"Missing SDSS features: {missing_sdss}")
        logger.info(f"✅ All 26 SDSS features verified")
        
        # Verify DECaLS features
        missing_decals = [col for col in self.decals_feature_columns if col not in self.decals_catalog.columns]
        if missing_decals:
            raise ValueError(f"Missing DECaLS features: {missing_decals}")
        logger.info(f"✅ All 26 DECaLS features verified")
        
        # Store the standardized feature columns
        self.common_features_sdss = self.sdss_feature_columns
        self.common_features_decals = self.decals_feature_columns
        
        # Output features are always in standardized order (26 features)
        self.output_features = FEATURE_NAMES
            
        logger.info(f"Using standardized 26-feature mapping for consistent evaluation")
        
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
        if self.high_quality:
            # Select SDSS galaxies with highest classification counts (total_votes)
            # First, ensure total_votes column exists and handle NaN values
            sdss_votes = self.sdss_catalog['total_votes'].fillna(0)
            # Sort by total_votes in descending order and take top sdss_count
            sdss_sample = self.sdss_catalog.loc[sdss_votes.nlargest(sdss_count).index]
            logger.info(f"Selected SDSS galaxies with vote counts: {sdss_votes.nlargest(sdss_count).min():.0f} to {sdss_votes.nlargest(sdss_count).max():.0f}")
        else:
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
            
            # Add standardized features in order
            features = self.extract_standardized_features(row, 'sdss')
            for i, feature_name in enumerate(FEATURE_NAMES):
                entry[feature_name] = features[i]
            
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
            
            # Add standardized features in order
            features = self.extract_standardized_features(row, 'decals')
            for i, feature_name in enumerate(FEATURE_NAMES):
                entry[feature_name] = features[i]
            
            mixed_data.append(entry)
        
        # Create DataFrame and shuffle
        self.mixed_catalog = pd.DataFrame(mixed_data)
        self.mixed_catalog = self.mixed_catalog.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        # Handle NaN values
        for feature in self.output_features:
            self.mixed_catalog[feature] = self.mixed_catalog[feature].fillna(0.0)
    
    def extract_standardized_features(self, row, survey):
        """
        Extract features in standardized order.
        
        This ensures both SDSS and DECaLS output features in the same order
        corresponding to the standard 26-feature mapping.
        """
        if survey == 'sdss':
            feature_columns = self.sdss_feature_columns
        elif survey == 'decals':
            feature_columns = self.decals_feature_columns
        else:
            raise ValueError(f"Unknown survey: {survey}")
        
        # Extract features in standardized order
        features = []
        for col in feature_columns:
            value = row.get(col, 0.0)  # Default to 0.0 if missing
            if pd.isna(value):
                value = 0.0
            features.append(float(value))
        
        return np.array(features, dtype=np.float32)
            
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


class MixedSplitDataset(Dataset):
    """Dataset for train/val/test splits that reuses master dataset's processed data"""
    
    def __init__(self, master_dataset, indices, transform=None):
        """
        Args:
            master_dataset: MixedSDSSDECaLSDataset instance with processed data
            indices: List of indices for this split
            transform: Transform to apply to images
        """
        self.master_dataset = master_dataset
        self.indices = indices
        self.transform = transform
        
        # Use the master dataset's processed catalog
        if hasattr(master_dataset, 'mixed_catalog_full'):
            self.mixed_catalog = master_dataset.mixed_catalog_full.iloc[indices].reset_index(drop=True)
        else:
            self.mixed_catalog = master_dataset.mixed_catalog.iloc[indices].reset_index(drop=True)
        
        # Copy output features from master dataset
        self.output_features = master_dataset.output_features
        
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
        }

def create_mixed_data_loaders(config, transforms_dict, sample_size=None, sdss_fraction=0.5, high_quality=False):
    """Create mixed SDSS+DECaLS data loaders"""
    
    # Config is already a dictionary, no need to load from file
    
    data_config = config['data']
    training_config = config['training']
    
    # Paths
    sdss_catalog = Path(data_config['catalogs_dir']) / 'gz2_master_catalog_corrected.csv'
    decals_catalog = Path(data_config['catalogs_dir']) / 'gz_decals_volunteers_1_and_2.csv'
    
    # Create master dataset that does all expensive filtering once
    master_dataset = MixedSDSSDECaLSDataset(
        sdss_catalog_path=sdss_catalog,
        decals_catalog_path=decals_catalog,
        sdss_image_dir=data_config['sdss_dir'],
        decals_image_dir=data_config['decals_dir'],
        sdss_fraction=sdss_fraction,
        max_galaxies=sample_size,
        transform=None,
        feature_set='sdss',
        random_seed=42,
        high_quality=high_quality,
        master_dataset=True  # This tells it to keep full data for splits
    )
    
    # Split dataset  
    dataset_size = len(master_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size) 
    test_size = dataset_size - train_size - val_size
    
    logger.info(f"Dataset splits: train={train_size}, val={val_size}, test={test_size}")
    
    # Create indices for splits
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, dataset_size))
    
    # Create efficient split datasets that reuse master's processed data
    train_dataset = MixedSplitDataset(
        master_dataset=master_dataset,
        indices=train_indices,
        transform=transforms_dict['train']
    )
    
    val_dataset = MixedSplitDataset(
        master_dataset=master_dataset,
        indices=val_indices,
        transform=transforms_dict['val']
    )
    
    test_dataset = MixedSplitDataset(
        master_dataset=master_dataset,
        indices=test_indices,
        transform=transforms_dict['test']
    )
    
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