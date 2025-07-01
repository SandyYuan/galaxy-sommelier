#!/usr/bin/env python3
"""
Triple Mixed Dataset: SDSS + DECaLS + HST
Combines three galaxy surveys for cross-survey morphology training.
Uses standardized 26-feature mapping for consistent evaluation.
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

class TripleMixedDataset(Dataset):
    """
    Dataset combining SDSS, DECaLS, and HST galaxies.
    
    Uses standardized 26-feature mapping to ensure consistent
    output order across all surveys for fair model comparison.
    """
    
    def __init__(self, sdss_catalog_path, decals_catalog_path, hst_catalog_path,
                 sdss_image_dir, decals_image_dir, hst_image_dir,
                 survey_fractions={'sdss': 0.33, 'decals': 0.33, 'hst': 0.34},
                 max_galaxies=150000, transform=None, split='train', 
                 train_ratio=0.8, val_ratio=0.1, random_seed=42):
        
        self.transform = transform
        self.split = split
        self.random_seed = random_seed
        
        # Load the standardized 26-feature columns for each survey
        self.sdss_feature_columns = get_survey_columns('sdss')
        self.decals_feature_columns = get_survey_columns('decals')
        self.hst_feature_columns = get_survey_columns('hst')
        
        print(f"Initializing TripleMixedDataset with standardized 26 features...")
        print(f"Survey fractions: {survey_fractions}")
        print(f"Max galaxies: {max_galaxies}")
        print(f"Feature mapping: {len(FEATURE_NAMES)} features in standard order")
        
        # Load catalogs
        self.load_catalogs(sdss_catalog_path, decals_catalog_path, hst_catalog_path)
        
        # Set image directories
        self.sdss_image_dir = Path(sdss_image_dir)
        self.decals_image_dir = Path(decals_image_dir)
        self.hst_image_dir = Path(hst_image_dir)
        
        # Create mixed dataset
        self.create_mixed_dataset(survey_fractions, max_galaxies)
        
        # Split dataset
        self.split_dataset(train_ratio, val_ratio)
        
        print(f"Dataset created: {len(self.data)} {split} samples")
        print(f"  SDSS: {sum(1 for x in self.data if x['survey'] == 'sdss')} samples")
        print(f"  DECaLS: {sum(1 for x in self.data if x['survey'] == 'decals')} samples")
        print(f"  HST: {sum(1 for x in self.data if x['survey'] == 'hst')} samples")
    
    def load_catalogs(self, sdss_path, decals_path, hst_path):
        """Load and validate feature columns in all catalogs."""
        print("Loading catalogs and validating standardized features...")
        
        # Load SDSS catalog
        print(f"Loading SDSS catalog: {sdss_path}")
        self.sdss_catalog = pd.read_csv(sdss_path)
        print(f"  SDSS: {len(self.sdss_catalog)} galaxies")
        
        # Verify SDSS features
        missing_sdss = [col for col in self.sdss_feature_columns if col not in self.sdss_catalog.columns]
        if missing_sdss:
            raise ValueError(f"Missing SDSS features: {missing_sdss}")
        print(f"  ✅ All 26 SDSS features verified")
        
        # Load DECaLS catalog  
        print(f"Loading DECaLS catalog: {decals_path}")
        self.decals_catalog = pd.read_csv(decals_path)
        print(f"  DECaLS: {len(self.decals_catalog)} galaxies")
        
        # Verify DECaLS features
        missing_decals = [col for col in self.decals_feature_columns if col not in self.decals_catalog.columns]
        if missing_decals:
            raise ValueError(f"Missing DECaLS features: {missing_decals}")
        print(f"  ✅ All 26 DECaLS features verified")
        
        # Load HST catalog
        print(f"Loading HST catalog: {hst_path}")
        self.hst_catalog = pd.read_csv(hst_path)
        print(f"  HST: {len(self.hst_catalog)} galaxies")
        
        # Verify HST features
        missing_hst = [col for col in self.hst_feature_columns if col not in self.hst_catalog.columns]
        if missing_hst:
            raise ValueError(f"Missing HST features: {missing_hst}")
        print(f"  ✅ All 26 HST features verified")
    
    def create_mixed_dataset(self, survey_fractions, max_galaxies):
        """Create mixed dataset with specified survey fractions."""
        
        # Calculate sample sizes
        sdss_size = int(max_galaxies * survey_fractions['sdss'])
        decals_size = int(max_galaxies * survey_fractions['decals'])
        hst_size = int(max_galaxies * survey_fractions['hst'])
        
        print(f"Target sample sizes:")
        print(f"  SDSS: {sdss_size}")
        print(f"  DECaLS: {decals_size}")
        print(f"  HST: {hst_size}")
        
        # Sample from each catalog
        np.random.seed(self.random_seed)
        
        # SDSS sampling
        sdss_sample_size = min(sdss_size, len(self.sdss_catalog))
        sdss_indices = np.random.choice(len(self.sdss_catalog), sdss_sample_size, replace=False)
        sdss_sample = self.sdss_catalog.iloc[sdss_indices].reset_index(drop=True)
        
        # DECaLS sampling
        decals_sample_size = min(decals_size, len(self.decals_catalog))
        decals_indices = np.random.choice(len(self.decals_catalog), decals_sample_size, replace=False)
        decals_sample = self.decals_catalog.iloc[decals_indices].reset_index(drop=True)
        
        # HST sampling
        hst_sample_size = min(hst_size, len(self.hst_catalog))
        hst_indices = np.random.choice(len(self.hst_catalog), hst_sample_size, replace=False)
        hst_sample = self.hst_catalog.iloc[hst_indices].reset_index(drop=True)
        
        print(f"Actual sample sizes:")
        print(f"  SDSS: {len(sdss_sample)}")
        print(f"  DECaLS: {len(decals_sample)}")
        print(f"  HST: {len(hst_sample)}")
        
        # Create unified dataset
        self.full_data = []
        
        # Add SDSS samples
        for idx, row in sdss_sample.iterrows():
            self.full_data.append({
                'survey': 'sdss',
                'catalog_idx': idx,
                'objid': row.get('dr7objid', f'sdss_{idx}'),
                'features': self.extract_standardized_features(row, 'sdss')
            })
        
        # Add DECaLS samples
        for idx, row in decals_sample.iterrows():
            self.full_data.append({
                'survey': 'decals',
                'catalog_idx': idx,
                'objid': row.get('iauname', f'decals_{idx}'),
                'features': self.extract_standardized_features(row, 'decals')
            })
        
        # Add HST samples
        for idx, row in hst_sample.iterrows():
            self.full_data.append({
                'survey': 'hst',
                'catalog_idx': idx,
                'objid': row.get('iauname', f'hst_{idx}'),
                'features': self.extract_standardized_features(row, 'hst')
            })
        
        # Store samples for image loading
        self.sdss_sample = sdss_sample
        self.decals_sample = decals_sample
        self.hst_sample = hst_sample
        
        print(f"Mixed dataset created: {len(self.full_data)} total samples")
    
    def extract_standardized_features(self, row, survey):
        """
        Extract features in standardized order.
        
        This ensures all surveys output features in the same order
        corresponding to the standard 26-feature mapping.
        """
        if survey == 'sdss':
            feature_columns = self.sdss_feature_columns
        elif survey == 'decals':
            feature_columns = self.decals_feature_columns
        elif survey == 'hst':
            feature_columns = self.hst_feature_columns
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
    
    def split_dataset(self, train_ratio, val_ratio):
        """Split dataset into train/val/test."""
        
        # Shuffle the full dataset
        np.random.seed(self.random_seed)
        indices = np.random.permutation(len(self.full_data))
        
        # Calculate split sizes
        n_total = len(self.full_data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Create splits
        if self.split == 'train':
            selected_indices = indices[:n_train]
        elif self.split == 'val':
            selected_indices = indices[n_train:n_train+n_val]
        elif self.split == 'test':
            selected_indices = indices[n_train+n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        # Filter data for this split
        self.data = [self.full_data[i] for i in selected_indices]
    
    def load_image(self, survey, catalog_idx, objid):
        """Load image based on survey type."""
        
        if survey == 'sdss':
            return self.load_sdss_image(catalog_idx, objid)
        elif survey == 'decals':
            return self.load_decals_image(catalog_idx, objid)
        elif survey == 'hst':
            return self.load_hst_image(catalog_idx, objid)
        else:
            raise ValueError(f"Unknown survey: {survey}")
    
    def load_sdss_image(self, catalog_idx, objid):
        """Load SDSS image."""
        # Use the same logic as existing SDSS dataset
        row = self.sdss_sample.iloc[catalog_idx]
        
        # Try different filename patterns
        possible_names = [
            f"{objid}.jpg",
            f"{row.get('dr7objid', objid)}.jpg"
        ]
        
        for filename in possible_names:
            image_path = self.sdss_image_dir / filename
            if image_path.exists():
                try:
                    image = Image.open(image_path).convert('RGB')
                    return image
                except Exception as e:
                    continue
        
        # Return black image if not found
        return Image.new('RGB', (256, 256), color='black')
    
    def load_decals_image(self, catalog_idx, objid):
        """Load DECaLS image."""
        row = self.decals_sample.iloc[catalog_idx]
        
        # DECaLS uses iauname for filenames
        filename = f"{row.get('iauname', objid)}.jpg"
        image_path = self.decals_image_dir / filename
        
        if image_path.exists():
            try:
                image = Image.open(image_path).convert('RGB')
                return image
            except Exception as e:
                pass
        
        # Return black image if not found
        return Image.new('RGB', (256, 256), color='black')
    
    def load_hst_image(self, catalog_idx, objid):
        """Load HST image."""
        row = self.hst_sample.iloc[catalog_idx]
        
        # HST images are PNG files in subdirectories
        subfolder = row.get('subfolder', 'unknown')
        filename = row.get('filename', f"{objid}.png")
        
        image_path = self.hst_image_dir / subfolder / filename
        
        if image_path.exists():
            try:
                image = Image.open(image_path).convert('RGB')
                return image
            except Exception as e:
                pass
        
        # Return black image if not found
        return Image.new('RGB', (256, 256), color='black')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Load image
        image = self.load_image(
            sample['survey'], 
            sample['catalog_idx'], 
            sample['objid']
        )
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get features (already in standardized order)
        features = sample['features']
        
        return {
            'image': image,
            'labels': torch.tensor(features, dtype=torch.float32),
            'survey': sample['survey'],
            'objid': sample['objid']
        }

def get_triple_mixed_data_loaders(config, transforms_dict):
    """
    Create data loaders for triple mixed dataset.
    
    Returns train, validation, and test loaders using standardized
    26-feature mapping.
    """
    from torch.utils.data import DataLoader
    
    # Extract paths from config
    sdss_catalog = config['data']['sdss_catalog_path']
    decals_catalog = config['data']['decals_catalog_path']
    hst_catalog = config['data']['hst_catalog_path']
    
    sdss_images = config['data']['sdss_dir']
    decals_images = config['data']['decals_dir']
    hst_images = config['data']['hst_dir']
    
    # Get dataset parameters
    mixed_config = config['mixed_data']
    survey_fractions = mixed_config['survey_fractions']
    max_galaxies = mixed_config['max_galaxies']
    
    batch_size = config['training']['batch_size']
    num_workers = config.get('num_workers', 4)
    
    print(f"Creating triple mixed data loaders with standardized 26 features...")
    
    # Create datasets
    train_dataset = TripleMixedDataset(
        sdss_catalog, decals_catalog, hst_catalog,
        sdss_images, decals_images, hst_images,
        survey_fractions=survey_fractions,
        max_galaxies=max_galaxies,
        transform=transforms_dict['train'],
        split='train'
    )
    
    val_dataset = TripleMixedDataset(
        sdss_catalog, decals_catalog, hst_catalog,
        sdss_images, decals_images, hst_images,
        survey_fractions=survey_fractions,
        max_galaxies=max_galaxies,
        transform=transforms_dict['val'],
        split='val'
    )
    
    test_dataset = TripleMixedDataset(
        sdss_catalog, decals_catalog, hst_catalog,
        sdss_images, decals_images, hst_images,
        survey_fractions=survey_fractions,
        max_galaxies=max_galaxies,
        transform=transforms_dict['test'],
        split='test'
    )
    
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
    
    print(f"Data loaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the dataset
    train_loader, val_loader, test_loader = get_triple_mixed_data_loaders({})
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Test loading a batch
    batch = next(iter(train_loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    print(f"Surveys in batch: {set(batch['survey'])}") 