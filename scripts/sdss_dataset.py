#!/usr/bin/env python3
"""
SDSS Dataset for Galaxy Sommelier
Handles SDSS-only Galaxy Zoo data with standardized 26-feature mapping.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py
from pathlib import Path
from PIL import Image
from astropy.io import fits
from astropy.visualization import make_lupton_rgb, MinMaxInterval, PercentileInterval
import logging
from tqdm import tqdm
import yaml

# Import the unified feature registry
import sys
sys.path.append(str(Path(__file__).parent.parent))
from feature_registry import FeatureRegistry

logger = logging.getLogger(__name__)

class SDSSDataset(Dataset):
    """Dataset class for SDSS Galaxy Zoo data with standardized 26-feature mapping"""
    
    def __init__(self, catalog_path, image_dir, transform=None, 
                 cache_dir='/pscratch/sd/s/sihany/galaxy-sommelier-data/processed',
                 use_cache=True, sample_size=None, master_dataset=False):
        
        self.catalog_path = Path(catalog_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.sample_size = sample_size
        self.master_dataset = master_dataset
        
        # Load the standardized 26-feature columns for SDSS
        self.sdss_feature_columns = FeatureRegistry.get_survey_columns('sdss')
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load catalog
        logger.info(f"Loading SDSS catalog from {self.catalog_path}")
        self.catalog = pd.read_csv(self.catalog_path)
        
        if sample_size:
            logger.info(f"Using sample of {sample_size} galaxies")
            self.catalog = self.catalog.head(sample_size)
        
        logger.info(f"SDSS dataset initialized with {len(self.catalog)} galaxies")
        logger.info(f"Using standardized 26-feature mapping")
        
        # Load image filename mapping
        self.load_filename_mapping()
        
        # Prepare morphology labels using standardized features
        self.prepare_standardized_labels()
        
        # Filter for available images
        self.filter_available_images()
        
        if master_dataset:
            # Master dataset keeps all data for splits to use
            self.catalog_full = self.catalog.copy()
            logger.info(f"Master dataset created with {len(self.catalog_full)} SDSS galaxies")
        else:
            logger.info(f"Final dataset: {len(self.catalog)} SDSS galaxies with standardized 26 features")
        
        # Create HDF5 cache for faster loading
        self.cache_file = self.cache_dir / f'sdss_cache_{len(self.catalog)}.h5'
        if self.use_cache and not self.cache_file.exists():
            self.create_cache()
    
    def load_filename_mapping(self):
        """Load the mapping from dr7objid to asset_id (image filename)"""
        mapping_path = self.catalog_path.parent / 'gz2_filename_mapping.csv'
        if mapping_path.exists():
            logger.info(f"Loading filename mapping from {mapping_path}")
            mapping_df = pd.read_csv(mapping_path)
            # Create dictionary mapping objid to asset_id
            self.objid_to_asset = dict(zip(mapping_df['objid'], mapping_df['asset_id']))
            logger.info(f"Loaded mapping for {len(self.objid_to_asset)} objects")
        else:
            logger.warning(f"Mapping file not found at {mapping_path}, will try direct objid matching")
            self.objid_to_asset = {}
    
    def prepare_standardized_labels(self):
        """Prepare labels using standardized 26-feature mapping"""
        logger.info("Preparing standardized morphology labels...")
        
        # Verify all required SDSS features are present
        missing_features = [col for col in self.sdss_feature_columns if col not in self.catalog.columns]
        if missing_features:
            raise ValueError(f"Missing SDSS features in catalog: {missing_features}")
        
        logger.info(f"✅ All 26 SDSS features verified")
        
        # Store the standardized feature columns
        self.label_columns = self.sdss_feature_columns
        
        logger.info(f"Using standardized 26-feature mapping for consistent evaluation")
        
        # Use existing total_votes or total_classifications column for weighting
        if 'total_votes' not in self.catalog.columns:
            if 'total_classifications' in self.catalog.columns:
                self.catalog['total_votes'] = self.catalog['total_classifications']
            else:
                # Fallback: sum all count columns
                count_cols = [col for col in self.catalog.columns if '_count' in col]
                if count_cols:
                    self.catalog['total_votes'] = self.catalog[count_cols].sum(axis=1)
                else:
                    self.catalog['total_votes'] = 1.0  # Default weight
    
    def filter_available_images(self):
        """Filter catalog to only include galaxies with available images"""
        logger.info("Filtering for available images...")
        
        available_indices = []
        for idx, row in self.catalog.iterrows():
            objid = row['dr7objid']
            
            # Use mapping to get asset_id if available
            if objid in self.objid_to_asset:
                asset_id = self.objid_to_asset[objid]
                image_path = self.image_dir / f'{asset_id}.jpg'
                if image_path.exists():
                    available_indices.append(idx)
            else:
                # Fallback: try direct objid matching
                fits_path = self.image_dir / f'sdss_{objid}.fits'
                jpg_path = self.image_dir / f'{objid}.jpg'
                if fits_path.exists() or jpg_path.exists():
                    available_indices.append(idx)
        
        logger.info(f"Found {len(available_indices)} galaxies with available images")
        self.catalog = self.catalog.loc[available_indices].reset_index(drop=True)
    
    def load_image(self, objid):
        """Load image using asset_id mapping or direct objid"""
        # Use mapping to get asset_id if available
        if objid in self.objid_to_asset:
            asset_id = self.objid_to_asset[objid]
            jpg_path = self.image_dir / f'{asset_id}.jpg'
            if jpg_path.exists():
                return self.load_jpg_image(jpg_path)
        
        # Fallback: try direct objid matching
        fits_path = self.image_dir / f'sdss_{objid}.fits'
        if fits_path.exists():
            return self.load_fits_image(fits_path)
        
        jpg_path = self.image_dir / f'{objid}.jpg'
        if jpg_path.exists():
            return self.load_jpg_image(jpg_path)
        
        return None
    
    def load_fits_image(self, fits_path):
        """Load and process FITS image"""
        try:
            with fits.open(fits_path) as hdul:
                data = hdul[0].data
                
                # Handle different data shapes
                if data.ndim == 2:
                    # Single band image - convert to RGB
                    rgb_data = np.stack([data] * 3, axis=-1)
                elif data.ndim == 3:
                    # Multi-band - take first 3 bands
                    rgb_data = data[:3].transpose(1, 2, 0)
                else:
                    raise ValueError(f"Unexpected data shape: {data.shape}")
                
                # Normalize to [0, 1]
                rgb_data = (rgb_data - rgb_data.min()) / (rgb_data.max() - rgb_data.min() + 1e-8)
                
                # Convert to uint8 for PIL
                rgb_data = (rgb_data * 255).astype(np.uint8)
                
                return Image.fromarray(rgb_data)
                
        except Exception as e:
            logger.error(f"Error loading {fits_path}: {e}")
            return None
    
    def load_jpg_image(self, jpg_path):
        """Load JPG image"""
        try:
            return Image.open(jpg_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading {jpg_path}: {e}")
            return None
    
    def extract_standardized_features(self, row):
        """
        Extract features in standardized order.
        
        This ensures SDSS outputs features in the same order as other datasets.
        """
        # Extract features in standardized order
        features = []
        for col in self.sdss_feature_columns:
            value = row.get(col, 0.0)  # Default to 0.0 if missing
            if pd.isna(value):
                value = 0.0
            features.append(float(value))
        
        return np.array(features, dtype=np.float32)
    
    def create_cache(self):
        """Create HDF5 cache for preprocessed images"""
        logger.info(f"Creating cache at {self.cache_file}")
        
        with h5py.File(self.cache_file, 'w') as f:
            # Create datasets
            n_samples = len(self.catalog)
            
            # Pre-allocate datasets
            images_dset = f.create_dataset('images', (n_samples, 224, 224, 3), dtype=np.uint8,
                                         compression='gzip', compression_opts=1)
            labels_dset = f.create_dataset('labels', (n_samples, len(self.label_columns)), dtype=np.float32)
            weights_dset = f.create_dataset('weights', (n_samples,), dtype=np.float32)
            objids_dset = f.create_dataset('objids', (n_samples,), dtype='S20')
            
            valid_count = 0
            
            for idx in tqdm(range(len(self.catalog)), desc="Creating cache"):
                row = self.catalog.iloc[idx]
                objid = row['dr7objid']
                
                # Load image
                image = self.load_image(objid)
                
                if image is not None:
                    # Resize image
                    image = image.resize((224, 224), Image.Resampling.LANCZOS)
                    
                    # Convert to numpy array
                    image_array = np.array(image)
                    
                    # Get standardized labels
                    labels = self.extract_standardized_features(row)
                    weight = row['total_votes']
                    
                    # Store in cache
                    images_dset[valid_count] = image_array
                    labels_dset[valid_count] = labels
                    weights_dset[valid_count] = weight
                    objids_dset[valid_count] = str(objid).encode('utf-8')
                    
                    valid_count += 1
            
            # Resize datasets to actual valid count
            if valid_count < n_samples:
                logger.info(f"Resizing cache to {valid_count} valid samples")
                images_dset.resize((valid_count, 224, 224, 3))
                labels_dset.resize((valid_count, len(self.label_columns)))
                weights_dset.resize((valid_count,))
                objids_dset.resize((valid_count,))
            
            f.attrs['n_samples'] = valid_count
            f.attrs['n_features'] = len(self.label_columns)
            
        logger.info(f"Cache created with {valid_count} samples")
    
    def __len__(self):
        if self.use_cache and self.cache_file.exists():
            with h5py.File(self.cache_file, 'r') as f:
                return f.attrs.get('n_samples', len(self.catalog))
        return len(self.catalog)
    
    def __getitem__(self, idx):
        if self.use_cache and self.cache_file.exists():
            return self._get_cached_item(idx)
        else:
            return self._get_item_from_disk(idx)
    
    def _get_cached_item(self, idx):
        """Get item from HDF5 cache (faster)"""
        with h5py.File(self.cache_file, 'r') as f:
            image_array = f['images'][idx]
            labels = f['labels'][idx]
            weight = f['weights'][idx]
            objid = f['objids'][idx].decode('utf-8')
        
        # Convert to PIL Image
        image = Image.fromarray(image_array)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default normalization
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])(image)
        
        return {
            'image': image,
            'labels': torch.tensor(labels, dtype=torch.float32),
            'weight': torch.tensor(weight, dtype=torch.float32),
            'objid': objid
        }
    
    def _get_item_from_disk(self, idx):
        """Get item directly from disk (slower)"""
        row = self.catalog.iloc[idx]
        objid = row['dr7objid']
        
        # Load image
        image = self.load_image(objid)
        
        if image is None:
            # Return dummy data if image can't be loaded
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        # Resize
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default normalization
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])(image)
        
        # Get standardized labels
        labels = self.extract_standardized_features(row)
        weight = row['total_votes']
        
        return {
            'image': image,
            'labels': torch.tensor(labels, dtype=torch.float32),
            'weight': torch.tensor(weight, dtype=torch.float32),
            'objid': str(objid)
        }


class SDSSSplitDataset(Dataset):
    """Dataset for train/val/test splits that reuses master dataset's processed data"""
    
    def __init__(self, master_dataset, indices, transform=None):
        """
        Args:
            master_dataset: SDSSDataset instance with processed data
            indices: List of indices for this split
            transform: Transform to apply to images
        """
        self.master_dataset = master_dataset
        self.indices = indices
        self.transform = transform
        
        # Use the master dataset's processed catalog
        if hasattr(master_dataset, 'catalog_full'):
            self.catalog = master_dataset.catalog_full.iloc[indices].reset_index(drop=True)
        else:
            self.catalog = master_dataset.catalog.iloc[indices].reset_index(drop=True)
        
        # Copy attributes from master dataset
        self.label_columns = master_dataset.label_columns
        self.objid_to_asset = master_dataset.objid_to_asset
        self.image_dir = master_dataset.image_dir
        
    def __len__(self):
        return len(self.catalog)
    
    def __getitem__(self, idx):
        row = self.catalog.iloc[idx]
        objid = row['dr7objid']
        
        # Load image using master dataset's methods
        image = self.master_dataset.load_image(objid)
        
        if image is None:
            # Return dummy data if image can't be loaded
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        # Resize
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default normalization
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])(image)
        
        # Get standardized labels
        labels = self.master_dataset.extract_standardized_features(row)
        weight = row['total_votes']
        
        return {
            'image': image,
            'labels': torch.tensor(labels, dtype=torch.float32),
            'weight': torch.tensor(weight, dtype=torch.float32),
            'objid': str(objid)
        }

def get_transforms(mode='train'):
    """Get image transforms for training or validation"""
    
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_data_loaders(config, transforms_dict, sample_size=None):
    """Create SDSS train/val/test data loaders with standardized 26-feature mapping"""
    
    data_config = config['data']
    training_config = config['training']
    
    # Load dataset
    catalog_name = data_config.get('catalog_name', 'gz2_master_catalog_corrected.csv')
    catalog_path = Path(data_config['catalogs_dir']) / catalog_name
    if not catalog_path.exists():
        # Fallback to default catalog
        catalog_path = Path(data_config['catalogs_dir']) / 'gz2_master_catalog.csv'
        if not catalog_path.exists():
            # Try sample catalog
            sample_catalogs = list(Path(data_config['catalogs_dir']).glob('gz2_sample_*.csv'))
            if sample_catalogs:
                catalog_path = sample_catalogs[0]
                logger.info(f"Using sample catalog: {catalog_path}")
            else:
                raise FileNotFoundError("No Galaxy Zoo catalog found")
    
    # Create master dataset that does all expensive filtering once
    master_dataset = SDSSDataset(
        catalog_path=catalog_path,
        image_dir=data_config['sdss_dir'],
        transform=None,
        cache_dir=data_config['processed_dir'],
        use_cache=False,  # Disable caching for faster startup
        sample_size=sample_size,
        master_dataset=True  # This tells it to keep full data for splits
    )
    
    # Split dataset
    dataset_size = len(master_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    logger.info(f"SDSS dataset splits: train={train_size}, val={val_size}, test={test_size}")
    
    # Create indices for splits
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, dataset_size))
    
    # Create efficient split datasets that reuse master's processed data
    train_dataset = SDSSSplitDataset(
        master_dataset=master_dataset,
        indices=train_indices,
        transform=transforms_dict['train']
    )
    
    val_dataset = SDSSSplitDataset(
        master_dataset=master_dataset,
        indices=val_indices,
        transform=transforms_dict['val']
    )
    
    test_dataset = SDSSSplitDataset(
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

def test_data_loading():
    """Test SDSS data loading functionality"""
    logging.basicConfig(level=logging.INFO)
    
    # Test dataset creation
    catalog_path = "/pscratch/sd/s/sihany/galaxy-sommelier-data/catalogs/gz2_sample_100.csv"
    image_dir = "/pscratch/sd/s/sihany/galaxy-sommelier-data/sdss"
    
    if not Path(catalog_path).exists():
        logger.error(f"Test catalog not found: {catalog_path}")
        return
    
    logger.info("Testing SDSS dataset creation...")
    dataset = SDSSDataset(
        catalog_path=catalog_path,
        image_dir=image_dir,
        transform=get_transforms('train'),
        sample_size=10
    )
    
    logger.info(f"SDSS dataset created with {len(dataset)} samples")
    
    # Test data loading
    if len(dataset) > 0:
        sample = dataset[0]
        logger.info(f"Sample keys: {sample.keys()}")
        logger.info(f"Image shape: {sample['image'].shape}")
        logger.info(f"Labels shape: {sample['labels'].shape}")
        logger.info(f"Labels range: [{sample['labels'].min():.4f}, {sample['labels'].max():.4f}]")
        logger.info(f"Weight: {sample['weight']:.2f}")
        logger.info(f"Object ID: {sample['objid']}")
        logger.info(f"✅ SDSS dataset outputs exactly {sample['labels'].shape[0]} standardized features")
    
    logger.info("SDSS data loading test completed!")

if __name__ == "__main__":
    test_data_loading() 