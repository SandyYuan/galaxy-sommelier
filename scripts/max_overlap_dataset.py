#!/usr/bin/env python3
"""
Max Overlap SDSS + DECaLS Dataset for Galaxy Sommelier
Pytorch Dataset loader for the "maximum overlap" training catalogs.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import logging
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from sdss_decals_feature_mapping import SDSS_TO_DECALS

logger = logging.getLogger(__name__)

class MaxOverlapDataset(Dataset):
    """
    Dataset class that combines the pre-computed 'maximum overlap'
    SDSS and DECaLS training catalogs.
    """
    
    def __init__(self, sdss_catalog_path, decals_catalog_path, 
                 sdss_image_dir, decals_image_dir,
                 transform=None, feature_set='sdss'):
        """
        Args:
            sdss_catalog_path: Path to the matched SDSS training catalog CSV
            decals_catalog_path: Path to the matched DECaLS training catalog CSV  
            sdss_image_dir: Directory containing SDSS images
            decals_image_dir: Directory containing DECaLS images
            transform: Image transforms to apply
            feature_set: 'sdss' or 'decals' feature naming convention to use
        """
        self.sdss_image_dir = Path(sdss_image_dir)
        self.decals_image_dir = Path(decals_image_dir)
        self.transform = transform
        self.feature_set = feature_set
        
        logger.info("Loading Max Overlap dataset...")
        
        # Load pre-computed catalogs
        sdss_cat = pd.read_csv(sdss_catalog_path)
        decals_cat = pd.read_csv(decals_catalog_path)
        
        sdss_cat['survey'] = 'sdss'
        decals_cat['survey'] = 'decals'
        
        # Combine the two catalogs into one
        self.catalog = pd.concat([sdss_cat, decals_cat], ignore_index=True)
        
        # Add image paths and get common features
        self.prepare_data()
        
        logger.info(f"Max Overlap dataset created with {len(self.catalog)} total galaxies.")
        logger.info(f"SDSS: {(self.catalog['survey'] == 'sdss').sum()}, DECaLS: {(self.catalog['survey'] == 'decals').sum()}")

    def prepare_data(self):
        """Adds image paths to catalog and identifies common features."""
        
        logger.info("Assigning image paths...")
        self.catalog['image_path'] = self.catalog.apply(self.get_image_path, axis=1)
        
        # Drop galaxies where image path could not be found
        original_count = len(self.catalog)
        self.catalog = self.catalog.dropna(subset=['image_path']).reset_index(drop=True)
        if len(self.catalog) < original_count:
            logger.warning(f"Dropped {original_count - len(self.catalog)} galaxies due to missing image files.")

        self.get_common_features()

    def get_image_path(self, row):
        """Gets the full path to a galaxy's image file."""
        if row['survey'] == 'sdss':
            objid = row['dr7objid']
            # Try different SDSS image naming conventions
            for name_template in ["{objid}.jpg", "sdss_{objid}.jpg"]:
                path = self.sdss_image_dir / name_template.format(objid=objid)
                if path.exists():
                    return str(path)
        elif row['survey'] == 'decals':
            iauname = row.get('iauname')
            if pd.notna(iauname) and isinstance(iauname, str) and len(iauname) >= 4:
                directory = iauname[:4]
                for ext in ['.png', '.jpg']:
                    path = self.decals_image_dir / directory / (iauname + ext)
                    if path.exists():
                        return str(path)
        return None

    def get_common_features(self):
        """Identifies common morphological features between SDSS and DECaLS."""
        sdss_morphology = [col for col in self.catalog.columns if '_fraction' in col or '_debiased' in col]
        
        self.common_features_sdss = []
        self.common_features_decals = []
        
        for sdss_col, decals_col in SDSS_TO_DECALS.items():
            if sdss_col in sdss_morphology and decals_col in self.catalog.columns:
                self.common_features_sdss.append(sdss_col)
                self.common_features_decals.append(decals_col)
        
        if self.feature_set == 'sdss':
            self.output_features = self.common_features_sdss
        else: # 'decals'
            self.output_features = self.common_features_decals
            # We need to rename the columns in the catalog to match DECaLS names
            rename_map = {sdss_col: decals_col for sdss_col, decals_col in SDSS_TO_DECALS.items() if sdss_col in self.catalog.columns}
            self.catalog.rename(columns=rename_map, inplace=True)
            
        logger.info(f"Found {len(self.output_features)} overlapping morphological features.")

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, idx):
        row = self.catalog.iloc[idx]
        
        # Load image
        try:
            image = Image.open(row['image_path']).convert('RGB')
        except FileNotFoundError:
            logger.error(f"Image not found at {row['image_path']}. Skipping.")
            # Return a dummy item or handle this case appropriately
            return self.__getitem__((idx + 1) % len(self)) 

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        # Get labels
        labels = torch.tensor(row[self.output_features].values.astype(float), dtype=torch.float32)
        
        # Get sample weight (if available)
        weight = torch.tensor(row.get('weight', 1.0), dtype=torch.float32)
        
        return image, labels, weight 