#!/usr/bin/env python3
"""
Create a training dataset that is balanced between SDSS and DECaLS,
anchored by the set of galaxies that appear in both surveys.

This script first determines the total number of DECaLS galaxies that have
available image files, replicating the data preparation steps of previous
experiments. This number becomes the target size for each survey's
training set.

The DECaLS training set consists of all galaxies with available images.

The SDSS training set is constructed to be the same size. It includes all
SDSS galaxies that were cross-matched with DECaLS objects, with the
remainder of the set filled by a random sample of non-matched SDSS galaxies.

This ensures the final datasets are directly comparable to previous runs
while testing the hypothesis that training on matched multi-survey objects
is beneficial.
"""

import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

# --- Configuration ---
# Base path for all input and output data
BASE_DATA_PATH = Path('/pscratch/sd/s/sihany/galaxy-sommelier-data/')

# Input paths
DECAL_CATALOG_PATH = BASE_DATA_PATH / 'catalogs/gz_decals_volunteers_1_and_2.csv'
SDSS_CATALOG_PATH = BASE_DATA_PATH / 'catalogs/gz2_master_catalog_corrected.csv'
MATCHED_PAIRS_PATH = BASE_DATA_PATH / 'catalogs/gz_sdss_decals_matched_catalog.csv'
DECAL_IMAGE_DIR = BASE_DATA_PATH / 'decals'

# Output paths
OUTPUT_DIR = BASE_DATA_PATH / 'catalogs/max_overlap_catalogs'
OUTPUT_SDSS_CATALOG = OUTPUT_DIR / 'sdss_max_overlap_training_catalog.csv'
OUTPUT_DECAL_CATALOG = OUTPUT_DIR / 'decals_max_overlap_training_catalog.csv'


def check_decals_image_availability(catalog, image_dir):
    """Filters the DECaLS catalog to include only galaxies with available images."""
    logging.info(f"Checking DECaLS image availability in {image_dir}...")
    image_dir = Path(image_dir)
    available_indices = []
    
    for idx, row in tqdm(catalog.iterrows(), total=len(catalog), desc="Checking DECaLS images"):
        iauname = row.get('iauname')
        if pd.isna(iauname) or not isinstance(iauname, str) or len(iauname) < 4:
            continue
            
        # DECaLS naming convention: J103438.28-005109.6 -> J103/J103438.28-005109.6.png
        # Note: The images from the bulk download might be in .jpg format
        directory = iauname[:4]
        png_path = image_dir / directory / f"{iauname}.png"
        jpg_path = image_dir / directory / f"{iauname}.jpg"
        
        if png_path.exists() or jpg_path.exists():
            available_indices.append(idx)
            
    filtered_catalog = catalog.loc[available_indices].reset_index(drop=True)
    logging.info(f"Found {len(filtered_catalog)} DECaLS galaxies with available images.")
    return filtered_catalog

def main():
    """
    Main function to create the matched training catalogs.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load catalogs
    logging.info("Loading catalogs...")
    decals_cat = pd.read_csv(DECAL_CATALOG_PATH, low_memory=False)
    sdss_cat = pd.read_csv(SDSS_CATALOG_PATH, low_memory=False)
    matched_pairs = pd.read_csv(MATCHED_PAIRS_PATH)
    logging.info(f"Loaded {len(decals_cat)} DECaLS, {len(sdss_cat)} SDSS, and {len(matched_pairs)} matched pairs.")

    # --------------------------------------------------------------------------
    # Determine the target dataset size by replicating the original data cuts
    # --------------------------------------------------------------------------
    decals_with_images = check_decals_image_availability(decals_cat, DECAL_IMAGE_DIR)
    TARGET_SIZE_PER_SURVEY = len(decals_with_images)
    logging.info(f"Target size per survey set to {TARGET_SIZE_PER_SURVEY} based on available DECaLS images.")

    # The DECaLS training set is simply all DECaLS galaxies that have images
    decals_training_catalog = decals_with_images
    
    # --------------------------------------------------------------------------
    # Create the SDSS training catalog
    # It will contain:
    # 1. All SDSS galaxies that have a match in DECaLS.
    # 2. A random sample of non-matched SDSS galaxies to reach the target size.
    # --------------------------------------------------------------------------
    logging.info("Constructing the matched SDSS training catalog...")
    
    # Get the list of SDSS objids that are in the matched pairs file
    # The matched catalog uses 'sdss_objid', not 'sdss_dr7objid'
    matched_sdss_ids = set(matched_pairs['sdss_objid'])
    logging.info(f"Found {len(matched_sdss_ids)} unique SDSS galaxies in the matched pairs list.")

    # Get the full catalog entries for these matched SDSS galaxies
    # The main SDSS catalog uses 'dr7objid'
    matched_sdss_sample = sdss_cat[sdss_cat['dr7objid'].isin(matched_sdss_ids)]
    logging.info(f"Retrieved {len(matched_sdss_sample)} full records for matched SDSS galaxies.")

    # Get the SDSS galaxies that are NOT in the matched set
    non_matched_sdss_catalog = sdss_cat[~sdss_cat['dr7objid'].isin(matched_sdss_ids)]
    
    # Determine how many more galaxies we need to fill the SDSS set
    needed_count = TARGET_SIZE_PER_SURVEY - len(matched_sdss_sample)

    if needed_count < 0:
        logging.warning(f"Warning: More matched SDSS galaxies ({len(matched_sdss_sample)}) than target size ({TARGET_SIZE_PER_SURVEY}).")
        logging.warning("The SDSS training set will be larger than the DECaLS set.")
        sdss_training_catalog = matched_sdss_sample.sample(n=TARGET_SIZE_PER_SURVEY, random_state=42)

    elif needed_count > len(non_matched_sdss_catalog):
        logging.warning(f"Warning: Not enough non-matched SDSS galaxies ({len(non_matched_sdss_catalog)}) to reach target size.")
        logging.warning("The SDSS training set will be smaller than the DECaLS set.")
        random_sample = non_matched_sdss_catalog
        sdss_training_catalog = pd.concat([matched_sdss_sample, random_sample], ignore_index=True)
    else:
        logging.info(f"Sampling {needed_count} additional random galaxies from the non-matched SDSS set.")
        # Sample the required number of additional galaxies randomly
        random_sample = non_matched_sdss_catalog.sample(n=needed_count, random_state=42)
        # Combine the matched sample with the random sample
        sdss_training_catalog = pd.concat([matched_sdss_sample, random_sample], ignore_index=True)

    # Ensure asset_id is preserved, as it is used for locating image files
    if 'asset_id' not in sdss_training_catalog.columns:
        logging.error("Critical error: 'asset_id' column is missing from the final SDSS catalog.")
        logging.error("This column is required to find the image files for training.")
        # Attempt to merge it back in from the original catalog
        if 'asset_id' in sdss_cat.columns:
            logging.info("Attempting to merge 'asset_id' back from the master SDSS catalog.")
            sdss_training_catalog = pd.merge(
                sdss_training_catalog.drop(columns=['asset_id'], errors='ignore'),
                sdss_cat[['dr7objid', 'asset_id']],
                on='dr7objid',
                how='left'
            )
            logging.info(f"Asset IDs merged. Null count: {sdss_training_catalog['asset_id'].isnull().sum()}")
        else:
             logging.error("Master SDSS catalog also lacks 'asset_id'. Cannot proceed.")
             
    logging.info(f"Final SDSS training catalog size: {len(sdss_training_catalog)}")
    logging.info(f"Final DECaLS training catalog size: {len(decals_training_catalog)}")

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the final catalogs
    logging.info(f"Saving SDSS training catalog to {OUTPUT_SDSS_CATALOG}")
    sdss_training_catalog.to_csv(OUTPUT_SDSS_CATALOG, index=False)
    
    logging.info(f"Saving DECaLS training catalog to {OUTPUT_DECAL_CATALOG}")
    decals_training_catalog.to_csv(OUTPUT_DECAL_CATALOG, index=False)

    logging.info("Done.")

if __name__ == '__main__':
    main() 