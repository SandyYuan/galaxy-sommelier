#!/usr/bin/env python3
"""
Test script to demonstrate the high-quality SDSS galaxy selection functionality.
Shows comparison between random vs high-quality selection.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our modified mixed dataset
from scripts.mixed_dataset import MixedSDSSDECaLSDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_selection_strategies(config_path="configs/mixed_high_quality_config.yaml", 
                                sample_size=1000, sdss_fraction=0.5):
    """Compare random vs high-quality SDSS galaxy selection"""
    
    import yaml
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    
    # Paths  
    sdss_catalog = Path(data_config['catalogs_dir']) / 'gz2_master_catalog_corrected.csv'
    decals_catalog = Path(data_config['catalogs_dir']) / 'gz_decals_volunteers_1_and_2.csv'
    
    print("=" * 70)
    print("TESTING HIGH-QUALITY SDSS GALAXY SELECTION")
    print("=" * 70)
    
    # Test 1: Random selection
    print("\n1. Creating dataset with RANDOM SDSS selection...")
    dataset_random = MixedSDSSDECaLSDataset(
        sdss_catalog_path=sdss_catalog,
        decals_catalog_path=decals_catalog,
        sdss_image_dir=data_config['sdss_dir'],
        decals_image_dir=data_config['decals_dir'],
        sdss_fraction=sdss_fraction,
        max_galaxies=sample_size,
        transform=None,
        feature_set='sdss',
        random_seed=42,
        high_quality=False  # Random selection
    )
    
    # Test 2: High-quality selection
    print("\n2. Creating dataset with HIGH-QUALITY SDSS selection...")
    dataset_hq = MixedSDSSDECaLSDataset(
        sdss_catalog_path=sdss_catalog,
        decals_catalog_path=decals_catalog,
        sdss_image_dir=data_config['sdss_dir'],
        decals_image_dir=data_config['decals_dir'],
        sdss_fraction=sdss_fraction,
        max_galaxies=sample_size,
        transform=None,
        feature_set='sdss',
        random_seed=42,
        high_quality=True  # High-quality selection
    )
    
    # Analysis
    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS")
    print("=" * 50)
    
    # Get SDSS galaxies from each dataset
    sdss_random = dataset_random.mixed_catalog[dataset_random.mixed_catalog['survey'] == 'sdss']
    sdss_hq = dataset_hq.mixed_catalog[dataset_hq.mixed_catalog['survey'] == 'sdss']
    
    print(f"\nDataset sizes:")
    print(f"  Random SDSS galaxies: {len(sdss_random)}")
    print(f"  High-quality SDSS galaxies: {len(sdss_hq)}")
    
    print(f"\nClassification count statistics:")
    print(f"  Random selection:")
    print(f"    Min votes: {sdss_random['weight'].min():.0f}")
    print(f"    Max votes: {sdss_random['weight'].max():.0f}")
    print(f"    Mean votes: {sdss_random['weight'].mean():.1f}")
    print(f"    Median votes: {sdss_random['weight'].median():.1f}")
    
    print(f"  High-quality selection:")
    print(f"    Min votes: {sdss_hq['weight'].min():.0f}")
    print(f"    Max votes: {sdss_hq['weight'].max():.0f}")
    print(f"    Mean votes: {sdss_hq['weight'].mean():.1f}")
    print(f"    Median votes: {sdss_hq['weight'].median():.1f}")
    
    improvement = sdss_hq['weight'].mean() / sdss_random['weight'].mean()
    print(f"\nQuality improvement: {improvement:.2f}x higher average classification count")
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(sdss_random['weight'], bins=30, alpha=0.7, label='Random Selection', color='blue')
    plt.hist(sdss_hq['weight'], bins=30, alpha=0.7, label='High-Quality Selection', color='red')
    plt.xlabel('Classification Count (total_votes)')
    plt.ylabel('Number of Galaxies')
    plt.title('Distribution of Classification Counts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    strategies = ['Random', 'High-Quality']
    means = [sdss_random['weight'].mean(), sdss_hq['weight'].mean()]
    medians = [sdss_random['weight'].median(), sdss_hq['weight'].median()]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    plt.bar(x - width/2, means, width, label='Mean', alpha=0.8)
    plt.bar(x + width/2, medians, width, label='Median', alpha=0.8)
    plt.xlabel('Selection Strategy')
    plt.ylabel('Classification Count')
    plt.title('Mean vs Median Classification Counts')
    plt.xticks(x, strategies)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path("plots") / "high_quality_comparison.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {plot_path}")
    
    plt.show()
    
    return {
        'random': dataset_random,
        'high_quality': dataset_hq,
        'improvement_factor': improvement
    }

if __name__ == "__main__":
    print("Testing high-quality SDSS galaxy selection...")
    
    # Run comparison
    try:
        results = compare_selection_strategies(sample_size=2000)
        print(f"\n✅ Test completed successfully!")
        print(f"Quality improvement factor: {results['improvement_factor']:.2f}x")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 