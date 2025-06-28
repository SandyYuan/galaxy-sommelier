#!/usr/bin/env python3
"""
Analyze overlap between SDSS and DECaLS galaxy datasets
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u
from scipy.stats import pearsonr
import logging
import random

logger = logging.getLogger(__name__)

class SurveyOverlapAnalyzer:
    """Analyze overlap between different galaxy surveys"""
    
    def __init__(self, scratch_dir='/pscratch/sd/s/sihany/galaxy-sommelier-data'):
        self.scratch_dir = Path(scratch_dir)
        self.catalogs_dir = self.scratch_dir / 'catalogs'
        self.results = {}
        
    def load_catalogs(self):
        """Load SDSS and DECaLS catalogs"""
        print("Loading survey catalogs...")
        
        # Load SDSS catalog
        sdss_path = self.catalogs_dir / 'gz2_master_catalog_corrected.csv'
        if sdss_path.exists():
            self.sdss_catalog = pd.read_csv(sdss_path)
            print(f"SDSS catalog: {len(self.sdss_catalog)} galaxies")
        else:
            print(f"SDSS catalog not found at {sdss_path}")
            return False
            
        # Load DECaLS catalog  
        decals_path = self.catalogs_dir / 'gz_decals_volunteers_1_and_2.csv'
        if decals_path.exists():
            self.decals_catalog = pd.read_csv(decals_path)
            print(f"DECaLS catalog: {len(self.decals_catalog)} galaxies")
        else:
            print(f"DECaLS catalog not found at {decals_path}")
            return False
            
        return True
        
    def find_coordinate_matches(self, max_sep_arcsec=2.0):
        """Find galaxies that appear in both surveys based on coordinates"""
        print(f"Finding coordinate matches within {max_sep_arcsec}\"...")
        
        # Create coordinate objects
        sdss_coords = SkyCoord(
            ra=self.sdss_catalog['ra'].values*u.deg, 
            dec=self.sdss_catalog['dec'].values*u.deg
        )
        decals_coords = SkyCoord(
            ra=self.decals_catalog['ra'].values*u.deg, 
            dec=self.decals_catalog['dec'].values*u.deg
        )
        
        # Perform matching
        print("Performing coordinate matching...")
        idx, d2d, d3d = match_coordinates_sky(sdss_coords, decals_coords)
        
        # Select good matches
        good_matches = d2d < (max_sep_arcsec * u.arcsec)
        n_matches = good_matches.sum()
        
        print(f"Found {n_matches} coordinate matches")
        print(f"Match rate: {n_matches/len(self.sdss_catalog)*100:.2f}% of SDSS galaxies")
        print(f"Match rate: {n_matches/len(self.decals_catalog)*100:.2f}% of DECaLS galaxies")
        
        # Create matched catalog
        matched_indices = np.where(good_matches)[0]
        self.matched_catalog = pd.DataFrame({
            'sdss_idx': matched_indices,
            'decals_idx': idx[good_matches],
            'separation_arcsec': d2d[good_matches].arcsec,
            'sdss_objid': self.sdss_catalog.iloc[matched_indices]['dr7objid'].values,
            'decals_iauname': self.decals_catalog.iloc[idx[good_matches]]['iauname'].values,
            'sdss_ra': self.sdss_catalog.iloc[matched_indices]['ra'].values,
            'sdss_dec': self.sdss_catalog.iloc[matched_indices]['dec'].values,
            'decals_ra': self.decals_catalog.iloc[idx[good_matches]]['ra'].values,
            'decals_dec': self.decals_catalog.iloc[idx[good_matches]]['dec'].values,
        })
        
        return self.matched_catalog
        
    def analyze_morphology_consistency(self):
        """Compare morphological classifications for matched galaxies"""
        if not hasattr(self, 'matched_catalog'):
            print("No matched catalog found. Run find_coordinate_matches() first.")
            return
            
        print("Analyzing morphological consistency...")
        
        # Import feature mapping
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from sdss_decals_feature_mapping import SDSS_TO_DECALS
        
        correlations = {}
        
        # Compare overlapping morphological features
        for sdss_feature, decals_feature in SDSS_TO_DECALS.items():
            if ('_fraction' in sdss_feature or '_debiased' in sdss_feature):
                # Get values for matched galaxies
                sdss_indices = self.matched_catalog['sdss_idx'].values
                decals_indices = self.matched_catalog['decals_idx'].values
                
                if (sdss_feature in self.sdss_catalog.columns and 
                    decals_feature in self.decals_catalog.columns):
                    
                    sdss_values = self.sdss_catalog.iloc[sdss_indices][sdss_feature].values
                    decals_values = self.decals_catalog.iloc[decals_indices][decals_feature].values
                    
                    # Remove NaN values
                    valid_mask = ~(np.isnan(sdss_values) | np.isnan(decals_values))
                    if valid_mask.sum() > 10:  # Need at least 10 valid pairs
                        correlation, p_value = pearsonr(
                            sdss_values[valid_mask], 
                            decals_values[valid_mask]
                        )
                        correlations[sdss_feature] = {
                            'correlation': correlation,
                            'p_value': p_value,
                            'n_samples': valid_mask.sum()
                        }
        
        self.morphology_correlations = correlations
        
        # Print results
        print(f"\nMorphological consistency for {len(correlations)} features:")
        print("-" * 60)
        for feature, stats in correlations.items():
            print(f"{feature[:30]:30s} r={stats['correlation']:5.3f} (n={stats['n_samples']:4d})")
            
        return correlations
        
    def plot_overlap_analysis(self, save_path=None):
        """Create visualization of survey overlap"""
        if not hasattr(self, 'matched_catalog'):
            print("No matched catalog found. Run find_coordinate_matches() first.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Sky distribution of matches
        ax = axes[0, 0]
        ax.scatter(self.sdss_catalog['ra'], self.sdss_catalog['dec'], 
                  alpha=0.1, s=1, color='blue', label='SDSS only')
        ax.scatter(self.decals_catalog['ra'], self.decals_catalog['dec'], 
                  alpha=0.1, s=1, color='red', label='DECaLS only')
        ax.scatter(self.matched_catalog['sdss_ra'], self.matched_catalog['sdss_dec'], 
                  s=2, color='purple', label='Matched', alpha=0.7)
        ax.set_xlabel('RA (degrees)')
        ax.set_ylabel('Dec (degrees)')
        ax.set_title('Sky Distribution of Survey Overlap')
        ax.legend()
        
        # 2. Separation distribution
        ax = axes[0, 1]
        ax.hist(self.matched_catalog['separation_arcsec'], bins=50, alpha=0.7)
        ax.set_xlabel('Separation (arcsec)')
        ax.set_ylabel('Number of matches')
        ax.set_title('Coordinate Match Quality')
        ax.axvline(np.median(self.matched_catalog['separation_arcsec']), 
                  color='red', linestyle='--', label=f'Median: {np.median(self.matched_catalog["separation_arcsec"]):.2f}"')
        ax.legend()
        
        # 3. Morphology correlation comparison (if available)
        if hasattr(self, 'morphology_correlations'):
            ax = axes[1, 0]
            features = list(self.morphology_correlations.keys())[:10]  # Top 10
            correlations = [self.morphology_correlations[f]['correlation'] for f in features]
            
            y_pos = np.arange(len(features))
            ax.barh(y_pos, correlations)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f.replace('t01_smooth_or_features_', '').replace('_fraction', '') 
                               for f in features], fontsize=8)
            ax.set_xlabel('Correlation coefficient')
            ax.set_title('Morphology Consistency (Matched Galaxies)')
            ax.axvline(0.7, color='red', linestyle='--', alpha=0.7, label='r=0.7')
            ax.legend()
        
        # 4. Sample overlap statistics
        ax = axes[1, 1]
        n_sdss = len(self.sdss_catalog)
        n_decals = len(self.decals_catalog)
        n_matched = len(self.matched_catalog)
        
        categories = ['SDSS only', 'DECaLS only', 'Matched']
        counts = [n_sdss - n_matched, n_decals - n_matched, n_matched]
        colors = ['blue', 'red', 'purple']
        
        ax.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%')
        ax.set_title('Dataset Composition')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Overlap analysis plot saved to {save_path}")
        
        return fig
        
    def create_overlap_report(self):
        """Generate comprehensive overlap analysis report"""
        if not hasattr(self, 'matched_catalog'):
            print("No matched catalog found. Run find_coordinate_matches() first.")
            return
            
        report = {
            'summary': {
                'sdss_galaxies': len(self.sdss_catalog),
                'decals_galaxies': len(self.decals_catalog),
                'matched_galaxies': len(self.matched_catalog),
                'sdss_match_rate': len(self.matched_catalog) / len(self.sdss_catalog),
                'decals_match_rate': len(self.matched_catalog) / len(self.decals_catalog),
                'median_separation_arcsec': np.median(self.matched_catalog['separation_arcsec'])
            }
        }
        
        if hasattr(self, 'morphology_correlations'):
            # Calculate mean correlation across all features
            correlations = [stats['correlation'] for stats in self.morphology_correlations.values()]
            report['morphology_consistency'] = {
                'mean_correlation': np.mean(correlations),
                'median_correlation': np.median(correlations),
                'min_correlation': np.min(correlations),
                'max_correlation': np.max(correlations),
                'n_features_compared': len(correlations),
                'high_correlation_features': len([r for r in correlations if r > 0.7])
            }
        
        # Training implications
        total_unique = (len(self.sdss_catalog) + len(self.decals_catalog) - 
                       len(self.matched_catalog))
        
        report['training_implications'] = {
            'total_unique_galaxies': total_unique,
            'overlap_percentage': len(self.matched_catalog) / total_unique * 100,
            'effective_sample_size_reduction': len(self.matched_catalog) / 
                                             (len(self.sdss_catalog) + len(self.decals_catalog)) * 100
        }
        
        return report
        
    def simulate_mixed_dataset_sampling(self, sdss_fraction=0.5, max_galaxies=None, random_seed=42):
        """Simulate exact same sampling as MixedSDSSDECaLSDataset to analyze training data overlap"""
        print("=" * 60)
        print("Simulating Mixed Dataset Sampling")
        print("=" * 60)
        
        # Set random seeds exactly like MixedSDSSDECaLSDataset
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        print(f"Using random seed: {random_seed}")
        print(f"SDSS fraction: {sdss_fraction:.1%}")
        print(f"Max galaxies: {max_galaxies}")
        
        # Apply image availability filtering (same as MixedSDSSDECaLSDataset)
        print("\nFiltering for image availability...")
        self.filter_for_available_images()
        
        # Determine sample sizes (same logic as MixedSDSSDECaLSDataset)
        if max_galaxies is None:
            max_galaxies = min(len(self.sdss_available), len(self.decals_available)) * 2
        
        sdss_count = int(max_galaxies * sdss_fraction)
        decals_count = max_galaxies - sdss_count
        
        # Ensure we don't exceed available data
        sdss_count = min(sdss_count, len(self.sdss_available))
        decals_count = min(decals_count, len(self.decals_available))
        
        print(f"Sampling {sdss_count} SDSS + {decals_count} DECaLS galaxies")
        
        # Sample galaxies (same as MixedSDSSDECaLSDataset)
        self.sdss_sample = self.sdss_available.sample(n=sdss_count, random_state=random_seed)
        self.decals_sample = self.decals_available.sample(n=decals_count, random_state=random_seed)
        
        print(f"SDSS sample: {len(self.sdss_sample)} galaxies")
        print(f"DECaLS sample: {len(self.decals_sample)} galaxies")
        
        return self.sdss_sample, self.decals_sample
        
    def filter_for_available_images(self):
        """Filter catalogs for available images (same logic as MixedSDSSDECaLSDataset)"""
        from tqdm import tqdm
        
        # Filter SDSS catalog for available images
        print("Checking SDSS image availability...")
        sdss_available = []
        sdss_image_dir = self.scratch_dir / 'sdss'
        
        for idx, row in tqdm(self.sdss_catalog.iterrows(), total=len(self.sdss_catalog), desc="SDSS images"):
            objid = row['dr7objid']
            # Try different image naming conventions (same as MixedSDSSDECaLSDataset)
            jpg_path = sdss_image_dir / f"{row.get('asset_id', objid)}.jpg"
            if not jpg_path.exists():
                jpg_path = sdss_image_dir / f"sdss_{objid}.jpg"
            if not jpg_path.exists():
                jpg_path = sdss_image_dir / f"{objid}.jpg"
            
            if jpg_path.exists():
                sdss_available.append(idx)
        
        self.sdss_available = self.sdss_catalog.loc[sdss_available].reset_index(drop=True)
        print(f"SDSS galaxies with images: {len(self.sdss_available)}")
        
        # Filter DECaLS catalog for available images
        print("Checking DECaLS image availability...")
        decals_available = []
        decals_image_dir = self.scratch_dir / 'decals'
        
        for idx, row in tqdm(self.decals_catalog.iterrows(), total=len(self.decals_catalog), desc="DECaLS images"):
            iauname = row['iauname']
            if pd.isna(iauname) or not isinstance(iauname, str) or len(iauname) < 4:
                continue
                
            # DECaLS naming (same as MixedSDSSDECaLSDataset)
            directory = iauname[:4]  # e.g., 'J103'
            image_path = decals_image_dir / directory / f"{iauname}.png"
            
            if image_path.exists():
                decals_available.append(idx)
        
        self.decals_available = self.decals_catalog.loc[decals_available].reset_index(drop=True)
        print(f"DECaLS galaxies with images: {len(self.decals_available)}")

    def find_training_data_overlap(self, max_sep_arcsec=2.0):
        """Find overlap in the actual training data samples"""
        if not (hasattr(self, 'sdss_sample') and hasattr(self, 'decals_sample')):
            print("No training samples found. Run simulate_mixed_dataset_sampling() first.")
            return
            
        print(f"\nFinding coordinate matches in training samples within {max_sep_arcsec}\"...")
        
        # Create coordinate objects for training samples
        sdss_coords = SkyCoord(
            ra=self.sdss_sample['ra'].values*u.deg, 
            dec=self.sdss_sample['dec'].values*u.deg
        )
        decals_coords = SkyCoord(
            ra=self.decals_sample['ra'].values*u.deg, 
            dec=self.decals_sample['dec'].values*u.deg
        )
        
        # Perform matching
        print("Performing coordinate matching on training samples...")
        idx, d2d, d3d = match_coordinates_sky(sdss_coords, decals_coords)
        
        # Select good matches
        good_matches = d2d < (max_sep_arcsec * u.arcsec)
        n_matches = good_matches.sum()
        
        print(f"Found {n_matches} coordinate matches in training data")
        print(f"Match rate: {n_matches/len(self.sdss_sample)*100:.2f}% of SDSS training galaxies")
        print(f"Match rate: {n_matches/len(self.decals_sample)*100:.2f}% of DECaLS training galaxies")
        
        # Create matched catalog for training data
        matched_indices = np.where(good_matches)[0]
        self.training_matched_catalog = pd.DataFrame({
            'sdss_idx': matched_indices,
            'decals_idx': idx[good_matches],
            'separation_arcsec': d2d[good_matches].arcsec,
            'sdss_objid': self.sdss_sample.iloc[matched_indices]['dr7objid'].values,
            'decals_iauname': self.decals_sample.iloc[idx[good_matches]]['iauname'].values,
            'sdss_ra': self.sdss_sample.iloc[matched_indices]['ra'].values,
            'sdss_dec': self.sdss_sample.iloc[matched_indices]['dec'].values,
            'decals_ra': self.decals_sample.iloc[idx[good_matches]]['ra'].values,
            'decals_dec': self.decals_sample.iloc[idx[good_matches]]['dec'].values,
        })
        
        return self.training_matched_catalog

    def analyze_training_morphology_consistency(self):
        """Compare morphological classifications for matched galaxies in training data"""
        if not hasattr(self, 'training_matched_catalog'):
            print("No training matched catalog found. Run find_training_data_overlap() first.")
            return
            
        print("Analyzing morphological consistency in training data...")
        
        # Import feature mapping
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from sdss_decals_feature_mapping import SDSS_TO_DECALS
        
        correlations = {}
        
        # Compare overlapping morphological features
        for sdss_feature, decals_feature in SDSS_TO_DECALS.items():
            if ('_fraction' in sdss_feature or '_debiased' in sdss_feature):
                # Get values for matched galaxies in training samples
                sdss_indices = self.training_matched_catalog['sdss_idx'].values
                decals_indices = self.training_matched_catalog['decals_idx'].values
                
                if (sdss_feature in self.sdss_sample.columns and 
                    decals_feature in self.decals_sample.columns):
                    
                    sdss_values = self.sdss_sample.iloc[sdss_indices][sdss_feature].values
                    decals_values = self.decals_sample.iloc[decals_indices][decals_feature].values
                    
                    # Remove NaN values
                    valid_mask = ~(np.isnan(sdss_values) | np.isnan(decals_values))
                    if valid_mask.sum() > 5:  # Need at least 5 valid pairs for training data
                        correlation, p_value = pearsonr(
                            sdss_values[valid_mask], 
                            decals_values[valid_mask]
                        )
                        correlations[sdss_feature] = {
                            'correlation': correlation,
                            'p_value': p_value,
                            'n_samples': valid_mask.sum()
                        }
        
        self.training_morphology_correlations = correlations
        
        # Print results
        print(f"\nMorphological consistency for {len(correlations)} features in training data:")
        print("-" * 60)
        for feature, stats in correlations.items():
            print(f"{feature[:30]:30s} r={stats['correlation']:5.3f} (n={stats['n_samples']:4d})")
            
        return correlations

    def create_training_overlap_report(self):
        """Generate overlap analysis report for actual training data"""
        if not hasattr(self, 'training_matched_catalog'):
            print("No training matched catalog found. Run find_training_data_overlap() first.")
            return
            
        report = {
            'training_data_summary': {
                'sdss_training_galaxies': len(self.sdss_sample),
                'decals_training_galaxies': len(self.decals_sample),
                'matched_training_galaxies': len(self.training_matched_catalog),
                'sdss_match_rate': len(self.training_matched_catalog) / len(self.sdss_sample),
                'decals_match_rate': len(self.training_matched_catalog) / len(self.decals_sample),
                'median_separation_arcsec': np.median(self.training_matched_catalog['separation_arcsec'])
            }
        }
        
        if hasattr(self, 'training_morphology_correlations'):
            # Calculate mean correlation across all features
            correlations = [stats['correlation'] for stats in self.training_morphology_correlations.values()]
            report['training_morphology_consistency'] = {
                'mean_correlation': np.mean(correlations),
                'median_correlation': np.median(correlations),
                'min_correlation': np.min(correlations),
                'max_correlation': np.max(correlations),
                'n_features_compared': len(correlations),
                'high_correlation_features': len([r for r in correlations if r > 0.7])
            }
        
        # Training implications
        total_training = len(self.sdss_sample) + len(self.decals_sample)
        unique_training = total_training - len(self.training_matched_catalog)
        
        report['training_implications'] = {
            'total_training_galaxies': total_training,
            'unique_training_galaxies': unique_training,
            'overlap_percentage': len(self.training_matched_catalog) / unique_training * 100,
            'effective_sample_size_reduction': len(self.training_matched_catalog) / total_training * 100
        }
        
        return report

    def run_training_data_analysis(self, sdss_fraction=0.5, max_galaxies=None, random_seed=42, max_sep_arcsec=2.0, save_plots=True):
        """Run complete overlap analysis for actual training data"""
        print("=" * 80)
        print("SDSS-DECaLS Training Data Overlap Analysis")
        print("=" * 80)
        
        # Load catalogs
        if not self.load_catalogs():
            return None
            
        # Simulate mixed dataset sampling (exact same logic as training)
        self.simulate_mixed_dataset_sampling(sdss_fraction, max_galaxies, random_seed)
        
        # Find coordinate matches in training data
        training_matches = self.find_training_data_overlap(max_sep_arcsec)
        
        # Analyze morphological consistency in training data
        self.analyze_training_morphology_consistency()
        
        # Create report
        report = self.create_training_overlap_report()
        
        # Print summary
        print("\n" + "=" * 80)
        print("TRAINING DATA OVERLAP RESULTS")
        print("=" * 80)
        
        s = report['training_data_summary']
        print(f"SDSS training galaxies:     {s['sdss_training_galaxies']:,}")
        print(f"DECaLS training galaxies:   {s['decals_training_galaxies']:,}")
        print(f"Matched training galaxies:  {s['matched_training_galaxies']:,}")
        print(f"SDSS match rate:            {s['sdss_match_rate']*100:.1f}%")
        print(f"DECaLS match rate:          {s['decals_match_rate']*100:.1f}%")
        print(f"Median separation:          {s['median_separation_arcsec']:.2f}\"")
        
        if 'training_morphology_consistency' in report:
            m = report['training_morphology_consistency']
            print(f"\nMorphological consistency in training data:")
            print(f"Mean correlation:  {m['mean_correlation']:.3f}")
            print(f"Features with r>0.7: {m['high_correlation_features']}/{m['n_features_compared']}")
        
        t = report['training_implications']
        print(f"\nTraining implications:")
        print(f"Total training galaxies: {t['total_training_galaxies']:,}")
        print(f"Unique training galaxies: {t['unique_training_galaxies']:,}")
        print(f"Overlap percentage: {t['overlap_percentage']:.1f}%")
        print(f"Sample size reduction: {t['effective_sample_size_reduction']:.1f}%")
        
        # Create plots (modified for training data)
        if save_plots:
            plot_path = self.scratch_dir / 'training_overlap_analysis.png'
            self.plot_training_overlap_analysis(plot_path)
            
        # Save detailed results
        training_matches.to_csv(self.scratch_dir / 'training_sdss_decals_matches.csv', index=False)
        
        import json
        with open(self.scratch_dir / 'training_overlap_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\nDetailed results saved to {self.scratch_dir}")
        
        return report
        
    def plot_training_overlap_analysis(self, save_path=None):
        """Create visualization of training data overlap"""
        if not hasattr(self, 'training_matched_catalog'):
            print("No training matched catalog found.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Sky distribution of training matches
        ax = axes[0, 0]
        ax.scatter(self.sdss_sample['ra'], self.sdss_sample['dec'], 
                  alpha=0.5, s=3, color='blue', label='SDSS training')
        ax.scatter(self.decals_sample['ra'], self.decals_sample['dec'], 
                  alpha=0.5, s=3, color='red', label='DECaLS training')
        ax.scatter(self.training_matched_catalog['sdss_ra'], self.training_matched_catalog['sdss_dec'], 
                  s=10, color='purple', label='Matched in training', alpha=0.8)
        ax.set_xlabel('RA (degrees)')
        ax.set_ylabel('Dec (degrees)')
        ax.set_title('Sky Distribution of Training Data Overlap')
        ax.legend()
        
        # 2. Separation distribution
        ax = axes[0, 1]
        ax.hist(self.training_matched_catalog['separation_arcsec'], bins=20, alpha=0.7)
        ax.set_xlabel('Separation (arcsec)')
        ax.set_ylabel('Number of matches')
        ax.set_title('Training Data Match Quality')
        ax.axvline(np.median(self.training_matched_catalog['separation_arcsec']), 
                  color='red', linestyle='--', 
                  label=f'Median: {np.median(self.training_matched_catalog["separation_arcsec"]):.2f}"')
        ax.legend()
        
        # 3. Morphology correlation comparison (if available)
        if hasattr(self, 'training_morphology_correlations'):
            ax = axes[1, 0]
            features = list(self.training_morphology_correlations.keys())[:10]  # Top 10
            correlations = [self.training_morphology_correlations[f]['correlation'] for f in features]
            
            y_pos = np.arange(len(features))
            ax.barh(y_pos, correlations)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f.replace('t01_smooth_or_features_', '').replace('_fraction', '') 
                               for f in features], fontsize=8)
            ax.set_xlabel('Correlation coefficient')
            ax.set_title('Training Data Morphology Consistency')
            ax.axvline(0.7, color='red', linestyle='--', alpha=0.7, label='r=0.7')
            ax.legend()
        
        # 4. Training sample overlap statistics
        ax = axes[1, 1]
        n_sdss = len(self.sdss_sample)
        n_decals = len(self.decals_sample)
        n_matched = len(self.training_matched_catalog)
        
        categories = ['SDSS only', 'DECaLS only', 'Matched']
        counts = [n_sdss - n_matched, n_decals - n_matched, n_matched]
        colors = ['blue', 'red', 'purple']
        
        ax.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%')
        ax.set_title('Training Data Composition')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training overlap analysis plot saved to {save_path}")
        
        return fig

    def run_full_analysis(self, max_sep_arcsec=2.0, save_plots=True):
        """Run complete overlap analysis (legacy method - use run_training_data_analysis instead)"""
        print("=" * 60)
        print("SDSS-DECaLS Survey Overlap Analysis")
        print("=" * 60)
        
        # Load catalogs
        if not self.load_catalogs():
            return None
            
        # Find coordinate matches
        matches = self.find_coordinate_matches(max_sep_arcsec)
        
        # Analyze morphological consistency
        self.analyze_morphology_consistency()
        
        # Create report
        report = self.create_overlap_report()
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY RESULTS")
        print("=" * 60)
        
        s = report['summary']
        print(f"SDSS galaxies:     {s['sdss_galaxies']:,}")
        print(f"DECaLS galaxies:   {s['decals_galaxies']:,}")
        print(f"Matched galaxies:  {s['matched_galaxies']:,}")
        print(f"SDSS match rate:   {s['sdss_match_rate']*100:.1f}%")
        print(f"DECaLS match rate: {s['decals_match_rate']*100:.1f}%")
        print(f"Median separation: {s['median_separation_arcsec']:.2f}\"")
        
        if 'morphology_consistency' in report:
            m = report['morphology_consistency']
            print(f"\nMorphological consistency:")
            print(f"Mean correlation:  {m['mean_correlation']:.3f}")
            print(f"Features with r>0.7: {m['high_correlation_features']}/{m['n_features_compared']}")
        
        t = report['training_implications']
        print(f"\nTraining implications:")
        print(f"Total unique galaxies: {t['total_unique_galaxies']:,}")
        print(f"Overlap percentage: {t['overlap_percentage']:.1f}%")
        print(f"Sample size reduction: {t['effective_sample_size_reduction']:.1f}%")
        
        # Create plots
        if save_plots:
            plot_path = self.scratch_dir / 'overlap_analysis.png'
            self.plot_overlap_analysis(plot_path)
            
        # Save detailed results
        matches.to_csv(self.scratch_dir / 'sdss_decals_matches.csv', index=False)
        
        import json
        with open(self.scratch_dir / 'overlap_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\nDetailed results saved to {self.scratch_dir}")
        
        return report


def main():
    """Run training data overlap analysis"""
    analyzer = SurveyOverlapAnalyzer()
    
    # Analyze training data overlap using same parameters as mixed dataset
    report = analyzer.run_training_data_analysis(
        sdss_fraction=0.5,  # Same as mixed dataset default
        max_galaxies=None,  # Use all available
        random_seed=42,     # Same as mixed dataset default
        max_sep_arcsec=2.0,
        save_plots=True
    )
    
    if report:
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS FOR TRAINING")
        print("=" * 80)
        
        overlap_pct = report['training_implications']['overlap_percentage']
        
        if overlap_pct < 5:
            print("✅ LOW OVERLAP: Minimal impact on training diversity")
            print("   Recommendation: Include overlapping galaxies for transfer learning benefits")
            print("   The overlaps will help the model learn survey-invariant features.")
        elif overlap_pct < 15:
            print("⚠️  MODERATE OVERLAP: Some impact on training diversity")
            print("   Recommendation: Include overlaps but ensure proper train/val splitting")
            print("   Consider stratified splitting to ensure overlaps don't leak between sets.")
        else:
            print("🚨 HIGH OVERLAP: Significant impact on training diversity")
            print("   Recommendation: Consider separate analysis of overlap vs non-overlap performance")
            print("   May want to deduplicate or carefully manage overlapping galaxies.")
            
        if 'training_morphology_consistency' in report:
            mean_corr = report['training_morphology_consistency']['mean_correlation']
            if mean_corr > 0.8:
                print("✅ EXCELLENT consistency between surveys in training data")
                print("   Overlapping galaxies show strong morphological agreement.")
            elif mean_corr > 0.6:
                print("⚠️  GOOD consistency between surveys in training data")
                print("   Some morphological differences but generally consistent.")
            else:
                print("🚨 POOR consistency between surveys in training data")
                print("   Investigate systematic differences - may indicate data quality issues.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 