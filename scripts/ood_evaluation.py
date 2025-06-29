#!/usr/bin/env python3
"""
Out-of-Distribution Evaluation Script for Galaxy Sommelier
Tests the fine-tuned model on UKIDSS data and compares against in-distribution SDSS results.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from PIL import Image
import torchvision.transforms as transforms

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from model_setup import GalaxySommelier
from data_processing import get_transforms

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class UKIDSSDataset:
    """Dataset class for UKIDSS Galaxy Zoo data"""
    
    def __init__(self, catalog_path, image_base_dir, transform=None, sample_size=None):
        self.catalog_path = Path(catalog_path)
        self.image_base_dir = Path(image_base_dir)
        self.transform = transform
        
        # Load catalog
        print(f"Loading UKIDSS catalog from {self.catalog_path}")
        self.catalog = pd.read_csv(self.catalog_path)
        
        if sample_size:
            print(f"Using sample of {sample_size} galaxies")
            self.catalog = self.catalog.head(sample_size)
        
        print(f"UKIDSS dataset initialized with {len(self.catalog)} galaxies")
        
        # Prepare labels - map UKIDSS columns to Galaxy Zoo format
        self.prepare_labels()
        
        # Filter for available images
        self.filter_available_images()
        
        print(f"Final dataset: {len(self.catalog)} galaxies with available images")
    
    def prepare_labels(self):
        """Map UKIDSS labels to Galaxy Zoo format for comparison"""
        print("Mapping UKIDSS labels to Galaxy Zoo format...")
        
        # Define mapping from UKIDSS to Galaxy Zoo task names
        self.label_mapping = {
            # Task 1: Smooth or featured
            't01_smooth_or_features_a01_smooth_fraction': 'smooth-or-featured-ukidss_smooth_fraction',
            't01_smooth_or_features_a02_features_or_disk_fraction': 'smooth-or-featured-ukidss_featured-or-disk_fraction',
            't01_smooth_or_features_a03_star_or_artifact_fraction': 'smooth-or-featured-ukidss_artifact_fraction',
            
            # Task 2: Edge-on
            't02_edgeon_a04_yes_fraction': 'disk-edge-on-ukidss_yes_fraction',
            't02_edgeon_a05_no_fraction': 'disk-edge-on-ukidss_no_fraction',
            
            # Task 3: Bar
            't03_bar_a06_bar_fraction': 'bar-ukidss_yes_fraction',
            't03_bar_a07_no_bar_fraction': 'bar-ukidss_no_fraction',
            
            # Task 4: Spiral
            't04_spiral_a08_spiral_fraction': 'has-spiral-arms-ukidss_yes_fraction',
            't04_spiral_a09_no_spiral_fraction': 'has-spiral-arms-ukidss_no_fraction',
            
            # Task 6: Odd features
            't06_odd_a14_yes_fraction': 'something-odd-ukidss_yes_fraction',
            't06_odd_a15_no_fraction': 'something-odd-ukidss_no_fraction',
        }
        
        # Create label columns that match Galaxy Zoo format
        self.label_columns = []
        for gz_col, ukidss_col in self.label_mapping.items():
            if ukidss_col in self.catalog.columns:
                self.catalog[gz_col] = self.catalog[ukidss_col]
                self.label_columns.append(gz_col)
                print(f"Mapped {ukidss_col} -> {gz_col}")
            else:
                print(f"Warning: {ukidss_col} not found in UKIDSS catalog")
        
        print(f"Successfully mapped {len(self.label_columns)} label columns")
    
    def filter_available_images(self):
        """Filter catalog to only include galaxies with available images"""
        print("Filtering for available images...")
        
        available_indices = []
        for idx, row in self.catalog.iterrows():
            subfolder = row['subfolder']
            filename = row['filename']
            image_path = self.image_base_dir / subfolder / filename
            
            if image_path.exists():
                available_indices.append(idx)
        
        print(f"Found {len(available_indices)} galaxies with available images out of {len(self.catalog)}")
        self.catalog = self.catalog.loc[available_indices].reset_index(drop=True)
    
    def load_image(self, idx):
        """Load image by index"""
        row = self.catalog.iloc[idx]
        subfolder = row['subfolder']
        filename = row['filename']
        image_path = self.image_base_dir / subfolder / filename
        
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.catalog)
    
    def __getitem__(self, idx):
        # Load image
        image = self.load_image(idx)
        if image is None:
            # Return dummy data if image fails to load
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        labels = self.catalog.iloc[idx][self.label_columns].values.astype(np.float32)
        
        # Handle any NaN values
        labels = np.nan_to_num(labels, nan=0.0)
        
        return {
            'image': image,
            'labels': torch.tensor(labels),
            'objid': f"{self.catalog.iloc[idx]['subfolder']}_{self.catalog.iloc[idx]['filename']}"
        }

class OODEvaluator:
    """Out-of-Distribution Evaluator for Galaxy Sommelier"""
    
    def __init__(self, model_path, config_path, output_dir='./ood_results'):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"OOD evaluation results will be saved to: {self.output_dir}")
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.load_model()
        
        # Define key features for comparison
        self.key_features = {
            'disk_fraction': 't01_smooth_or_features_a02_features_or_disk_fraction',
            'spiral_fraction': 't04_spiral_a08_spiral_fraction',
            'bar_fraction': 't03_bar_a06_bar_fraction',
            'edge_on_fraction': 't02_edgeon_a04_yes_fraction',
            'odd_features_fraction': 't06_odd_a14_yes_fraction'
        }
        
        # Results storage
        self.results = {}
    
    def load_model(self):
        """Load the fine-tuned model"""
        print(f"Loading model from {self.model_path}")
        
        # Initialize model architecture
        model_config = self.config['model']
        self.model = GalaxySommelier(
            config=self.config,
            model_name=model_config['name'],
            num_outputs=model_config['num_outputs'],
            dropout_rate=model_config['dropout_rate'],
            freeze_backbone=model_config.get('freeze_backbone', False)
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully")
    
    def evaluate_dataset(self, dataset, dataset_name):
        """Evaluate model on a dataset"""
        print(f"Evaluating on {dataset_name} dataset...")
        
        # Create data loader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        all_predictions = []
        all_labels = []
        all_objids = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {dataset_name}"):
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                objids = batch['objid']
                
                # Forward pass
                predictions = self.model(images)
                
                # Store results
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_objids.extend(objids)
        
        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=0)
        true_labels = np.concatenate(all_labels, axis=0)
        
        print(f"Generated predictions for {len(predictions)} galaxies")
        print(f"Model predictions shape: {predictions.shape}")
        print(f"True labels shape: {true_labels.shape}")
        
        # Handle dimension mismatch: extract only relevant predictions
        if predictions.shape[1] != true_labels.shape[1]:
            print(f"Dimension mismatch detected: {predictions.shape[1]} vs {true_labels.shape[1]}")
            
            # Map dataset label columns to model output indices by name
            if dataset_name == 'UKIDSS' and 'SDSS' in self.results:
                # Use SDSS label columns as reference for model output order
                sdss_label_columns = self.results['SDSS']['label_columns']
                print(f"Mapping UKIDSS tasks to SDSS model outputs...")
                
                # Find indices of UKIDSS tasks in the full model output
                prediction_indices = []
                for ukidss_col in dataset.label_columns:
                    if ukidss_col in sdss_label_columns:
                        idx = sdss_label_columns.index(ukidss_col)
                        prediction_indices.append(idx)
                        print(f"  {ukidss_col} -> model output index {idx}")
                    else:
                        print(f"  Warning: {ukidss_col} not found in SDSS model outputs")
                
                if len(prediction_indices) == len(dataset.label_columns):
                    # Extract the corresponding model predictions
                    predictions_subset = predictions[:, prediction_indices]
                    print(f"Successfully mapped {len(prediction_indices)} tasks")
                    print(f"Using mapped predictions: {predictions_subset.shape}")
                else:
                    print(f"Warning: Could only map {len(prediction_indices)}/{len(dataset.label_columns)} tasks")
                    # Fallback to first N columns
                    n_labels = true_labels.shape[1]
                    predictions_subset = predictions[:, :n_labels]
                    print(f"Using fallback subset: {predictions_subset.shape}")
            else:
                # Fallback: just use the first N columns
                n_labels = true_labels.shape[1]
                predictions_subset = predictions[:, :n_labels]
                print(f"Using subset of predictions: {predictions_subset.shape}")
            
            # Store both full and subset predictions
            predictions_full = predictions
            predictions = predictions_subset
        else:
            predictions_full = predictions
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, true_labels, dataset.label_columns)
        
        # Store results
        self.results[dataset_name] = {
            'predictions': predictions,
            'predictions_full': predictions_full,
            'true_labels': true_labels,
            'objids': all_objids,
            'label_columns': dataset.label_columns,
            'metrics': metrics
        }
        
        return metrics
    
    def compute_metrics(self, predictions, true_labels, label_columns):
        """Compute comprehensive metrics"""
        # Overall metrics
        mse = mean_squared_error(true_labels, predictions)
        mae = mean_absolute_error(true_labels, predictions)
        
        # Flatten for correlation
        pred_flat = predictions.flatten()
        true_flat = true_labels.flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(pred_flat) | np.isnan(true_flat))
        correlation = np.corrcoef(pred_flat[mask], true_flat[mask])[0, 1]
        r_squared = correlation ** 2
        
        # Per-task metrics
        task_metrics = {}
        for i, col in enumerate(label_columns):
            pred_task = predictions[:, i]
            true_task = true_labels[:, i]
            
            # Remove NaN values for this task
            task_mask = ~(np.isnan(pred_task) | np.isnan(true_task))
            if task_mask.sum() > 0:
                task_corr = np.corrcoef(pred_task[task_mask], true_task[task_mask])[0, 1]
                task_mse = mean_squared_error(true_task[task_mask], pred_task[task_mask])
                task_mae = mean_absolute_error(true_task[task_mask], pred_task[task_mask])
                
                task_metrics[col] = {
                    'correlation': float(task_corr) if not np.isnan(task_corr) else 0.0,
                    'mse': float(task_mse),
                    'mae': float(task_mae),
                    'n_samples': int(task_mask.sum())
                }
        
        return {
            'overall': {
                'mse': float(mse),
                'mae': float(mae),
                'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                'r_squared': float(r_squared) if not np.isnan(r_squared) else 0.0,
                'n_samples': len(predictions)
            },
            'per_task': task_metrics
        }
    
    def compare_performance(self):
        """Compare in-distribution vs out-of-distribution performance"""
        if 'SDSS' not in self.results or 'UKIDSS' not in self.results:
            print("Need both SDSS and UKIDSS results for comparison")
            return
        
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON: IN-DISTRIBUTION vs OUT-OF-DISTRIBUTION")
        print("="*60)
        
        sdss_metrics = self.results['SDSS']['metrics']['overall']
        ukidss_metrics = self.results['UKIDSS']['metrics']['overall']
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"{'Metric':<20} {'SDSS (ID)':<15} {'UKIDSS (OOD)':<15} {'Degradation':<15}")
        print("-" * 70)
        
        for metric in ['correlation', 'r_squared', 'mae', 'mse']:
            sdss_val = sdss_metrics[metric]
            ukidss_val = ukidss_metrics[metric]
            
            if metric in ['mae', 'mse']:
                # Lower is better - compute relative increase
                degradation = (ukidss_val - sdss_val) / sdss_val * 100
                deg_str = f"+{degradation:.1f}%" if degradation > 0 else f"{degradation:.1f}%"
            else:
                # Higher is better - compute relative decrease
                degradation = (sdss_val - ukidss_val) / sdss_val * 100
                deg_str = f"-{degradation:.1f}%" if degradation > 0 else f"+{abs(degradation):.1f}%"
            
            print(f"{metric:<20} {sdss_val:<15.4f} {ukidss_val:<15.4f} {deg_str:<15}")
        
        # Key features comparison
        print(f"\nKEY FEATURES PERFORMANCE:")
        print(f"{'Feature':<25} {'SDSS r':<12} {'UKIDSS r':<12} {'Degradation':<15}")
        print("-" * 70)
        
        for feature_name, column in self.key_features.items():
            if column in self.results['SDSS']['metrics']['per_task'] and \
               column in self.results['UKIDSS']['metrics']['per_task']:
                
                sdss_r = self.results['SDSS']['metrics']['per_task'][column]['correlation']
                ukidss_r = self.results['UKIDSS']['metrics']['per_task'][column]['correlation']
                
                degradation = (sdss_r - ukidss_r) / sdss_r * 100 if sdss_r > 0 else 0
                deg_str = f"-{degradation:.1f}%" if degradation > 0 else f"+{abs(degradation):.1f}%"
                
                print(f"{feature_name:<25} {sdss_r:<12.3f} {ukidss_r:<12.3f} {deg_str:<15}")
        
        # Store comparison
        self.results['comparison'] = {
            'overall_degradation': {
                'correlation': (sdss_metrics['correlation'] - ukidss_metrics['correlation']) / sdss_metrics['correlation'] * 100,
                'mae_increase': (ukidss_metrics['mae'] - sdss_metrics['mae']) / sdss_metrics['mae'] * 100
            }
        }
    
    def create_visualizations(self):
        """Create comparison visualizations"""
        print("Creating visualizations...")
        
        # Create output dir if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Overall performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('In-Distribution vs Out-of-Distribution Performance', fontsize=16, fontweight='bold')
        
        # Correlation comparison
        datasets = ['SDSS (ID)', 'UKIDSS (OOD)']
        correlations = [
            self.results['SDSS']['metrics']['overall']['correlation'],
            self.results['UKIDSS']['metrics']['overall']['correlation']
        ]
        
        axes[0, 0].bar(datasets, correlations, color=['skyblue', 'lightcoral'])
        axes[0, 0].set_title('Overall Correlation')
        axes[0, 0].set_ylabel('Correlation (r)')
        axes[0, 0].set_ylim(0, 1)
        
        # MAE comparison
        maes = [
            self.results['SDSS']['metrics']['overall']['mae'],
            self.results['UKIDSS']['metrics']['overall']['mae']
        ]
        
        axes[0, 1].bar(datasets, maes, color=['skyblue', 'lightcoral'])
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_ylabel('MAE')
        
        # Key features comparison
        feature_names = list(self.key_features.keys())
        sdss_corrs = []
        ukidss_corrs = []
        
        for feature_name, column in self.key_features.items():
            if column in self.results['SDSS']['metrics']['per_task'] and \
               column in self.results['UKIDSS']['metrics']['per_task']:
                sdss_corrs.append(self.results['SDSS']['metrics']['per_task'][column]['correlation'])
                ukidss_corrs.append(self.results['UKIDSS']['metrics']['per_task'][column]['correlation'])
            else:
                sdss_corrs.append(0)
                ukidss_corrs.append(0)
        
        x = np.arange(len(feature_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, sdss_corrs, width, label='SDSS (ID)', color='skyblue')
        axes[1, 0].bar(x + width/2, ukidss_corrs, width, label='UKIDSS (OOD)', color='lightcoral')
        axes[1, 0].set_title('Key Features Correlation')
        axes[1, 0].set_ylabel('Correlation (r)')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([name.replace('_', '\n') for name in feature_names], rotation=45)
        axes[1, 0].legend()
        
        # Performance degradation
        degradations = []
        for i, (feature_name, column) in enumerate(self.key_features.items()):
            if sdss_corrs[i] > 0:
                deg = (sdss_corrs[i] - ukidss_corrs[i]) / sdss_corrs[i] * 100
                degradations.append(deg)
            else:
                degradations.append(0)
        
        colors = ['red' if d > 0 else 'green' for d in degradations]
        axes[1, 1].bar(feature_names, degradations, color=colors, alpha=0.7)
        axes[1, 1].set_title('Performance Degradation (%)')
        axes[1, 1].set_ylabel('Degradation (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ood_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {self.output_dir}")
    
    def save_results(self):
        """Save detailed results to files"""
        print("Saving results...")
        
        # Save metrics
        metrics_only = {}
        for dataset_name, results in self.results.items():
            if 'metrics' in results:
                metrics_only[dataset_name] = results['metrics']
        
        with open(self.output_dir / 'ood_metrics.json', 'w') as f:
            json.dump(metrics_only, f, indent=2)
        
        # Save predictions for further analysis
        for dataset_name, results in self.results.items():
            if 'predictions' in results:
                output_file = self.output_dir / f'{dataset_name.lower()}_predictions.npz'
                np.savez_compressed(
                    output_file,
                    predictions=results['predictions'],
                    true_labels=results['true_labels'],
                    objids=results['objids'],
                    label_columns=results['label_columns']
                )
                print(f"Saved {dataset_name} predictions to {output_file}")
        
        print(f"All results saved to {self.output_dir}")
    
    def run_evaluation(self, sdss_config_path=None, ukidss_catalog_path=None, 
                      ukidss_image_dir=None, sample_size=None):
        """Run complete OOD evaluation"""
        
        # Default paths
        if ukidss_catalog_path is None:
            ukidss_catalog_path = '/pscratch/sd/s/sihany/galaxy-sommelier-data/catalogs/ukidss_catalog.csv'
        if ukidss_image_dir is None:
            ukidss_image_dir = '/pscratch/sd/s/sihany/galaxy-sommelier-data/ukidss'
        
        # Setup transforms
        transform = get_transforms('test')
        
        print("="*60)
        print("GALAXY SOMMELIER: OUT-OF-DISTRIBUTION EVALUATION")
        print("="*60)
        
        # 1. Evaluate on SDSS (in-distribution) first for comparison
        if sdss_config_path:
            print("\n1. Evaluating on SDSS (In-Distribution)...")
            from data_processing import create_data_loaders
            _, _, sdss_test_loader = create_data_loaders(sdss_config_path, sample_size=sample_size)
            
            # Convert test loader to dataset format for evaluation
            sdss_dataset = sdss_test_loader.dataset.dataset  # Get underlying dataset from Subset
            sdss_metrics = self.evaluate_dataset(sdss_dataset, 'SDSS')
        
        # 2. Evaluate on UKIDSS (out-of-distribution)
        print("\n2. Evaluating on UKIDSS (Out-of-Distribution)...")
        ukidss_dataset = UKIDSSDataset(
            catalog_path=ukidss_catalog_path,
            image_base_dir=ukidss_image_dir,
            transform=transform,
            sample_size=sample_size
        )
        
        ukidss_metrics = self.evaluate_dataset(ukidss_dataset, 'UKIDSS')
        
        # 3. Compare performance
        if 'SDSS' in self.results:
            print("\n3. Comparing Performance...")
            self.compare_performance()
        
        # 4. Create visualizations
        print("\n4. Creating Visualizations...")
        self.create_visualizations()
        
        # 5. Save results
        print("\n5. Saving Results...")
        self.save_results()
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)
        
        return self.results

def main():
    parser = argparse.ArgumentParser(description='Out-of-Distribution Evaluation for Galaxy Sommelier')
    parser.add_argument('--model-path', required=True, help='Path to the fine-tuned model')
    parser.add_argument('--config-path', required=True, help='Path to config file')
    parser.add_argument('--sdss-config-path', help='Path to SDSS config for in-distribution comparison')
    parser.add_argument('--ukidss-catalog', default='/pscratch/sd/s/sihany/galaxy-sommelier-data/catalogs/ukidss_catalog.csv')
    parser.add_argument('--ukidss-images', default='/pscratch/sd/s/sihany/galaxy-sommelier-data/ukidss')
    parser.add_argument('--output-dir', default='./ood_results')
    parser.add_argument('--sample-size', type=int, help='Sample size for testing (optional)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = OODEvaluator(
        model_path=args.model_path,
        config_path=args.config_path,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    results = evaluator.run_evaluation(
        sdss_config_path=args.sdss_config_path,
        ukidss_catalog_path=args.ukidss_catalog,
        ukidss_image_dir=args.ukidss_images,
        sample_size=args.sample_size
    )
    
    return results

if __name__ == "__main__":
    main() 