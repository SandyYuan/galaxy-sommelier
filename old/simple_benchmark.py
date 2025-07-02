#!/usr/bin/env python3
"""
Simple Benchmark Script for Galaxy Sommelier
Evaluates fine-tuned model performance with clear, interpretable metrics.
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
from sklearn.metrics import confusion_matrix, classification_report
import json

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from model_setup import GalaxySommelier
from sdss_dataset import create_data_loaders

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GalaxyBenchmark:
    """Simple benchmark suite for Galaxy Sommelier model"""
    
    def __init__(self, model_path, config_path, output_dir='./benchmark_results', max_features=None, feature_indices=None):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        
        # Create model-specific subdirectory based on model name
        model_name = self.model_path.stem  # e.g., 'best_model' from 'best_model.pt'
        self.output_dir = Path(output_dir) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Benchmark results will be saved to: {self.output_dir}")
        
        # Define our 6 key morphological features
        self.key_features = {
            'disk_fraction': 't01_smooth_or_features_a02_features_or_disk_fraction',
            'spiral_fraction': 't04_spiral_a08_spiral_fraction',
            'bar_fraction': 't03_bar_a06_bar_fraction', 
            'bulge_dominant_fraction': 't05_bulge_prominence_a13_dominant_fraction',
            'edge_on_fraction': 't02_edgeon_a04_yes_fraction',
            'odd_features_fraction': 't06_odd_a14_yes_fraction'
        }
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.load_model()
        
        # Get test data
        self.setup_test_data()
        
        # Map feature names to indices
        self.map_key_features()
        
        # Results storage
        self.results = {}
        
        self.max_features = max_features
        self.feature_indices = feature_indices
        
        if feature_indices is not None:
            print(f"Using specific feature indices: {len(feature_indices)} features")
        elif max_features is not None:
            print(f"Using max features constraint: {max_features} features")
        
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
        
        # Load checkpoint (allow non-weights data since this is our trusted model)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully")
    
    def setup_test_data(self):
        """Setup test data loader"""
        print("Setting up test data...")
        
        # Get data loaders (we'll use the test split)
        _, _, self.test_loader = create_data_loaders(
            self.config_path,
            sample_size=None  # Use full test set
        )
        
        print(f"Test set: {len(self.test_loader)} batches")
    
    def map_key_features(self):
        """Map key feature names to column indices"""
        # Get label columns from the dataset
        temp_dataset = self.test_loader.dataset.dataset  # Get underlying dataset from Subset
        label_columns = temp_dataset.label_columns
        
        self.key_feature_indices = {}
        self.key_feature_names = {}
        
        for display_name, column_name in self.key_features.items():
            if column_name in label_columns:
                idx = label_columns.index(column_name)
                self.key_feature_indices[display_name] = idx
                self.key_feature_names[display_name] = column_name
                print(f"Mapped {display_name} -> column {idx}: {column_name}")
            else:
                print(f"Warning: {column_name} not found in dataset")
        
        print(f"Successfully mapped {len(self.key_feature_indices)}/6 key features")
    
    def predict_test_set(self):
        """Generate predictions on test set"""
        print("Generating predictions on test set...")
        
        all_predictions = []
        all_labels = []
        all_objids = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Predicting"):
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                objids = batch['objid']
                
                # Forward pass
                predictions = self.model(images)
                
                # Store results
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_objids.extend(objids)
        
        # Concatenate all results
        self.predictions = np.concatenate(all_predictions, axis=0)
        self.true_labels = np.concatenate(all_labels, axis=0)
        self.objids = all_objids
        
        print(f"Predictions generated for {len(self.predictions)} galaxies")
        print(f"Shape: predictions={self.predictions.shape}, labels={self.true_labels.shape}")
    
    def compute_regression_metrics(self):
        """Compute overall regression metrics"""
        print("Computing regression metrics...")
        
        # Handle shape mismatch OR max_features constraint OR specific feature indices
        if self.feature_indices is not None:
            print(f"Using specific feature indices: {len(self.feature_indices)} features")
            predictions_subset = self.predictions
            true_labels_subset = self.true_labels[:, self.feature_indices]
        elif self.max_features is not None:
            print(f"Max features constraint: Using only first {self.max_features} features")
            predictions_subset = self.predictions[:, :self.max_features]
            true_labels_subset = self.true_labels[:, :self.max_features]
        elif self.predictions.shape[1] != self.true_labels.shape[1]:
            print(f"Shape mismatch detected:")
            print(f"  Predictions: {self.predictions.shape}")
            print(f"  Labels: {self.true_labels.shape}")
            print(f"  Using only the first {self.predictions.shape[1]} features for evaluation")
            
            # Use only the features that the model was trained on
            predictions_subset = self.predictions
            true_labels_subset = self.true_labels[:, :self.predictions.shape[1]]
        else:
            predictions_subset = self.predictions
            true_labels_subset = self.true_labels
        
        # Overall metrics
        mse = np.mean((predictions_subset - true_labels_subset) ** 2)
        mae = np.mean(np.abs(predictions_subset - true_labels_subset))
        
        # Flatten for correlation
        pred_flat = predictions_subset.flatten()
        true_flat = true_labels_subset.flatten()
        
        # Remove any NaN values
        mask = ~(np.isnan(pred_flat) | np.isnan(true_flat))
        pred_clean = pred_flat[mask]
        true_clean = true_flat[mask]
        
        correlation = np.corrcoef(pred_clean, true_clean)[0, 1]
        r_squared = correlation ** 2
        
        self.results['overall_metrics'] = {
            'mse': float(mse),
            'mae': float(mae),
            'correlation': float(correlation),
            'r_squared': float(r_squared),
            'n_samples': len(pred_clean),
            'n_features_evaluated': predictions_subset.shape[1]
        }
        
        print(f"Overall Metrics:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  R²: {r_squared:.4f}")
        print(f"  Features evaluated: {predictions_subset.shape[1]}")
        
        # Store the subset for other methods to use
        self.predictions_subset = predictions_subset
        self.true_labels_subset = true_labels_subset
    
    def compute_task_metrics(self):
        """Compute per-task metrics"""
        print("Computing per-task metrics...")
        
        # Use subset data if available (for mixed models or max_features constraint)
        predictions_to_use = getattr(self, 'predictions_subset', self.predictions)
        labels_to_use = getattr(self, 'true_labels_subset', self.true_labels)
        
        # Get task names from data loader
        # For now, use generic task names
        n_tasks = predictions_to_use.shape[1]
        task_names = [f'task_{i:02d}' for i in range(n_tasks)]
        
        task_metrics = {}
        
        for i, task_name in enumerate(task_names):
            pred_task = predictions_to_use[:, i]
            true_task = labels_to_use[:, i]
            
            # Remove NaN values
            mask = ~(np.isnan(pred_task) | np.isnan(true_task))
            pred_clean = pred_task[mask]
            true_clean = true_task[mask]
            
            if len(pred_clean) > 0:
                mse = np.mean((pred_clean - true_clean) ** 2)
                mae = np.mean(np.abs(pred_clean - true_clean))
                correlation = np.corrcoef(pred_clean, true_clean)[0, 1] if len(pred_clean) > 1 else 0
                r_squared = correlation ** 2
                
                task_metrics[task_name] = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'correlation': float(correlation),
                    'r_squared': float(r_squared),
                    'n_samples': len(pred_clean)
                }
        
        self.results['task_metrics'] = task_metrics
        
        # Print top and bottom performing tasks
        sorted_tasks = sorted(task_metrics.items(), key=lambda x: x[1]['correlation'], reverse=True)
        
        print("\nTop 5 performing tasks (by correlation):")
        for task_name, metrics in sorted_tasks[:5]:
            print(f"  {task_name}: r={metrics['correlation']:.3f}, MAE={metrics['mae']:.4f}")
        
        print("\nBottom 5 performing tasks (by correlation):")
        for task_name, metrics in sorted_tasks[-5:]:
            print(f"  {task_name}: r={metrics['correlation']:.3f}, MAE={metrics['mae']:.4f}")
    
    def create_scatter_plots(self):
        """Create predicted vs true scatter plots for our 6 key features"""
        print("Creating scatter plots for 6 key morphological features...")
        
        # Use subset labels if available (for mixed models)
        labels_to_use = getattr(self, 'true_labels_subset', self.true_labels)
        
        # Create subplots (2x3 grid for 6 features)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (feature_name, feature_idx) in enumerate(self.key_feature_indices.items()):
            ax = axes[idx]
            
            pred_task = self.predictions[:, feature_idx]
            true_task = labels_to_use[:, feature_idx]
            
            # Remove NaN values
            mask = ~(np.isnan(pred_task) | np.isnan(true_task))
            pred_clean = pred_task[mask]
            true_clean = true_task[mask]
            
            # Create scatter plot with small, faint points
            ax.scatter(true_clean, pred_clean, s=0.5, alpha=0.3, color='blue')
            
            # Add 1:1 line
            min_val = min(true_clean.min(), pred_clean.min())
            max_val = max(true_clean.max(), pred_clean.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            # Get metrics for this feature
            task_metrics = self.results['task_metrics']
            task_key = f'task_{feature_idx:02d}'
            if task_key in task_metrics:
                metrics = task_metrics[task_key]
                correlation = metrics['correlation']
                mse = metrics['mse']
            else:
                correlation = np.corrcoef(pred_clean, true_clean)[0, 1]
                mse = np.mean((pred_clean - true_clean) ** 2)
            
            # Annotations
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{feature_name.replace("_", " ").title()}\nr={correlation:.3f}, MSE={mse:.4f}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'key_features_scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Key feature scatter plots saved to {self.output_dir / 'key_features_scatter_plots.png'}")
    
    def compare_distributions(self):
        """Compare predicted vs true distributions for our 6 key features"""
        print("Comparing distributions for 6 key morphological features...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        ks_stats = {}
        
        for idx, (feature_name, feature_idx) in enumerate(self.key_feature_indices.items()):
            ax = axes[idx]
            
            pred_task = self.predictions[:, feature_idx]
            true_task = self.true_labels[:, feature_idx]
            
            # Remove NaN values
            mask = ~(np.isnan(pred_task) | np.isnan(true_task))
            pred_clean = pred_task[mask]
            true_clean = true_task[mask]
            
            # Plot histograms
            ax.hist(true_clean, bins=50, alpha=0.7, label='True', density=True, color='blue')
            ax.hist(pred_clean, bins=50, alpha=0.7, label='Predicted', density=True, color='red')
            
            # KS test
            ks_stat, ks_p = stats.ks_2samp(true_clean, pred_clean)
            ks_stats[feature_name] = {'ks_stat': float(ks_stat), 'ks_p': float(ks_p)}
            
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.set_title(f'{feature_name.replace("_", " ").title()}\nKS stat={ks_stat:.3f}, p={ks_p:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'key_features_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.results['key_features_ks_tests'] = ks_stats
        print(f"Key feature distributions saved to {self.output_dir / 'key_features_distributions.png'}")
        
    def analyze_key_features(self):
        """Detailed analysis of our 6 key morphological features"""
        print("Analyzing key morphological features...")
        
        key_feature_metrics = {}
        
        for feature_name, feature_idx in self.key_feature_indices.items():
            pred_task = self.predictions[:, feature_idx]
            true_task = self.true_labels[:, feature_idx]
            
            # Remove NaN values
            mask = ~(np.isnan(pred_task) | np.isnan(true_task))
            pred_clean = pred_task[mask]
            true_clean = true_task[mask]
            
            if len(pred_clean) > 0:
                mse = np.mean((pred_clean - true_clean) ** 2)
                mae = np.mean(np.abs(pred_clean - true_clean))
                correlation = np.corrcoef(pred_clean, true_clean)[0, 1] if len(pred_clean) > 1 else 0
                r_squared = correlation ** 2
                
                # Additional metrics
                rmse = np.sqrt(mse)
                mean_pred = np.mean(pred_clean)
                mean_true = np.mean(true_clean)
                bias = mean_pred - mean_true
                
                key_feature_metrics[feature_name] = {
                    'column_name': self.key_feature_names[feature_name],
                    'mse': float(mse),
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'correlation': float(correlation),
                    'r_squared': float(r_squared),
                    'bias': float(bias),
                    'mean_predicted': float(mean_pred),
                    'mean_true': float(mean_true),
                    'n_samples': len(pred_clean)
                }
        
        self.results['key_features_analysis'] = key_feature_metrics
        
        # Print summary
        print("\nKey Features Performance Summary:")
        print("=" * 80)
        for feature_name, metrics in key_feature_metrics.items():
            print(f"{feature_name.replace('_', ' ').title():25s} | "
                  f"r={metrics['correlation']:.3f} | "
                  f"MAE={metrics['mae']:.4f} | "
                  f"Bias={metrics['bias']:+.4f}")
        print("=" * 80)
    
    def analyze_main_morphology_task(self):
        """Analyze the main morphology classification (t01-like task)"""
        print("Analyzing main morphology classification...")
        
        # Assume first 3 tasks are the main morphology classes (smooth, featured, star)
        if self.predictions.shape[1] >= 3:
            # Get the first 3 columns as main morphology
            main_pred = self.predictions[:, :3]
            main_true = self.true_labels[:, :3]
            
            # Convert to categorical (argmax)
            pred_classes = np.argmax(main_pred, axis=1)
            true_classes = np.argmax(main_true, axis=1)
            
            # Confusion matrix
            cm = confusion_matrix(true_classes, pred_classes)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Smooth', 'Featured', 'Star'],
                       yticklabels=['Smooth', 'Featured', 'Star'])
            plt.title('Main Morphology Classification\nConfusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(self.output_dir / 'morphology_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Classification accuracy
            accuracy = np.mean(pred_classes == true_classes)
            
            # Per-class metrics
            class_report = classification_report(true_classes, pred_classes, 
                                               target_names=['Smooth', 'Featured', 'Star'],
                                               output_dict=True)
            
            self.results['morphology_classification'] = {
                'accuracy': float(accuracy),
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report
            }
            
            print(f"Main morphology classification accuracy: {accuracy:.3f}")
            print("Confusion matrix saved")
    
    def save_results(self):
        """Save all results to files"""
        print("Saving results...")
        
        # Save detailed results as JSON
        results_file = self.output_dir / 'benchmark_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create summary report
        summary_file = self.output_dir / 'summary_report.txt'
        with open(summary_file, 'w') as f:
            f.write("Galaxy Sommelier Benchmark Results\n")
            f.write("=" * 40 + "\n\n")
            
            # Overall metrics
            overall = self.results['overall_metrics']
            f.write("Overall Performance:\n")
            f.write(f"  MSE: {overall['mse']:.6f}\n")
            f.write(f"  MAE: {overall['mae']:.6f}\n")
            f.write(f"  Correlation: {overall['correlation']:.4f}\n")
            f.write(f"  R²: {overall['r_squared']:.4f}\n")
            f.write(f"  Samples: {overall['n_samples']}\n")
            f.write(f"  Features evaluated: {overall['n_features_evaluated']}\n\n")
            
            # Top performing tasks
            task_metrics = self.results['task_metrics']
            sorted_tasks = sorted(task_metrics.items(), key=lambda x: x[1]['correlation'], reverse=True)
            
            f.write("Top 10 Performing Tasks (by correlation):\n")
            for i, (task_name, metrics) in enumerate(sorted_tasks[:10]):
                f.write(f"  {i+1:2d}. {task_name}: r={metrics['correlation']:.3f}, "
                       f"MAE={metrics['mae']:.4f}, MSE={metrics['mse']:.6f}\n")
            
            # Morphology classification
            if 'morphology_classification' in self.results:
                morph = self.results['morphology_classification']
                f.write(f"\nMain Morphology Classification:\n")
                f.write(f"  Accuracy: {morph['accuracy']:.3f}\n")
        
        print(f"Results saved to {self.output_dir}")
    
    def run_benchmark(self):
        """Run complete benchmark suite focused on key morphological features"""
        print("Starting Galaxy Sommelier Benchmark...")
        print("Focusing on 6 key morphological features:")
        for display_name, column_name in self.key_features.items():
            print(f"  - {display_name.replace('_', ' ').title()}")
        print("=" * 50)
        
        # Generate predictions
        self.predict_test_set()
        
        # Compute metrics
        self.compute_regression_metrics()
        self.compute_task_metrics()
        
        # Analyze our key features specifically
        self.analyze_key_features()
        
        # Create visualizations focused on key features
        self.create_scatter_plots()
        self.compare_distributions()
        self.analyze_main_morphology_task()
        
        # Save results
        self.save_results()
        
        print("\n" + "=" * 50)
        print("Benchmark completed!")
        print(f"Results saved to: {self.output_dir}")
        print("\nKey outputs:")
        print(f"  - Key features scatter plots: key_features_scatter_plots.png")
        print(f"  - Key features distributions: key_features_distributions.png") 
        print(f"  - Summary report: summary_report.txt")
        print(f"  - Detailed results: benchmark_results.json")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Galaxy Sommelier Model Benchmark")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--output", default="./benchmark_results", help="Output directory")
    parser.add_argument("--max-features", type=int, help="Limit evaluation to first N features (for fair comparison)")
    parser.add_argument("--feature-indices", nargs='+', type=int, help="Specific feature indices to evaluate")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Galaxy Sommelier Model Benchmark")
    print("=" * 50)
    
    # Create benchmark instance
    benchmark = GalaxyBenchmark(
        model_path=args.model,
        config_path=args.config,
        output_dir=args.output,
        max_features=args.max_features,
        feature_indices=args.feature_indices
    )
    
    # Run benchmark
    benchmark.run_benchmark()
    
    print(f"\nBenchmark completed!")
    print(f"Results saved to: {benchmark.output_dir}")
    print(f"\nKey outputs:")
    print(f"  - Key features scatter plots: key_features_scatter_plots.png")
    print(f"  - Key features distributions: key_features_distributions.png") 
    print(f"  - Summary report: summary_report.txt")
    print(f"  - Detailed results: benchmark_results.json")

if __name__ == "__main__":
    main() 