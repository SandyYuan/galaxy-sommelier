#!/usr/bin/env python3
"""
Compare SDSS-only and Mixed (SDSS+DECaLS) models on UKIDSS data,
or benchmark a single model.

Run in comparison mode:
python scripts/compare_models.py \\
    --model1_path /path/to/model1.pt \\
    --model1_config /path/to/config1.yaml \\
    --model1_features 74 \\
    --model2_path /path/to/model2.pt \\
    --model2_config /path/to/config2.yaml \\
    --model2_features 52

Run in single benchmark mode:
python scripts/compare_models.py \\
    --model1_path /path/to/model1.pt \\
    --model1_config /path/to/config1.yaml \\
    --model1_features 74
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import argparse
import json

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))
# Add root directory to path for feature_registry
sys.path.append(str(Path(__file__).parent.parent))

from model_setup import GalaxySommelier
from sdss_dataset import get_transforms
from ood_evaluation import UKIDSSDataset # Reusing dataset
from feature_registry import FeatureRegistry

# --- Configuration ---
SDSS_MODEL_PATH = "/pscratch/sd/s/sihany/galaxy-sommelier-data/models/best_model.pt"
SDSS_CONFIG_PATH = "configs/full_finetuning_config.yaml"
SDSS_NUM_FEATURES = 74 # Determined from catalog

MIXED_MODEL_PATH = "/pscratch/sd/s/sihany/galaxy-sommelier-data/models/mixed/best_model.pt"
MIXED_CONFIG_PATH = "configs/mixed_full_finetuning_config.yaml"
MIXED_NUM_FEATURES = 52 # From mixed config

UKIDSS_CATALOG_PATH = "/pscratch/sd/s/sihany/galaxy-sommelier-data/catalogs/ukidss_catalog.csv"
UKIDSS_IMAGE_DIR = "/pscratch/sd/s/sihany/galaxy-sommelier-data/ukidss"

# Legacy paths no longer needed - using centralized registry

def load_model(model_path, config_path, num_features):
    """Loads a trained model."""
    print(f"Loading model from {model_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    # Override num_outputs to be sure
    model_config['num_outputs'] = num_features

    model = GalaxySommelier(
        config=config,
        num_outputs=num_features
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model, device

def get_predictions(model, device, dataset):
    """Generate predictions for a given model and dataset."""
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            predictions = model(images)
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    return np.concatenate(all_predictions), np.concatenate(all_labels)

# get_sdss_feature_columns function removed - using centralized registry

def compute_metrics(predictions, true_labels, label_columns):
    """Compute comprehensive metrics, adapted from ood_evaluation.py"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Overall metrics
    mse = np.mean((predictions - true_labels) ** 2)
    mae = np.mean(np.abs(predictions - true_labels))
    
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
        
        task_mask = ~(np.isnan(pred_task) | np.isnan(true_task))
        if task_mask.sum() > 0:
            task_corr = np.corrcoef(pred_task[task_mask], true_task[task_mask])[0, 1]
            task_metrics[col] = {
                'correlation': float(task_corr) if not np.isnan(task_corr) else 0.0,
            }
            
    return {
        'overall': {
            'mse': float(mse),
            'mae': float(mae),
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'r_squared': float(r_squared) if not np.isnan(r_squared) else 0.0,
        },
        'per_task': task_metrics
    }

# load_shared_features function removed - using centralized registry

def get_model_info_from_config(config_path):
    """Extract model path, features, and other info from config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get checkpoint directory and look for best_model.pt
    checkpoint_dir = Path(config.get('checkpoint_dir', './models'))
    model_path = checkpoint_dir / 'best_model.pt'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Best model not found: {model_path}")
    
    # Get number of features from model config
    num_features = config['model']['num_outputs']
    
    # Determine model type for logging
    mixed_config = config.get('mixed_data', {})
    if mixed_config.get('use_mixed_dataset', False):
        dataset_name = mixed_config.get('dataset_name', 'MixedSDSSDECaLSDataset')
        if dataset_name == "MaxOverlapDataset":
            model_type = "MaxOverlapDataset"
        else:
            model_type = "Mixed SDSS+DECaLS"
    else:
        model_type = "SDSS-only"
    
    print(f"Detected {model_type} model: {model_path}")
    print(f"  Features: {num_features}")
    
    return {
        'model_path': str(model_path),
        'config_path': config_path,
        'num_features': num_features,
        'model_type': model_type,
        'config': config
    }

def get_model_feature_indices(config_path, model_path):
    """Get the model feature indices using the unified registry."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Use the unified registry to get feature indices
    return FeatureRegistry.get_feature_indices(
        model_path=model_path,
        config=config
    )

def run_comparison(args):
    """Runs the two-model comparison."""
    print("Running in comparison mode...")
    
    # Get model info from configs
    model1_info = get_model_info_from_config(args.config1)
    model2_info = get_model_info_from_config(args.config2)
    
    # Load models
    model1, device = load_model(model1_info['model_path'], model1_info['config_path'], model1_info['num_features'])
    model2, _ = load_model(model2_info['model_path'], model2_info['config_path'], model2_info['num_features'])

    # Load data
    ukidss_dataset = load_ukidss_data()

    # Get predictions
    model1_preds, true_labels = get_predictions(model1, device, ukidss_dataset)
    model2_preds, _ = get_predictions(model2, device, ukidss_dataset)
    
    print(f"\nModel 1 ({model1_info['model_type']}) predictions shape: {model1_preds.shape}")
    print(f"Model 2 ({model2_info['model_type']}) predictions shape: {model2_preds.shape}")
    print(f"True labels shape: {true_labels.shape}")
    
    # Get benchmark features (same 9 features for all models)
    benchmark_features = FeatureRegistry.get_benchmark_features()
    ukidss_label_columns = ukidss_dataset.label_columns
    
    # Get model-specific indices for these benchmark features
    model1_indices = get_model_feature_indices(model1_info['config_path'], model1_info['model_path'])
    model2_indices = get_model_feature_indices(model2_info['config_path'], model2_info['model_path'])
    
    # Validate we have the right number of features
    if len(model1_indices) != len(benchmark_features):
        print(f"WARNING: Model1 config suggests {len(benchmark_features)} features, but model has {model1_info['num_features']}")
    
    if len(model2_indices) != len(benchmark_features):
        print(f"WARNING: Model2 config suggests {len(benchmark_features)} features, but model has {model2_info['num_features']}")
    
    # Find which benchmark features are available in UKIDSS
    comparison_features = []
    ukidss_indices = []
    final_model1_indices = []
    final_model2_indices = []
    
    for i, feature in enumerate(benchmark_features):
        if feature in ukidss_label_columns:
            comparison_features.append(feature)
            ukidss_indices.append(ukidss_label_columns.index(feature))
            final_model1_indices.append(model1_indices[i])
            final_model2_indices.append(model2_indices[i])
    
    print(f"\nFound {len(comparison_features)} features common to both models and UKIDSS dataset.")
    print(f"Model 1 features: {len(benchmark_features)} total")
    print(f"Model 2 features: {len(benchmark_features)} total")
    
    # Extract relevant predictions and labels using the correct indices
    model1_preds_final = model1_preds[:, final_model1_indices]
    model2_preds_final = model2_preds[:, final_model2_indices]
    true_labels_final = true_labels[:, ukidss_indices]
    
    print(f"Model 1 final predictions: {model1_preds_final.shape}")
    print(f"Model 2 final predictions: {model2_preds_final.shape}")
    print(f"True labels final: {true_labels_final.shape}")

    print_metrics_table(model1_preds_final, model2_preds_final, true_labels_final, comparison_features)

def run_single_benchmark(args):
    """Runs a benchmark on a single model."""
    print("Running in single benchmark mode...")
    
    # Get model info from config
    model_info = get_model_info_from_config(args.config1)
    
    # Load model
    model, device = load_model(model_info['model_path'], model_info['config_path'], model_info['num_features'])

    # Load data
    ukidss_dataset = load_ukidss_data()

    # Get predictions
    predictions, true_labels = get_predictions(model, device, ukidss_dataset)

    print(f"\nPredictions shape: {predictions.shape}")
    print(f"True labels shape: {true_labels.shape}")

    # Get benchmark features and model-specific indices
    benchmark_features = FeatureRegistry.get_benchmark_features()
    model_indices = get_model_feature_indices(model_info['config_path'], model_info['model_path'])
    ukidss_label_columns = ukidss_dataset.label_columns
    
    # Find which benchmark features are available in UKIDSS
    comparison_features = []
    pred_indices = []
    label_indices = []
    
    for i, feature in enumerate(benchmark_features):
        if feature in ukidss_label_columns:
            comparison_features.append(feature)
            pred_indices.append(model_indices[i])
            label_indices.append(ukidss_label_columns.index(feature))
    
    preds_final = predictions[:, pred_indices]
    labels_final = true_labels[:, label_indices]

    print(f"Comparing {len(comparison_features)} features.")

    metrics = compute_metrics(preds_final, labels_final, comparison_features)
    
    print(f"\n--- Benchmark Results for {model_info['model_type']} Model ---")
    results_df = pd.DataFrame({
        'Metric': ['Correlation', 'R-squared', 'MAE', 'MSE'],
        'Score': [
            metrics['overall']['correlation'],
            metrics['overall']['r_squared'],
            metrics['overall']['mae'],
            metrics['overall']['mse']
        ]
    })
    print(results_df.to_string(index=False))

    print("\n\n--- Per-Task Correlation ---")
    task_corr = {f: metrics['per_task'].get(f, {}).get('correlation', 0.0) for f in comparison_features}
    task_corr_df = pd.DataFrame(task_corr.items(), columns=['Feature', 'Correlation']).sort_values(by='Correlation', ascending=False)
    print(task_corr_df.to_string(index=False))

    # Save results to output directory
    if hasattr(args, 'output_dir') and args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving results to {output_dir}...")
        
        # Save metrics
        with open(output_dir / 'benchmark_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Save predictions
        np.savez_compressed(
            output_dir / 'benchmark_predictions.npz',
            predictions=preds_final,
            true_labels=labels_final,
            label_columns=comparison_features
        )
        print("Results saved successfully.")

    print("\nBenchmark finished.")

def load_ukidss_data():
    """Loads the UKIDSS dataset."""
    print("Loading UKIDSS dataset...")
    ukidss_transform = get_transforms('val')
    dataset = UKIDSSDataset(
        catalog_path=UKIDSS_CATALOG_PATH,
        image_base_dir=UKIDSS_IMAGE_DIR,
        transform=ukidss_transform
    )
    print(f"UKIDSS dataset loaded with {len(dataset)} samples.")
    return dataset

def print_metrics_table(model1_preds, model2_preds, true_labels, shared_features):
    """Calculates and prints the metrics tables for comparison."""
    print("\n--- Computing Metrics for Model 1 (on comparison features) ---")
    model1_metrics = compute_metrics(model1_preds, true_labels, shared_features)
    
    print("\n--- Computing Metrics for Model 2 ---")
    model2_metrics = compute_metrics(model2_preds, true_labels, shared_features)
    
    # --- Display Results ---
    print("\n\n--- Model Comparison on UKIDSS Data (Shared Features) ---")
    
    results_df = pd.DataFrame({
        'Metric': ['Correlation', 'R-squared', 'MAE', 'MSE'],
        'Model 1': [
            model1_metrics['overall']['correlation'],
            model1_metrics['overall']['r_squared'],
            model1_metrics['overall']['mae'],
            model1_metrics['overall']['mse']
        ],
        'Model 2': [
            model2_metrics['overall']['correlation'],
            model2_metrics['overall']['r_squared'],
            model2_metrics['overall']['mae'],
            model2_metrics['overall']['mse']
        ]
    })
    print(results_df.to_string(index=False))

    print("\n\n--- Per-Task Correlation Comparison ---")
    
    task_corr_df = pd.DataFrame({
        'Feature': shared_features,
        'Model 1 Correlation': [model1_metrics['per_task'].get(f, {}).get('correlation', 0.0) for f in shared_features],
        'Model 2 Correlation': [model2_metrics['per_task'].get(f, {}).get('correlation', 0.0) for f in shared_features]
    })
    
    task_corr_df['Difference'] = task_corr_df['Model 2 Correlation'] - task_corr_df['Model 1 Correlation']
    task_corr_df = task_corr_df.sort_values(by='Difference', ascending=False)
    print(task_corr_df.to_string(index=False))
    print("\nComparison finished.")

def main():
    parser = argparse.ArgumentParser(description="Benchmark or compare galaxy morphology models using config files.")
    
    # Config-based approach
    parser.add_argument('--config1', required=True, type=str, help="Path to the first model's config file.")
    parser.add_argument('--config2', type=str, help="Path to the second model's config file (for comparison mode).")
    
    # Optional output directory
    parser.add_argument('--output_dir', type=str, help="Directory to save benchmark results and predictions.")

    args = parser.parse_args()

    # Decide which mode to run
    if args.config2:
        run_comparison(args)
    else:
        run_single_benchmark(args)

if __name__ == "__main__":
    main() 