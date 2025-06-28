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

from model_setup import GalaxySommelier
from data_processing import get_transforms
from ood_evaluation import UKIDSSDataset # Reusing dataset

# --- Configuration ---
SDSS_MODEL_PATH = "/pscratch/sd/s/sihany/galaxy-sommelier-data/models/best_model.pt"
SDSS_CONFIG_PATH = "configs/full_finetuning_config.yaml"
SDSS_NUM_FEATURES = 74 # Determined from catalog

MIXED_MODEL_PATH = "/pscratch/sd/s/sihany/galaxy-sommelier-data/models/mixed/best_model.pt"
MIXED_CONFIG_PATH = "configs/mixed_full_finetuning_config.yaml"
MIXED_NUM_FEATURES = 52 # From mixed config

UKIDSS_CATALOG_PATH = "/pscratch/sd/s/sihany/galaxy-sommelier-data/catalogs/ukidss_catalog.csv"
UKIDSS_IMAGE_DIR = "/pscratch/sd/s/sihany/galaxy-sommelier-data/ukidss"

SHARED_FEATURES_FILE = Path(__file__).parent / "mixed_feature_indices.txt"
SDSS_CATALOG_PATH = "/pscratch/sd/s/sihany/galaxy-sommelier-data/catalogs/gz2_master_catalog_corrected.csv"

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

def get_sdss_feature_columns(catalog_path):
    """Gets the full list of SDSS feature columns."""
    df = pd.read_csv(catalog_path, nrows=0)
    return [col for col in df.columns if '_fraction' in col and col.startswith('t')]

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

def load_shared_features(filepath):
    """Load shared feature names from the text file."""
    features = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            feature_name = line.split(' -> ')[0]
            features.append(feature_name)
    return features

def run_comparison(args):
    """Runs the two-model comparison."""
    print("Running in comparison mode...")
    
    # Load models
    model1, device = load_model(args.model1_path, args.model1_config, args.model1_features)
    model2, _ = load_model(args.model2_path, args.model2_config, args.model2_features)

    # Load data
    ukidss_dataset = load_ukidss_data()

    # Get predictions
    preds1, true_labels = get_predictions(model1, device, ukidss_dataset)
    preds2, _ = get_predictions(model2, device, ukidss_dataset)

    # ... (rest of comparison logic as before) ...
    # This part needs to be adapted to use the generic preds1, preds2 etc.
    # For now, let's assume model1 is SDSS and model2 is Mixed for simplicity of refactoring.
    
    sdss_preds, mixed_preds = preds1, preds2
    
    print(f"Model 1 predictions shape: {sdss_preds.shape}")
    print(f"Model 2 predictions shape: {mixed_preds.shape}")
    print(f"True labels shape: {true_labels.shape}")
    
    # The rest of the logic from the original main function follows...
    ukidss_label_columns = ukidss_dataset.label_columns
    all_shared_features = load_shared_features(SHARED_FEATURES_FILE)
    comparison_features = [f for f in all_shared_features if f in ukidss_label_columns and "fraction" in f]
    print(f"\nFound {len(comparison_features)} features common to all models and the UKIDSS dataset for comparison.")

    sdss_all_features = get_sdss_feature_columns(SDSS_CATALOG_PATH)
    sdss_indices = [sdss_all_features.index(f) for f in comparison_features]
    sdss_preds_final = sdss_preds[:, sdss_indices]

    mixed_indices = [all_shared_features.index(f) for f in comparison_features]
    mixed_preds_final = mixed_preds[:, mixed_indices]

    true_labels_indices = [ukidss_label_columns.index(f) for f in comparison_features]
    true_labels_final = true_labels[:, true_labels_indices]

    print(f"  True labels: {true_labels_final.shape}")

    print_metrics_table(sdss_preds_final, mixed_preds_final, true_labels_final, comparison_features)

def run_single_benchmark(args):
    """Runs a benchmark on a single model."""
    print("Running in single benchmark mode...")
    
    # Load model
    model, device = load_model(args.model1_path, args.model1_config, args.model1_features)

    # Load data
    ukidss_dataset = load_ukidss_data()

    # Get predictions
    predictions, true_labels = get_predictions(model, device, ukidss_dataset)

    print(f"\nPredictions shape: {predictions.shape}")
    print(f"True labels shape: {true_labels.shape}")

    # For a single model, we compare all its outputs against available labels
    model_output_features = get_sdss_feature_columns(SDSS_CATALOG_PATH)[:predictions.shape[1]]
    ukidss_label_columns = ukidss_dataset.label_columns
    
    comparison_features = [f for f in model_output_features if f in ukidss_label_columns]
    
    pred_indices = [model_output_features.index(f) for f in comparison_features]
    label_indices = [ukidss_label_columns.index(f) for f in comparison_features]
    
    preds_final = predictions[:, pred_indices]
    labels_final = true_labels[:, label_indices]

    print(f"Comparing {len(comparison_features)} features.")

    metrics = compute_metrics(preds_final, labels_final, comparison_features)
    
    print("\n--- Benchmark Results on UKIDSS Data ---")
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
    if args.output_dir:
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
    parser = argparse.ArgumentParser(description="Benchmark or compare galaxy morphology models.")
    
    # Model 1 (required)
    parser.add_argument('--model1_path', required=True, type=str, help="Path to the first model checkpoint.")
    parser.add_argument('--model1_config', required=True, type=str, help="Path to the first model's config file.")
    parser.add_argument('--model1_features', required=True, type=int, help="Number of output features for the first model.")

    # Model 2 (optional, for comparison)
    parser.add_argument('--model2_path', type=str, help="Path to the second model checkpoint.")
    parser.add_argument('--model2_config', type=str, help="Path to the second model's config file.")
    parser.add_argument('--model2_features', type=int, help="Number of output features for the second model.")
    
    # Output directory (optional)
    parser.add_argument('--output_dir', type=str, help="Directory to save benchmark results and predictions.")

    args = parser.parse_args()

    # Decide which mode to run
    if args.model2_path and args.model2_config and args.model2_features:
        run_comparison(args)
    else:
        run_single_benchmark(args)

if __name__ == "__main__":
    main() 