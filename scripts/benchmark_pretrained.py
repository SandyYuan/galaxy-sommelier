#!/usr/bin/env python3
"""
Benchmark Pre-trained DINOv2 Model (Frozen Backbone)
Creates a baseline for comparison with fine-tuned model.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from simple_benchmark import GalaxyBenchmark
from model_setup import GalaxySommelier
import yaml

def create_pretrained_baseline(config_path, output_dir='./benchmark_results'):
    """
    Create a pre-trained baseline model for comparison.
    This loads DINOv2 with frozen backbone and randomly initialized head.
    """
    print("Creating pre-trained baseline model...")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with frozen backbone
    model_config = config['model']
    model = GalaxySommelier(
        config=config,
        model_name=model_config['name'],
        num_outputs=model_config['num_outputs'],
        dropout_rate=model_config['dropout_rate'],
        freeze_backbone=True  # Keep backbone frozen
    )
    
    model.to(device)
    
    # Save the model for benchmarking
    baseline_dir = Path('./models')
    baseline_dir.mkdir(exist_ok=True)
    baseline_path = baseline_dir / 'pretrained_baseline.pt'
    
    # Save just the state dict
    torch.save(model.state_dict(), baseline_path)
    
    print(f"Pre-trained baseline model saved to: {baseline_path}")
    return baseline_path

def main():
    """Main function to create and benchmark pre-trained baseline"""
    parser = argparse.ArgumentParser(description="Benchmark pre-trained DINOv2 baseline")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--output", default="./benchmark_results", help="Output directory")
    parser.add_argument("--compare-with", help="Path to fine-tuned model for comparison")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Benchmarking Pre-trained DINOv2 Baseline")
    print("=" * 60)
    
    # Create pre-trained baseline
    baseline_path = create_pretrained_baseline(args.config, args.output)
    
    # Benchmark the baseline
    print("\nRunning benchmark on pre-trained baseline...")
    baseline_benchmark = GalaxyBenchmark(
        model_path=baseline_path,
        config_path=args.config,
        output_dir=args.output
    )
    baseline_benchmark.run_benchmark()
    
    # If comparison model provided, benchmark it too
    if args.compare_with:
        print("\n" + "=" * 60)
        print("Benchmarking Fine-tuned Model for Comparison")
        print("=" * 60)
        
        finetuned_benchmark = GalaxyBenchmark(
            model_path=args.compare_with,
            config_path=args.config,
            output_dir=args.output
        )
        finetuned_benchmark.run_benchmark()
        
        # Create comparison report
        create_comparison_report(
            baseline_benchmark.output_dir,
            finetuned_benchmark.output_dir,
            args.output
        )

def create_comparison_report(baseline_dir, finetuned_dir, output_dir):
    """Create a comparison report between baseline and fine-tuned models"""
    import json
    
    print("\nCreating comparison report...")
    
    # Load results
    baseline_results = json.load(open(baseline_dir / 'benchmark_results.json'))
    finetuned_results = json.load(open(finetuned_dir / 'benchmark_results.json'))
    
    # Create comparison
    comparison_file = Path(output_dir) / 'model_comparison.txt'
    with open(comparison_file, 'w') as f:
        f.write("Galaxy Sommelier Model Comparison\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall metrics comparison
        baseline_overall = baseline_results['overall_metrics']
        finetuned_overall = finetuned_results['overall_metrics']
        
        f.write("Overall Performance Comparison:\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Metric':<15} {'Baseline':<12} {'Fine-tuned':<12} {'Improvement':<12}\n")
        f.write("-" * 55 + "\n")
        
        for metric in ['correlation', 'mse', 'mae', 'r_squared']:
            baseline_val = baseline_overall[metric]
            finetuned_val = finetuned_overall[metric]
            
            if metric in ['correlation', 'r_squared']:
                improvement = ((finetuned_val - baseline_val) / baseline_val) * 100
                f.write(f"{metric:<15} {baseline_val:<12.4f} {finetuned_val:<12.4f} {improvement:+.1f}%\n")
            else:  # MSE, MAE (lower is better)
                improvement = ((baseline_val - finetuned_val) / baseline_val) * 100
                f.write(f"{metric:<15} {baseline_val:<12.4f} {finetuned_val:<12.4f} {improvement:+.1f}%\n")
        
        # Key features comparison if available
        if 'key_features_analysis' in baseline_results and 'key_features_analysis' in finetuned_results:
            f.write(f"\n\nKey Features Performance Comparison:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Feature':<25} {'Baseline r':<12} {'Fine-tuned r':<12} {'Improvement':<12}\n")
            f.write("-" * 65 + "\n")
            
            baseline_features = baseline_results['key_features_analysis']
            finetuned_features = finetuned_results['key_features_analysis']
            
            for feature_name in baseline_features.keys():
                if feature_name in finetuned_features:
                    baseline_r = baseline_features[feature_name]['correlation']
                    finetuned_r = finetuned_features[feature_name]['correlation']
                    improvement = ((finetuned_r - baseline_r) / abs(baseline_r)) * 100 if baseline_r != 0 else 0
                    
                    display_name = feature_name.replace('_', ' ').title()
                    f.write(f"{display_name:<25} {baseline_r:<12.3f} {finetuned_r:<12.3f} {improvement:+.1f}%\n")
    
    print(f"Comparison report saved to: {comparison_file}")

if __name__ == "__main__":
    main() 