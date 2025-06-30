#!/bin/bash
# High-Quality Mixed SDSS + DECaLS Training Pipeline
# Runs head training followed by full fine-tuning with high-quality SDSS selection

set -e  # Exit on error

echo "=== Galaxy Sommelier High-Quality Mixed SDSS+DECaLS Training Pipeline ==="
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Check if we're in the right directory
if [ ! -f "scripts/train_baseline.py" ]; then
    echo "Error: Please run this script from the galaxy-sommelier root directory"
    exit 1
fi

# Check if feature mapping exists
if [ ! -f "sdss_decals_feature_mapping.py" ]; then
    echo "Error: SDSS-DECaLS feature mapping not found"
    exit 1
fi

echo "Step 1: Starting High-Quality Head Training (Frozen Backbone)..."
echo "Configuration: configs/mixed_high_quality_head_training_config.yaml"
echo "Training for 5 epochs with frozen DINOv2 backbone"
echo "Using SDSS galaxies with highest classification counts"

# Run head training
python scripts/train_baseline.py \
    --config configs/mixed_high_quality_head_training_config.yaml \
    --wandb \
    --no-resume

# Check if head training completed successfully
if [ ! -f "/pscratch/sd/s/sihany/galaxy-sommelier-data/models/mixed_high_quality/best_model.pt" ]; then
    echo "Warning: Head training checkpoint not found at expected location"
    echo "Looking for alternative checkpoint locations..."
    
    # List available checkpoints
    find /pscratch/sd/s/sihany/galaxy-sommelier-data/models -name "*.pt" -ls
fi

echo "Step 2: Starting High-Quality Full Fine-tuning (Unfrozen Backbone)..."
echo "Configuration: configs/mixed_high_quality_config.yaml"
echo "Training for 10 epochs with unfrozen DINOv2 backbone"
echo "Loading from high-quality head training checkpoint"

# Run full fine-tuning (will automatically load from head training checkpoint)
python scripts/train_baseline.py \
    --config configs/mixed_high_quality_config.yaml \
    --wandb \
    --no-resume

echo "=== High-Quality Training Pipeline Completed ==="
echo "Final model saved at: /pscratch/sd/s/sihany/galaxy-sommelier-data/models/mixed_high_quality/best_model.pt"
echo "Results saved at: /pscratch/sd/s/sihany/galaxy-sommelier-data/results/mixed_high_quality/"

# Show final model info
if [ -f "/pscratch/sd/s/sihany/galaxy-sommelier-data/models/mixed_high_quality/best_model.pt" ]; then
    echo "Final model file size: $(du -h /pscratch/sd/s/sihany/galaxy-sommelier-data/models/mixed_high_quality/best_model.pt)"
    echo "High-quality training completed successfully!"
else
    echo "Warning: Final model not found at expected location"
    echo "Please check the training logs for errors"
fi 