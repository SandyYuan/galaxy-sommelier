#!/bin/bash
# SDSS-Only Training Pipeline
# Runs head training followed by full fine-tuning

set -e  # Exit on error

echo "=== Galaxy Sommelier SDSS-Only Training Pipeline ==="
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Check if we're in the right directory
if [ ! -f "scripts/train_baseline.py" ]; then
    echo "Error: Please run this script from the galaxy-sommelier root directory"
    exit 1
fi

# Check if feature mapping exists
if [ ! -f "feature_registry.py" ]; then
    echo "Error: Unified feature registry not found"
    exit 1
fi

echo "Step 1: Starting Head Training (Frozen Backbone)..."
echo "Configuration: configs/head_training_config.yaml"
echo "Training for 5 epochs with frozen DINOv2 backbone"

# Run head training
python scripts/train_baseline.py \
    --config configs/head_training_config.yaml \
    --wandb \
    --no-resume

# Check if head training completed successfully
if [ ! -f "/pscratch/sd/s/sihany/galaxy-sommelier-data/models/best_model.pt" ]; then
    echo "Warning: Head training checkpoint not found at expected location"
    echo "Looking for alternative checkpoint locations..."
    
    # List available checkpoints
    find /pscratch/sd/s/sihany/galaxy-sommelier-data/models -name "*.pt" -ls
fi

echo "Step 2: Starting Full Fine-tuning (Unfrozen Backbone)..."
echo "Configuration: configs/full_finetuning_config.yaml"
echo "Training for 10 epochs with unfrozen DINOv2 backbone"

# Run full fine-tuning (will automatically load from head training checkpoint)
python scripts/train_baseline.py \
    --config configs/full_finetuning_config.yaml \
    --wandb \
    --no-resume

echo "=== Training Pipeline Completed ==="
echo "Final model saved at: /pscratch/sd/s/sihany/galaxy-sommelier-data/models/best_model.pt"
echo "Results saved at: /pscratch/sd/s/sihany/galaxy-sommelier-data/results/"

# Show final model info
if [ -f "/pscratch/sd/s/sihany/galaxy-sommelier-data/models/best_model.pt" ]; then
    echo "Final model file size: $(du -h /pscratch/sd/s/sihany/galaxy-sommelier-data/models/best_model.pt)"
    echo "SDSS-only training completed successfully!"
else
    echo "Warning: Final model not found at expected location"
    echo "Please check the training logs for errors"
fi 