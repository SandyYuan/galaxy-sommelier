#!/bin/bash
# Triple Mixed SDSS + DECaLS + HST Training Pipeline
# Runs two-stage training: head training → full fine-tuning

set -e  # Exit on error

echo "=== Galaxy Sommelier Triple Mixed (SDSS+DECaLS+HST) Training Pipeline ==="
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Check if we're in the right directory
if [ ! -f "scripts/train_baseline.py" ]; then
    echo "Error: Please run this script from the galaxy-sommelier root directory"
    exit 1
fi

# Check required files
echo "Checking required files..."
for file in "sdss_decals_feature_mapping.py" "standard_26_features.py" "scripts/triple_mixed_dataset.py"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file not found: $file"
        exit 1
    fi
done
echo "✓ All required files found"

# Check HST data availability
echo "Checking HST data availability..."
hst_png_count=$(find /pscratch/sd/s/sihany/galaxy-sommelier-data/hubble/images/ -name "cosmos_*.png" | wc -l)
echo "HST PNG files with cosmos_ pattern: $hst_png_count"

if [ $hst_png_count -lt 10000 ]; then
    echo "Warning: Only $hst_png_count HST images found. Training will use only available images."
else
    echo "✓ Good HST image availability ($hst_png_count images)"
fi

echo ""
echo "=== STAGE 1: Head Training (Frozen Backbone) ==="
echo "Configuration: configs/triple_mixed_head_training_config.yaml"
echo "Training parameters: 32 batch, 1e-4 LR, 5 epochs (same as successful mixed training)"
echo "HST filtering: Only galaxies with downloaded images will be used"

# Run head training
python scripts/train_baseline.py \
    --config configs/triple_mixed_head_training_config.yaml \
    --wandb \
    --no-resume

echo ""
echo "=== Head Training Completed ==="

# Check if head training completed successfully
head_model="/pscratch/sd/s/sihany/galaxy-sommelier-data/models/triple_mixed/best_model.pt"
if [ ! -f "$head_model" ]; then
    echo "Error: Head training checkpoint not found at $head_model"
    echo "Please check the training logs for errors"
    exit 1
fi
echo "✓ Head training checkpoint found: $head_model"

echo ""
echo "=== STAGE 2: Full Fine-tuning (Unfrozen Backbone) ==="
echo "Configuration: configs/triple_mixed_full_finetuning_config.yaml"  
echo "Training parameters: 16 batch, 1e-5 LR, 10 epochs (same as successful mixed training)"
echo "Will automatically load from head training checkpoint"

# Run full fine-tuning
python scripts/train_baseline.py \
    --config configs/triple_mixed_full_finetuning_config.yaml \
    --wandb \
    --no-resume

echo ""
echo "=== Triple Mixed Training Pipeline Completed ==="

# Check final results
final_model="/pscratch/sd/s/sihany/galaxy-sommelier-data/models/triple_mixed/best_model.pt"
if [ -f "$final_model" ]; then
    echo "✓ Final model saved: $final_model"
    echo "✓ Training completed successfully!"
    
    # Show model directory contents
    echo ""
    echo "Final model directory contents:"
    ls -la /pscratch/sd/s/sihany/galaxy-sommelier-data/models/triple_mixed/
else
    echo "Warning: Final model not found at expected location"
    echo "Please check the training logs for errors"
fi

echo ""
echo "Next steps:"
echo "1. Run benchmark evaluation to compare with SDSS-only and Mixed models"
echo "2. Test generalization on UKIDSS out-of-distribution data"
echo "3. Create performance comparison plots" 