#!/bin/bash
# Triple Mixed SDSS + DECaLS + HST Training Pipeline
# Runs full fine-tuning with 3-survey mixed dataset

set -e  # Exit on error

echo "=== Galaxy Sommelier Triple Mixed (SDSS+DECaLS+HST) Training Pipeline ==="
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

# Check if triple mixed dataset exists
if [ ! -f "scripts/triple_mixed_dataset.py" ]; then
    echo "Error: Triple mixed dataset not found"
    exit 1
fi

# Check HST data availability
echo "Checking HST data availability..."
hst_png_count=$(find /pscratch/sd/s/sihany/galaxy-sommelier-data/hubble/images/ -name "*.png" | wc -l)
hst_fits_count=$(find /pscratch/sd/s/sihany/galaxy-sommelier-data/hubble/images/ -name "*.fits" | wc -l)
echo "HST PNG files: $hst_png_count"
echo "HST FITS files: $hst_fits_count"

if [ $hst_png_count -eq 0 ] && [ $hst_fits_count -eq 0 ]; then
    echo "Error: No HST images found. Please ensure HST data is available."
    exit 1
fi

echo "Step 1: Starting Head Training (Frozen Backbone)..."
echo "Configuration: configs/triple_mixed_head_training_config.yaml"
echo "Training for 5 epochs with frozen DINOv2 backbone"

# Run head training
python scripts/train_baseline.py \
    --config configs/triple_mixed_head_training_config.yaml \
    --wandb \
    --no-resume

# Check if head training completed successfully
if [ ! -f "/pscratch/sd/s/sihany/galaxy-sommelier-data/models/triple_mixed/best_model.pt" ]; then
    echo "Warning: Head training checkpoint not found at expected location"
    echo "Looking for alternative checkpoint locations..."
    find /pscratch/sd/s/sihany/galaxy-sommelier-data/models/triple_mixed -name "*.pt" -ls 2>/dev/null || echo "No triple_mixed model directory found yet"
fi

echo "Step 2: Starting Full Fine-tuning (Unfrozen Backbone)..."
echo "Configuration: configs/triple_mixed_full_finetuning_config.yaml"
echo "Training for 25 epochs with unfrozen DINOv2 backbone"

# Run full fine-tuning (will automatically load from head training checkpoint)
python scripts/train_baseline.py \
    --config configs/triple_mixed_full_finetuning_config.yaml \
    --wandb \
    --no-resume

echo "=== Triple Mixed Training Pipeline Completed ==="
echo "Final model saved at: /pscratch/sd/s/sihany/galaxy-sommelier-data/models/triple_mixed/"

# Show final model info
if [ -d "/pscratch/sd/s/sihany/galaxy-sommelier-data/models/triple_mixed/" ]; then
    echo "Final model directory contents:"
    ls -la /pscratch/sd/s/sihany/galaxy-sommelier-data/models/triple_mixed/
    echo "Training completed successfully!"
else
    echo "Warning: Final model directory not found"
    echo "Please check the training logs for errors"
fi 