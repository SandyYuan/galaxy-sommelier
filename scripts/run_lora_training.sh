#!/bin/bash
# LoRA SDSS Training Pipeline
# Conservative LoRA fine-tuning for efficient training

set -e  # Exit on error

echo "=== Galaxy Sommelier LoRA Training Pipeline ==="
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Check if we're in the right directory
if [ ! -f "scripts/train_lora.py" ]; then
    echo "Error: Please run this script from the galaxy-sommelier root directory"
    exit 1
fi

# Check if PEFT is installed
python -c "import peft" 2>/dev/null || {
    echo "Error: PEFT library not found. Install with: pip install peft"
    exit 1
}

# Check if feature mapping exists
if [ ! -f "feature_registry.py" ]; then
    echo "Error: Unified feature registry not found"
    exit 1
fi

echo "Starting LoRA Training on Mixed SDSS+DECaLS Dataset..."
echo "Configuration: configs/lora_head_training_config.yaml"
echo "Using conservative LoRA settings (rank=16, alpha=32)"
echo "Dataset: 50% SDSS + 50% DECaLS with 26 standardized features"

# Run LoRA training
python scripts/train_lora.py \
    --config configs/lora_head_training_config.yaml \
    --wandb \
    --no-resume

# Check if training completed successfully
if [ -f "/pscratch/sd/s/sihany/galaxy-sommelier-data/models/lora/lora_best_model.pt" ]; then
    echo "=== LoRA Training Completed Successfully ==="
    echo "Model saved at: /pscratch/sd/s/sihany/galaxy-sommelier-data/models/lora/lora_best_model.pt"
    echo "Results saved at: /pscratch/sd/s/sihany/galaxy-sommelier-data/results/lora/"
    
    # Show model file size
    echo "Model file size: $(du -h /pscratch/sd/s/sihany/galaxy-sommelier-data/models/lora/lora_best_model.pt)"
    
    # Show parameter efficiency
    echo ""
    echo "LoRA Benefits:"
    echo "- Parameter Efficiency: Only ~1-5% of parameters trained"
    echo "- Faster Training: Reduced memory usage and computation"
    echo "- Better Generalization: Preserves pre-trained features"
    
else
    echo "Warning: LoRA model not found at expected location"
    echo "Please check the training logs for errors"
    exit 1
fi 