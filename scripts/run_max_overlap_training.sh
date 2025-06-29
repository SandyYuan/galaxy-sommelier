#!/bin/bash
# Max Overlap SDSS + DECaLS Training Pipeline
# Runs head training followed by full fine-tuning on the max-overlap dataset.

set -e

echo "=== Galaxy Sommelier Max Overlap Training Pipeline ==="
echo "Date: $(date)"

# Check if we're in the right directory
if [ ! -f "scripts/train_baseline.py" ]; then
    echo "Error: Please run this script from the galaxy-sommelier root directory"
    exit 1
fi

echo "Step 1: Starting Head Training (Frozen Backbone)..."
echo "Configuration: configs/max_overlap_head_training_config.yaml"
echo "Training for 5 epochs with frozen DINOv2 backbone"

python scripts/train_baseline.py \
    --config configs/max_overlap_head_training_config.yaml \
    --wandb \
    --no-resume

echo "Step 2: Starting Full Fine-tuning (Unfrozen Backbone)..."
echo "Configuration: configs/max_overlap_full_finetuning_config.yaml"
echo "Training for 10 epochs with unfrozen DINOv2 backbone"

python scripts/train_baseline.py \
    --config configs/max_overlap_full_finetuning_config.yaml \
    --wandb \
    # The --no-resume flag is intentionally omitted here to allow
    # the fine-tuning to load the checkpoint from the head training.

echo "=== Max Overlap Training Pipeline Completed ==="
FINAL_MODEL_PATH="/pscratch/sd/s/sihany/galaxy-sommelier-data/models/max_overlap/best_model.pt"
echo "Final model saved at: ${FINAL_MODEL_PATH}"
echo "Results saved at: /pscratch/sd/s/sihany/galaxy-sommelier-data/results/max_overlap/"

if [ -f "${FINAL_MODEL_PATH}" ]; then
    echo "Final model file size: $(du -h ${FINAL_MODEL_PATH})"
    echo "Training completed successfully!"
else
    echo "Warning: Final model not found at ${FINAL_MODEL_PATH}"
    echo "Please check the training logs for errors"
fi 