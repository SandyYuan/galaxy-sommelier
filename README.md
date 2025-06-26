# Galaxy Sommelier: Vision Transformer for Galaxy Morphology Classification

A state-of-the-art galaxy morphology classifier using DINOv2 fine-tuning for robust generalization across astronomical surveys.

## Project Overview

Galaxy Sommelier leverages the power of Vision Transformers (DINOv2) to classify galaxy morphologies using Galaxy Zoo citizen science data. The model is designed to generalize well across different astronomical surveys including SDSS, DESI Legacy Imaging, and HST.

## Results

Our training progression demonstrates significant improvements across all morphological classification tasks:

### Performance Summary
- **Overall Correlation**: 0.85 (R² = 0.72)
- **Mean Absolute Error**: 0.106 
- **Main Morphology Classification Accuracy**: 62.5%

### Training Stage Progression
| Stage | Overall Correlation | Overall R² | MAE | Notes |
|-------|-------------------|------------|-----|--------|
| Pretrained Baseline | 0.116 | 0.013 | 0.403 | Poor performance across all features |
| Head Training | 0.759 | 0.576 | 0.153 | Major improvement, learned basic concepts |
| Full Fine-tuning | **0.850** | **0.722** | **0.106** | Best performance, refined all features |

### Key Morphological Features Performance
- **Disk Fraction** (smooth vs featured): r = 0.968 (excellent)
- **Edge-on Detection**: r = 0.935 (excellent) 
- **Odd Features Detection**: r = 0.932 (excellent)
- **Spiral Detection**: r = 0.857 (very good)
- **Bar Detection**: r = 0.772 (good)
- **Bulge Dominance**: r = 0.506 (moderate - most challenging feature)

The model successfully learned to classify most galaxy morphological characteristics, with geometric and structural features showing the strongest performance. Bulge prominence assessment remains the most challenging task, likely requiring additional specialized techniques.

### Performance Visualization

**Overall Performance Comparison**
![Performance Summary](comparison_plots/performance_summary.png)

**Distribution Comparison: True vs Predicted**
![Distribution Comparison](comparison_plots/distribution_comparison.png)

The plots above demonstrate the dramatic improvement from pretrained baseline through head training to full fine-tuning. The distribution comparison shows how well the final model captures the underlying data distributions for key morphological features.

## Phase 1: Foundation Setup ✅

**Status**: Implementation Complete

### Completed Components

- [x] Project structure setup
- [x] Data directories on NERSC scratch space
- [x] Galaxy Zoo data downloader (`scripts/download_galaxy_zoo_data.py`)
- [x] DINOv2-based model architecture (`scripts/model_setup.py`)
- [x] Data processing pipeline (`scripts/data_processing.py`)
- [x] Training infrastructure (`scripts/train_baseline.py`)
- [x] Configuration management (`configs/base_config.yaml`)
- [x] Requirements specification (`requirements.txt`)

### Key Features

- **DINOv2 Backbone**: Pre-trained Vision Transformer for robust feature extraction
- **Galaxy Zoo Integration**: Automated download and processing of Galaxy Zoo catalogs
- **SDSS Image Pipeline**: FITS image loading and preprocessing
- **Mixed Precision Training**: Efficient training with automatic mixed precision
- **Comprehensive Logging**: Integration with Weights & Biases for experiment tracking
- **Flexible Configuration**: YAML-based configuration system
<!-- 
## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n galaxy-sommelier python=3.10
conda activate galaxy-sommelier

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

```bash
# Download Galaxy Zoo catalogs
python scripts/download_galaxy_zoo_data.py --download-catalogs --sample-size 1000

# Download SDSS images
python scripts/download_galaxy_zoo_data.py --download-images --sample-size 100
```

### 3. Train Baseline Model

```bash
# Train with sample data
python scripts/train_baseline.py --sample-size 100 --epochs 5

# Full training with W&B logging
python scripts/train_baseline.py --wandb
``` -->

## Project Structure

```
galaxy-sommelier/
├── configs/                 # Configuration files
│   └── base_config.yaml    # Base training configuration
├── data/                   # Symbolic link to scratch storage
├── models/                 # Model checkpoints
├── results/                # Training results and logs
├── scripts/                # Core implementation
│   ├── download_galaxy_zoo_data.py  # Data acquisition
│   ├── model_setup.py               # Model architecture
│   ├── data_processing.py           # Data pipeline
│   └── train_baseline.py            # Training script
├── notebooks/              # Analysis notebooks
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

<!-- ## Configuration

The project uses YAML configuration files for easy parameter management. Key settings:

- **Model**: DINOv2 variant, output dimensions, dropout rate
- **Training**: Learning rate, batch size, number of epochs
- **Data**: Paths to catalogs and images, preprocessing options
- **Logging**: Checkpoint frequency, W&B settings

## Data Storage

- **Scratch Directory**: `/pscratch/sd/s/sihany/galaxy-sommelier-data/`
- **SDSS Images**: `data/sdss/`
- **Catalogs**: `data/catalogs/`
- **Processed Data**: `data/processed/` -->

## Hardware Requirements

- **GPU**: NVIDIA A100 (available on NERSC Perlmutter)
- **Memory**: 16GB+ RAM recommended
- **Storage**: 100GB+ for full Galaxy Zoo dataset

## Next Steps (Phase 2)

- [ ] DESI Legacy Imaging data collection
- [ ] Cross-survey matching and validation
- [ ] Distribution shift analysis
- [ ] OOD testing framework

<!-- ## Contributing

This project follows the plan outlined in `plan.txt`. Please refer to the plan for detailed implementation roadmap and contribute according to the phase structure. -->

## License

Research use only. Please cite appropriately if using this code for academic purposes.
<!-- 
## Contact

Project developed as part of galaxy morphology research at NERSC. -->
