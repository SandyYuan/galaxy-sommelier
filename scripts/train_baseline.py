#!/usr/bin/env python3
"""
Baseline Training Script for Galaxy Sommelier
Trains DINOv2-based model on Galaxy Zoo data.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
from torchvision import transforms

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from model_setup import GalaxySommelier, GalaxyZooLoss, save_model_checkpoint, count_parameters
from sdss_dataset import create_data_loaders

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_transforms(config, mode='train'):
    """Gets the transformations for the dataset based on config."""
    image_size = config['preprocessing']['image_size']
    normalize_mean = config['preprocessing']['normalize_mean']
    normalize_std = config['preprocessing']['normalize_std']
    
    if mode == 'train':
        aug_config = config.get('augmentation', {})
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5 if aug_config.get('horizontal_flip', False) else 0),
            transforms.RandomVerticalFlip(p=0.5 if aug_config.get('vertical_flip', False) else 0),
            transforms.RandomRotation(degrees=aug_config.get('rotation_degrees', 0)),
            transforms.ColorJitter(
                brightness=aug_config.get('brightness_factor', 0),
                contrast=aug_config.get('contrast_factor', 0)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
    else: # 'val' or 'test'
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])

class GalaxyTrainer:
    """Main training class for Galaxy Sommelier"""
    
    def __init__(self, config_path, sample_size=None, use_wandb=False, resume=True):
        self.config_path = Path(config_path)
        self.sample_size = sample_size
        self.use_wandb = use_wandb
        self.resume = resume
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        logger.info(f"Using device: {self.device}")
        
        # Initialize wandb if requested
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config['wandb']['project'],
                    config=self.config,
                    name=f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                self.wandb = wandb
                logger.info("Weights & Biases initialized")
            except ImportError:
                logger.warning("wandb not available, skipping logging")
                self.use_wandb = False
        
        # Define transforms
        self.train_transform = get_transforms(self.config, mode='train')
        self.val_transform = get_transforms(self.config, mode='val')

        # Setup model
        self.setup_model()
        
        # Setup data loaders
        self.setup_data_loaders()
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Setup loss function
        self.criterion = GalaxyZooLoss()
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        # Early stopping state
        self.early_stopping_counter = 0
        self.best_val_metric = float('inf') if self.get_early_stopping_mode() == 'min' else float('-inf')
        
        # Try to resume from checkpoint if requested
        if self.resume:
            self.load_latest_checkpoint()
        
    def setup_model(self):
        """Initialize the Galaxy Sommelier model"""
        logger.info("Setting up model...")
        
        model_config = self.config['model']
        
        self.model = GalaxySommelier(
            config=self.config,
            model_name=model_config['name'],
            num_outputs=model_config['num_outputs'],
            dropout_rate=model_config['dropout_rate'],
            freeze_backbone=model_config.get('freeze_backbone', False)
        )
        
        self.model.to(self.device)
        
        # Load from pretrained checkpoint if specified
        if 'pretrained_checkpoint' in self.config:
            checkpoint_path = self.config['pretrained_checkpoint']
            if os.path.exists(checkpoint_path):
                logger.info(f"Loading pretrained checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                logger.info("Successfully loaded pretrained checkpoint")
            else:
                logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
        
        # Count parameters
        param_info = count_parameters(self.model)
        
        if self.use_wandb:
            self.wandb.log(param_info)
    
    def setup_data_loaders(self):
        """Setup train and validation data loaders"""
        logger.info("Setting up data loaders...")
        
        # Check if using mixed dataset
        mixed_config = self.config.get('mixed_data', {})
        use_mixed = mixed_config.get('use_mixed_dataset', False)
        
        if use_mixed:
            dataset_name = mixed_config.get('dataset_name', 'MixedSDSSDECaLSDataset') # Default to original
            logger.info(f"Using mixed dataset loader: {dataset_name}")

            if dataset_name == "MaxOverlapDataset":
                from max_overlap_dataset import MaxOverlapDataset # Import our new class
                
                # These paths now come directly from the config
                sdss_cat_path = self.config['data']['sdss_catalog_path']
                decals_cat_path = self.config['data']['decals_catalog_path']
                
                # Create datasets
                train_dataset = MaxOverlapDataset(
                    sdss_catalog_path=sdss_cat_path,
                    decals_catalog_path=decals_cat_path,
                    sdss_image_dir=self.config['data']['sdss_dir'],
                    decals_image_dir=self.config['data']['decals_dir'],
                    transform=self.train_transform,
                    feature_set=mixed_config.get('feature_set', 'sdss')
                )
                
                # For this dataset, we can use a simple random split for validation
                # as the catalogs are already finalized
                val_split = 0.1
                num_train = int((1.0 - val_split) * len(train_dataset))
                num_val = len(train_dataset) - num_train
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    train_dataset, [num_train, num_val],
                    generator=torch.Generator().manual_seed(42)
                )
                
                # Set transforms after split
                self.train_dataset.dataset.transform = self.train_transform
                self.val_dataset.dataset.transform = self.val_transform
                
                # Create data loaders
                batch_size = self.config['training']['batch_size']
                self.train_loader = torch.utils.data.DataLoader(
                    self.train_dataset, batch_size=batch_size, shuffle=True,
                    num_workers=self.config.get('num_workers', 4),
                    pin_memory=True
                )
                self.val_loader = torch.utils.data.DataLoader(
                    self.val_dataset, batch_size=batch_size, shuffle=False,
                    num_workers=self.config.get('num_workers', 4),
                    pin_memory=True
                )
                
            elif dataset_name == "TripleMixedDataset":
                from triple_mixed_dataset import get_triple_mixed_data_loaders
                
                logger.info("Using TripleMixedDataset with standardized 26 features")
                
                # Create transforms dict
                transforms_dict = {
                    'train': self.train_transform,
                    'val': self.val_transform,
                    'test': self.val_transform
                }
                
                # Create data loaders using the standardized approach
                self.train_loader, self.val_loader, self.test_loader = get_triple_mixed_data_loaders(
                    self.config, transforms_dict
                )
                
                logger.info("TripleMixedDataset data loaders created successfully")
                
            else:
                from mixed_dataset import create_mixed_data_loaders
                
                transforms_dict = {
                    'train': self.train_transform,
                    'val': self.val_transform,
                    'test': self.val_transform
                }
                
                # Use the standard mixed dataset (SDSS + DECaLS)
                self.train_loader, self.val_loader, self.test_loader = create_mixed_data_loaders(
                    self.config, transforms_dict, sample_size=self.sample_size
                )
        else:
            # Use standard SDSS Galaxy Zoo dataset
            transforms_dict = {
                'train': self.train_transform,
                'val': self.val_transform,
                'test': self.val_transform
            }
            
            self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
                self.config, transforms_dict, sample_size=self.sample_size
            )
        
        logger.info(f"Data loaders created - Train: {len(self.train_loader)} batches, "
                   f"Val: {len(self.val_loader)} batches")
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler with differential learning rates"""
        training_config = self.config['training']
        
        # Check if we're doing full fine-tuning with different LRs
        if 'backbone_lr' in training_config and 'head_lr' in training_config:
            # Differential learning rates for backbone vs head
            backbone_params = []
            head_params = []
            
            for name, param in self.model.named_parameters():
                if 'dinov2' in name:  # Backbone parameters
                    backbone_params.append(param)
                else:  # Head parameters
                    head_params.append(param)
            
            # Create parameter groups with different learning rates
            param_groups = [
                {
                    'params': backbone_params,
                    'lr': float(training_config['backbone_lr']),
                    'weight_decay': float(training_config['weight_decay'])
                },
                {
                    'params': head_params,
                    'lr': float(training_config['head_lr']),
                    'weight_decay': float(training_config['weight_decay'])
                }
            ]
            
            self.optimizer = optim.AdamW(param_groups)
            
            logger.info(f"Differential Learning Rates:")
            logger.info(f"  Backbone LR: {training_config['backbone_lr']}")
            logger.info(f"  Head LR: {training_config['head_lr']}")
            
        else:
            # Standard single learning rate
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=float(training_config['learning_rate']),
                weight_decay=float(training_config['weight_decay'])
            )
            
            logger.info(f"Single Learning Rate: {training_config['learning_rate']}")
        
        # Scheduler
        total_steps = len(self.train_loader) * training_config['num_epochs']
        
        # Use base learning rate for scheduler
        base_lr = float(training_config.get('backbone_lr', training_config['learning_rate']))
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=base_lr * 0.01
        )
        
        logger.info(f"Scheduler: CosineAnnealingLR, Total steps: {total_steps}")
    
    def load_latest_checkpoint(self):
        """Load the most recent checkpoint to resume training"""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', './models'))
        if not checkpoint_dir.exists():
            logger.info("No checkpoint directory found, starting from scratch")
            return
        
        # Find all checkpoint files
        checkpoint_files = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if not checkpoint_files:
            logger.info("No checkpoint files found, starting from scratch")
            return
        
        # Sort by epoch number and get the latest
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        latest_checkpoint = checkpoint_files[-1]
        
        logger.info(f"Loading checkpoint from {latest_checkpoint}")
        
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=self.device, weights_only=False)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training state
            self.current_epoch = checkpoint['epoch'] + 1  # Start from next epoch
            self.best_val_loss = checkpoint.get('val_loss', float('inf'))
            
            # Load early stopping state if available
            self.early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
            self.best_val_metric = checkpoint.get('best_val_metric', 
                                                 float('inf') if self.get_early_stopping_mode() == 'min' else float('-inf'))
            
            # Load scaler state if available
            if 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Load training history if available
            history_path = Path(self.config.get('results_dir', './results')) / 'training_history.json'
            if history_path.exists():
                with open(history_path, 'r') as f:
                    self.training_history = json.load(f)
            
            logger.info(f"Resumed training from epoch {self.current_epoch}")
            logger.info(f"Best validation loss so far: {self.best_val_loss:.4f}")
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            logger.info("Starting training from scratch")
            self.current_epoch = 0
            self.best_val_loss = float('inf')
            self.training_history = []
    
    def get_early_stopping_config(self):
        """Get early stopping configuration with defaults"""
        es_config = self.config.get('early_stopping', {})
        return {
            'patience': es_config.get('patience', 5),
            'min_delta': es_config.get('min_delta', 0.001),
            'monitor': es_config.get('monitor', 'val_loss'),
            'mode': es_config.get('mode', 'min')  # 'min' for loss, 'max' for correlation
        }
    
    def get_early_stopping_mode(self):
        """Get early stopping mode"""
        es_config = self.get_early_stopping_config()
        monitor = es_config['monitor']
        if 'loss' in monitor.lower():
            return 'min'
        elif 'correlation' in monitor.lower() or 'accuracy' in monitor.lower():
            return 'max'
        else:
            return es_config.get('mode', 'min')
    
    def check_early_stopping(self, current_metrics):
        """Check if we should stop training based on early stopping criteria"""
        es_config = self.get_early_stopping_config()
        monitor_metric = es_config['monitor']
        patience = es_config['patience']
        min_delta = es_config['min_delta']
        mode = self.get_early_stopping_mode()
        
        # Get current metric value
        current_value = current_metrics.get(monitor_metric)
        if current_value is None:
            logger.warning(f"Early stopping metric '{monitor_metric}' not found in metrics")
            return False
        
        # Check for improvement
        improved = False
        if mode == 'min':
            if current_value < self.best_val_metric - min_delta:
                improved = True
                self.best_val_metric = current_value
        else:  # mode == 'max'
            if current_value > self.best_val_metric + min_delta:
                improved = True
                self.best_val_metric = current_value
        
        if improved:
            self.early_stopping_counter = 0
            logger.info(f"Early stopping metric improved: {monitor_metric} = {current_value:.4f}")
        else:
            self.early_stopping_counter += 1
            logger.info(f"Early stopping metric did not improve: {monitor_metric} = {current_value:.4f} "
                       f"(counter: {self.early_stopping_counter}/{patience})")
        
        # Check if we should stop
        should_stop = self.early_stopping_counter >= patience
        if should_stop:
            logger.info(f"Early stopping triggered! No improvement in {monitor_metric} "
                       f"for {patience} epochs.")
        
        return should_stop
    
    def train_epoch(self):
        """Run one training epoch"""
        self.model.train()
        total_loss = 0
        
        loop = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for i, batch_data in enumerate(loop):
            self.optimizer.zero_grad()
            
            # Handle both dict and tuple/list batch formats for backward compatibility
            if isinstance(batch_data, dict):
                images = batch_data['image'].to(self.device, non_blocking=True)
                labels = batch_data['labels'].to(self.device, non_blocking=True)
                weights = batch_data['weight'].to(self.device, non_blocking=True)
            else:
                images, labels, weights = batch_data
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                weights = weights.to(self.device, non_blocking=True)
            
            with autocast():
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, labels, weights)
                loss = loss_dict['total_loss']
            
            self.scaler.scale(loss).backward()
            
            # Unscale and clip gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['gradient_clip_norm']
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Scheduler step
            self.scheduler.step()
            
            # Accumulate losses
            total_loss += loss.item()
            
            # Update progress bar
            loop.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"})
            
            # Log to wandb
            if self.use_wandb and i % self.config['wandb']['log_freq'] == 0:
                log_metrics = {
                    'train_loss_step': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'epoch': self.current_epoch,
                    'step': i + self.current_epoch * len(self.train_loader)
                }
                log_metrics.update({
                    f'train_{k}_step': v.item() if torch.is_tensor(v) else v 
                    for k, v in loss_dict.items() if k != 'total_loss'
                })
                self.wandb.log(log_metrics)
        
        return {'train_loss': total_loss / len(self.train_loader)}
    
    def validate(self):
        """Run one validation epoch"""
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_labels = []
        
        loop = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        
        with torch.no_grad():
            for batch_data in loop:
                # Handle both dict and tuple/list batch formats for backward compatibility
                if isinstance(batch_data, dict):
                    images = batch_data['image'].to(self.device, non_blocking=True)
                    labels = batch_data['labels'].to(self.device, non_blocking=True)
                    weights = batch_data['weight'].to(self.device, non_blocking=True)
                else:
                    images, labels, weights = batch_data
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    weights = weights.to(self.device, non_blocking=True)

                with autocast():
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, labels, weights)
                    loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Calculate average losses
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate additional metrics
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Mean Absolute Error
        mae = float(np.mean(np.abs(all_outputs - all_labels)))
        
        # Correlation coefficient
        correlation = float(np.corrcoef(all_outputs.flatten(), all_labels.flatten())[0, 1])
        
        return {
            'val_loss': avg_loss,
            'val_mae': mae,
            'val_correlation': correlation
        }
    
    def save_checkpoint(self, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', './models'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint with additional state
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_loss': metrics['val_loss'],
            'config': self.config,
            'training_history': self.training_history,
            'early_stopping_counter': self.early_stopping_counter,
            'best_val_metric': self.best_val_metric
        }
        
        # Regular checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{self.current_epoch:03d}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Best model checkpoint
        if is_best:
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with validation loss: {metrics['val_loss']:.4f}")
    
    def train(self, num_epochs=None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        start_epoch = self.current_epoch
        total_epochs = start_epoch + num_epochs
        
        if start_epoch > 0:
            logger.info(f"Resuming training from epoch {start_epoch + 1}")
            logger.info(f"Training for {num_epochs} more epochs (until epoch {total_epochs})")
        else:
            logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(start_epoch, total_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Log metrics
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"  Val MAE: {val_metrics['val_mae']:.4f}")
            logger.info(f"  Val Correlation: {val_metrics['val_correlation']:.4f}")
            
            # Log to wandb
            if self.use_wandb:
                epoch_metrics['epoch'] = epoch
                self.wandb.log(epoch_metrics)
            
            # Save training history
            self.training_history.append(epoch_metrics)
            
            # Check for best model
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            # Save checkpoint
            if (epoch + 1) % self.config['logging']['save_frequency'] == 0 or is_best:
                self.save_checkpoint(epoch_metrics, is_best)
            
            # Check for early stopping
            if self.check_early_stopping(epoch_metrics):
                logger.info("Early stopping triggered. Training stopped.")
                break
        
        # Save final training history
        history_path = Path(self.config.get('results_dir', './results')) / 'training_history.json'
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        if self.use_wandb:
            self.wandb.finish()

def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Galaxy Sommelier Baseline Training")
    parser.add_argument('--config', type=str, required=True, help="Path to the training configuration file")
    parser.add_argument('--sample_size', type=int, default=None, help="Use a subset of the data for quick testing")
    parser.add_argument('--wandb', action='store_true', help="Enable Weights & Biases logging")
    parser.add_argument('--no-resume', action='store_false', dest='resume', help="Do not resume from latest checkpoint")
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Loading configuration from: {config_path}")
    
    # Initialize trainer
    trainer = GalaxyTrainer(
        config_path=args.config,
        sample_size=args.sample_size,
        use_wandb=args.wandb,
        resume=args.resume
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 