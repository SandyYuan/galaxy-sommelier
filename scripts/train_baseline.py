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

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from model_setup import GalaxySommelier, GalaxyZooLoss, save_model_checkpoint, count_parameters
from data_processing import create_data_loaders

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            logger.info("Using mixed SDSS+DECaLS dataset")
            from mixed_dataset import create_mixed_data_loaders
            
            sdss_fraction = mixed_config.get('sdss_fraction', 0.5)
            logger.info(f"SDSS fraction: {sdss_fraction:.1%}")
            
            self.train_loader, self.val_loader, self.test_loader = create_mixed_data_loaders(
                self.config_path, 
                sdss_fraction=sdss_fraction,
                sample_size=self.sample_size
            )
        else:
            logger.info("Using standard SDSS dataset")
            self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
                self.config_path, 
                sample_size=self.sample_size
            )
        
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
        
        # Log data info
        wandb_data = {
            'train_batches': len(self.train_loader),
            'val_batches': len(self.val_loader),
            'sample_size': self.sample_size or 'full'
        }
        
        if use_mixed:
            wandb_data.update({
                'dataset_type': 'mixed_sdss_decals',
                'sdss_fraction': mixed_config.get('sdss_fraction', 0.5)
            })
        else:
            wandb_data['dataset_type'] = 'sdss_only'
            
        if self.use_wandb:
            self.wandb.log(wandb_data)
    
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
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_mse_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            weights = batch['weight'].to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                predictions = self.model(images)
                
                # Compute loss
                loss_dict = self.criterion(predictions, labels, weights)
                loss = loss_dict['total_loss']
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
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
            total_mse_loss += loss_dict['mse_loss'].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mse': f"{loss_dict['mse_loss'].item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % self.config['wandb']['log_freq'] == 0:
                self.wandb.log({
                    'train_loss_step': loss.item(),
                    'train_mse_step': loss_dict['mse_loss'].item(),
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'epoch': self.current_epoch,
                    'step': batch_idx + self.current_epoch * len(self.train_loader)
                })
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_mse_loss = total_mse_loss / num_batches
        
        return {
            'train_loss': avg_loss,
            'train_mse_loss': avg_mse_loss
        }
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        total_mse_loss = 0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                weights = batch['weight'].to(self.device, non_blocking=True)
                
                # Forward pass
                with autocast():
                    predictions = self.model(images)
                    loss_dict = self.criterion(predictions, labels, weights)
                
                # Accumulate losses
                total_loss += loss_dict['total_loss'].item()
                total_mse_loss += loss_dict['mse_loss'].item()
                num_batches += 1
                
                # Store predictions for metrics
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_mse_loss = total_mse_loss / num_batches
        
        # Calculate additional metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Mean Absolute Error
        mae = torch.mean(torch.abs(all_predictions - all_labels)).item()
        
        # Correlation coefficient
        pred_flat = all_predictions.flatten()
        label_flat = all_labels.flatten()
        correlation = torch.corrcoef(torch.stack([pred_flat, label_flat]))[0, 1].item()
        
        return {
            'val_loss': avg_loss,
            'val_mse_loss': avg_mse_loss,
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
            'training_history': self.training_history
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
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Train Galaxy Sommelier baseline model")
    parser.add_argument("--config", default="configs/base_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Use subset of data for testing")
    parser.add_argument("--wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs to train")
    parser.add_argument("--no-resume", action="store_true",
                       help="Start training from scratch instead of resuming")
    
    args = parser.parse_args()
    
    # Check if config exists
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        return
    
    try:
        # Initialize trainer
        trainer = GalaxyTrainer(
            config_path=args.config,
            sample_size=args.sample_size,
            use_wandb=args.wandb,
            resume=not args.no_resume
        )
        
        # Start training
        trainer.train(num_epochs=args.epochs)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 