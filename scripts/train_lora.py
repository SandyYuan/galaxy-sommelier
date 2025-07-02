#!/usr/bin/env python3
"""
LoRA Training Script for Galaxy Sommelier
Trains DINOv2-based model using LoRA adapters for efficient fine-tuning.
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

# Try to import PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    logging.warning("PEFT library not available. Install with: pip install peft")
    PEFT_AVAILABLE = False

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

class GalaxyLoRATrainer:
    """LoRA training class for Galaxy Sommelier"""
    
    def __init__(self, config_path, sample_size=None, use_wandb=False, resume=True):
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library is required for LoRA training. Install with: pip install peft")
            
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
                    name=f"lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        """Initialize the Galaxy Sommelier model with LoRA adapters"""
        logger.info("Setting up model with LoRA adapters...")
        
        model_config = self.config['model']
        
        # Create base model (always freeze backbone for LoRA)
        self.base_model = GalaxySommelier(
            config=self.config,
            model_name=model_config['name'],
            num_outputs=model_config['num_outputs'],
            dropout_rate=model_config['dropout_rate'],
            freeze_backbone=True  # Always freeze for LoRA
        )
        
        # Configure LoRA
        lora_config = self.get_lora_config()
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.to(self.device)
        
        # Load from pretrained checkpoint if specified
        if 'pretrained_checkpoint' in self.config:
            checkpoint_path = self.config['pretrained_checkpoint']
            if os.path.exists(checkpoint_path):
                logger.info(f"Loading pretrained checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                
                # For LoRA, we need to be careful about loading weights
                if 'model_state_dict' in checkpoint:
                    # Try to load compatible weights
                    try:
                        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        logger.info("Successfully loaded compatible weights from checkpoint")
                    except Exception as e:
                        logger.warning(f"Could not load checkpoint weights: {e}")
                        logger.info("Continuing with freshly initialized LoRA adapters")
        
        # Count parameters
        param_info = self.count_lora_parameters()
        logger.info(f"LoRA Model - Total: {param_info['total_parameters']:,}, "
                   f"Trainable: {param_info['trainable_parameters']:,} "
                   f"({param_info['trainable_percentage']:.2f}%)")
        
        if self.use_wandb:
            self.wandb.log(param_info)
    
    def get_lora_config(self):
        """Get LoRA configuration with conservative settings"""
        lora_config_dict = self.config.get('lora', {})
        
        # Conservative defaults
        return LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_config_dict.get('rank', 16),  # Conservative rank
            lora_alpha=lora_config_dict.get('alpha', 32),  # Standard scaling
            lora_dropout=lora_config_dict.get('dropout', 0.1),
            target_modules=lora_config_dict.get('target_modules', [
                "query", "value", "key", "dense"  # Standard attention modules
            ]),
            bias=lora_config_dict.get('bias', "none"),  # Conservative: no bias adaptation
        )
    
    def count_lora_parameters(self):
        """Count LoRA parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'trainable_percentage': (trainable_params / total_params) * 100
        }
    
    def setup_data_loaders(self):
        """Setup train and validation data loaders - reuse from baseline"""
        logger.info("Setting up data loaders...")
        
        # Use same data loading logic as baseline
        self.train_loader, self.val_loader = create_data_loaders(
            config=self.config,
            train_transform=self.train_transform,
            val_transform=self.val_transform,
            sample_size=self.sample_size
        )
        
        logger.info(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        
    def setup_optimizer(self):
        """Setup optimizer and scheduler for LoRA parameters"""
        optimizer_config = self.config['optimizer']
        
        # Only optimize trainable (LoRA) parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if optimizer_config['type'] == 'adamw':
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 0.01),
                betas=optimizer_config.get('betas', [0.9, 0.999])
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['type']}")
        
        # Setup scheduler
        scheduler_config = self.config.get('scheduler', {})
        if scheduler_config.get('type') == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 10),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        else:
            self.scheduler = None
            
        logger.info(f"Optimizer: {optimizer_config['type']}, "
                   f"LR: {optimizer_config['learning_rate']}, "
                   f"Trainable params: {len(trainable_params)}")
    
    def load_latest_checkpoint(self):
        """Load latest checkpoint if available"""
        checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        checkpoint_pattern = f"*lora*checkpoint*.pt"
        
        checkpoints = list(checkpoint_dir.glob(checkpoint_pattern))
        if not checkpoints:
            logger.info("No LoRA checkpoints found, starting fresh")
            return
        
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading checkpoint: {latest_checkpoint}")
        
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=self.device, weights_only=False)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.training_history = checkpoint.get('training_history', [])
            
            logger.info(f"Resumed from epoch {self.current_epoch}")
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            logger.info("Starting fresh training")
    
    def get_early_stopping_config(self):
        """Get early stopping configuration"""
        return self.config.get('early_stopping', {
            'patience': 10,
            'min_delta': 1e-4,
            'monitor': 'val_loss'
        })
    
    def get_early_stopping_mode(self):
        """Get early stopping mode (min or max)"""
        monitor = self.get_early_stopping_config()['monitor']
        return 'min' if 'loss' in monitor or 'error' in monitor else 'max'
    
    def check_early_stopping(self, current_metrics):
        """Check if early stopping criteria are met"""
        config = self.get_early_stopping_config()
        patience = config['patience']
        min_delta = config['min_delta']
        monitor = config['monitor']
        mode = self.get_early_stopping_mode()
        
        if monitor not in current_metrics:
            logger.warning(f"Early stopping metric '{monitor}' not found in metrics")
            return False
        
        current_value = current_metrics[monitor]
        
        if mode == 'min':
            improved = current_value < (self.best_val_metric - min_delta)
        else:
            improved = current_value > (self.best_val_metric + min_delta)
        
        if improved:
            self.best_val_metric = current_value
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            logger.info(f"Early stopping counter: {self.early_stopping_counter}/{patience}")
            return self.early_stopping_counter >= patience
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['images'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward pass with autocast
            with autocast():
                predictions = self.model(images)
                loss_dict = self.criterion(predictions, targets)
                loss = loss_dict['total_loss']
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 50 == 0:
                self.wandb.log({
                    'train_batch_loss': loss.item(),
                    'epoch': self.current_epoch + 1,
                    'batch': batch_idx
                })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['images'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                with autocast():
                    predictions = self.model(images)
                    loss_dict = self.criterion(predictions, targets)
                    loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
        correlation = torch.corrcoef(torch.stack([
            all_predictions.flatten(), 
            all_targets.flatten()
        ]))[0, 1].item()
        
        metrics = {
            'val_loss': avg_loss,
            'val_mae': mae,
            'val_correlation': correlation
        }
        
        return metrics
    
    def save_checkpoint(self, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"lora_checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / "lora_best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: {best_path}")
    
    def train(self, num_epochs=None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        logger.info(f"Starting LoRA training for {num_epochs} epochs")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Check if best model
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            # Save checkpoint
            self.save_checkpoint(val_metrics, is_best)
            
            # Log metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                **val_metrics
            }
            
            self.training_history.append(epoch_metrics)
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs}: "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_metrics['val_loss']:.4f}, "
                       f"Val Corr: {val_metrics['val_correlation']:.4f}")
            
            if self.use_wandb:
                self.wandb.log(epoch_metrics)
            
            # Early stopping check
            if self.check_early_stopping(val_metrics):
                logger.info("Early stopping triggered")
                break
        
        logger.info("Training completed!")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Galaxy Sommelier with LoRA')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Number of samples to use for training (for debugging)')
    parser.add_argument('--wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--no-resume', action='store_true',
                        help='Do not resume from checkpoint')
    
    args = parser.parse_args()
    
    # Check if PEFT is available
    if not PEFT_AVAILABLE:
        logger.error("PEFT library is not available. Install with: pip install peft")
        sys.exit(1)
    
    # Initialize trainer
    trainer = GalaxyLoRATrainer(
        config_path=args.config,
        sample_size=args.sample_size,
        use_wandb=args.wandb,
        resume=not args.no_resume
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 