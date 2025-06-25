#!/usr/bin/env python3
"""
Galaxy Sommelier Model Setup
Defines the main model architecture using DINOv2 backbone with galaxy morphology head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model, Dinov2Config, AutoImageProcessor
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class GalaxySommelier(nn.Module):
    """
    Galaxy morphology classifier based on DINOv2
    
    Architecture:
    - DINOv2 backbone for feature extraction
    - Custom classification head for Galaxy Zoo vote fractions
    - Batch normalization and dropout for regularization
    """
    
    def __init__(self, config=None, model_name='facebook/dinov2-base', 
                 num_outputs=37, dropout_rate=0.2, freeze_backbone=False):
        super().__init__()
        
        # Store configuration
        self.config = config or {}
        self.model_name = model_name
        self.num_outputs = num_outputs
        self.dropout_rate = dropout_rate
        self._is_backbone_frozen = freeze_backbone
        
        # Load pre-trained DINOv2
        logger.info(f"Loading DINOv2 model: {model_name}")
        self.dinov2 = Dinov2Model.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Freeze backbone if requested
        if freeze_backbone:
            logger.info("Freezing DINOv2 backbone parameters")
            for param in self.dinov2.parameters():
                param.requires_grad = False
        
        # Get feature dimension
        hidden_size = self.dinov2.config.hidden_size
        logger.info(f"DINOv2 hidden size: {hidden_size}")
        
        # Galaxy morphology specific head with batch norm
        self.morphology_head = nn.Sequential(
            # First layer: hidden_size -> 512
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Second layer: 512 -> 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Output layer: 256 -> num_outputs
            nn.Linear(256, num_outputs),
            nn.Sigmoid()  # For vote fractions [0, 1]
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Model initialized with {num_outputs} outputs")
        
    def _init_weights(self):
        """Initialize the morphology head weights using Xavier initialization"""
        for module in self.morphology_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, pixel_values, return_features=False, return_attention=False):
        """
        Forward pass through the model
        
        Args:
            pixel_values: Input images [batch_size, channels, height, width]
            return_features: Whether to return intermediate features
            return_attention: Whether to return attention maps
            
        Returns:
            morphology_predictions: Galaxy morphology vote fractions
            features (optional): Intermediate feature representations
            attention_maps (optional): Attention maps from DINOv2
        """
        # Extract features using DINOv2
        outputs = self.dinov2(
            pixel_values=pixel_values,
            output_attentions=return_attention,
            output_hidden_states=return_features
        )
        
        # Get pooled features (CLS token representation)
        features = outputs.pooler_output
        
        # Pass through morphology classification head
        morphology_predictions = self.morphology_head(features)
        
        # Prepare return values
        result = morphology_predictions
        
        if return_features or return_attention:
            result = [morphology_predictions]
            
            if return_features:
                result.append(features)
                
            if return_attention:
                result.append(outputs.attentions)
                
            result = tuple(result)
        
        return result
    
    def get_feature_extractor(self):
        """Get the feature extraction part of the model"""
        return self.dinov2
    
    def get_classifier_head(self):
        """Get the classification head"""
        return self.morphology_head
    
    def freeze_backbone(self):
        """Freeze the DINOv2 backbone"""
        for param in self.dinov2.parameters():
            param.requires_grad = False
        self._is_backbone_frozen = True
        logger.info("Backbone frozen")
            
    def unfreeze_backbone(self):
        """Unfreeze the DINOv2 backbone"""
        for param in self.dinov2.parameters():
            param.requires_grad = True
        self._is_backbone_frozen = False
        logger.info("Backbone unfrozen")
    
    def get_trainable_parameters(self):
        """Get number of trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }

class GalaxyZooLoss(nn.Module):
    """
    Custom loss function for Galaxy Zoo vote fractions
    Combines MSE loss with additional regularization
    """
    
    def __init__(self, vote_weight=1.0, consistency_weight=0.1):
        super().__init__()
        self.vote_weight = vote_weight
        self.consistency_weight = consistency_weight
        
    def forward(self, predictions, targets, vote_counts=None):
        """
        Compute loss for Galaxy Zoo predictions
        
        Args:
            predictions: Model predictions [batch_size, num_outputs]
            targets: Target vote fractions [batch_size, num_outputs]
            vote_counts: Total vote counts for weighting (optional)
        """
        # Main MSE loss on vote fractions
        mse_loss = F.mse_loss(predictions, targets, reduction='none')
        
        # Weight by vote counts if provided
        if vote_counts is not None:
            # Normalize vote counts to use as weights
            weights = vote_counts / vote_counts.max()
            weights = weights.unsqueeze(-1)  # [batch_size, 1]
            mse_loss = mse_loss * weights
        
        mse_loss = mse_loss.mean()
        
        # Consistency loss: predictions should sum to 1 for each task
        # This is optional depending on the target format
        consistency_loss = 0.0
        
        total_loss = self.vote_weight * mse_loss + self.consistency_weight * consistency_loss
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'consistency_loss': consistency_loss
        }

def load_model_from_config(config_path):
    """Load model from configuration file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config.get('model', {})
    
    model = GalaxySommelier(
        config=config,
        model_name=model_config.get('name', 'facebook/dinov2-base'),
        num_outputs=model_config.get('num_outputs', 37),
        dropout_rate=model_config.get('dropout_rate', 0.2),
        freeze_backbone=model_config.get('freeze_backbone', False)
    )
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, save_path):
    """Save model checkpoint"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': {
            'model_name': model.model_name,
            'num_outputs': model.num_outputs,
            'dropout_rate': model.dropout_rate,
            'freeze_backbone': model.freeze_backbone
        }
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")

def load_model_checkpoint(checkpoint_path, model=None, optimizer=None):
    """Load model checkpoint"""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model if not provided
    if model is None:
        model_config = checkpoint['model_config']
        model = GalaxySommelier(**model_config)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    logger.info(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']

def count_parameters(model):
    """Count model parameters"""
    param_info = model.get_trainable_parameters()
    
    logger.info("Model Parameter Count:")
    logger.info(f"  Total parameters: {param_info['total_parameters']:,}")
    logger.info(f"  Trainable parameters: {param_info['trainable_parameters']:,}")
    logger.info(f"  Frozen parameters: {param_info['frozen_parameters']:,}")
    
    return param_info

def test_model():
    """Test model creation and forward pass"""
    logger.info("Testing model creation...")
    
    # Create model
    model = GalaxySommelier(num_outputs=37)
    
    # Count parameters
    count_parameters(model)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    logger.info(f"Testing forward pass with input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        # Basic forward pass
        predictions = model(dummy_input)
        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        # Forward pass with features
        predictions, features = model(dummy_input, return_features=True)
        logger.info(f"Features shape: {features.shape}")
        
        # Forward pass with attention
        predictions, features, attention = model(dummy_input, return_features=True, return_attention=True)
        logger.info(f"Attention layers: {len(attention)}")
        logger.info(f"Attention shape: {attention[0].shape}")
    
    logger.info("Model test completed successfully!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_model() 