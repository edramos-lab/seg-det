import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import wandb
from tensorboardX import SummaryWriter

from src.models.model_factory import create_model, get_model_info
from src.models.losses import get_loss_function
from src.utils.metrics import calculate_metrics
from src.utils.visualization import plot_training_curves, plot_predictions


class SegmentationTrainer:
    """
    Trainer class for semantic segmentation models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
        device: str = "cuda"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = get_loss_function(config)
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if config['hardware']['mixed_precision'] else None
        
        # Initialize logging
        self.writer = SummaryWriter(config['logging']['log_dir']) if config['logging']['tensorboard'] else None
        
        # Initialize wandb
        if config['logging']['wandb']:
            wandb.init(
                project=config['logging']['wandb_project'],
                config=config
            )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': []
        }
        
        # Create output directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results', exist_ok=True)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = self.config['optimizer']['name'].lower()
        lr = self.config['optimizer']['lr']
        weight_decay = self.config['optimizer']['weight_decay']
        
        if optimizer_name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler based on configuration."""
        scheduler_name = self.config['scheduler']['name'].lower()
        warmup_epochs = self.config['scheduler']['warmup_epochs']
        min_lr = self.config['scheduler']['min_lr']
        
        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=min_lr
            )
        elif scheduler_name == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=min_lr
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Calculate metrics
            metric = calculate_metrics(outputs, masks, 'dice')
            
            total_loss += loss.item()
            total_metric += metric
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Dice': f"{metric:.4f}"
            })
        
        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches
        
        return avg_loss, avg_metric
    
    def validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                metric = calculate_metrics(outputs, masks, 'dice')
                
                total_loss += loss.item()
                total_metric += metric
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches
        
        return avg_loss, avg_metric
    
    def train(self) -> Dict:
        """Train the model."""
        print("Starting training...")
        print(f"Model info: {get_model_info(self.model)}")
        
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_metric = self.train_epoch()
            
            # Validate
            val_loss, val_metric = self.validate_epoch()
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Log metrics
            self._log_metrics(train_loss, val_loss, train_metric, val_metric)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_model('best_loss.pth')
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self._save_model('best_metric.pth')
            
            # Save checkpoint
            if (epoch + 1) % self.config['logging']['save_frequency'] == 0:
                self._save_checkpoint(epoch)
            
            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Final evaluation
        test_metrics = self.evaluate()
        
        # Save training history
        self._save_training_history()
        
        return {
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
            'test_metrics': test_metrics,
            'training_history': self.training_history
        }
    
    def evaluate(self) -> Dict:
        """Evaluate the model on test set."""
        print("Evaluating on test set...")
        
        # Load best model
        self._load_model('best_metric.pth')
        
        self.model.eval()
        total_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                
                # Calculate various metrics
                metrics = calculate_metrics(outputs, masks, 'all')
                
                for metric_name, metric_value in metrics.items():
                    if metric_name not in total_metrics:
                        total_metrics[metric_name] = 0.0
                    total_metrics[metric_name] += metric_value
                
                num_batches += 1
        
        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        print("Test Results:")
        for metric_name, metric_value in avg_metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
        return avg_metrics
    
    def _log_metrics(self, train_loss: float, val_loss: float, train_metric: float, val_metric: float):
        """Log training and validation metrics."""
        # Update history
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['train_metric'].append(train_metric)
        self.training_history['val_metric'].append(val_metric)
        
        # Log to tensorboard
        if self.writer is not None:
            self.writer.add_scalar('Loss/Train', train_loss, self.current_epoch)
            self.writer.add_scalar('Loss/Val', val_loss, self.current_epoch)
            self.writer.add_scalar('Metric/Train', train_metric, self.current_epoch)
            self.writer.add_scalar('Metric/Val', val_metric, self.current_epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        
        # Log to wandb
        if self.config['logging']['wandb']:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metric': train_metric,
                'val_metric': val_metric,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch': self.current_epoch
            })
        
        print(f"Epoch {self.current_epoch + 1}: "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Dice: {train_metric:.4f}, Val Dice: {val_metric:.4f}")
    
    def _save_model(self, filename: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
            'config': self.config
        }, os.path.join('models', filename))
    
    def _load_model(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(os.path.join('models', filename), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        self._save_model(f'checkpoint_epoch_{epoch + 1}.pth')
    
    def _save_training_history(self):
        """Save training history to JSON."""
        with open('results/training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def plot_training_curves(self):
        """Plot training curves."""
        plot_training_curves(self.training_history, save_path='results/training_curves.png')
    
    def cleanup(self):
        """Cleanup resources."""
        if self.writer is not None:
            self.writer.close()
        if self.config['logging']['wandb']:
            wandb.finish() 