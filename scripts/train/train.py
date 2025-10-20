"""
Training script for Glacier Movement Prediction
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import pickle

from scripts.utils.config import Config
from scripts.preprocess.dataset import create_dataloaders
from scripts.train.model import create_model
from scripts.train.simple_model import create_simple_model
from scripts.train.losses import CombinedLoss

class Trainer:
    
    def __init__(self, config=Config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Create model
        if config.MODEL_TYPE == 'timesformer':
            self.model = create_model(config)
        else:
            self.model = create_simple_model(config)
        
        self.model = self.model.to(self.device)
        
        # Loss function
        self.criterion = CombinedLoss(
            dice_weight=config.DICE_WEIGHT,
            bce_weight=config.BCE_WEIGHT,
            focal_weight=config.FOCAL_WEIGHT,
            focal_alpha=config.FOCAL_ALPHA,
            focal_gamma=config.FOCAL_GAMMA
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.USE_AMP else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_dice = 0
        epoch_bce = 0
        epoch_focal = 0
        
        valid_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Skip error samples
            if batch['region'][0] == 'error':
                continue
            
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            if self.config.USE_AMP:
                with autocast():
                    outputs = self.model(inputs)
                    loss, loss_dict = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(inputs)
                loss, loss_dict = self.criterion(outputs, targets)
                
                loss.backward()
                
                if (batch_idx + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            epoch_loss += loss.item()
            epoch_dice += loss_dict['dice']
            epoch_bce += loss_dict['bce']
            epoch_focal += loss_dict['focal']
            valid_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{loss_dict['dice']:.4f}"
            })
        
        n_batches = max(valid_batches, 1)
        return {
            'loss': epoch_loss / n_batches,
            'dice': epoch_dice / n_batches,
            'bce': epoch_bce / n_batches,
            'focal': epoch_focal / n_batches
        }
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        epoch_loss = 0
        epoch_dice = 0
        valid_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                if batch['region'][0] == 'error':
                    continue
                
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                if self.config.USE_AMP:
                    with autocast():
                        outputs = self.model(inputs)
                        loss, loss_dict = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss, loss_dict = self.criterion(outputs, targets)
                
                epoch_loss += loss.item()
                epoch_dice += loss_dict['dice']
                valid_batches += 1
        
        n_batches = max(valid_batches, 1)
        return {
            'loss': epoch_loss / n_batches,
            'dice': epoch_dice / n_batches
        }
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        checkpoint_path = self.config.CHECKPOINT_DIR / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = self.config.CHECKPOINT_DIR / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        print(f"\n{'='*60}")
        print(f"TRAINING GLACIER MOVEMENT PREDICTION MODEL")
        print(f"{'='*60}\n")
        
        for epoch in range(self.current_epoch, self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['loss'])
            
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics['loss'])
            
            self.scheduler.step()
            
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} (Dice: {train_metrics['dice']:.4f})")
            print(f"  Val Loss: {val_metrics['loss']:.4f} (Dice: {val_metrics['dice']:.4f})")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            if (epoch + 1) % self.config.SAVE_INTERVAL == 0:
                self.save_checkpoint(is_best=is_best)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")

def main():
    """Main training function with quality checks"""
    print("Checking data quality...")
    
    good_regions = []
    for region in Config.RGI_REGIONS:
        dynamic_path = Config.DYNAMIC_FEATURES_DIR / f"{region}_dynamic.pkl"
        if dynamic_path.exists():
            try:
                with open(dynamic_path, 'rb') as f:
                    dynamic = pickle.load(f)
                if len(dynamic) >= 6:
                    good_regions.append(region)
            except:
                continue
    
    if len(good_regions) < 2:
        print(f"Error: Only {len(good_regions)} regions have sufficient data")
        print("Available regions:", good_regions)
        return
    
    # Split regions
    split_idx = max(1, int(len(good_regions) * 0.8))
    train_regions = good_regions[:split_idx]
    val_regions = good_regions[split_idx:]
    
    print(f"\nGood regions found: {len(good_regions)}")
    print(f"Training regions ({len(train_regions)}): {train_regions}")
    print(f"Validation regions ({len(val_regions)}): {val_regions}\n")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_regions, val_regions)
    
    if len(train_loader.dataset) == 0:
        print("Error: No valid training samples!")
        return
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}\n")
    
    # Create trainer
    trainer = Trainer()
    
    # Load checkpoint if exists
    latest_checkpoint = Config.CHECKPOINT_DIR / "best_model.pth"
    if latest_checkpoint.exists():
        print(f"Found existing checkpoint: {latest_checkpoint}")
        response = input("Load checkpoint? (y/n): ")
        if response.lower() == 'y':
            trainer.load_checkpoint(latest_checkpoint)
    
    # Train
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
