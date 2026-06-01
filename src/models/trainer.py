"""
Model trainer with mixed precision and class imbalance handling.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path

from src.utils import save_checkpoint, load_checkpoint


class Trainer:
    """Trainer for CNN models."""
    
    def __init__(
        self,
        model,
        device='cuda',
        learning_rate=1e-4,
        weight_decay=1e-5,
        class_weights=None,
        use_amp=True
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            class_weights: Weights for class imbalance
            use_amp: Use automatic mixed precision
        """
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp
        
        # Loss function with class weights
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)
    
    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs,
        save_dir='checkpoints',
        model_name='model',
        patience=10
    ):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            model_name: Name for saved model
            patience: Early stopping patience
        """
        from src.utils import ensure_dir
        ensure_dir(save_dir)
        
        best_model_path = f"{save_dir}/{model_name}_best.pth"
        epochs_no_improve = 0
        
        print(f"\nTraining {model_name} for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc, _, _ = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    best_model_path
                )
                epochs_no_improve = 0
                print(f"✓ New best model saved!")
            else:
                epochs_no_improve += 1
            
            # Early stopping
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
        
        print(f"\nTraining completed! Best val loss: {self.best_val_loss:.4f}")
        return best_model_path
