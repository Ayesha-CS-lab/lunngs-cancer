"""
Stacked ensemble learning module.
"""
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.models.base_models import get_model
from src.models.trainer import Trainer
from src.models.inference import predict


class StackedEnsemble:
    """Stacked ensemble using out-of-fold predictions."""
    
    def __init__(
        self,
        base_models,
        n_folds=5,
        device='cuda',
        seed=42
    ):
        """
        Initialize stacked ensemble.
        
        Args:
            base_models: List of model names ('efficientnet', 'densenet', 'resnet')
            n_folds: Number of folds for cross-validation
            device: Device to train on
            seed: Random seed
        """
        self.base_models = base_models
        self.n_folds = n_folds
        self.device = device
        self.seed = seed
        self.trained_models = {model_name: [] for model_name in base_models}
        self.meta_model = None
    
    def generate_oof_predictions(
        self,
        train_loader_fn,
        val_loader_fn,
        num_epochs=50,
        save_dir='checkpoints/ensemble'
    ):
        """
        Generate out-of-fold predictions for meta-features.
        
        Args:
            train_loader_fn: Function that returns train loader for a fold
            val_loader_fn: Function that returns validation loader for a fold
            num_epochs: Epochs per fold
            save_dir: Directory to save models
        
        Returns:
            oof_predictions: Out-of-fold predictions for meta-features [N, num_models * 2]
            oof_labels: True labels [N]
        """
        from src.utils import ensure_dir
        ensure_dir(save_dir)
        
        print(f"\n{'='*70}")
        print(f"Generating Out-of-Fold Predictions for {len(self.base_models)} models")
        print(f"{'='*70}\n")
        
        # Initialize storage for OOF predictions
        oof_preds_dict = {model_name: [] for model_name in self.base_models}
        oof_labels = []
        
        for fold in range(self.n_folds):
            print(f"\n{'='*70}")
            print(f"FOLD {fold + 1}/{self.n_folds}")
            print(f"{'='*70}\n")
            
            train_loader = train_loader_fn(fold)
            val_loader = val_loader_fn(fold)
            
            # Train each base model
            for model_name in self.base_models:
                print(f"\n--- Training {model_name} on Fold {fold + 1} ---")
                
                # Create model
                model = get_model(model_name, num_classes=2, pretrained=True)
                
                # Create trainer
                from configs.config import LEARNING_RATE, WEIGHT_DECAY, CLASS_WEIGHTS
                trainer = Trainer(
                    model,
                    device=self.device,
                    learning_rate=LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY,
                    class_weights=CLASS_WEIGHTS
                )
                
                # Train
                best_model_path = trainer.fit(
                    train_loader,
                    val_loader,
                    num_epochs=num_epochs,
                    save_dir=f"{save_dir}/fold_{fold}",
                    model_name=model_name
                )
                
                # Store trained model
                self.trained_models[model_name].append(best_model_path)
                
                # Generate predictions on validation set
                _, probs, labels = predict(model, val_loader, self.device)
                
                # Store OOF predictions (probabilities for both classes)
                oof_preds_dict[model_name].append(probs)
                
                if len(oof_labels) <= fold:
                    oof_labels.append(labels)
        
        # Concatenate all folds
        oof_predictions = []
        for model_name in self.base_models:
            model_preds = np.vstack(oof_preds_dict[model_name])
            oof_predictions.append(model_preds)
        
        # Stack horizontally: [N, num_models * 2]
        oof_predictions = np.hstack(oof_predictions)
        oof_labels = np.concatenate(oof_labels)
        
        print(f"\n✓ OOF Predictions shape: {oof_predictions.shape}")
        print(f"✓ OOF Labels shape: {oof_labels.shape}")
        
        return oof_predictions, oof_labels
    
    def train_meta_model(self, oof_predictions, oof_labels, meta_learner='xgboost'):
        """
        Train meta-learner on OOF predictions.
        
        Args:
            oof_predictions: Out-of-fold predictions [N, num_models * 2]
            oof_labels: True labels [N]
            meta_learner: Type of meta-learner
        
        Returns:
            Trained meta-model
        """
        from src.ensemble.meta_models import get_meta_model
        
        print(f"\n{'='*70}")
        print(f"Training Meta-Learner: {meta_learner}")
        print(f"{'='*70}\n")
        
        self.meta_model = get_meta_model(meta_learner)
        self.meta_model.fit(oof_predictions, oof_labels)
        
        # Calculate training accuracy
        meta_preds = self.meta_model.predict(oof_predictions)
        train_acc = np.mean(meta_preds == oof_labels) * 100
        print(f"✓ Meta-model training accuracy: {train_acc:.2f}%")
        
        return self.meta_model
    
    def predict_ensemble(self, dataloader):
        """
        Generate ensemble predictions.
        
        Args:
            dataloader: DataLoader for test data
        
        Returns:
            predictions, probabilities
        """
        # Get predictions from all base models (averaged across folds)
        base_predictions = []
        
        for model_name in self.base_models:
            fold_probs = []
            
            for fold_idx, model_path in enumerate(self.trained_models[model_name]):
                # Load model
                model = get_model(model_name, num_classes=2, pretrained=False)
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Predict
                _, probs, _ = predict(model, dataloader, self.device)
                fold_probs.append(probs)
            
            # Average across folds
            avg_probs = np.mean(fold_probs, axis=0)
            base_predictions.append(avg_probs)
        
        # Stack predictions
        meta_features = np.hstack(base_predictions)
        
        # Meta-model prediction
        final_preds = self.meta_model.predict(meta_features)
        final_probs = self.meta_model.predict_proba(meta_features) if hasattr(self.meta_model, 'predict_proba') else None
        
        return final_preds, final_probs
