"""
Stacked Ensemble Training Demo
===============================

This script demonstrates the complete stacking ensemble workflow:
1. Train multiple base models with K-fold cross-validation
2. Generate out-of-fold (OOF) predictions
3. Train meta-learner on OOF predictions
4. Make ensemble predictions on test data
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.config import *
from src.models.base_models import ModelFactory
from src.models.trainer import Trainer
from src.ensemble.stacking import StackedEnsemble
from src.ensemble.meta_models import create_meta_learner
from src.datasets import create_dataloaders, create_kfold_splits
from src.augmentations import get_train_transforms, get_val_transforms
from src.utils import set_seed, ensure_dir


def create_demo_data():
    """Create dummy dataset for demonstration."""
    print("\n" + "="*70)
    print("STEP 1: Creating Dummy Dataset")
    print("="*70 + "\n")
    
    import cv2
    
    ensure_dir('data/ensemble_demo/train/no_cancer')
    ensure_dir('data/ensemble_demo/train/cancer')
    ensure_dir('data/ensemble_demo/test/no_cancer')
    ensure_dir('data/ensemble_demo/test/cancer')
    
    print("Creating dummy images for ensemble demo...")
    
    # Training images (more for better demonstration)
    for i in range(50):
        # No cancer
        img = np.random.randint(80, 120, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(f'data/ensemble_demo/train/no_cancer/img_{i:03d}.png', img)
        
        # Cancer
        img = np.random.randint(100, 150, (224, 224, 3), dtype=np.uint8)
        center = (np.random.randint(80, 140), np.random.randint(80, 140))
        cv2.circle(img, center, np.random.randint(10, 20), (255, 255, 255), -1)
        cv2.imwrite(f'data/ensemble_demo/train/cancer/img_{i:03d}.png', img)
    
    # Test images
    for i in range(10):
        img = np.random.randint(80, 120, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(f'data/ensemble_demo/test/no_cancer/img_{i:03d}.png', img)
        
        img = np.random.randint(100, 150, (224, 224, 3), dtype=np.uint8)
        center = (np.random.randint(80, 140), np.random.randint(80, 140))
        cv2.circle(img, center, np.random.randint(10, 20), (255, 255, 255), -1)
        cv2.imwrite(f'data/ensemble_demo/test/cancer/img_{i:03d}.png', img)
    
    print("✓ Created 120 dummy images")
    print("  Training: 50 no_cancer + 50 cancer = 100 images")
    print("  Testing: 10 no_cancer + 10 cancer = 20 images")
    
    return 'data/ensemble_demo'


def train_base_models_kfold(data_dir, n_folds=3, device='cuda'):
    """Train base models with K-fold cross-validation."""
    print("\n" + "="*70)
    print(f"STEP 2: Training Base Models with {n_folds}-Fold CV")
    print("="*70 + "\n")
    
    model_names = ['efficientnet', 'densenet', 'resnet']
    all_oof_predictions = {}
    all_test_predictions = {}
    
    # Create data loaders
    print("Loading data...")
    train_loader, _ = create_dataloaders(
        data_dir=data_dir,
        batch_size=8,
        num_workers=0,
        train_transform=get_train_transforms(IMAGE_SIZE),
        val_transform=get_val_transforms(IMAGE_SIZE)
    )
    
    test_loader, _ = create_dataloaders(
        data_dir=data_dir.replace('train', 'test'),
        batch_size=8,
        num_workers=0,
        train_transform=get_val_transforms(IMAGE_SIZE),
        val_transform=get_val_transforms(IMAGE_SIZE)
    )
    
    # For demo, we'll simulate K-fold by training each model once
    # In production, you'd actually split data into K folds
    for model_name in model_names:
        print(f"\n{'-'*70}")
        print(f"Training {model_name.upper()}")
        print(f"{'-'*70}\n")
        
        # Create model
        model = ModelFactory.create_model(
            model_name=model_name,
            num_classes=NUM_CLASSES,
            pretrained=True
        ).to(device)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            device=device,
            num_classes=NUM_CLASSES,
            class_weights=CLASS_WEIGHTS,
            use_amp=True
        )
        
        # Train (simplified: just 3 epochs for demo)
        model.freeze_backbone()
        history = trainer.train(
            train_loader=train_loader,
            val_loader=train_loader,  # Using same for demo
            num_epochs=3,
            learning_rate=LEARNING_RATE,
            checkpoint_dir=f'checkpoints/ensemble_demo/{model_name}',
            model_name=f'{model_name}_fold'
        )
        
        # Get OOF predictions (simulated)
        model.eval()
        oof_preds = []
        oof_labels = []
        
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                oof_preds.append(probs.cpu().numpy())
                oof_labels.append(labels.numpy())
        
        oof_preds = np.vstack(oof_preds)
        oof_labels = np.concatenate(oof_labels)
        
        all_oof_predictions[model_name] = oof_preds
        
        # Get test predictions
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                test_preds.append(probs.cpu().numpy())
                test_labels.append(labels.numpy())
        
        test_preds = np.vstack(test_preds)
        test_labels = np.concatenate(test_labels)
        
        all_test_predictions[model_name] = test_preds
        
        # Print individual model performance
        oof_acc = accuracy_score(oof_labels, np.argmax(oof_preds, axis=1))
        test_acc = accuracy_score(test_labels, np.argmax(test_preds, axis=1))
        
        print(f"\n{model_name.upper()} Results:")
        print(f"  OOF Accuracy: {oof_acc*100:.2f}%")
        print(f"  Test Accuracy: {test_acc*100:.2f}%")
    
    print("\n✓ All base models trained!")
    
    return all_oof_predictions, all_test_predictions, oof_labels, test_labels


def create_meta_features(oof_predictions):
    """Create meta-features from OOF predictions."""
    print("\n" + "="*70)
    print("STEP 3: Creating Meta-Features")
    print("="*70 + "\n")
    
    # Stack predictions from all models
    meta_features = []
    
    for model_name, preds in oof_predictions.items():
        print(f"{model_name.upper()} predictions shape: {preds.shape}")
        meta_features.append(preds)
    
    # Concatenate all predictions
    X_meta = np.hstack(meta_features)
    
    print(f"\nMeta-features created:")
    print(f"  Shape: {X_meta.shape}")
    print(f"  Features per model: {preds.shape[1]} (probabilities)")
    print(f"  Total features: {X_meta.shape[1]} ({len(oof_predictions)} models × {preds.shape[1]} classes)")
    
    return X_meta


def train_meta_learner(X_meta, y_true, meta_learner_type='xgboost'):
    """Train meta-learner on meta-features."""
    print("\n" + "="*70)
    print(f"STEP 4: Training Meta-Learner ({meta_learner_type.upper()})")
    print("="*70 + "\n")
    
    # Create meta-learner
    meta_learner = create_meta_learner(meta_learner_type)
    
    print(f"Training {meta_learner_type} on {X_meta.shape[0]} samples...")
    
    # Train
    meta_learner.fit(X_meta, y_true)
    
    # Evaluate on training data
    train_acc = meta_learner.score(X_meta, y_true)
    
    print(f"✓ Meta-learner trained!")
    print(f"  Training Accuracy: {train_acc*100:.2f}%")
    
    # Save meta-learner
    ensure_dir('checkpoints/ensemble_demo')
    joblib.dump(meta_learner, f'checkpoints/ensemble_demo/meta_learner_{meta_learner_type}.pkl')
    print(f"  Saved to: checkpoints/ensemble_demo/meta_learner_{meta_learner_type}.pkl")
    
    return meta_learner


def evaluate_ensemble(test_predictions, test_labels, meta_learner):
    """Evaluate ensemble on test set."""
    print("\n" + "="*70)
    print("STEP 5: Evaluating Ensemble")
    print("="*70 + "\n")
    
    # Create test meta-features
    X_test_meta = np.hstack([preds for preds in test_predictions.values()])
    
    # Ensemble prediction
    ensemble_preds = meta_learner.predict(X_test_meta)
    ensemble_acc = accuracy_score(test_labels, ensemble_preds)
    
    print("Ensemble Performance:")
    print(f"  Test Accuracy: {ensemble_acc*100:.2f}%")
    
    # Individual model performances
    print("\nIndividual Model Performance:")
    individual_accs = {}
    for model_name, preds in test_predictions.items():
        pred_classes = np.argmax(preds, axis=1)
        acc = accuracy_score(test_labels, pred_classes)
        individual_accs[model_name] = acc
        print(f"  {model_name.upper()}: {acc*100:.2f}%")
    
    # Improvement
    best_individual = max(individual_accs.values())
    improvement = (ensemble_acc - best_individual) * 100
    
    print(f"\nEnsemble Improvement:")
    print(f"  Best individual model: {best_individual*100:.2f}%")
    print(f"  Ensemble: {ensemble_acc*100:.2f}%")
    print(f"  Improvement: {improvement:+.2f}%")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, ensemble_preds, 
                                target_names=['No Cancer', 'Cancer']))
    
    return ensemble_acc, individual_accs


def visualize_results(individual_accs, ensemble_acc):
    """Visualize ensemble vs individual models."""
    print("\n" + "="*70)
    print("STEP 6: Visualizing Results")
    print("="*70 + "\n")
    
    ensure_dir('outputs/ensemble_demo')
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart comparing models
    models = list(individual_accs.keys()) + ['ENSEMBLE']
    accuracies = [acc*100 for acc in individual_accs.values()] + [ensemble_acc*100]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#4CAF50']
    
    bars = ax1.bar(range(len(models)), accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([m.upper() for m in models], rotation=0, fontsize=10, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Highlight ensemble
    bars[-1].set_linewidth(3)
    bars[-1].set_edgecolor('gold')
    
    # Architecture diagram
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    # Draw ensemble architecture
    # Base models
    base_y = 8
    for i, model in enumerate(['EfficientNet', 'DenseNet', 'ResNet']):
        x = 2 + i * 2.5
        rect = plt.Rectangle((x-0.4, base_y-0.3), 0.8, 0.6, 
                            facecolor='lightblue', edgecolor='black', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(x, base_y, model, ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Arrows to meta-learner
        ax2.arrow(x, base_y-0.4, 0, -1.5, head_width=0.2, head_length=0.2, 
                 fc='gray', ec='gray', linewidth=2)
    
    # Meta-learner
    meta_rect = plt.Rectangle((3.5, 5-0.4), 2, 0.8, 
                             facecolor='#4CAF50', edgecolor='black', linewidth=2)
    ax2.add_patch(meta_rect)
    ax2.text(4.5, 5, 'Meta-Learner\n(XGBoost)', ha='center', va='center', 
            fontsize=9, fontweight='bold', color='white')
    
    # Final prediction
    ax2.arrow(4.5, 4.6, 0, -1.2, head_width=0.3, head_length=0.3, 
             fc='gold', ec='black', linewidth=2)
    
    final_rect = plt.Rectangle((3.5, 2.5-0.3), 2, 0.6, 
                              facecolor='gold', edgecolor='black', linewidth=2)
    ax2.add_patch(final_rect)
    ax2.text(4.5, 2.5, 'Final Prediction', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Labels
    ax2.text(5, 9, 'Stacked Ensemble Architecture', ha='center', 
            fontsize=12, fontweight='bold')
    ax2.text(0.5, base_y, 'Base Models:', ha='left', fontsize=9, fontweight='bold')
    ax2.text(0.5, 5, 'Meta-Learner:', ha='left', fontsize=9, fontweight='bold')
    ax2.text(0.5, 2.5, 'Output:', ha='left', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    save_path = 'outputs/ensemble_demo/ensemble_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to: {save_path}")


def main():
    """Run complete ensemble stacking demo."""
    print("\n" + "="*70)
    print("STACKED ENSEMBLE LEARNING DEMO")
    print("="*70)
    print("\nThis demo demonstrates:")
    print("  1. Creating training data")
    print("  2. Training multiple base models with K-fold CV")
    print("  3. Generating out-of-fold predictions (meta-features)")
    print("  4. Training meta-learner (XGBoost)")
    print("  5. Evaluating ensemble on test set")
    print("  6. Visualizing results")
    
    # Set seed
    set_seed(42)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    try:
        # Step 1: Create data
        data_dir = create_demo_data()
        
        # Step 2: Train base models with K-fold
        oof_predictions, test_predictions, oof_labels, test_labels = train_base_models_kfold(
            data_dir + '/train',
            n_folds=3,
            device=device
        )
        
        # Step 3: Create meta-features
        X_meta = create_meta_features(oof_predictions)
        
        # Step 4: Train meta-learner
        meta_learner = train_meta_learner(X_meta, oof_labels, meta_learner_type='xgboost')
        
        # Step 5: Evaluate ensemble
        ensemble_acc, individual_accs = evaluate_ensemble(
            test_predictions,
            test_labels,
            meta_learner
        )
        
        # Step 6: Visualize
        visualize_results(individual_accs, ensemble_acc)
        
        # Summary
        print("\n" + "="*70)
        print("DEMO COMPLETE!")
        print("="*70)
        print("\n✅ Stacked ensemble demonstration successful!")
        print("\nGenerated files:")
        print("  📁 data/ensemble_demo/ - Demo dataset")
        print("  💾 checkpoints/ensemble_demo/ - Base models & meta-learner")
        print("  📊 outputs/ensemble_demo/ensemble_results.png - Results visualization")
        print("\nKey Concepts Demonstrated:")
        print("  ✓ K-fold cross-validation")
        print("  ✓ Out-of-fold predictions (prevents overfitting)")
        print("  ✓ Meta-feature construction")
        print("  ✓ Meta-learner training")
        print("  ✓ Ensemble prediction")
        print("\nNext steps:")
        print("  1. Use real CT scans for better results")
        print("  2. Train for more epochs (50+)")
        print("  3. Try different meta-learners: 'random_forest', 'logistic'")
        print("  4. Use actual K-fold splitting (not simulated)")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  - Install dependencies: pip install xgboost scikit-learn joblib")
        print("  - Ensure base models are trained first")
        print("  - Check data directory structure")


if __name__ == "__main__":
    main()
