"""
PyTorch Dataset class for lung cancer images.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import cv2


class LungCancerDataset(Dataset):
    """Dataset for lung cancer CT images."""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of paths to images
            labels: List of labels (0 or 1)
            transform: Albumentations transforms
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, torch.tensor(label, dtype=torch.long)


def create_dataloaders(data_dir, batch_size=32, num_workers=4, train_transform=None, val_transform=None):
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Directory containing train/val subdirectories
        batch_size: Batch size
        num_workers: Number of worker processes
        train_transform: Augmentations for training
        val_transform: Augmentations for validation
    
    Returns:
        train_loader, val_loader
    """
    data_dir = Path(data_dir)
    
    # Load data
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    
    for split in ['train', 'val']:
        split_dir = data_dir / split
        for label, class_name in enumerate(['no_cancer', 'cancer']):
            class_dir = split_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.dcm']:
                        if split == 'train':
                            train_images.append(str(img_path))
                            train_labels.append(label)
                        else:
                            val_images.append(str(img_path))
                            val_labels.append(label)
    
    # Create datasets
    train_dataset = LungCancerDataset(train_images, train_labels, train_transform)
    val_dataset = LungCancerDataset(val_images, val_labels, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_kfold_splits(data_dir, n_folds=5, save_dir='data/splits', seed=42):
    """
    Create stratified K-fold splits for ensemble learning.
    
    Args:
        data_dir: Directory containing images
        n_folds: Number of folds
        save_dir: Directory to save split information
        seed: Random seed
    
    Returns:
        List of (train_indices, val_indices) tuples
    """
    from src.utils import ensure_dir
    
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    ensure_dir(save_dir)
    
    # Collect all images and labels
    all_images = []
    all_labels = []
    
    for label, class_name in enumerate(['no_cancer', 'cancer']):
        class_dir = data_dir / class_name
        if class_dir.exists():
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.dcm']:
                    all_images.append(str(img_path))
                    all_labels.append(label)
    
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    
    # Create stratified K-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_images, all_labels)):
        # Save split information
        pd.DataFrame({
            'image_path': all_images[train_idx],
            'label': all_labels[train_idx]
        }).to_csv(save_dir / f'fold_{fold_idx}_train.csv', index=False)
        
        pd.DataFrame({
            'image_path': all_images[val_idx],
            'label': all_labels[val_idx]
        }).to_csv(save_dir / f'fold_{fold_idx}_val.csv', index=False)
        
        splits.append((train_idx, val_idx))
        print(f"Fold {fold_idx}: Train={len(train_idx)}, Val={len(val_idx)}")
    
    return splits
