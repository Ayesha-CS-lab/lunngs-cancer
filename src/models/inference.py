"""
Inference module for trained models.
"""
import torch
import numpy as np
from tqdm import tqdm


def predict(model, dataloader, device='cuda'):
    """
    Generate predictions for a dataset.
    
    Args:
        model: Trained model
        dataloader: DataLoader
        device: Device to use
    
    Returns:
        predictions, probabilities, labels
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Inference'):
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def predict_single_image(model, image, transform, device='cuda'):
    """
    Predict for a single image.
    
    Args:
        model: Trained model
        image: Image array [H, W, C]
        transform: Albumentations transform
        device: Device to use
    
    Returns:
        predicted_class, probability
    """
    model.eval()
    model.to(device)
    
    # Apply transform
    if transform:
        augmented = transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(device)
    else:
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0][pred_class].item()
    
    return pred_class, pred_prob, probs[0].cpu().numpy()


def predict_with_tta(model, image, tta_transforms, device='cuda'):
    """
    Predict with Test-Time Augmentation.
    
    Args:
        model: Trained model
        image: Image array
        tta_transforms: List of transforms
        device: Device to use
    
    Returns:
        averaged prediction and probabilities
    """
    model.eval()
    model.to(device)
    
    all_probs = []
    
    with torch.no_grad():
        for transform in tta_transforms:
            augmented = transform(image=image)
            image_tensor = augmented['image'].unsqueeze(0).to(device)
            
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    # Average probabilities
    avg_probs = np.mean(all_probs, axis=0)[0]
    pred_class = np.argmax(avg_probs)
    
    return pred_class, avg_probs[pred_class], avg_probs
