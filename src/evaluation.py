"""
Evaluation metrics and visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve
from pathlib import Path


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, y_true, y_pred, y_proba=None):
        """
        Initialize evaluator.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities [N, 2]
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
    
    def calculate_metrics(self):
        """Calculate all metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        metrics['precision'] = precision_score(self.y_true, self.y_pred, average='binary')
        metrics['recall'] = recall_score(self.y_true, self.y_pred, average='binary')
        metrics['specificity'] = self._calculate_specificity()
        metrics['f1'] = f1_score(self.y_true, self.y_pred, average='binary')
        
        # ROC AUC
        if self.y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_proba[:, 1])
        
        return metrics
    
    def _calculate_specificity(self):
        """Calculate specificity (True Negative Rate)."""
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return specificity
    
    def print_report(self):
        """Print comprehensive evaluation report."""
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60 + "\n")
        
        metrics = self.calculate_metrics()
        
        print(f"Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print(f"Specificity:  {metrics['specificity']:.4f}")
        print(f"F1-Score:     {metrics['f1']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"ROC AUC:      {metrics['roc_auc']:.4f}")
        
        print("\n" + "-"*60)
        print("Classification Report:")
        print("-"*60 + "\n")
        print(classification_report(self.y_true, self.y_pred, target_names=['No Cancer', 'Cancer']))
        
        return metrics
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No Cancer', 'Cancer'],
            yticklabels=['No Cancer', 'Cancer']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            from src.utils import ensure_dir
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve."""
        if self.y_proba is None:
            print("Warning: No probabilities provided, cannot plot ROC curve")
            return
        
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_proba[:, 1])
        roc_auc = roc_auc_score(self.y_true, self.y_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            from src.utils import ensure_dir
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_calibration_curve(self, n_bins=10, save_path=None):
        """Plot calibration curve."""
        if self.y_proba is None:
            print("Warning: No probabilities provided, cannot plot calibration curve")
            return
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.y_true,
            self.y_proba[:, 1],
            n_bins=n_bins
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            from src.utils import ensure_dir
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Calibration curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_all_plots(self, output_dir='outputs/plots'):
        """Save all evaluation plots."""
        from src.utils import ensure_dir
        ensure_dir(output_dir)
        
        self.plot_confusion_matrix(f"{output_dir}/confusion_matrix.png")
        self.plot_roc_curve(f"{output_dir}/roc_curve.png")
        self.plot_calibration_curve(save_path=f"{output_dir}/calibration_curve.png")
        
        print(f"\n✓ All plots saved to {output_dir}")


def evaluate_model(model, dataloader, device='cuda', output_dir='outputs/plots'):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        dataloader: Test data loader
        device: Device to use
        output_dir: Directory to save plots
    
    Returns:
        metrics dictionary
    """
    from src.models.inference import predict
    
    # Get predictions
    y_pred, y_proba, y_true = predict(model, dataloader, device)
    
    # Create evaluator
    evaluator = ModelEvaluator(y_true, y_pred, y_proba)
    
    # Print report
    metrics = evaluator.print_report()
    
    # Save plots
    evaluator.save_all_plots(output_dir)
    
    return metrics
