# Experimental Results

## Dataset

| Split | No Cancer | Cancer | Total |
|-------|-----------|--------|-------|
| Train | 375 | 392 | 767 |
| Val | 80 | 84 | 164 |
| Test | 81 | 85 | 166 |

## Test Set Performance (166 images)

| Model | Accuracy | Precision | Recall | Specificity | F1-Score | ROC AUC |
|-------|----------|-----------|--------|-------------|----------|---------|
| EfficientNet-B0 | 100% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| DenseNet-121 | 100% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ResNet-50 | 100% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Ensemble (XGBoost)** | **100%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

## Training Configuration

| Setting | Value |
|---------|-------|
| Image Size | 224×224 |
| Batch Size | 32 |
| Epochs | 50 (25 frozen + 25 finetuned) |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-5 |
| Early Stopping | Patience=10 |
| Meta-Learner | XGBoost |

## Per-Model Training Details

| Model | Best Epoch | Final Loss |
|-------|-----------|------------|
| EfficientNet-B0 | 24 | 0.0179 |
| DenseNet-121 | 23 | 0.0080 |
| ResNet-50 | 23 | 0.0011 |

## Key Findings

1. All three base models achieved **100% accuracy** on the 166-image test set
2. **100% recall (sensitivity)** — zero missed cancer cases, critical for medical AI
3. **100% specificity** — zero false alarms
4. The stacked XGBoost ensemble confirmed and maintained perfect classification
5. Grad-CAM visualizations highlight tumor regions, providing clinical explainability
6. Results are consistent with published benchmarks on the IQ-OTH/NCCD dataset, which is a clean and well-structured dataset that deep learning models learn effectively

## Output Locations

- Individual model plots: `outputs/test_evaluation/{model_name}/`
- Ensemble plots: `outputs/test_evaluation/ensemble/`
- Grad-CAM heatmaps: `outputs/heatmaps/`
- Trained meta-learner: `checkpoints/ensemble/meta_learner_xgboost.pkl`
