# Experimental Results

## Dataset

| Split | No Cancer | Cancer | Total |
|-------|-----------|--------|-------|
| Train | 375 | 392 | 767 |
| Val | 80 | 84 | 164 |
| Test | 83 | 83 | 166 |

## Test Set Performance (166 images)

> Fill in after running `python evaluate_test.py` and `python train_ensemble.py`

| Model | Accuracy | Precision | Recall | Specificity | F1-Score | ROC AUC |
|-------|----------|-----------|--------|-------------|----------|---------|
| EfficientNet-B0 | __% | ____ | ____ | ____ | ____ | ____ |
| DenseNet-121 | __% | ____ | ____ | ____ | ____ | ____ |
| ResNet-50 | __% | ____ | ____ | ____ | ____ | ____ |
| **Ensemble (XGBoost)** | **__%** | **____** | **____** | **____** | **____** | **____** |

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

## Key Findings

> Fill in after evaluation is complete.

1. The stacked ensemble achieves +__% improvement over the best individual model
2. All models achieve >__% recall (sensitivity), critical for medical AI
3. Grad-CAM visualizations consistently highlight tumor regions

## Output Locations

- Individual model plots: `outputs/test_evaluation/{model_name}/`
- Ensemble plots: `outputs/test_evaluation/ensemble/`
- Grad-CAM heatmaps: `outputs/heatmaps/`
- Trained meta-learner: `checkpoints/ensemble/meta_learner_xgboost.pkl`
