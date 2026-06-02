# Lung Cancer Detection AI — Supervisor Explanation Guide

---

## 1. What is This Project?

This project is an **AI-powered lung cancer detection system** that analyzes CT scan images and automatically classifies them as:
- **Cancer** (malignant)
- **No Cancer** (benign or normal)

It uses **Deep Learning** — a type of AI that learns patterns from thousands of images, similar to how a radiologist learns from years of experience, but much faster and consistent.

---

## 2. Why is This Important?

- Lung cancer is the **leading cause of cancer death** worldwide
- Early detection increases survival rate from **~15% to ~56%**
- Radiologists are overworked — AI can assist as a **second opinion**
- This system provides not just a prediction, but also **visual explanation** of which part of the lung it detected as suspicious

---

## 3. The Dataset

- **Source:** IQ-OTH/NCCD Lung Cancer Dataset (publicly available on Kaggle)
- **Total Images:** 1,097 CT scan images
- **Classes:** Cancer (malignant) and No Cancer (benign + normal)
- **Split used for training and testing:**

| Split | No Cancer | Cancer | Total |
|-------|-----------|--------|-------|
| Training | 375 | 392 | 767 |
| Validation | 80 | 84 | 164 |
| Test (unseen) | 81 | 85 | 166 |

> The **test set was never seen during training** — it gives an honest measure of real-world performance.

---

## 4. How the System Works (Step by Step)

### Step 1 — Image Preprocessing
When a CT scan image is fed in:
1. It is resized to **224×224 pixels** (standard size for deep learning)
2. Normalized using ImageNet statistics (standard practice)
3. During training, random flips, rotations, and brightness changes are applied (**data augmentation**) to make the model more robust

### Step 2 — Three CNN Models Learn Independently
Three different deep learning architectures are trained on the same dataset:

| Model | Parameters | Specialty |
|-------|-----------|-----------|
| **EfficientNet-B0** | 5.3 million | Best accuracy per parameter — efficient and fast |
| **DenseNet-121** | 8 million | Reuses features across layers — reduces overfitting |
| **ResNet-50** | 25.6 million | Deep residual learning — handles vanishing gradients |

Each model is trained in **two phases**:
- **Phase 1 (Frozen, 25 epochs):** Only the final classification layer trains — the rest of the model stays fixed with ImageNet weights. This is called **Transfer Learning**.
- **Phase 2 (Fine-tuning, 25 epochs):** All layers are unfrozen and the whole model fine-tunes on our lung cancer data.

> **Transfer Learning analogy:** Instead of teaching a student from scratch, we hire someone who already knows medicine (trained on ImageNet) and just teach them the specific skill of reading lung scans.

### Step 3 — Stacked Ensemble (Meta-Learning)
Rather than picking one model's answer, all three models vote together through a **meta-learner**:

1. Each model outputs two probabilities: `P(No Cancer)` and `P(Cancer)`
2. These 6 values (3 models × 2 probabilities) are combined into a **meta-feature vector**
3. An **XGBoost classifier** (a powerful machine learning algorithm) learns the best way to combine these predictions
4. XGBoost gives the final verdict

> **Ensemble analogy:** Like getting a second and third medical opinion and having an experienced consultant make the final decision based on all three doctors' views.

### Step 4 — Grad-CAM Explainability
After prediction, **Grad-CAM** (Gradient-weighted Class Activation Mapping) creates a heatmap showing WHICH part of the image influenced the decision:
- **Red/warm areas** = regions most suspicious for cancer
- **Blue/cool areas** = regions not relevant to the decision

> This is critical for clinical trust — doctors can verify the AI is looking at the right place, not making decisions for wrong reasons.

---

## 5. Results

All models were evaluated on the **166 unseen test images**:

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| EfficientNet-B0 | **100%** | 1.000 | 1.000 | 1.000 | 1.000 |
| DenseNet-121 | **100%** | 1.000 | 1.000 | 1.000 | 1.000 |
| ResNet-50 | **100%** | 1.000 | 1.000 | 1.000 | 1.000 |
| **Ensemble (XGBoost)** | **100%** | 1.000 | 1.000 | 1.000 | 1.000 |

### Why 100%?
- The **IQ-OTH/NCCD dataset** is a clean, well-curated dataset with clearly distinguishable images
- This is consistent with other published research on this exact dataset
- **Transfer learning** from ImageNet gives the models a strong starting point
- **100% recall is the most important metric** in medical AI — it means zero missed cancer cases

### Key metrics explained:
- **Accuracy** — out of all 166 images, how many were correctly classified
- **Recall (Sensitivity)** — out of all actual cancer cases, how many did we catch? (most critical — missing cancer = dangerous)
- **Precision** — out of all cases flagged as cancer, how many actually were?
- **Specificity** — out of all healthy cases, how many did we correctly identify as healthy?
- **AUC (Area Under ROC Curve)** — overall ability to distinguish cancer from non-cancer (1.0 = perfect)

---

## 6. System Architecture Diagram

```
CT Scan Image
      │
      ▼
┌─────────────┐
│ Preprocessing│  → Resize, Normalize, Augment
└─────────────┘
      │
      ▼
┌────────────────────────────────────┐
│           Three CNN Models          │
│  ┌──────────┐ ┌────────┐ ┌───────┐ │
│  │EfficientN│ │DenseNet│ │ResNet │ │
│  │    B0    │ │  121   │ │  50   │ │
│  └────┬─────┘ └───┬────┘ └───┬───┘ │
└───────┼───────────┼──────────┼─────┘
        │           │          │
        ▼           ▼          ▼
   [P0, P1]    [P0, P1]   [P0, P1]   ← 6 probability values
        │           │          │
        └───────────┴──────────┘
                    │
                    ▼
         ┌──────────────────┐
         │ XGBoost          │
         │ Meta-Learner     │  ← learns best combination
         └────────┬─────────┘
                  │
                  ▼
         Final Prediction
         + Grad-CAM Heatmap
```

---

## 7. The Demo Application

A **web-based demo app** (built with Gradio) allows anyone to:
1. Upload a CT scan image
2. Choose a model (EfficientNet, DenseNet, ResNet, or Ensemble)
3. Click **"Analyze"**
4. See the diagnosis + confidence score + Grad-CAM heatmap

**To run it locally:**
```bash
python demo_app.py
# Opens at http://localhost:7860
```

---

## 8. Project File Structure

```
lung_cancer_ai/
│
├── train.py                  ← Trains individual CNN models
├── train_ensemble.py         ← Trains the XGBoost meta-learner
├── evaluate_test.py          ← Evaluates all models on test set
├── generate_gradcam.py       ← Generates Grad-CAM heatmaps
├── demo_app.py               ← Interactive web demo
│
├── src/
│   ├── models/               ← EfficientNet, DenseNet, ResNet code
│   ├── ensemble/             ← XGBoost stacking code
│   ├── explainability/       ← Grad-CAM implementation
│   ├── datasets.py           ← Data loading
│   ├── augmentations.py      ← Image augmentation
│   └── evaluation.py         ← Metrics and plots
│
├── checkpoints/              ← Saved trained model weights (.pth files)
│   └── ensemble/             ← Saved XGBoost meta-learner
│
├── data/kaggle_lung_cancer/  ← Dataset (train/val/test splits)
│
├── outputs/
│   ├── test_evaluation/      ← Confusion matrices, ROC curves
│   └── heatmaps/             ← Grad-CAM visualization images
│
└── docs/                     ← All documentation
    ├── RESULTS.md            ← Final performance numbers
    ├── ARCHITECTURE.md       ← System design diagrams
    └── PROJECT_OVERVIEW.md   ← Full project description
```

---

## 9. Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.x** | Main programming language |
| **PyTorch** | Deep learning framework for building and training CNNs |
| **Torchvision** | Pre-trained models (DenseNet, ResNet) |
| **EfficientNet-PyTorch** | EfficientNet model |
| **Albumentations** | Fast image augmentation |
| **XGBoost** | Meta-learner for ensemble stacking |
| **Grad-CAM** | Visual explainability (custom implementation) |
| **Gradio** | Web-based demo interface |
| **Scikit-learn** | Evaluation metrics |
| **Kaggle GPU** | Training platform (T4 GPU, ~2 hours total) |

---

## 10. Answers to Common Supervisor Questions

**Q: Why three models instead of one?**
> Each model has different strengths. Combining them (ensemble) reduces the chance of all three making the same mistake. It's the same reason hospitals get second opinions.

**Q: Why use Transfer Learning?**
> Training from scratch needs millions of images. We only have 767. Transfer learning lets us reuse knowledge from ImageNet (1.2 million images) and adapt it to our problem — giving much better results with less data.

**Q: Is 100% accuracy realistic?**
> On this specific dataset, yes. The IQ-OTH/NCCD dataset has clearly distinguishable images and published papers report similarly high results. In a real clinical setting with more diverse data, accuracy would be lower — this is a research prototype, not a clinical tool.

**Q: What makes this different from a basic CNN classifier?**
> Three things: (1) **Ensemble of 3 architectures** instead of one, (2) **XGBoost meta-learner** that intelligently combines predictions, (3) **Grad-CAM explainability** that shows WHY the model made its decision — which is critical for medical AI trust.

**Q: What are the limitations?**
> - Trained on one dataset — needs validation on diverse, real-world CT scans
> - Not FDA-approved — research prototype only
> - Requires good quality CT images
> - Should always be used alongside, not instead of, a radiologist

---

## 11. What to Show in the Demo

1. **Run** `python demo_app.py` → open `http://localhost:7860`
2. **Upload** a cancer image from `data/kaggle_lung_cancer/test/cancer/`
3. **Select** "Ensemble (Recommended)" → click Analyze
4. **Show** the prediction + Grad-CAM heatmap
5. **Upload** a no-cancer image from `data/kaggle_lung_cancer/test/no_cancer/`
6. **Show** the different result + heatmap
7. **Explain** that the red regions = where the model detected suspicious tissue

---

*This system was developed as a Final Year Project demonstrating the application of deep learning, transfer learning, ensemble methods, and explainable AI in medical image analysis.*
