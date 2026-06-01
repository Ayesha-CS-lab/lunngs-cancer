# 🏗️ System Architecture

## Overall System Architecture

```mermaid
graph TB
    A["📷 CT Scan Image"] --> B["🔧 Preprocessing<br/>Resize, Normalize, HU Windowing"]
    B --> C["🎨 Data Augmentation<br/>Flip, Rotate, Brightness"]
    C --> D{"🧠 Base Models"}
    D --> E["EfficientNet-B0<br/>5.3M params"]
    D --> F["DenseNet-121<br/>8.0M params"]
    D --> G["ResNet-50<br/>25.6M params"]
    E --> H["P(no_cancer), P(cancer)"]
    F --> I["P(no_cancer), P(cancer)"]
    G --> J["P(no_cancer), P(cancer)"]
    H --> K["📊 Meta-Feature Stack<br/>[6 features]"]
    I --> K
    J --> K
    K --> L["🎯 XGBoost Meta-Learner"]
    L --> M["✅ Final Prediction"]
    M --> N["🔍 Grad-CAM<br/>Explainability"]
    N --> O["🖼️ Heatmap + Diagnosis"]
```

---

## Training Pipeline

```mermaid
graph LR
    A["Raw CT Scans<br/>PNG/JPEG/DICOM"] --> B["Preprocessing<br/>Resize 224x224<br/>Normalize"]
    B --> C["Augmentation<br/>Flip, Rotate<br/>Brightness, Blur"]
    C --> D["Phase 1<br/>Frozen Backbone<br/>25 epochs, LR=1e-4"]
    D --> E["Phase 2<br/>Fine-tuning<br/>25 epochs, LR=1e-5"]
    E --> F["Best Model<br/>Checkpoint"]
    F --> G["Evaluation<br/>Metrics + Plots"]
```

---

## Stacked Ensemble Architecture

```mermaid
graph TB
    subgraph "Level 1: Base Models"
        E["EfficientNet-B0<br/>⚡ Fast, ⭐⭐⭐⭐⭐ Accuracy"]
        D["DenseNet-121<br/>⚡⚡ Medium, ⭐⭐⭐⭐ Accuracy"]
        R["ResNet-50<br/>⚡⚡⚡ Slower, ⭐⭐⭐⭐ Accuracy"]
    end

    subgraph "Level 2: Meta-Learner"
        X["XGBoost<br/>Learns optimal combination"]
    end

    I["Input CT Scan"] --> E
    I --> D
    I --> R
    E -->|"[P0, P1]"| X
    D -->|"[P0, P1]"| X
    R -->|"[P0, P1]"| X
    X --> O["Final Prediction<br/>Cancer / No Cancer"]
```

---

## Out-of-Fold Prediction Generation

```mermaid
graph TB
    subgraph "5-Fold Cross Validation"
        F1["Fold 1: Train on 2-5, Predict on 1"]
        F2["Fold 2: Train on 1,3-5, Predict on 2"]
        F3["Fold 3: Train on 1-2,4-5, Predict on 3"]
        F4["Fold 4: Train on 1-3,5, Predict on 4"]
        F5["Fold 5: Train on 1-4, Predict on 5"]
    end
    
    F1 --> OOF["Out-of-Fold Predictions<br/>(No data leakage!)"]
    F2 --> OOF
    F3 --> OOF
    F4 --> OOF
    F5 --> OOF
    OOF --> META["Meta-Features for<br/>XGBoost Training"]
```

---

## Grad-CAM Explainability Pipeline

```mermaid
graph LR
    A["Input Image<br/>[1, 3, 224, 224]"] --> B["Forward Pass<br/>Through CNN"]
    B --> C["Prediction<br/>Cancer: 85%"]
    C --> D["Backward Pass<br/>Compute Gradients"]
    D --> E["Global Avg Pool<br/>Gradient Weights"]
    E --> F["Weighted Sum<br/>of Feature Maps"]
    F --> G["ReLU + Normalize<br/>[0, 1]"]
    G --> H["Resize to<br/>Original Size"]
    H --> I["Apply Colormap<br/>JET"]
    I --> J["Overlay on<br/>Original Image"]
```

---

## Data Flow

```mermaid
graph TB
    subgraph "Data Sources"
        K["Kaggle IQ-OTHNCCD<br/>Dataset"]
        G["GAN Synthetic<br/>Images (Optional)"]
    end

    subgraph "Binary Classification"
        N["Normal"] -->|merge| NC["no_cancer"]
        B["Benign"] -->|merge| NC
        M["Malignant"] --> CA["cancer"]
    end

    subgraph "Stratified Splits"
        TR["Train: 767 images<br/>375 no_cancer + 392 cancer"]
        VA["Val: 164 images<br/>80 no_cancer + 84 cancer"]
        TE["Test: 166 images<br/>81 no_cancer + 85 cancer"]
    end

    K --> N
    K --> B
    K --> M
    NC --> TR
    NC --> VA
    NC --> TE
    CA --> TR
    CA --> VA
    CA --> TE
    G -.->|"Optional<br/>Augmentation"| TR
```

---

## Project Module Structure

```mermaid
graph TB
    subgraph "Entry Points"
        TRAIN["train.py"]
        EVAL["evaluate.py"]
        DEMO["demo_app.py"]
    end

    subgraph "Core Modules (src/)"
        PP["preprocessing.py<br/>Image Loading"]
        AUG["augmentations.py<br/>Albumentations"]
        DS["datasets.py<br/>PyTorch Dataset"]
        UT["utils.py<br/>Utilities"]
    end

    subgraph "Models (src/models/)"
        BM["base_models.py<br/>EfficientNet, DenseNet, ResNet"]
        TR2["trainer.py<br/>AMP Training Loop"]
        INF["inference.py<br/>Prediction & TTA"]
    end

    subgraph "GAN (src/gan/)"
        GEN["generator.py"]
        DIS["discriminator.py"]
        TG["train_gan.py"]
        SA["sample.py"]
    end

    subgraph "Ensemble (src/ensemble/)"
        ST["stacking.py<br/>K-Fold OOF"]
        ML["meta_models.py<br/>XGBoost/RF/LR"]
    end

    subgraph "Explainability (src/explainability/)"
        GC["gradcam.py<br/>Grad-CAM"]
    end

    TRAIN --> DS
    TRAIN --> BM
    TRAIN --> TR2
    EVAL --> INF
    EVAL --> BM
    DEMO --> BM
    DEMO --> GC
    DS --> PP
    DS --> AUG
    TR2 --> UT
```

---

## CNN Model Architectures

### EfficientNet-B0

```
Input [3, 224, 224]
    ↓
Pretrained EfficientNet-B0 Backbone (from ImageNet)
    ↓ [1280 features]
Dropout (0.3)
    ↓
Linear (1280 → 512)
    ↓
ReLU
    ↓
Dropout (0.3)
    ↓
Linear (512 → 2)
    ↓
Output [2] → Softmax → [P(no_cancer), P(cancer)]
```

### DenseNet-121

```
Input [3, 224, 224]
    ↓
Pretrained DenseNet-121 Backbone (Dense Blocks × 4)
    ↓ [1024 features]
Dropout (0.3)
    ↓
Linear (1024 → 512)
    ↓
ReLU
    ↓
Dropout (0.3)
    ↓
Linear (512 → 2)
    ↓
Output [2] → Softmax → [P(no_cancer), P(cancer)]
```

### ResNet-50

```
Input [3, 224, 224]
    ↓
Pretrained ResNet-50 Backbone (Residual Blocks × 4)
    ↓ [2048 features]
Dropout (0.3)
    ↓
Linear (2048 → 512)
    ↓
ReLU
    ↓
Dropout (0.3)
    ↓
Linear (512 → 2)
    ↓
Output [2] → Softmax → [P(no_cancer), P(cancer)]
```

---

## GAN Architecture

### Generator

```
Input: Noise z [100] + Label Embedding [10]
    ↓ Concat [110]
Linear (110 → 256 × 4 × 4)
    ↓ Reshape [256, 4, 4]
ConvTranspose2d (256 → 128) + BN + ReLU
    ↓ [128, 8, 8]
ConvTranspose2d (128 → 64) + BN + ReLU
    ↓ [64, 16, 16]
ConvTranspose2d (64 → 32) + BN + ReLU
    ↓ [32, 32, 32]
ConvTranspose2d (32 → 3) + Tanh
    ↓ [3, 64, 64]
Output: Synthetic CT Scan Image
```

### Discriminator

```
Input: Image [3, 64, 64] + Label Embedding [64, 64]
    ↓ Concat [4, 64, 64]
Conv2d (4 → 32) + LeakyReLU + Dropout
    ↓ [32, 32, 32]
Conv2d (32 → 64) + BN + LeakyReLU + Dropout
    ↓ [64, 16, 16]
Conv2d (64 → 128) + BN + LeakyReLU + Dropout
    ↓ [128, 8, 8]
Flatten → Linear → Sigmoid
    ↓
Output: Real/Fake Probability [0, 1]
```

---

## Evaluation Pipeline

```mermaid
graph LR
    A["Trained Model"] --> B["Test DataLoader"]
    B --> C["Forward Pass<br/>(No Gradients)"]
    C --> D["Predictions<br/>+ Probabilities"]
    D --> E["Metrics<br/>Acc, Prec, Recall<br/>F1, AUC"]
    D --> F["Confusion Matrix"]
    D --> G["ROC Curve"]
    D --> H["Calibration Curve"]
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | PyTorch 2.0+ | Deep learning |
| **Models** | EfficientNet, DenseNet, ResNet | Feature extraction |
| **Augmentation** | Albumentations | Image transforms |
| **Meta-Learner** | XGBoost / scikit-learn | Ensemble combination |
| **Explainability** | Grad-CAM | Visual explanations |
| **Demo** | Gradio | Web interface |
| **Medical Imaging** | pydicom, OpenCV | DICOM/image processing |
| **Visualization** | Matplotlib, Seaborn | Plots & charts |

---

**Architecture designed for clinical-grade explainable AI!** 🏗️✨
