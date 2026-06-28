# Front Matter — Table of Contents, List of Figures, List of Tables

> **Working note:** Page numbers below are placeholders/estimates. They will be
> finalised automatically once the full chapter content is written and the thesis
> is compiled to Word (`.docx`), where the Table of Contents, List of Figures, and
> List of Tables are generated from the document's headings and captions.

---

# TABLE OF CONTENTS

| | | Page |
|---|---|---:|
| | **Declaration** | i |
| | **Certificate of Approval** | ii |
| | **Dedication** | iii |
| | **Acknowledgements** | iv |
| | **Abstract** | v |
| | **Table of Contents** | vi |
| | **List of Figures** | ix |
| | **List of Tables** | xi |
| | **List of Abbreviations** | xii |
| | | |
| **CHAPTER 1** | **INTRODUCTION** | 1 |
| 1.1 | Background | 1 |
| 1.2 | Lung Cancer and the Importance of Early Detection | 2 |
| 1.3 | Medical Imaging for Lung Cancer Screening | 3 |
| 1.4 | Problem Statement | 4 |
| 1.5 | Motivation | 5 |
| 1.6 | Aim and Objectives | 6 |
| 1.7 | Research Questions | 7 |
| 1.8 | Scope of the Study | 7 |
| 1.9 | Significance and Contributions | 8 |
| 1.10 | Organisation of the Thesis | 9 |
| | | |
| **CHAPTER 2** | **LITERATURE REVIEW** | 10 |
| 2.1 | Introduction | 10 |
| 2.2 | Traditional Methods of Lung Cancer Diagnosis | 11 |
| 2.3 | Medical Imaging Modalities for the Lungs | 12 |
| 2.4 | The Role of Deep Learning in Medical Image Analysis | 13 |
| 2.5 | Convolutional Neural Networks for Lung Cancer Detection | 14 |
| 2.6 | Transfer Learning Approaches | 16 |
| 2.7 | Generative Adversarial Networks for Data Augmentation | 17 |
| 2.8 | Ensemble Learning Methods | 18 |
| 2.9 | Explainable AI (Grad-CAM) in Medical Imaging | 19 |
| 2.10 | Benchmark Datasets (IQ-OTH/NCCD and LIDC-IDRI) | 20 |
| 2.11 | Research Gap | 21 |
| 2.12 | Summary | 22 |
| | | |
| **CHAPTER 3** | **MATERIALS AND METHODS** | 23 |
| 3.1 | Overview of the Proposed Methodology | 23 |
| 3.2 | Datasets | 25 |
| 3.2.1 | IQ-OTH/NCCD Dataset | 25 |
| 3.2.2 | LIDC-IDRI Dataset | 26 |
| 3.3 | Data Preprocessing | 28 |
| 3.3.1 | Image Loading and Supported Formats (PNG, JPEG, DICOM) | 28 |
| 3.3.2 | HU Windowing for DICOM Slices | 29 |
| 3.3.3 | Resizing and Normalisation | 30 |
| 3.4 | Data Augmentation | 31 |
| 3.4.1 | Classical Image Augmentation | 31 |
| 3.4.2 | Conditional GAN Augmentation | 32 |
| 3.4.2.1 | Generator Architecture | 33 |
| 3.4.2.2 | Discriminator Architecture | 34 |
| 3.4.2.3 | GAN Training Procedure | 35 |
| 3.5 | Data Splitting Strategy | 36 |
| 3.6 | Base CNN Models | 37 |
| 3.6.1 | EfficientNet-B0 | 37 |
| 3.6.2 | DenseNet-121 | 38 |
| 3.6.3 | ResNet-50 | 39 |
| 3.6.4 | Transfer Learning and Two-Phase Training | 40 |
| 3.7 | Stacked Ensemble | 41 |
| 3.7.1 | Out-of-Fold Predictions and Meta-Features | 41 |
| 3.7.2 | XGBoost Meta-Learner | 42 |
| 3.8 | Explainability with Grad-CAM | 43 |
| 3.9 | Evaluation Metrics | 44 |
| 3.10 | Experimental Setup and Implementation Details | 45 |
| 3.11 | Web Demonstration Application | 46 |
| | | |
| **CHAPTER 4** | **RESULTS AND DISCUSSION** | 47 |
| 4.1 | Introduction | 47 |
| 4.2 | GAN Augmentation Results | 47 |
| 4.3 | Training and Validation Performance | 49 |
| 4.4 | Results on the IQ-OTH/NCCD Dataset | 50 |
| 4.4.1 | Individual Model Performance | 50 |
| 4.4.2 | Confusion Matrices | 51 |
| 4.4.3 | ROC Curves | 52 |
| 4.4.4 | Stacked Ensemble Performance | 53 |
| 4.5 | Results on the LIDC-IDRI Dataset | 54 |
| 4.5.1 | Individual Model Performance | 54 |
| 4.5.2 | Confusion Matrices and ROC Curves | 55 |
| 4.5.3 | Stacked Ensemble Performance | 56 |
| 4.6 | Cross-Dataset Comparison | 57 |
| 4.7 | Explainability Analysis (Grad-CAM) | 59 |
| 4.8 | Web Application Demonstration | 61 |
| 4.9 | Discussion | 62 |
| 4.10 | Comparison with Existing Literature | 63 |
| | | |
| **CHAPTER 5** | **CONCLUSION AND FUTURE WORK** | 64 |
| 5.1 | Conclusion | 64 |
| 5.2 | Key Contributions | 65 |
| 5.3 | Limitations | 66 |
| 5.4 | Future Work | 67 |
| | | |
| | **REFERENCES** | 68 |
| | **APPENDICES** | 72 |
| | Appendix A — Additional Grad-CAM Heatmaps | 72 |
| | Appendix B — Hyperparameter Configuration | 73 |
| | Appendix C — Sample Source Code | 74 |

---

# LIST OF FIGURES

| Figure | Caption | Page |
|---|---|---:|
| Figure 1.1 | Global incidence and mortality of lung cancer | 2 |
| Figure 1.2 | Example CT scans: cancerous (malignant) vs. benign lung tissue | 3 |
| Figure 3.1 | Overall architecture of the proposed lung cancer detection system | 24 |
| Figure 3.2 | Sample images from the IQ-OTH/NCCD dataset | 26 |
| Figure 3.3 | Sample LIDC-IDRI DICOM slices after lung windowing (HU −1000 to +400) | 27 |
| Figure 3.4 | Data preprocessing pipeline | 30 |
| Figure 3.5 | Conditional GAN architecture (generator and discriminator) | 33 |
| Figure 3.6 | EfficientNet-B0 architecture | 38 |
| Figure 3.7 | DenseNet-121 architecture | 39 |
| Figure 3.8 | ResNet-50 architecture | 40 |
| Figure 3.9 | Two-phase transfer-learning strategy (frozen → fine-tuned) | 41 |
| Figure 3.10 | Stacked ensemble architecture with XGBoost meta-learner | 42 |
| Figure 3.11 | Grad-CAM explainability workflow | 43 |
| Figure 4.1 | GAN generator and discriminator training losses | 48 |
| Figure 4.2 | Synthetic CT images generated by the GAN across training epochs | 48 |
| Figure 4.3 | Training and validation accuracy and loss curves | 49 |
| Figure 4.4 | Confusion matrix — EfficientNet-B0 (IQ-OTH/NCCD) | 51 |
| Figure 4.5 | Confusion matrix — DenseNet-121 (IQ-OTH/NCCD) | 51 |
| Figure 4.6 | Confusion matrix — ResNet-50 (IQ-OTH/NCCD) | 51 |
| Figure 4.7 | Confusion matrix — Stacked Ensemble (IQ-OTH/NCCD) | 53 |
| Figure 4.8 | ROC curves for all models (IQ-OTH/NCCD) | 52 |
| Figure 4.9 | Calibration curve — Stacked Ensemble (IQ-OTH/NCCD) | 53 |
| Figure 4.10 | Confusion matrix — EfficientNet-B0 (LIDC-IDRI) | 55 |
| Figure 4.11 | Confusion matrix — DenseNet-121 (LIDC-IDRI) | 55 |
| Figure 4.12 | Confusion matrix — ResNet-50 (LIDC-IDRI) | 55 |
| Figure 4.13 | Confusion matrix — Stacked Ensemble (LIDC-IDRI) | 56 |
| Figure 4.14 | ROC curves for all models (LIDC-IDRI) | 56 |
| Figure 4.15 | Accuracy comparison across both datasets | 57 |
| Figure 4.16 | ROC-AUC comparison across both datasets | 58 |
| Figure 4.17 | Full metrics heatmap (all models, both datasets) | 58 |
| Figure 4.18 | Grad-CAM heatmaps for malignant (cancer) cases | 60 |
| Figure 4.19 | Grad-CAM heatmaps for benign (no-cancer) cases | 60 |
| Figure 4.20 | Multi-model Grad-CAM comparison | 61 |
| Figure 4.21 | Web demonstration application interface | 62 |

---

# LIST OF TABLES

| Table | Caption | Page |
|---|---|---:|
| Table 2.1 | Summary of related work on deep-learning-based lung cancer detection | 21 |
| Table 3.1 | Class distribution of the IQ-OTH/NCCD dataset | 25 |
| Table 3.2 | Class distribution of the LIDC-IDRI dataset | 27 |
| Table 3.3 | Train / validation / test split for both datasets | 36 |
| Table 3.4 | Conditional GAN training hyperparameters | 35 |
| Table 3.5 | Summary of base CNN architectures | 40 |
| Table 3.6 | Training configuration and hyperparameters | 45 |
| Table 3.7 | Evaluation metrics and their definitions | 44 |
| Table 4.1 | Test-set results on IQ-OTH/NCCD (166 images) | 50 |
| Table 4.2 | Test-set results on LIDC-IDRI (300 images) | 54 |
| Table 4.3 | Cross-dataset performance comparison | 57 |
| Table 4.4 | Factors contributing to the cross-dataset generalisation gap | 59 |
| Table 4.5 | Comparison of the proposed system with existing literature | 63 |

---

# LIST OF ABBREVIATIONS

| Abbreviation | Meaning |
|---|---|
| AI | Artificial Intelligence |
| AMP | Automatic Mixed Precision |
| AUC | Area Under the Curve |
| CAD | Computer-Aided Diagnosis |
| CNN | Convolutional Neural Network |
| CT | Computed Tomography |
| DICOM | Digital Imaging and Communications in Medicine |
| GAN | Generative Adversarial Network |
| Grad-CAM | Gradient-weighted Class Activation Mapping |
| HU | Hounsfield Unit |
| IQ-OTH/NCCD | Iraq-Oncology Teaching Hospital / National Centre for Cancer Diseases (dataset) |
| LIDC-IDRI | Lung Image Database Consortium – Image Database Resource Initiative |
| ROC | Receiver Operating Characteristic |
| TTA | Test-Time Augmentation |
| XAI | Explainable Artificial Intelligence |
| XGBoost | Extreme Gradient Boosting |
