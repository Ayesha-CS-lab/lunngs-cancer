# ABSTRACT

Lung cancer remains the leading cause of cancer-related mortality worldwide, yet its
prognosis is strongly dependent on the stage at which it is detected. Manual
interpretation of Computed Tomography (CT) scans is slow, resource-intensive and prone
to inter-observer variability, motivating the development of automated, accurate and
explainable computer-aided diagnosis systems.

This thesis presents an **Explainable Stacked Ensemble Model for Lung Cancer Detection
Using Transfer Learning and Grad-CAM Visualisation**. The proposed framework addresses
three critical limitations of existing deep-learning approaches: class imbalance and
data scarcity, the instability of single-model predictions, and the post-hoc — rather
than integrated — application of explainability tools. A **Conditional Generative
Adversarial Network (cGAN)** is trained to generate synthetic CT images for the
minority class, correcting the training-set class imbalance. Three complementary
transfer learning architectures — **EfficientNet-B0, DenseNet-121 and ResNet-50** —
are fine-tuned using a two-phase strategy (frozen backbone followed by full fine-tuning)
and combined through an **XGBoost stacked meta-learner** to produce a robust ensemble
prediction. **Gradient-weighted Class Activation Mapping (Grad-CAM)** is integrated
as a first-class output component, generating a visual heatmap that highlights the CT
regions responsible for each prediction alongside the classification score.

The system is evaluated on two publicly available benchmark datasets:
**IQ-OTH/NCCD** — a clean, pre-processed CT dataset — and **LIDC-IDRI** — a raw,
multi-institutional DICOM dataset. On IQ-OTH/NCCD, the stacked ensemble achieves
**98.80% accuracy, 100% recall (zero missed cancer cases) and AUC = 0.9997**,
surpassing the target accuracy of 97%. On the more challenging LIDC-IDRI dataset, the
ensemble achieves **84.00% accuracy and AUC = 0.9218** — the highest AUC among all
tested models on that dataset. Grad-CAM heatmaps consistently highlight clinically
relevant focal regions in malignant cases and produce diffuse, non-focal activations
for benign cases, confirming that the model's attention aligns with pathologically
meaningful features. The complete system is deployed as an interactive web application
providing real-time prediction with integrated visual explanation.

The 14.8-percentage-point accuracy gap between the two datasets quantifies the
generalisation challenge of cross-domain medical AI and is presented as a research
finding underscoring the importance of multi-dataset evaluation. The proposed system
contributes a unified, leakage-free, explainability-first pipeline that advances
clinical confidence in AI-assisted lung cancer diagnosis.

**Keywords:** Lung cancer detection, deep learning, transfer learning, stacked ensemble,
Generative Adversarial Network, Grad-CAM, explainable artificial intelligence, CT
imaging, IQ-OTH/NCCD, LIDC-IDRI.
