# ABSTRACT

Lung cancer continues to claim more lives than any other malignancy, and a patient's
outcome depends heavily on how early the disease is identified. Since reading Computed
Tomography (CT) scans manually is slow and varies from one radiologist to another, there
is a pressing need for automated diagnostic tools that are both dependable and easy to
interpret. This thesis introduces an **Explainable Stacked Ensemble Model for Lung
Cancer Detection Using Transfer Learning and Grad-CAM Visualisation** that targets three
recurring weaknesses of current deep-learning methods: limited and imbalanced training
data, the fragility of depending on a single model, and the tendency to bolt on
explainability only after the fact. To rebalance the data, a **Conditional Generative
Adversarial Network (cGAN)** synthesises CT images for the under-represented class, and
three backbones — **EfficientNet-B0, DenseNet-121 and ResNet-50** — are fine-tuned in
two stages and merged through an **XGBoost meta-learner**, with **Grad-CAM** heatmaps
generated alongside every prediction to reveal the regions behind each decision. On the
clean IQ-OTH/NCCD benchmark the ensemble attains **98.80% accuracy, 100% recall and
AUC = 0.9997**, while on the raw multi-institutional LIDC-IDRI dataset it reaches
**84.00% accuracy and AUC = 0.9218**. The 14.8-point gap between the two is reported
honestly as a measure of the cross-domain generalisation challenge in medical AI.
Deployed as an interactive web application, the proposed pipeline delivers a unified,
leakage-free and explainability-first approach that reinforces clinical trust in
AI-assisted lung cancer diagnosis.

**Keywords:** Lung cancer detection, deep learning, transfer learning, stacked ensemble,
Generative Adversarial Network, Grad-CAM, explainable artificial intelligence, CT
imaging, IQ-OTH/NCCD, LIDC-IDRI.
