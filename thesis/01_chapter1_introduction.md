# CHAPTER 1

# INTRODUCTION

## 1.1 Background

Cancer is among the most devastating and widespread illnesses of the modern era,
claiming millions of lives each year across every region of the world. According to the
World Health Organization, cancer is responsible for approximately 10 million deaths
annually, making it one of the leading causes of mortality globally [1]. Among the
numerous forms of cancer, lung cancer consistently ranks as the most lethal, accounting
for more deaths each year than any other type. Its high mortality is not primarily a
consequence of aggressive biology alone, but is closely linked to the fact that the
disease is rarely detected at a stage where curative treatment is still possible.

Lung cancer arises when abnormal cells in the lung tissue begin to multiply in an
uncontrolled manner, forming malignant tumours that may eventually spread through the
bloodstream or lymphatic system to distant organs — a process known as metastasis. By
the time clinical symptoms such as a persistent cough, chest pain, haemoptysis,
shortness of breath or unexplained weight loss become evident, the disease has typically
progressed to an advanced stage. At this point the available treatment options are
limited, and the five-year survival rate drops significantly compared with cases detected
at an early, localised stage. This stark contrast in prognosis underlines the critical
importance of early and accurate detection.

Medical imaging, in particular **Computed Tomography (CT)**, has become the standard
modality for examining the lungs in both screening and diagnostic settings. CT produces
detailed, cross-sectional images in which small nodules and subtle abnormalities can be
visualised long before they cause noticeable symptoms. However, interpreting these images
is a demanding and time-intensive task. A single CT examination may contain hundreds of
axial slices, each of which must be scrutinised by a qualified radiologist for signs of
malignancy. This process is not only resource-intensive but is also susceptible to
inter-observer variability and human fatigue, both of which can result in missed
diagnoses or unnecessary follow-up procedures.

These limitations have motivated a growing body of research into **Computer-Aided
Diagnosis (CAD)** systems — automated tools that analyse medical images and support
the clinician's decision-making process. Recent advances in **deep learning**, and in
particular in **Convolutional Neural Networks (CNNs)**, have produced remarkable results
in image classification tasks across many domains, and medical imaging is no exception.
CNN-based systems have demonstrated the ability to match or even surpass experienced
radiologists on specific classification tasks, creating genuine opportunities to improve
the speed, consistency and accuracy of lung cancer screening.

## 1.2 Lung Cancer and the Importance of Early Detection

Lung cancer is broadly divided into two major categories: **non-small-cell lung cancer
(NSCLC)**, which accounts for approximately 85% of cases, and **small-cell lung cancer
(SCLC)**, which is less common but tends to grow and spread more rapidly. Regardless of
subtype, the single most important determinant of survival is the stage at which the
disease is identified. When detected at Stage I — confined to a small region of one lung
— the five-year survival rate may be as high as 80–90%. When detected at Stage IV —
having spread to both lungs and distant organs — the five-year survival rate falls to
below 10%.

Despite significant advances in surgical, chemotherapeutic and immunotherapeutic
treatments, improving early detection remains the most impactful lever available for
reducing lung cancer mortality. National lung cancer screening programmes based on
low-dose CT have been shown to reduce mortality among high-risk populations by detecting
nodules before they cause symptoms. However, early-stage nodules are frequently small,
may resemble benign structures such as scar tissue or granulomas, and often require
expert interpretation to characterise correctly. This diagnostic uncertainty creates a
clear need for automated systems that can provide a consistent, reproducible and
objective "second opinion" to assist the radiologist.

## 1.3 Medical Imaging and Deep Learning for Lung Cancer Screening

The work presented in this thesis is based on the analysis of lung CT images drawn from
two distinct sources. The first, the **IQ-OTH/NCCD** dataset, consists of curated,
pre-processed CT images labelled at folder level as cancerous or non-cancerous. The
second, the **LIDC-IDRI** dataset, consists of raw CT data stored in the Digital Imaging
and Communications in Medicine (DICOM) format, which must be windowed and converted
before use. Working with both a clean benchmark dataset and a raw, multi-institutional
clinical dataset allows the robustness and generalisation ability of the proposed system
to be assessed under realistic, rather than only idealised, conditions.

Although deep learning has achieved remarkable classification performance on many
medical imaging benchmarks, two significant obstacles limit its clinical adoption:
**a lack of training data** (particularly for rare or minority-class conditions) and
**a lack of transparency** in model predictions. The former is addressed through
GAN-based data augmentation; the latter through Grad-CAM visualisation. Together,
these components form the core of the proposed system.

## 1.4 Problem Statement

Although deep learning models for lung cancer detection have achieved impressive results
in accuracy, several critical challenges continue to limit their reliability and clinical
adoption:

1. **Class imbalance and data scarcity.** Annotated medical images are expensive to
   acquire. Malignant cases are frequently under-represented relative to benign ones,
   causing models trained without correction to be biased toward the majority class and
   to perform poorly on the minority class — precisely the class with the greatest
   clinical consequence.

2. **Dependence on a single model.** Many existing systems rely on a single CNN
   architecture, whose predictions are sensitive to the specific characteristics of the
   training data and may be unstable or unreliable when applied to new, unseen images.

3. **Lack of transparency and explainability.** Deep neural networks are widely regarded
   as "black boxes." In a clinical environment, a prediction that cannot be explained or
   justified is unlikely to be trusted or acted upon by a physician. Existing systems
   often apply explainability tools such as Grad-CAM as a post-hoc add-on rather than
   treating interpretability as a core design requirement [15].

4. **Poor cross-dataset generalisation.** A model that performs well on one dataset may
   perform considerably worse on data from a different source or institution, raising
   serious doubts about real-world applicability.

This thesis addresses all four of these problems through an integrated framework that
combines GAN-based synthetic augmentation, a stacked ensemble of complementary transfer
learning architectures, and Grad-CAM visualisation — designed from the outset as a
unified, explainable system.

## 1.5 Motivation

The motivation for this work arises from both the clinical significance of the problem
and the limitations that persist in current approaches. Despite the excellent accuracy
reported by many deep-learning systems for lung cancer detection, two properties
essential for clinical trust are frequently absent: **robustness** (consistent
performance across different data sources) and **transparency** (the ability to explain
*why* a prediction was made). A system that is highly accurate on a specific benchmark
but provides no visual justification for its outputs is of limited practical value to a
radiologist who must make a treatment decision for a real patient.

This project is motivated by the goal of developing a system that is accurate,
robust *and* explainable — and of evaluating it honestly on two contrasting datasets
so that its strengths and limitations are clearly understood. The research aligns with
the broader movement toward **Explainable Artificial Intelligence (XAI)** in medicine,
where the interpretability of a model's reasoning is regarded as a prerequisite for
clinical adoption, not merely a desirable feature.

## 1.6 Aim and Objectives

The **aim** of this project is to develop an **Explainable Stacked Ensemble Model for
Lung Cancer Detection** that combines multiple transfer learning architectures with
GAN-based augmentation and Grad-CAM visualisation to achieve both high classification
performance and transparent, clinically interpretable predictions.

To achieve this aim, the following **objectives** are pursued:

1. To develop a stacked ensemble deep learning framework by combining multiple transfer
   learning architectures (EfficientNet-B0, DenseNet-121, and ResNet-50) trained on CT
   images from two benchmark lung cancer datasets.
2. To address class imbalance and data scarcity through GAN-based synthetic CT image
   generation, using a Conditional GAN to augment the minority class in the training
   split.
3. To enhance the transparency and interpretability of the model by integrating
   Gradient-weighted Class Activation Mapping (Grad-CAM), producing visual heatmaps
   that identify the regions of a CT image influencing each prediction.
4. To compare and evaluate the performance of individual base models against the
   stacked ensemble model, demonstrating the benefit of combining complementary
   architectures.
5. To evaluate model reliability using standard performance indicators — accuracy,
   precision, sensitivity (recall), F1-score and ROC-AUC — on both the IQ-OTH/NCCD
   and LIDC-IDRI datasets.
6. To improve clinical trust and confidence in AI-assisted diagnosis by delivering a
   system whose predictions are accompanied by visual evidence of the regions driving
   the classification decision.
7. To deploy the trained system as an interactive web application that demonstrates
   real-time prediction and Grad-CAM explanation.

## 1.7 Research Questions

The following research questions guide the investigation:

- **RQ1.** Can a Conditional GAN generate synthetic lung CT images of sufficient quality
  to mitigate class imbalance and improve classification performance?
- **RQ2.** Does a stacked ensemble of complementary transfer learning architectures
  outperform individual base models in terms of accuracy, F1-score and AUC?
- **RQ3.** Can Grad-CAM, integrated as a core component rather than a post-hoc tool,
  produce clinically meaningful heatmaps that localise the tumour regions responsible
  for each prediction?
- **RQ4.** How well does a system trained on a clean benchmark dataset (IQ-OTH/NCCD)
  generalise to raw multi-institutional clinical data (LIDC-IDRI), and what factors
  explain any performance gap?

## 1.8 Scope of the Study

This study focuses on the **binary classification** of two-dimensional lung CT images
into *cancer* (malignant) and *no-cancer* (benign) categories. The scope encompasses
the complete pipeline: data acquisition, preprocessing (including HU windowing for
DICOM data), GAN-based augmentation, transfer learning with multiple CNN architectures,
stacked ensemble construction, Grad-CAM explainability and web-based deployment. The
study uses two publicly available datasets — **IQ-OTH/NCCD** and **LIDC-IDRI** —
and evaluates the system using accuracy, sensitivity, F1-score, and AUC-ROC.

The study does **not** address three-dimensional volumetric analysis, nodule
segmentation, tumour staging, or histological subtype classification. Clinical
validation with prospective patient cohorts, regulatory approval and HIPAA/GDPR
compliance are likewise beyond the scope of this academic project, though their
importance for eventual clinical deployment is acknowledged.

## 1.9 Significance and Contributions

The principal contributions of this thesis are:

1. **An explainable stacked ensemble framework** that unifies GAN-based data
   augmentation, an ensemble of three transfer learning architectures, and integrated
   Grad-CAM visualisation within a single, end-to-end pipeline for lung cancer
   detection.
2. **A dual-dataset evaluation** on both a curated benchmark (IQ-OTH/NCCD) and a raw
   multi-institutional clinical dataset (LIDC-IDRI), providing an honest and realistic
   assessment of generalisation that is absent from many single-dataset studies.
3. **A leakage-free stacked ensemble**, in which the XGBoost meta-learner is trained on
   base-model predictions from a held-out validation set, ensuring that reported
   performance reflects genuine generalisation rather than training-set memorisation.
4. **Integrated, not post-hoc, explainability** — Grad-CAM is implemented as a core
   output of the system alongside the classification score, reinforcing clinical
   confidence in every prediction.
5. **A deployable web application** that makes the trained ensemble accessible and
   demonstrates real-time prediction with visual explanation.

## 1.10 Organisation of the Thesis

The remainder of this thesis is structured as follows:

- **Chapter 2 — Literature Review** surveys the clinical background of lung cancer
  diagnosis, traditional and deep-learning-based detection methods, transfer learning,
  GAN augmentation, ensemble learning, and explainable AI, concluding with the research
  gap that this work addresses.
- **Chapter 3 — Materials and Methods** describes the datasets, the preprocessing
  pipeline, the GAN architecture and training procedure, the three CNN base models,
  the stacked ensemble with XGBoost meta-learner, the Grad-CAM explainability module,
  the evaluation metrics and the full experimental setup.
- **Chapter 4 — Results and Discussion** presents the experimental results on both
  datasets, the cross-dataset comparison, the Grad-CAM analysis and the web application
  demonstration, and discusses the findings in the context of existing literature.
- **Chapter 5 — Conclusion and Future Work** summarises the contributions of the study,
  acknowledges its limitations and proposes directions for future research.
