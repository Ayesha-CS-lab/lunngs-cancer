# CHAPTER 2

# LITERATURE REVIEW

## 2.1 Introduction

The development of automated systems for lung cancer detection sits at the intersection
of medical imaging, machine learning and clinical decision support. A substantial body
of research has accumulated over the past decade, progressing from classical
image-processing techniques and shallow machine learning models through to the modern
deep-learning methods that now dominate the field. This chapter reviews the most
relevant work in each of these areas. It begins with the clinical and diagnostic
background of lung cancer, moves through the evolution of imaging-based detection
methods, examines the specific deep-learning techniques employed in the proposed system
— transfer learning, GAN augmentation, ensemble learning and explainable AI — and
concludes by identifying the research gap that motivates the present work.

## 2.2 Traditional Methods of Lung Cancer Diagnosis

Historically, the diagnosis of lung cancer relied on a combination of clinical
examination, chest X-ray, bronchoscopy and, ultimately, histopathological biopsy.
While biopsy remains the definitive diagnostic standard, it is invasive, carries
procedural risk and is impractical as a screening tool for asymptomatic populations.
Chest X-ray, though widely available and inexpensive, has limited sensitivity for small
nodules, which can be obscured by overlapping anatomical structures. These limitations
led to the adoption of **low-dose Computed Tomography (LDCT)** as the preferred
screening modality, following landmark clinical trials demonstrating a significant
reduction in lung cancer mortality among high-risk individuals screened with LDCT
compared with chest X-ray [1].

However, the interpretation of CT images is highly demanding. A typical CT examination
of the thorax produces between 300 and 500 axial slices, each of which must be
evaluated by a radiologist. The volume of data, combined with the subtle appearance of
early-stage nodules, creates conditions for diagnostic errors. Studies have reported
both false-negative rates (missed cancers) and high rates of false positives (benign
nodules flagged for unnecessary biopsy), underlining the need for reliable automated
assistance.

Early computer-assisted detection (CAD) systems used handcrafted image features —
such as Histogram of Oriented Gradients (HOG), Haralick texture features and geometric
shape descriptors — fed into classical classifiers such as Support Vector Machines
(SVMs) or Random Forests. While these approaches offered some improvement over
unaided reading, their performance was fundamentally limited by the quality of the
hand-engineered features, which required extensive domain expertise to design and
struggled to capture the full complexity of natural image variation.

## 2.3 Medical Imaging Modalities for the Lungs

CT remains the gold standard for lung cancer screening and nodule characterisation.
Unlike planar X-ray, CT provides three-dimensional volumetric information that allows
the size, shape, density and location of nodules to be assessed with precision.
Contrast-enhanced CT can further reveal vascular involvement, while **Positron
Emission Tomography (PET)** combined with CT (PET-CT) is used for metabolic staging.
In certain research contexts, **Magnetic Resonance Imaging (MRI)** has also been
explored for lung nodule characterisation, as demonstrated by Klangburanawat et al.
(2024), who benchmarked transfer learning models on MRI-derived datasets [11].

For computational modelling, raw CT data is typically stored in the **DICOM** format,
which encodes pixel intensities in **Hounsfield Units (HU)** — a linear scale in which
air is represented as −1000 HU and bone as approximately +1000 HU. Soft tissue and
lung parenchyma fall in the intermediate range. To make DICOM images suitable for a
deep learning model, it is standard practice to apply **lung windowing** (typically
HU −1000 to +400) to isolate and enhance the relevant anatomical structures before
converting to a standard image format. This step is essential when working with raw
clinical datasets such as LIDC-IDRI, and is incorporated into the preprocessing
pipeline described in Chapter 3.

## 2.4 The Role of Deep Learning in Medical Image Analysis

The publication of deep convolutional neural networks (CNNs) capable of surpassing
human-level performance on large-scale image recognition benchmarks marked a turning
point for medical image analysis. Deep learning models, trained end-to-end on labelled
data, learn hierarchical feature representations automatically, eliminating the need
for handcrafted features and offering far greater expressive power than classical
approaches.

In the specific domain of lung cancer detection, deep learning has been applied to
nodule detection, malignancy classification, staging, and survival prediction. The
early work of Yan et al. demonstrated that CNNs could learn discriminative features
directly from CT slices for nodule classification. More recently, the availability of
large public datasets — most notably the LIDC-IDRI dataset [8] — has enabled the
training and rigorous evaluation of increasingly sophisticated deep learning systems.
Hamoud et al. (2024) demonstrated that even a classical CNN architecture, when combined
with explainable deep learning outputs, can achieve strong classification performance
on lung CT data [17].

Despite these successes, deep learning models for medical imaging continue to face
several fundamental challenges. First, labelled medical datasets are typically far
smaller than those available in natural-image domains, making overfitting a persistent
concern. Second, class imbalance — where cancerous cases are significantly
under-represented — can distort training and lead to misleadingly high accuracy
alongside poor sensitivity. Third, and perhaps most critically for clinical adoption,
deep networks provide no inherent explanation for their predictions.

## 2.5 Convolutional Neural Networks for Lung Cancer Detection

CNNs have become the dominant architecture for medical image classification. Several
specific architectures have been widely adopted in lung cancer detection research, each
offering a different combination of depth, width, connectivity and computational
efficiency.

**ResNet** (Residual Networks), introduced by He et al. [3], addressed the
vanishing-gradient problem by introducing skip connections that allow gradients to flow
directly across many layers. This innovation made it possible to train very deep
networks (50, 101 or even 152 layers) without degradation, and ResNet-50 in particular
has become a standard baseline in medical imaging. **DenseNet** (Densely Connected
Convolutional Networks), introduced by Huang et al. [4], extended this idea by
connecting every layer to every subsequent layer within a dense block, maximising
feature reuse and reducing the number of parameters. DenseNet-121 has been shown to be
particularly effective in lung imaging tasks due to its ability to preserve fine-grained
features across network depth. **EfficientNet**, introduced by Tan and Le [5], proposed
a principled compound scaling method that simultaneously scales network depth, width
and resolution using a single efficiency coefficient. EfficientNet-B0 achieves
state-of-the-art performance with significantly fewer parameters than comparable
architectures, making it well suited to smaller medical datasets.

Kumaran et al. (2024) proposed an explainable ensemble combining multiple CNN
architectures for CT image analysis, finding that no single architecture consistently
outperformed the others across all evaluation metrics, and that combining them yielded
more stable and reliable predictions [13]. This finding directly motivates the ensemble
approach adopted in the present work.

## 2.6 Transfer Learning Approaches

A key practical challenge in medical imaging is the relative scarcity of labelled
training data. **Transfer learning** addresses this by initialising a CNN with weights
pre-trained on a large general-purpose dataset — typically ImageNet — and then
fine-tuning the network on the target medical imaging task. Because the lower layers of
a pre-trained network have already learned general low-level features (edges, textures,
gradients) that are useful for any image classification task, transfer learning allows
high performance to be achieved with far smaller datasets than would be required when
training from random initialisation.

Two-phase transfer learning — in which the pre-trained backbone is first held frozen
while only the new classification head is trained, and then the entire network is
fine-tuned with a lower learning rate — has become a standard and effective strategy.
This approach reduces the risk of catastrophically overwriting the pre-trained features
in the early stages of training.

Noman et al. (2025) proposed **LungCT-NET**, a transfer learning framework for
automated lung nodule detection from CT images, augmented with explainability
techniques, achieving high detection accuracy and demonstrating the value of combining
transfer learning with XAI outputs [10]. Klangburanawat et al. (2024) benchmarked
multiple transfer learning models for lung cancer classification, providing a systematic
comparison that highlights the sensitivity of performance to architecture choice and the
importance of dataset-specific fine-tuning [11]. Rana and Rana (2025) proposed
**SEMLCC**, a stacked ensemble model that combines transfer learning architectures for
lung cancer classification, reporting high accuracy and noting that the combination of
models consistently outperformed any single architecture [9]. Together, these studies
confirm that transfer learning is an essential component of any modern lung cancer
detection system and that the specific choice of backbone architecture matters.

## 2.7 Generative Adversarial Networks for Data Augmentation

**Generative Adversarial Networks (GANs)**, introduced by Goodfellow et al. [6], are
a class of generative model in which two neural networks — a generator and a
discriminator — are trained in an adversarial process. The generator learns to produce
synthetic images that are indistinguishable from real ones; the discriminator learns to
tell them apart. At equilibrium, the generator produces highly realistic samples that
can be used to augment the training data.

In medical imaging, GAN-based augmentation has been used to address class imbalance
by generating synthetic examples of the minority class. Jang et al. (2025) applied
GAN-based data synthesis specifically for lung cancer classification, demonstrating
that augmenting the training set with synthetic minority-class images measurably
improved classification performance and reduced the sensitivity of the model to
imbalanced class distributions [12]. Mendez et al. (2024) explored Cycle-GAN for CT
image synthesis, showing that unpaired image-to-image translation could generate
realistic CT images for domain adaptation and augmentation [19].

A **Conditional GAN (cGAN)** extends the standard GAN framework by conditioning both
the generator and discriminator on a class label, enabling the generation of images
belonging to a specific target class. This makes cGANs particularly suitable for
medical augmentation, where the goal is to generate synthetic images of a specific
pathological class rather than arbitrary images. The use of cGAN-based augmentation
in the present system is described in detail in Chapter 3.

Classical augmentation techniques — such as random rotation, horizontal flipping,
brightness and contrast adjustment, and Gaussian blur — provide diversity within the
existing training images and reduce overfitting. While effective and computationally
inexpensive, classical augmentation cannot generate truly new examples that lie beyond
the distribution of the original data; GAN-based augmentation can.

## 2.8 Ensemble Learning Methods

**Ensemble learning** combines the predictions of multiple individual models to produce
a single, more accurate and more stable prediction. The theoretical justification is
straightforward: if different models make different errors on different examples, then
combining their outputs tends to cancel out individual errors and produce a result
closer to the ground truth.

Several ensemble strategies have been applied to CNN-based medical image classification.
**Voting** ensembles take the majority or average of the individual predictions.
**Bagging** trains multiple models on different bootstrap samples of the data.
**Stacking** (or **stacked generalisation**) trains a second-level "meta-learner" on
the outputs of the base models, learning to combine them optimally. Among these,
stacking is the most flexible, as the meta-learner can learn complex, nonlinear
relationships between the base models' predictions.

Saba et al. (2024) proposed **VER-Net**, a stacked transfer learning model for
multi-cancer diagnosis using deep learning, demonstrating that a stacked ensemble
consistently outperformed the individual architectures that composed it [14]. Sabu and
Prakash (2025) proposed **SMLCC**, a multi-attention stacked network for lung cancer
classification, reporting that the stacking strategy improved both accuracy and
robustness compared with single-model baselines [16]. Rana and Rana (2025) similarly
found that their SEMLCC stacked ensemble achieved superior performance to each
constituent model individually [9]. These findings collectively support the use of a
stacked ensemble in the present system, where **XGBoost** is used as the meta-learner,
trained on base-model predictions from a held-out validation set to avoid data leakage.

## 2.9 Explainable Artificial Intelligence (XAI) and Grad-CAM

The opacity of deep neural networks is one of the most significant obstacles to their
clinical adoption. A radiologist presented with a classification output accompanied by
no explanation cannot verify whether the model has identified a genuine lesion or has
responded to an imaging artefact, scanner noise or patient positioning. This is not a
theoretical concern — studies have shown that even high-accuracy CNNs sometimes base
their predictions on spurious correlations in the training data that have no clinical
significance.

**Explainable AI (XAI)** encompasses a range of methods designed to make the
reasoning of black-box models interpretable to human users. In the context of image
classification, **Gradient-weighted Class Activation Mapping (Grad-CAM)**, introduced
by Selvaraju et al. [7], has become the most widely used approach. Grad-CAM computes
the gradient of the class score with respect to the feature maps of the final
convolutional layer and uses these gradients to produce a coarse heatmap highlighting
the regions of the input image that most strongly influenced the prediction. These
heatmaps are overlaid on the original image, providing an intuitive visual explanation
that can be reviewed by a clinician.

Verramat et al. (2024) proposed **SICEXes**, a hybrid explainable AI framework that
integrated Grad-CAM with attention mechanisms for CT image classification, showing
that integrated explainability tools produced more clinically relevant saliency maps
than post-hoc approaches [15]. Mendez et al. (2021) demonstrated that Grad-CAM, when
applied to deep learning models for lung nodule detection, consistently highlighted the
nodule region rather than background structures, supporting its validity as an
interpretability tool for this domain [19]. Noman et al. (2025) incorporated
explainability outputs into their LungCT-NET framework, confirming that XAI features
strengthened radiologist confidence in the system's recommendations [10].

A key observation from the literature — and one that directly motivates the present
work — is that XAI tools are most commonly applied *post-hoc*: that is, they are
bolted onto an existing model as an afterthought, rather than being treated as a core
design requirement. This means that the explainability of a system is often
inconsistent, incomplete or difficult to reproduce. The proposed system addresses this
by incorporating Grad-CAM as a first-class output component, produced alongside the
classification score for every prediction.

## 2.10 Benchmark Datasets: IQ-OTH/NCCD and LIDC-IDRI

Two datasets are used in this work.

The **IQ-OTH/NCCD** (Iraq Oncology Teaching Hospital / National Centre for Cancer
Diseases) dataset is a publicly available collection of lung CT images collected in
Iraq, organised into labelled folders corresponding to cancerous and non-cancerous
cases. The images are pre-processed PNGs, making them straightforward to work with and
suitable as a clean benchmark for evaluating classification performance under controlled
conditions.

The **LIDC-IDRI** (Lung Image Database Consortium – Image Database Resource
Initiative) dataset, described by Armato et al. [8], is a large, multi-institutional
collection of thoracic CT scans annotated by up to four experienced radiologists using
a standardised XML annotation format. It contains 1,018 cases from seven different
academic medical centres and is widely regarded as the most comprehensive and
challenging public benchmark for lung nodule analysis. The raw DICOM format and the
variability in nodule appearance, scanner type and imaging protocol make LIDC-IDRI a
significantly more demanding dataset than IQ-OTH/NCCD, and performance differences
between the two datasets reflect meaningful information about a model's ability to
generalise.

## 2.11 Summary of Related Work

The following table summarises the most relevant recent studies and identifies the key
limitation of each that the present work addresses.

| Author(s) & Year | Technique Used | Focus Area | Key Limitation |
|---|---|---|---|
| Rana & Rana (2025) [9] | Stacked Ensemble + Transfer Learning (SEMLCC) | High-accuracy lung cancer classification from CT images | Limited dataset diversity; XAI explanations limited |
| Noman et al. (2025) [10] | LungCT-NET: Transfer learning + XAI | Automated lung nodule detection | High computational cost; limited evaluation on external datasets |
| Klangburanawat et al. (2024) [11] | VGG16 Transfer Learning — benchmarking | Transfer learning evaluation for lung cancer (MRI / CT) | Moderate accuracy; dataset-specific, may not generalise |
| Jang et al. (2025) [12] | GAN-based Data Synthesis + CNN classification | Synthetic data generation to address class imbalance | Limited architectural diversity; no explainability |
| Kumaran et al. (2024) [13] | Ensemble VGG16 + ResNeXt + InceptionV3 + XAI | Explainable ensemble for CT image classification | Potential overfitting on small datasets; no cross-dataset validation |
| Saba et al. (2024) [14] | VER-Net (Stacked Transfer Learning Model) | Deep learning for multi-cancer diagnosis using imaging data | Model complexity; limited visual interpretability |
| Verramat et al. (2024) [15] | SICEXes: Hybrid XAI + Attention | XAI validation and confidence in model attention maps | XAI outputs inconsistent across different model architectures |
| Sabu & Prakash (2025) [16] | SMLCC: Multi-Attention Stacked Networks | Lung cancer detection with stacked attention | High computational requirements; XAI dependent on visual inspection |
| Hamoud et al. (2024) [17] | Classical CNN + Explainable deep learning | CT image classification with model interpretability | Limited scalability; lack of quantitative XAI evaluation |
| Oncu et al. (2025) [20] | Multimodal DL: CT images + clinical data | Integration of imaging and clinical features for lung diagnosis | Dataset availability; difficulty combining modalities at inference |

## 2.12 Research Gap

The review of existing literature reveals several consistent limitations in current
approaches to deep-learning-based lung cancer detection:

1. **Isolated explainability.** Although Grad-CAM and related XAI techniques have been
   applied in several studies, they are typically used as post-hoc tools applied after
   training, rather than being integrated as a core output of the classification
   pipeline [15]. This limits the consistency and clinical utility of the explanations
   produced.

2. **Single-dataset evaluation.** The majority of published systems are evaluated on
   a single dataset, making it impossible to determine whether their reported accuracy
   reflects genuine generalisation or merely overfitting to the idiosyncrasies of
   one particular data source.

3. **Incomplete integration of augmentation and ensembling.** While GAN-based
   augmentation and stacked ensemble methods have been studied separately, few systems
   combine both within a single, coherent framework alongside explainability, and fewer
   still evaluate this combination rigorously across multiple datasets [9, 12].

4. **Limited clinical interpretability evidence.** Studies that do apply Grad-CAM
   rarely evaluate whether the highlighted regions correspond to clinically meaningful
   structures. Providing visual evidence that the model's attention aligns with the
   tumour region is essential for building clinical trust.

As stated in the proposal: *"Although the current lung cancer detection models achieve
outstanding accuracy using deep learning and transfer models, they lack integration
methods, such as Grad-CAM, for providing clinical insights. Grad-CAM and similar XAI
tools are often applied post-hoc rather than integrated within models design."*

The present work directly addresses this gap by: (i) integrating Grad-CAM as a
first-class component of the prediction pipeline; (ii) evaluating on two contrasting
datasets to provide evidence of generalisation; and (iii) combining GAN augmentation,
a stacked ensemble and integrated XAI within a single framework.

## 2.13 Summary

This chapter has reviewed the clinical context of lung cancer detection, the evolution
from traditional CAD to deep learning, and the specific techniques — CNN architectures,
transfer learning, GAN augmentation, stacked ensemble learning and Grad-CAM
explainability — that form the building blocks of the proposed system. The literature
confirms that each of these components individually improves some aspect of detection
performance, but that no existing work combines all of them in a unified, integrated
and rigorously evaluated system. The following chapter describes in full detail how
these components are brought together in the proposed methodology.

---

## References (Chapter 2)

[1] World Health Organization. (2024). *Cancer fact sheet*. World Health Organization.

[2] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for
large-scale image recognition. *arXiv preprint arXiv:1409.1556*.

[3] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image
recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition (CVPR)*, 770–778.

[4] Huang, G., Liu, Z., Van der Maaten, L., & Weinberger, K. Q. (2017). Densely
connected convolutional networks. *Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition (CVPR)*, 4700–4708.

[5] Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional
neural networks. *Proceedings of the 36th International Conference on Machine Learning
(ICML)*, 6105–6114.

[6] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S.,
Courville, A., & Bengio, Y. (2014). Generative adversarial nets. *Advances in Neural
Information Processing Systems (NeurIPS)*, 27, 2672–2680.

[7] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D.
(2017). Grad-CAM: Visual explanations from deep networks via gradient-based
localization. *Proceedings of the IEEE International Conference on Computer Vision
(ICCV)*, 618–626.

[8] Armato, S. G., McLennan, G., Bidaut, L., McNitt-Gray, M. F., Meyer, C. R.,
Reeves, A. P., ... & Clarke, L. P. (2011). The Lung Image Database Consortium (LIDC)
and Image Database Resource Initiative (IDRI): A completed reference database of lung
nodules on CT scans. *Medical Physics*, 38(2), 915–931.

[9] Rana, D. J., & Rana, K. (2025). SEMLCC: A stacked ensemble model with transfer
learning for lung cancer classification. *Cancer Oncology*. Taylor & Francis Online.

[10] Noman, M., et al. (2025). LungCT-NET: Automated lung nodule detection using
transfer learning and explainability. *Procedia Computer Science*.

[11] Klangburanawat, et al. (2024). Benchmarking transfer learning for lung cancer
classification. *Biomedical Signal Processing and Control*.

[12] Jang, et al. (2025). GAN-based data synthesis for improved lung cancer
classification. Taylor & Francis Online.

[13] Kumaran, R., et al. (2024). Explainable ensemble deep learning for lung cancer CT
classification. *arXiv Preprint*.

[14] Vanillis, M., et al. (2024). VER-Net: Stacked transfer learning model for
multi-cancer diagnosis. *BioMed Central*.

[15] Verramat, R., et al. (2024). SICEXes: A hybrid explainable AI framework with
attention for CT image classification. *arXiv Preprint*.

[16] Sabu, M., & Prakash, D. (2025). SMLCC: Multi-attention stacked networks for lung
cancer classification. *arXiv Preprint*.

[17] Hamoud, et al. (2024). Classical CNN with explainable deep learning for medical
image analysis. *BioMed Central*.

[18] Elshazly, D., & Aliyn, A. (2019). A comprehensive analysis of SMOTE for handling
class imbalance in biomedical data. *Biomedical Signal Processing and Control*.

[19] Mendez, P., et al. (2021). Enhanced deep feature extraction with Grad-CAM for
lung nodule detection. *BioMed Central*. *(See also: Mendez et al. (2024). Cycle-GAN
for CT image synthesis. ScienceDirect.)*

[20] Oncu, et al. (2025). Multimodal deep learning: Integration of CT images and
clinical data for lung cancer diagnosis. *arXiv Preprint*.
