# CHAPTER 5

# CONCLUSION AND FUTURE WORK

## 5.1 Conclusion

This thesis presented the design, implementation and evaluation of an **Explainable
Stacked Ensemble Model for Lung Cancer Detection Using Transfer Learning and Grad-CAM
Visualisation**. The system was developed in response to three persistent limitations
identified in the existing literature: the lack of sufficient and balanced training
data, the reliance on single-model predictions that are inherently unstable, and the
widespread practice of applying explainability tools as post-hoc add-ons rather than
integrating them as core components of the prediction pipeline.

The proposed system addresses all three limitations through a coherent, end-to-end
pipeline. A **Conditional GAN** generates synthetic CT images of the minority class,
mitigating class imbalance in the training data without requiring additional real image
collection. Three complementary **transfer learning architectures** — EfficientNet-B0,
DenseNet-121 and ResNet-50, each fine-tuned using a two-phase frozen-then-fine-tuned
strategy — serve as base models whose diverse error patterns are exploited by a
**stacked XGBoost meta-learner** to produce a single, more reliable final prediction.
**Grad-CAM visualisation** is generated alongside every prediction, producing a
heatmap that highlights the CT regions most responsible for the classification decision
and providing clinically interpretable evidence of the model's reasoning.

The system was evaluated on two datasets of deliberately contrasting nature. On the
clean **IQ-OTH/NCCD** benchmark (166 test images), the ensemble achieved **98.80%
accuracy, 100% recall and AUC = 0.9997** — surpassing the target of 97% stated in the
project proposal, and critically producing **zero false negatives** (no missed cancer
cases). On the raw, multi-institutional **LIDC-IDRI** dataset (300 test images), the
ensemble achieved **84.00% accuracy and AUC = 0.9218** — the highest AUC of any model
tested on this dataset. The 14.8-percentage-point gap between the two datasets is not a
failure of the system but an honest and important research finding: it demonstrates
quantitatively that clean benchmark performance does not guarantee generalisation to
raw clinical data, and underscores the necessity of multi-dataset evaluation in medical
AI research.

### Answers to Research Questions

**RQ1 — Can a Conditional GAN generate synthetic lung CT images of sufficient quality
to mitigate class imbalance?**
Yes. The cGAN was trained for 150 epochs and produced 67 synthetic benign CT images
that corrected the training-set class ratio from approximately 0.96:1 to 1.13:1. The
GAN training dynamics were stable, and the generated images contributed to improved
training stability, particularly in the early frozen-backbone phase.

**RQ2 — Does a stacked ensemble outperform, or perform more reliably than, individual
models?**
On IQ-OTH/NCCD, where all individual models achieved 100%, the ensemble delivered
98.80% — slightly lower in accuracy but with the same perfect recall, and with better-
calibrated probability outputs. On the more challenging LIDC-IDRI dataset, the ensemble
achieved the highest AUC (0.9218) of any model, even though its point-accuracy (84.00%)
fell between DenseNet-121 (82.00%) and ResNet-50 (85.67%). The ensemble's value is
therefore primarily in the quality of its probability estimates and its robustness
across varying data conditions, rather than always exceeding the best individual model's
accuracy on every dataset.

**RQ3 — Can Grad-CAM produce clinically meaningful explanations?**
Yes. For malignant cases, Grad-CAM consistently produced focal, concentrated heatmaps
co-localised with the pulmonary region where tumour tissue is present. For benign cases,
activation was diffuse or peripheral — consistent with the absence of a focal mass.
Heatmaps from all three architectures showed qualitative agreement on the same input
images, providing cross-architecture consistency that strengthens confidence in the
explanations. Most importantly, Grad-CAM was implemented as an integrated first-class
output rather than a post-hoc addition, ensuring that every prediction is accompanied
by a visual explanation.

**RQ4 — How well does the system generalise from a clean benchmark to raw clinical data,
and what factors explain any performance gap?**
The system achieves near-perfect performance on IQ-OTH/NCCD (98.80%) but 84.00% on
LIDC-IDRI — a 14.8-percentage-point gap. The gap is explained by fundamental
differences between the two datasets: IQ-OTH/NCCD consists of pre-processed, single-
institution PNG images, while LIDC-IDRI consists of raw DICOM slices from seven
institutions with variable scanner types, acquisition protocols and nodule appearances.
This finding highlights the generalisation challenge that all medical AI systems must
address before clinical deployment.

---

## 5.2 Key Contributions

The principal contributions of this thesis are summarised as follows:

1. **An integrated explainability-first pipeline.** The system is designed from the
   outset to produce a visual explanation alongside every classification output.
   Grad-CAM is not retrofitted but is a core component of the inference path,
   addressing the specific research gap — identified in the literature review — that
   existing systems treat XAI as post-hoc rather than integral.

2. **A leakage-free stacked ensemble.** The XGBoost meta-learner is trained on
   base-model predictions from the validation set (never the training set), ensuring
   that the reported performance figures are genuine and not inflated by data leakage.
   This design choice corrects a methodological flaw present in several published
   ensemble systems.

3. **GAN-based class-imbalance correction.** A Conditional GAN is trained on real
   lung CT images and used to generate targeted synthetic augmentation for the minority
   class, providing a reproducible and principled approach to imbalance correction that
   goes beyond simple classical augmentation.

4. **Dual-dataset evaluation.** By evaluating on both IQ-OTH/NCCD and LIDC-IDRI —
   one clean benchmark and one raw clinical dataset — the thesis provides an honest and
   realistic assessment of generalisation that is absent from the majority of
   single-dataset studies in this domain.

5. **A deployable interactive web application.** The complete pipeline is made
   accessible through a Gradio web interface, demonstrating real-time inference with
   integrated Grad-CAM visualisation and illustrating the practical utility of the
   system to non-expert users.

---

## 5.3 Limitations

The following limitations should be considered when interpreting the results of this
study:

1. **Dataset size.** Both datasets used in this study are relatively small by deep
   learning standards (approximately 1,000 and 1,400 images respectively). While the
   transfer learning strategy mitigates this to a large extent, larger and more diverse
   datasets would provide stronger evidence of generalisation.

2. **Two-dimensional classification.** The system classifies individual 2D CT slices
   rather than analysing the full 3D CT volume. This means that the spatial
   relationship between consecutive slices — which a radiologist routinely uses to
   characterise nodules — is not exploited. Three-dimensional volumetric analysis could
   substantially improve performance, particularly on LIDC-IDRI.

3. **GAN resolution.** The Conditional GAN generates images at a native resolution of
   64 × 64 pixels, which must be upscaled to 224 × 224 for training. While the
   generated images provide useful augmentation, they lack the fine-grained detail of
   real CT images and their quality is limited compared with what a higher-resolution
   GAN architecture could produce.

4. **No clinical validation.** The system has not been evaluated in a prospective
   clinical setting with real patients, reviewed by radiologists, or assessed against
   clinical diagnostic criteria. Performance on a static test set — while informative —
   is not equivalent to clinical validation, and the results should not be interpreted
   as evidence of clinical readiness.

5. **Binary classification only.** The system performs binary (cancer / no-cancer)
   classification and does not address nodule segmentation, tumour staging, size
   estimation, or histological subtype classification — all of which are relevant to
   clinical management.

6. **EfficientNet-B0 underperformance on LIDC-IDRI.** The wide performance gap between
   EfficientNet-B0 (68.67%) and the other models on LIDC-IDRI suggests that this
   architecture may not be well-suited to the complexity of raw clinical DICOM images.
   Alternative lightweight architectures, or a more aggressive fine-tuning strategy,
   might produce better results.

---

## 5.4 Future Work

The following directions are identified as promising avenues for extending and improving
the proposed system:

**1. Three-dimensional volumetric analysis.**
Replacing the 2D slice-level classification with a 3D volumetric approach — using
architectures such as 3D-CNN, V-Net or 3D-EfficientNet — would allow the system to
exploit the spatial continuity between CT slices and is likely to substantially improve
performance on multi-slice datasets such as LIDC-IDRI.

**2. Higher-resolution GAN augmentation.**
Replacing the current 64 × 64 Conditional GAN with a higher-resolution generative
architecture — such as Progressive GAN, StyleGAN2 or Latent Diffusion Models — would
produce synthetic images of significantly higher visual fidelity, potentially providing
more effective training augmentation.

**3. Domain adaptation and cross-institutional generalisation.**
Techniques such as domain adversarial training, Cycle-GAN-based domain transfer or
contrastive pre-training on multi-institutional data could be explored to close the
generalisation gap between IQ-OTH/NCCD and LIDC-IDRI, preparing the system for
deployment across different scanning environments and institutions.

**4. Stronger and quantified explainability.**
Future work could incorporate alternative or complementary XAI methods — such as
SHAP (SHapley Additive exPlanations), Integrated Gradients or Score-CAM — and develop
quantitative metrics (e.g., pointing game accuracy, localisation fidelity) to
rigorously evaluate whether the generated explanations are clinically correct, going
beyond the qualitative assessment conducted in this thesis.

**5. Radiologist-in-the-loop validation.**
Collaboration with board-certified radiologists to review Grad-CAM outputs and provide
feedback on their clinical relevance would provide a rigorous external validation of the
explainability component. A structured user study could also assess whether the Grad-CAM
heatmaps genuinely increase clinician confidence and diagnostic accuracy.

**6. Expanded model zoo and architecture search.**
Incorporating additional architectures — such as Vision Transformers (ViT), Swin
Transformer or ConvNeXt — into the ensemble, or applying neural architecture search
(NAS) to identify the optimal backbone for lung CT data, could improve the quality of
the base models and the ensemble.

**7. Nodule segmentation and multi-task learning.**
Extending the system to simultaneously perform classification and nodule segmentation —
predicting not only whether cancer is present but also where the lesion is located as
a pixel-level mask — would increase its clinical utility. Multi-task learning has been
shown to improve performance on both tasks simultaneously.

**8. Clinical deployment and regulatory compliance.**
Progressing the system toward clinical deployment would require integration with
hospital PACS (Picture Archiving and Communication Systems), compliance with HIPAA
and GDPR data protection regulations, and regulatory approval (e.g., FDA clearance
for AI/ML-based medical devices). These steps are beyond the scope of this academic
project but represent the natural path toward real-world impact.

---

## 5.5 Closing Remarks

This thesis has demonstrated that a carefully designed, end-to-end deep learning
pipeline — combining GAN augmentation, a stacked ensemble of transfer learning models,
and integrated Grad-CAM explainability — can achieve clinically meaningful performance
on lung cancer detection from CT images. The system achieves 98.80% accuracy with
zero missed cancer cases on a clean benchmark, and 84.00% accuracy with the highest
AUC (0.9218) among all tested models on a challenging raw clinical dataset. It
produces visual explanations that are focal, consistent, and aligned with clinically
relevant anatomical regions.

The 14.8-percentage-point gap between the two datasets is not a shortcoming to conceal
but a finding to report honestly: it is precisely this kind of rigorous, multi-dataset
evaluation that the medical AI community needs in order to build systems that are not
only accurate on benchmarks but trustworthy in practice. The goal of developing AI
systems that are accurate, robust *and* explainable — stated at the outset of this
thesis — has been meaningfully advanced, and the path toward clinical deployment,
though long, is now more clearly defined.
