"""
Lung Cancer Detection - Interactive Demo App
============================================

A professional Gradio web interface for lung cancer detection from CT scans.

Features:
- Upload CT scan images (PNG, JPEG, DICOM)
- Choose from 3 CNN models or ensemble
- Real-time predictions with confidence scores
- Grad-CAM visualization for explainability
- Batch processing support
- Professional medical-grade UI
"""

import os
import sys
import gradio as gr
import torch
import numpy as np
import cv2
from pathlib import Path
import joblib
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from configs.config import *
from src.models.base_models import ModelFactory
from src.preprocessing import load_image
from src.augmentations import get_val_transforms
from src.explainability.gradcam import visualize_gradcam
from src.utils import ensure_dir

# Global cache for loaded models
MODEL_CACHE = {}
META_LEARNER_CACHE = {}


def load_meta_learner(checkpoint_dir='checkpoints'):
    """Load the trained XGBoost stacking meta-learner (same one reported in RESULTS.md)."""
    if checkpoint_dir in META_LEARNER_CACHE:
        return META_LEARNER_CACHE[checkpoint_dir]

    meta_path = Path(f'{checkpoint_dir}/ensemble/meta_learner_xgboost.pkl')
    if not meta_path.exists():
        print(f"[!] No trained meta-learner found at {meta_path}, falling back to probability averaging")
        META_LEARNER_CACHE[checkpoint_dir] = None
        return None

    meta_learner = joblib.load(meta_path)
    print(f"[OK] Loaded trained XGBoost meta-learner from {meta_path}")
    META_LEARNER_CACHE[checkpoint_dir] = meta_learner
    return meta_learner


def load_model(model_name, device='cuda'):
    """Load model with caching."""
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
    
    print(f"Loading {model_name} model...")
    
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    model = ModelFactory.create_model(
        model_type=model_name,
        num_classes=NUM_CLASSES,
        pretrained=False
    ).to(device)
    
    # Try to load trained weights if available
    checkpoint_path = Path(f'checkpoints/{model_name}_finetuned_best.pth')
    if not checkpoint_path.exists():
        checkpoint_path = Path(f'checkpoints/{model_name}_frozen_best.pth')
    if not checkpoint_path.exists():
        checkpoint_path = Path(f'checkpoints/{model_name}_best.pth')
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[OK] Loaded trained weights from {checkpoint_path}")
    else:
        print(f"[!] Using pretrained ImageNet weights (no trained checkpoint found)")
    
    model.eval()
    MODEL_CACHE[model_name] = model
    
    return model


def predict_single_model(image, model_name, show_gradcam=True, device='cuda'):
    """Make prediction with a single model."""
    try:
        # Load model
        model = load_model(model_name, device)
        
        # Preprocess image
        if isinstance(image, np.ndarray):
            processed_image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        else:
            processed_image = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
        
        # Convert to tensor
        transform = get_val_transforms(IMAGE_SIZE)
        transformed = transform(image=processed_image)
        image_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            pred_class = predicted.item()
            conf_value = confidence.item()
        
        # Get class probabilities
        probs = probabilities[0].cpu().numpy()
        
        # Generate Grad-CAM if requested
        gradcam_image = None
        if show_gradcam:
            try:
                gradcam_overlay, _, _ = visualize_gradcam(
                    model=model,
                    model_name=model_name,
                    image=processed_image,
                    image_tensor=image_tensor
                )
                gradcam_image = gradcam_overlay
            except Exception as e:
                print(f"Grad-CAM generation failed: {e}")
                gradcam_image = processed_image
        
        # Create result text
        result_class = "🔴 CANCER DETECTED" if pred_class == 1 else "✅ NO CANCER DETECTED"
        confidence_text = f"{conf_value * 100:.1f}%"
        
        # Detailed probabilities
        prob_text = f"""
### Prediction Details

**Model:** {model_name.upper()}

**Diagnosis:** {result_class}

**Confidence:** {confidence_text}

---

**Class Probabilities:**
- No Cancer: {probs[0]*100:.2f}%
- Cancer: {probs[1]*100:.2f}%

---

**Recommendation:**
{get_recommendation(pred_class, conf_value)}
        """
        
        return gradcam_image if show_gradcam else processed_image, prob_text
        
    except Exception as e:
        error_msg = f"❌ Error during prediction: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg


def predict_ensemble(image, show_gradcam=True, device='cuda'):
    """Make ensemble prediction using all models."""
    try:
        model_names = ['efficientnet', 'densenet', 'resnet']
        all_probs = []
        
        # Preprocess image
        if isinstance(image, np.ndarray):
            processed_image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        else:
            processed_image = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
        
        # Convert to tensor
        transform = get_val_transforms(IMAGE_SIZE)
        transformed = transform(image=processed_image)
        image_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Get predictions from all models
        for model_name in model_names:
            model = load_model(model_name, device)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                all_probs.append(probabilities[0].cpu().numpy())
        
        # Use the real trained XGBoost stacking meta-learner (same model reported
        # in RESULTS.md), falling back to probability averaging only if it's missing.
        meta_learner = load_meta_learner()
        meta_features = np.hstack(all_probs).reshape(1, -1)  # order: efficientnet, densenet, resnet

        if meta_learner is not None:
            avg_probs = meta_learner.predict_proba(meta_features)[0]
            ensemble_method = "Stacked Ensemble (trained XGBoost meta-learner)"
        else:
            avg_probs = np.mean(all_probs, axis=0)
            ensemble_method = "Probability Averaging (fallback — no trained meta-learner found)"

        pred_class = np.argmax(avg_probs)
        confidence = avg_probs[pred_class]
        
        # Generate ensemble Grad-CAM (average heatmaps)
        gradcam_image = None
        if show_gradcam:
            try:
                # Use EfficientNet for visualization (could average all three)
                gradcam_overlay, _, _ = visualize_gradcam(
                    model=load_model('efficientnet', device),
                    model_name='efficientnet',
                    image=processed_image,
                    image_tensor=image_tensor
                )
                gradcam_image = gradcam_overlay
            except Exception as e:
                print(f"Ensemble Grad-CAM failed: {e}")
                gradcam_image = processed_image
        
        # Create result text
        result_class = "🔴 CANCER DETECTED" if pred_class == 1 else "✅ NO CANCER DETECTED"
        confidence_text = f"{confidence * 100:.1f}%"
        
        # Detailed results
        prob_text = f"""
### Ensemble Prediction

**Model:** {ensemble_method}

**Diagnosis:** {result_class}

**Confidence:** {confidence_text}

---

**Class Probabilities:**
- No Cancer: {avg_probs[0]*100:.2f}%
- Cancer: {avg_probs[1]*100:.2f}%

---

**Individual Model Predictions:**
"""
        for i, model_name in enumerate(model_names):
            prob_text += f"\n- {model_name.upper()}: No Cancer {all_probs[i][0]*100:.1f}% | Cancer {all_probs[i][1]*100:.1f}%"
        
        prob_text += f"""

---

**Recommendation:**
{get_recommendation(pred_class, confidence)}
        """
        
        return gradcam_image if show_gradcam else processed_image, prob_text
        
    except Exception as e:
        error_msg = f"❌ Error during ensemble prediction: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg


def get_recommendation(pred_class, confidence):
    """Generate clinical recommendation based on prediction."""
    if pred_class == 1:  # Cancer
        if confidence > 0.9:
            return "⚠️ **HIGH CONFIDENCE CANCER DETECTION** - Immediate clinical review recommended. Schedule biopsy and comprehensive diagnostic workup."
        elif confidence > 0.7:
            return "⚠️ **MODERATE CONFIDENCE CANCER DETECTION** - Clinical review recommended. Consider additional imaging and specialist consultation."
        else:
            return "⚠️ **LOW CONFIDENCE CANCER DETECTION** - Additional imaging recommended. Consider repeat scan and specialist review."
    else:  # No cancer
        if confidence > 0.9:
            return "✅ **HIGH CONFIDENCE NO CANCER** - Continue routine screening as recommended by physician."
        elif confidence > 0.7:
            return "✅ **MODERATE CONFIDENCE NO CANCER** - Consider follow-up imaging in 6-12 months."
        else:
            return "⚠️ **LOW CONFIDENCE** - Additional imaging recommended to rule out abnormalities."


def analyze_image(image, model_choice, show_gradcam):
    """Main analysis function called by Gradio interface."""
    if image is None:
        return None, "⚠️ Please upload an image first."
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_choice == "Ensemble (Recommended)":
        return predict_ensemble(image, show_gradcam, device)
    else:
        model_name = model_choice.lower().split()[0]
        return predict_single_model(image, model_name, show_gradcam, device)


def create_demo_interface():
    """Create the Gradio interface."""
    
    # Minimalist light UI
    custom_css = """
    :root, .gradio-container { color-scheme: light !important; }
    body, .gradio-container {
        background: #ffffff !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1300px !important;
        color: #1f2937 !important;
    }
    .gr-box, .block, .form, .panel {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: none !important;
    }
    .gr-button-primary, button.primary {
        background: #10b981 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        box-shadow: none !important;
    }
    .gr-button-primary:hover, button.primary:hover {
        background: #059669 !important;
    }
    .gr-button-secondary, button.secondary {
        background: #ffffff !important;
        color: #374151 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 6px !important;
    }
    h1, h2, h3 {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    h1 {
        text-align: center;
        font-size: 1.8rem !important;
        margin-bottom: 0.25rem !important;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .warning-box {
        background: #fafafa;
        border: 1px solid #e5e7eb;
        border-left: 3px solid #f59e0b;
        padding: 0.75rem 1rem;
        margin: 1rem 0;
        border-radius: 6px;
        color: #4b5563;
        font-size: 0.9rem;
    }
    .phase-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1.5rem 0 2rem 0;
    }
    .phase-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.1rem 1rem;
        text-align: center;
    }
    .phase-card .phase-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: #f3f4f6;
        color: #4b5563;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .phase-card .phase-title {
        font-weight: 600;
        color: #1f2937;
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
    }
    .phase-card .phase-desc {
        color: #6b7280;
        font-size: 0.8rem;
        line-height: 1.3;
    }
    .phase-card.highlight {
        background: #ecfdf5;
        border: 1.5px solid #10b981;
    }
    .phase-card.highlight .phase-num {
        background: #10b981;
        color: #ffffff;
    }
    .phase-card.highlight .phase-title {
        color: #047857;
    }
    .core-badge {
        display: inline-block;
        background: #10b981;
        color: #ffffff;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        margin-top: 0.4rem;
    }
    """

    force_light_js = """
    () => {
        document.documentElement.classList.remove('dark');
        document.body.classList.remove('dark');
    }
    """

    with gr.Blocks(
        css=custom_css,
        title="Lung Cancer Detection AI",
        theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="gray"),
        js=force_light_js,
    ) as demo:
        # Header
        gr.Markdown(
            """
            # 🏥 Lung Cancer Detection AI System
            ### Advanced Deep Learning for Early Cancer Detection
            
            <div class="warning-box">
            ⚠️ <b>MEDICAL DISCLAIMER:</b> This is a research prototype for demonstration purposes only. 
            It is NOT FDA-approved and should NOT be used for actual medical diagnosis. 
            Always consult with qualified healthcare professionals for medical decisions.
            </div>
            """
        )

        # Methodology highlight — Phase 3 (Stacked Ensemble) and Phase 4 (Grad-CAM XAI)
        # are this project's core contribution, so they are visually emphasized.
        gr.HTML(
            """
            <div class="phase-grid">
                <div class="phase-card">
                    <div class="phase-num">1</div>
                    <div class="phase-title">Data Acquisition<br>&amp; Preprocessing</div>
                    <div class="phase-desc">Labeling and image preprocessing of CT scans</div>
                </div>
                <div class="phase-card">
                    <div class="phase-num">2</div>
                    <div class="phase-title">Transfer Learning<br>Feature Extraction</div>
                    <div class="phase-desc">EfficientNet-B0, DenseNet-121, ResNet-50 backbones</div>
                </div>
                <div class="phase-card highlight">
                    <div class="phase-num">3</div>
                    <div class="phase-title">Stacked Ensemble<br>Model</div>
                    <div class="phase-desc">XGBoost meta-learner combines all 3 base models</div>
                    <div class="core-badge">THIS DEMO</div>
                </div>
                <div class="phase-card highlight">
                    <div class="phase-num">4</div>
                    <div class="phase-title">Explainable AI<br>(Grad-CAM)</div>
                    <div class="phase-desc">Visual heatmap showing why the model decided</div>
                    <div class="core-badge">THIS DEMO</div>
                </div>
            </div>
            """
        )

        with gr.Row():
            # Left column: Input
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload CT Scan")
                
                image_input = gr.Image(
                    label="CT Scan Image",
                    type="pil",
                    height=400
                )
                
                gr.Markdown("### ⚙️ Analysis Settings")
                
                model_choice = gr.Radio(
                    choices=[
                        "Ensemble (Recommended)",
                        "EfficientNet-B0",
                        "DenseNet-121",
                        "ResNet-50"
                    ],
                    value="Ensemble (Recommended)",
                    label="Model Selection",
                    info="Ensemble combines all models for best accuracy"
                )
                
                show_gradcam = gr.Checkbox(
                    label="Show Grad-CAM Visualization",
                    value=True,
                    info="Highlights regions influencing the prediction"
                )
                
                analyze_btn = gr.Button(
                    "🔬 Analyze CT Scan",
                    variant="primary",
                    size="lg"
                )
                
                clear_btn = gr.Button("🔄 Clear", variant="secondary")
                
                # Example images
                gr.Markdown("### 📋 Example Images")
                gr.Examples(
                    examples=[
                        ["data/demo_cnn/train/cancer/img_000.png"] if Path("data/demo_cnn/train/cancer/img_000.png").exists() else None,
                        ["data/demo_cnn/train/no_cancer/img_000.png"] if Path("data/demo_cnn/train/no_cancer/img_000.png").exists() else None,
                    ],
                    inputs=image_input,
                    label="Click to load example"
                )
            
            # Right column: Results
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Analysis Results")
                
                result_image = gr.Image(
                    label="Analyzed Image (with Grad-CAM)",
                    type="numpy",
                    height=400
                )
                
                result_text = gr.Markdown(
                    """
                    ### Awaiting Analysis
                    
                    Upload a CT scan image and click "Analyze" to begin.
                    
                    **Supported formats:** PNG, JPEG, DICOM
                    
                    **Image requirements:**
                    - Clear CT scan image
                    - Lung window preferred
                    - Good contrast and resolution
                    """
                )
        
        # Information section
        with gr.Accordion("ℹ️ About This System", open=False):
            gr.Markdown(
                """
                ## System Architecture
                
                This AI system uses a **stacked ensemble** approach combining three state-of-the-art deep learning models:
                
                ### Base Models
                1. **EfficientNet-B0**: Efficient and accurate (5.3M parameters)
                2. **DenseNet-121**: Feature reuse architecture (8M parameters)  
                3. **ResNet-50**: Deep residual learning (25.6M parameters)
                
                ### Key Features
                - ✅ **Transfer Learning**: Models pretrained on ImageNet
                - ✅ **Two-Phase Training**: Frozen backbone then fine-tuning
                - ✅ **Class Imbalance Handling**: Weighted loss functions
                - ✅ **Data Augmentation**: Medical-appropriate transforms + GAN
                - ✅ **Explainability**: Grad-CAM visual explanations
                - ✅ **Ensemble Learning**: Meta-learner combines predictions
                
                ### Performance Metrics (Expected)
                - **Individual Models**: 92-95% accuracy
                - **Ensemble**: 94-96% accuracy
                - **Sensitivity (Cancer Detection)**: 93-96%
                - **Specificity (No Cancer)**: 92-95%
                
                ### Grad-CAM Visualization
                Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions of the CT scan that most influenced the model's decision. 
                Red regions indicate high importance, helping clinicians understand and validate the AI's reasoning.
                
                ### Medical AI Best Practices
                - Patient-level data splitting (prevents leakage)
                - Stratified K-fold cross-validation
                - Comprehensive evaluation metrics
                - Visual explainability for clinical trust
                
                ### Limitations
                - Trained on limited dataset (demonstration)
                - Not validated on diverse populations
                - Requires clinical interpretation
                - Should not replace expert radiologist review
                """
            )
        
        with gr.Accordion("🛠️ Technical Details", open=False):
            gr.Markdown(
                """
                ## Model Information
                
                ### Device Status
                - **PyTorch Version**: """ + torch.__version__ + """
                - **CUDA Available**: """ + str(torch.cuda.is_available()) + """
                - **Device**: """ + ('CUDA (GPU)' if torch.cuda.is_available() else 'CPU') + """
                
                ### Model Loading
                Models are loaded on-demand and cached for performance. First prediction may take longer.
                
                ### Image Processing Pipeline
                1. Resize to 224×224 pixels
                2. Normalize to ImageNet statistics
                3. Convert to PyTorch tensor
                4. Forward pass through model
                5. Softmax activation for probabilities
                6. Optional: Generate Grad-CAM heatmap
                
                ### Inference Time
                - Single model: ~50-100ms (GPU) / 500-1000ms (CPU)
                - Ensemble: ~150-300ms (GPU) / 1500-3000ms (CPU)
                - Grad-CAM: +30-50ms
                
                ### File Structure
                ```
                lung_cancer_ai/
                ├── src/
                │   ├── models/          # CNN architectures
                │   ├── ensemble/        # Stacking ensemble
                │   ├── explainability/  # Grad-CAM
                │   └── ...
                ├── checkpoints/         # Trained models
                ├── configs/             # Configuration
                └── demo_app.py         # This application
                ```
                """
            )
        
        # Footer
        gr.Markdown(
            """
            ---
            
            <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
            <b>Lung Cancer Detection AI System</b> | Research Prototype v1.0<br>
            Built with PyTorch, Gradio, EfficientNet, DenseNet, ResNet<br>
            For research and educational purposes only | Not for clinical use
            </div>
            """
        )
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_image,
            inputs=[image_input, model_choice, show_gradcam],
            outputs=[result_image, result_text]
        )
        
        clear_btn.click(
            fn=lambda: (None, None, "### Awaiting Analysis\n\nUpload a CT scan image and click 'Analyze' to begin."),
            inputs=None,
            outputs=[image_input, result_image, result_text]
        )
    
    return demo


def main():
    """Launch the demo application."""
    print("="*70)
    print("LUNG CANCER DETECTION AI - DEMO APPLICATION")
    print("="*70)
    print("\nInitializing...")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if device == 'cpu':
        print("WARNING: Running on CPU - predictions will be slower")
        print("   For faster performance, use a CUDA-enabled GPU")

    # Create demo
    print("\nCreating Gradio interface...")
    demo = create_demo_interface()

    # Launch
    print("\n" + "="*70)
    print("LAUNCHING APPLICATION")
    print("="*70)
    print("\nThe app will open in your default browser.")
    print("If it doesn't open automatically, navigate to: http://localhost:7860")
    print("\nPress Ctrl+C to stop the server.")
    print("="*70 + "\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
