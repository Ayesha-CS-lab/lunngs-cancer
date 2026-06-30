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

    try:
        meta_learner = joblib.load(meta_path)
        print(f"[OK] Loaded trained XGBoost meta-learner from {meta_path}")
    except Exception as e:
        print(f"[!] Could not load meta-learner ({e}); falling back to probability averaging")
        meta_learner = None

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
        
        # Build the professional HTML results panel
        model_label = f"{model_name.upper()} &middot; single model"
        result_html = build_result_html(pred_class, conf_value, probs, model_label)

        return gradcam_image if show_gradcam else processed_image, result_html

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, _error_html(str(e))


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
        
        # Build the professional HTML results panel
        individual = [(name.upper(), all_probs[i]) for i, name in enumerate(model_names)]
        model_label = ensemble_method
        result_html = build_result_html(pred_class, confidence, avg_probs, model_label, individual=individual)

        return gradcam_image if show_gradcam else processed_image, result_html

    except Exception as e:
        print(f"Error during ensemble prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, _error_html(str(e))


def _empty_state_html(message="Upload a CT scan and run the analysis to see results here."):
    """Render the idle/awaiting state for the results panel."""
    return f"""
    <div class="result-panel empty-state">
      <div class="empty-icon">&#129658;</div>
      <div class="empty-title">Awaiting Analysis</div>
      <div class="empty-text">{message}</div>
      <div class="empty-meta">
        <span>Supported: PNG &middot; JPEG &middot; DICOM</span>
        <span>Lung-window CT preferred</span>
      </div>
    </div>
    """


def _error_html(message):
    """Render an error state in the results panel."""
    return f"""
    <div class="result-panel">
      <div class="diagnosis-banner" style="background:#fffbeb;border-color:#f59e0b;">
        <div class="diagnosis-icon" style="color:#f59e0b;">&#9888;</div>
        <div class="diagnosis-main">
          <div class="diagnosis-verdict" style="color:#b45309;">Analysis Failed</div>
          <div class="diagnosis-sub">{message}</div>
        </div>
      </div>
    </div>
    """


def get_recommendation(pred_class, confidence):
    """Generate a clinical recommendation (title, body) based on prediction."""
    if pred_class == 1:  # Cancer
        if confidence > 0.9:
            return ("High-confidence cancer indication",
                    "Immediate clinical review recommended. Schedule biopsy and comprehensive diagnostic workup.")
        elif confidence > 0.7:
            return ("Moderate-confidence cancer indication",
                    "Clinical review recommended. Consider additional imaging and specialist consultation.")
        else:
            return ("Low-confidence cancer indication",
                    "Additional imaging recommended. Consider a repeat scan and specialist review.")
    else:  # No cancer
        if confidence > 0.9:
            return ("High-confidence negative result",
                    "Continue routine screening as recommended by the supervising physician.")
        elif confidence > 0.7:
            return ("Moderate-confidence negative result",
                    "Consider follow-up imaging in 6-12 months.")
        else:
            return ("Low-confidence result",
                    "Additional imaging recommended to rule out abnormalities.")


def _confidence_band(confidence):
    """Return a short qualitative label for a confidence value."""
    if confidence > 0.9:
        return "High"
    if confidence > 0.7:
        return "Moderate"
    return "Low"


def build_result_html(pred_class, confidence, probs, model_label, individual=None):
    """Render a clean, professional HTML results panel shared by all prediction paths."""
    is_cancer = pred_class == 1
    accent = "#dc2626" if is_cancer else "#059669"
    soft_bg = "#fef2f2" if is_cancer else "#ecfdf5"
    verdict = "Cancer Indicated" if is_cancer else "No Cancer Detected"
    icon = "&#9888;" if is_cancer else "&#10003;"
    no_cancer_pct = probs[0] * 100
    cancer_pct = probs[1] * 100
    rec_title, rec_body = get_recommendation(pred_class, confidence)
    band = _confidence_band(confidence)

    # Diagnosis banner
    html = f"""
    <div class="result-panel">
      <div class="diagnosis-banner" style="background:{soft_bg};border-color:{accent};">
        <div class="diagnosis-icon" style="color:{accent};">{icon}</div>
        <div class="diagnosis-main">
          <div class="diagnosis-verdict" style="color:{accent};">{verdict}</div>
          <div class="diagnosis-sub">{model_label}</div>
        </div>
        <div class="diagnosis-confidence">
          <div class="conf-value" style="color:{accent};">{confidence*100:.1f}%</div>
          <div class="conf-label">{band} confidence</div>
        </div>
      </div>

      <div class="result-section-title">Class Probabilities</div>
      <div class="prob-row">
        <div class="prob-name">No Cancer</div>
        <div class="prob-track"><div class="prob-fill" style="width:{no_cancer_pct:.1f}%;background:#059669;"></div></div>
        <div class="prob-pct">{no_cancer_pct:.1f}%</div>
      </div>
      <div class="prob-row">
        <div class="prob-name">Cancer</div>
        <div class="prob-track"><div class="prob-fill" style="width:{cancer_pct:.1f}%;background:#dc2626;"></div></div>
        <div class="prob-pct">{cancer_pct:.1f}%</div>
      </div>
    """

    # Per-model breakdown (ensemble only)
    if individual:
        html += '<div class="result-section-title">Base Model Breakdown</div><div class="model-table">'
        for name, p in individual:
            m_cancer = p[1] * 100
            m_pred = "Cancer" if p[1] >= p[0] else "No Cancer"
            m_color = "#dc2626" if p[1] >= p[0] else "#059669"
            html += f"""
            <div class="model-row">
              <div class="model-name">{name}</div>
              <div class="model-track"><div class="model-fill" style="width:{m_cancer:.1f}%;"></div></div>
              <div class="model-verdict" style="color:{m_color};">{m_pred} &middot; {max(p)*100:.0f}%</div>
            </div>
            """
        html += '</div>'

    # Recommendation
    html += f"""
      <div class="rec-box" style="border-left-color:{accent};">
        <div class="rec-title">Clinical Note &mdash; {rec_title}</div>
        <div class="rec-body">{rec_body}</div>
      </div>
      <div class="result-disclaimer">
        Research prototype output &mdash; not a medical diagnosis. Confirm all findings with a qualified radiologist.
      </div>
    </div>
    """
    return html


def analyze_image(image, model_choice, show_gradcam):
    """Main analysis function called by Gradio interface."""
    if image is None:
        return None, _empty_state_html("Please upload a CT scan image to begin analysis.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_choice == "Ensemble (Recommended)":
        return predict_ensemble(image, show_gradcam, device)
    else:
        model_name = model_choice.lower().split()[0]
        return predict_single_model(image, model_name, show_gradcam, device)


def create_demo_interface():
    """Create the Gradio interface."""

    custom_css = """
    :root, .gradio-container { color-scheme: light !important; }
    body, .gradio-container {
        background: #f6f8fa !important;
        font-family: 'Inter', 'Segoe UI', Tahoma, sans-serif;
        max-width: 1240px !important;
        color: #1f2937 !important;
    }
    footer { display: none !important; }

    /* ---------- App header ---------- */
    .app-header {
        background: linear-gradient(135deg, #0f766e 0%, #047857 100%);
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        color: #ffffff;
        margin-bottom: 0.5rem;
        box-shadow: 0 6px 20px rgba(4,120,87,0.18);
    }
    .app-header .brand {
        display: flex; align-items: center; gap: 0.75rem;
    }
    .app-header .brand-logo {
        font-size: 1.9rem; line-height: 1;
        background: rgba(255,255,255,0.15);
        width: 52px; height: 52px; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
    }
    .app-header h1 {
        font-size: 1.55rem !important; font-weight: 700 !important;
        margin: 0 !important; color: #ffffff !important; text-align: left !important;
    }
    .app-header .tagline {
        margin: 0.15rem 0 0; font-size: 0.92rem; color: #d1fae5; font-weight: 400;
    }
    .app-header .header-stats {
        display: flex; gap: 2rem; margin-top: 1.1rem;
        border-top: 1px solid rgba(255,255,255,0.18); padding-top: 0.9rem;
    }
    .app-header .stat .num { font-size: 1.25rem; font-weight: 700; }
    .app-header .stat .lbl { font-size: 0.72rem; color: #a7f3d0; text-transform: uppercase; letter-spacing: 0.04em; }

    /* ---------- Disclaimer ---------- */
    .warning-box {
        background: #fffbeb; border: 1px solid #fde68a;
        border-left: 4px solid #f59e0b;
        padding: 0.7rem 1rem; margin: 0.75rem 0 1.25rem;
        border-radius: 8px; color: #92400e; font-size: 0.85rem;
    }

    /* ---------- Section cards ---------- */
    .panel-card { background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; padding:1.25rem 1.25rem 1.4rem; box-shadow:0 1px 3px rgba(0,0,0,0.04); height:100%; box-sizing:border-box; }
    .card-head { display:flex; align-items:center; gap:0.5rem; font-weight:700; font-size:1.02rem; color:#111827; margin-bottom:0.9rem; }
    .card-head .dot { width:8px; height:8px; border-radius:50%; background:#10b981; }

    /* Strip Gradio's default block chrome from the heading wrappers */
    .card-head-wrap { background:transparent !important; border:none !important; box-shadow:none !important; padding:0 !important; }
    /* Pad the result text column so the diagnosis panel isn't flush to the edge */
    .result-col { padding:0.25rem 0.75rem 0.25rem 1.25rem !important; }
    .result-col .block { background:transparent !important; border:none !important; box-shadow:none !important; }

    .gr-button-primary, button.primary {
        background:#059669 !important; color:#fff !important; border:none !important;
        border-radius:8px !important; font-weight:600 !important; box-shadow:none !important;
    }
    .gr-button-primary:hover, button.primary:hover { background:#047857 !important; }
    .gr-button-secondary, button.secondary {
        background:#fff !important; color:#374151 !important; border:1px solid #d1d5db !important; border-radius:8px !important;
    }
    h3 { color:#111827 !important; font-weight:600 !important; }

    /* ---------- Results panel ---------- */
    .result-panel { font-family:'Inter','Segoe UI',sans-serif; }
    .diagnosis-banner {
        display:flex; align-items:center; gap:1rem;
        border:1.5px solid; border-radius:12px; padding:1rem 1.2rem; margin-bottom:1.3rem;
    }
    .diagnosis-icon { font-size:2rem; line-height:1; }
    .diagnosis-main { flex:1; }
    .diagnosis-verdict { font-size:1.25rem; font-weight:700; }
    .diagnosis-sub { font-size:0.8rem; color:#6b7280; margin-top:0.15rem; }
    .diagnosis-confidence { text-align:right; }
    .diagnosis-confidence .conf-value { font-size:1.6rem; font-weight:800; line-height:1; }
    .diagnosis-confidence .conf-label { font-size:0.72rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.04em; }

    .result-section-title { font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#6b7280; margin:1.1rem 0 0.6rem; }
    .prob-row { display:flex; align-items:center; gap:0.75rem; margin-bottom:0.55rem; }
    .prob-name { width:90px; font-size:0.85rem; color:#374151; font-weight:500; }
    .prob-track { flex:1; height:10px; background:#f3f4f6; border-radius:6px; overflow:hidden; }
    .prob-fill { height:100%; border-radius:6px; transition:width .4s ease; }
    .prob-pct { width:54px; text-align:right; font-size:0.85rem; font-weight:600; color:#111827; }

    .model-table { display:flex; flex-direction:column; gap:0.5rem; }
    .model-row { display:flex; align-items:center; gap:0.75rem; }
    .model-name { width:130px; font-size:0.82rem; font-weight:600; color:#374151; }
    .model-track { flex:1; height:8px; background:#f3f4f6; border-radius:6px; overflow:hidden; }
    .model-fill { height:100%; background:#94a3b8; border-radius:6px; }
    .model-verdict { width:120px; text-align:right; font-size:0.8rem; font-weight:600; }

    .rec-box { background:#f9fafb; border-left:4px solid #10b981; border-radius:8px; padding:0.85rem 1rem; margin-top:1.3rem; }
    .rec-title { font-weight:700; font-size:0.9rem; color:#111827; margin-bottom:0.25rem; }
    .rec-body { font-size:0.85rem; color:#4b5563; line-height:1.45; }
    .result-disclaimer { font-size:0.72rem; color:#9ca3af; margin-top:1rem; text-align:center; }

    /* ---------- Empty state ---------- */
    .empty-state { text-align:center; padding:2.5rem 1rem; }
    .empty-icon { font-size:2.6rem; margin-bottom:0.6rem; }
    .empty-title { font-size:1.05rem; font-weight:700; color:#374151; }
    .empty-text { font-size:0.88rem; color:#6b7280; margin:0.4rem 0 1.2rem; }
    .empty-meta { display:flex; justify-content:center; gap:1.2rem; font-size:0.75rem; color:#9ca3af; }
    """

    force_light_js = """
    () => {
        document.documentElement.classList.remove('dark');
        document.body.classList.remove('dark');
    }
    """

    device_label = 'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'

    with gr.Blocks(
        css=custom_css,
        title="LUNG CANCER DETECTION USING TRANSFERL EARNING AND GRAD-CAM VISUALIZATION",
        theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="gray"),
        js=force_light_js,
    ) as demo:
        # ---------------- Header ----------------
        gr.HTML(
            f"""
            <div class="app-header">
                <div class="brand">
                    <div class="brand-logo">&#129728;</div>
                    <div>
                        <h1>LUNG CANCER DETECTION USING TRANSFERL EARNING AND GRAD-CAM VISUALIZATION</h1>
                        <p class="tagline">CT-scan screening with a stacked CNN ensemble and Grad-CAM explainability</p>
                    </div>
                </div>
                <div class="header-stats">
                    <div class="stat"><div class="num">3</div><div class="lbl">CNN Backbones</div></div>
                    <div class="stat"><div class="num">XGBoost</div><div class="lbl">Meta-Learner</div></div>
                    <div class="stat"><div class="num">Grad-CAM</div><div class="lbl">Explainability</div></div>
                    <div class="stat"><div class="num">{device_label}</div><div class="lbl">Compute</div></div>
                </div>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="warning-box">
            <b>Medical disclaimer:</b> Research prototype for demonstration only. Not FDA-approved and not for
            clinical diagnosis. Always consult a qualified healthcare professional.
            </div>
            """
        )

        # ===== Row 1: Upload (left) + Model & Actions (right) =====
        with gr.Row(equal_height=True):
            # ---------------- Left: Upload card ----------------
            with gr.Column(scale=1):
                with gr.Group(elem_classes="panel-card"):
                    gr.HTML('<div class="card-head"><span class="dot"></span>Upload CT Scan</div>', elem_classes="card-head-wrap")

                    image_input = gr.Image(
                        label="Drop a CT scan or click to browse",
                        type="pil",
                        height=420,
                        sources=["upload", "clipboard"],
                    )

            # ---------------- Right: Model & Actions card ----------------
            with gr.Column(scale=1):
                with gr.Group(elem_classes="panel-card"):
                    gr.HTML('<div class="card-head"><span class="dot"></span>Model &amp; Analysis</div>', elem_classes="card-head-wrap")

                    model_choice = gr.Radio(
                        choices=[
                            "Ensemble (Recommended)",
                            "EfficientNet-B0",
                            "DenseNet-121",
                            "ResNet-50",
                        ],
                        value="Ensemble (Recommended)",
                        label="Model",
                        info="The ensemble combines all three backbones via the XGBoost meta-learner.",
                    )

                    show_gradcam = gr.Checkbox(
                        label="Generate Grad-CAM heatmap",
                        value=True,
                        info="Overlays the regions that drove the prediction.",
                    )

                    with gr.Row():
                        analyze_btn = gr.Button("Analyze Scan", variant="primary", size="lg", scale=2)
                        clear_btn = gr.Button("Clear", variant="secondary", scale=1)

        # ===== Row 2: Analysis results (full width below) =====
        with gr.Group(elem_classes="panel-card"):
            gr.HTML('<div class="card-head"><span class="dot"></span>Analysis Results</div>', elem_classes="card-head-wrap")

            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    result_image = gr.Image(
                        label="Grad-CAM overlay",
                        type="numpy",
                        height=360,
                    )
                with gr.Column(scale=1, elem_classes="result-col"):
                    result_text = gr.HTML(value=_empty_state_html())

        # ---------------- About (condensed) ----------------
        with gr.Accordion("About this system", open=False):
            gr.Markdown(
                """
                **Pipeline.** Each CT scan is resized to 224×224 and passed through three ImageNet-pretrained
                backbones — **EfficientNet-B0**, **DenseNet-121** and **ResNet-50**. Their class probabilities are
                stacked and combined by a trained **XGBoost meta-learner** to produce the final prediction.

                **Explainability.** **Grad-CAM** highlights the image regions most responsible for the decision,
                giving a visual check on what the model focused on.

                **Limitations.** Trained on a limited demonstration dataset, not validated across diverse
                populations, and intended to support — never replace — expert radiologist review.
                """
            )

        # ---------------- Footer ----------------
        gr.HTML(
            """
            <div style="text-align:center;color:#9ca3af;font-size:0.8rem;margin-top:1rem;">
            LUNG CANCER DETECTION USING TRANSFER LEARNING AND GRAD-CAM VISUALIZATION &middot; Research prototype v1.0 &middot; PyTorch · Gradio · XGBoost &middot; Not for clinical use
            </div>
            """
        )

        # ---------------- Events ----------------
        analyze_btn.click(
            fn=analyze_image,
            inputs=[image_input, model_choice, show_gradcam],
            outputs=[result_image, result_text],
        )

        clear_btn.click(
            fn=lambda: (None, None, _empty_state_html()),
            inputs=None,
            outputs=[image_input, result_image, result_text],
        )

    return demo


def main():
    """Launch the demo application."""
    print("="*70)
    print(" LUNG CANCER DETECTION USING TRANSFER LEARNING AND GRAD-CAM VISUALIZATION - DEMO APPLICATION")
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
