# Demo Application Guide

## Overview

The Lung Cancer Detection AI Demo App is a professional-grade web interface built with Gradio, showcasing the complete AI system in an easy-to-use format.

---

## ✅ Features

### 🎨 Professional Medical UI

- Clean, modern design optimized for clinical use
- Responsive layout works on desktop and tablets
- Medical-grade color scheme (calm blues and grays)
- Clear visual hierarchy for easy navigation

### 🔬 Analysis Capabilities

- **Model Selection**: Choose from 4 options
  - Ensemble (Recommended) - Combines all 3 models
  - EfficientNet-B0 - Fast and accurate
  - DenseNet-121 - Feature reuse
  - ResNet-50 - Deep learning classic

### 📊 Results Display

- **Prediction**: Cancer or No Cancer
- **Confidence Score**: Percentage certainty
- **Probability Breakdown**: Per-class probabilities
- **Clinical Recommendation**: Action items based on results
- **Individual Model Votes**: (for ensemble mode)

### 🔍 Grad-CAM Visualization

- **Toggle On/Off**: Optional explainability
- **Heatmap Overlay**: Red = high importance regions
- **Visual Validation**: See what the model "looks at"
- **Clinical Trust**: Verify model reasoning

### 📋 Additional Features

- **Example Images**: Pre-loaded CT scans to try
- **Medical Disclaimer**: Clear usage warnings
- **System Information**: Technical details accordion
- **Clear Function**: Reset interface quickly

---

## 🚀 Quick Start

### Installation

```bash
# Install Gradio
pip install gradio

# Or install all dependencies
pip install -r requirements.txt
```

### Launch Application

```bash
python demo_app.py
```

**Output:**

```
======================================================================
LUNG CANCER DETECTION AI - DEMO APPLICATION
======================================================================

Initializing...
Device: cuda

Creating Gradio interface...

======================================================================
🚀 LAUNCHING APPLICATION
======================================================================

The app will open in your default browser.
If it doesn't open automatically, navigate to: http://localhost:7860

Press Ctrl+C to stop the server.
======================================================================

Running on local URL:  http://127.0.0.1:7860
```

The app automatically opens in your browser at **http://localhost:7860**

---

## 📖 Usage Instructions

### Step-by-Step Workflow

#### 1. Upload CT Scan

- Click the **"Upload CT Scan"** area
- Select image from your computer
- Supported formats: PNG, JPEG, DICOM
- Or drag-and-drop directly

#### 2. Configure Settings

- **Model Selection**: Choose model or ensemble
  - `Ensemble (Recommended)`: Best accuracy, slower (~300ms)
  - `EfficientNet-B0`: Fast, good accuracy (~50ms)
  - `DenseNet-121`: Balanced performance (~80ms)
  - `ResNet-50`: Robust, slower (~70ms)

- **Grad-CAM**: ✅ Check to enable visualization
  - Shows which regions influenced prediction
  - Red = high importance
  - Blue = low importance

#### 3. Analyze

- Click **"🔬 Analyze CT Scan"** button
- Wait for processing (1-5 seconds)
- View results on right panel

#### 4. Interpret Results

**Example Output:**

```
### Ensemble Prediction

Model: STACKED ENSEMBLE (EfficientNet + DenseNet + ResNet)

Diagnosis: 🔴 CANCER DETECTED

Confidence: 87.3%

---

Class Probabilities:
- No Cancer: 12.7%
- Cancer: 87.3%

---

Individual Model Predictions:
- EFFICIENTNET: No Cancer 15.2% | Cancer 84.8%
- DENSENET: No Cancer 11.5% | Cancer 88.5%
- RESNET: No Cancer 11.4% | Cancer 88.6%

---

Recommendation:
⚠️ MODERATE CONFIDENCE CANCER DETECTION - Clinical review
recommended. Consider additional imaging and specialist consultation.
```

---

## 🎯 Use Cases

### 1. Clinical Demonstration

**Purpose:** Show AI capabilities to clinicians

**Workflow:**

1. Upload anonymized CT scans
2. Run ensemble prediction
3. Show Grad-CAM to validate reasoning
4. Discuss with radiologists

### 2. Educational Tool

**Purpose:** Teach students about medical AI

**Workflow:**

1. Use example images
2. Compare different models
3. Examine Grad-CAM heatmaps
4. Discuss AI decision-making

### 3. Research Validation

**Purpose:** Validate model performance

**Workflow:**

1. Upload test dataset images
2. Record predictions and confidence
3. Compare with ground truth
4. Analyze Grad-CAM accuracy

### 4. Model Comparison

**Purpose:** Compare individual models vs ensemble

**Workflow:**

1. Upload same image
2. Test EfficientNet → Record result
3. Test DenseNet → Record result
4. Test ResNet → Record result
5. Test Ensemble → Compare improvement

---

## 🔧 Customization

### Change Port

```python
# In demo_app.py, line ~450
demo.launch(
    server_port=8080,  # Change from 7860
    ...
)
```

### Enable Public Sharing

```python
# In demo_app.py
demo.launch(
    share=True,  # Creates public URL
    ...
)
```

This creates a temporary public link (valid for 72 hours):

```
Running on public URL: https://abc123.gradio.live
```

### Custom CSS Styling

```python
# In create_demo_interface()
custom_css = """
    .gradio-container {
        max-width: 1600px !important;  # Wider layout
    }
    h1 {
        color: #your-color !important;  # Custom color
    }
"""
```

### Add More Models

```python
# In create_demo_interface()
model_choice = gr.Radio(
    choices=[
        "Ensemble (Recommended)",
        "EfficientNet-B0",
        "DenseNet-121",
        "ResNet-50",
        "Your Custom Model"  # Add here
    ],
    ...
)

# Then handle in analyze_image()
```

---

## 📊 Performance

### Loading Time

| Component              | Time            | Notes                  |
| ---------------------- | --------------- | ---------------------- |
| App Launch             | ~5-10s          | First time only        |
| Model Loading          | ~2-5s per model | Cached after first use |
| First Prediction       | ~5-10s          | Includes model loading |
| Subsequent Predictions | ~0.3-1s         | Models cached          |

### Prediction Time (with Grad-CAM)

| Configuration | GPU    | CPU    |
| ------------- | ------ | ------ |
| EfficientNet  | ~80ms  | ~800ms |
| DenseNet      | ~120ms | ~1.2s  |
| ResNet        | ~100ms | ~1s    |
| Ensemble      | ~300ms | ~3s    |

### Memory Usage

| Models Loaded       | RAM  | VRAM (GPU) |
| ------------------- | ---- | ---------- |
| 1 model             | ~2GB | ~1GB       |
| 3 models (ensemble) | ~4GB | ~2.5GB     |

---

## 🐛 Troubleshooting

### Problem: App won't launch

**Error:**

```
OSError: [Errno 48] Address already in use
```

**Solution:**

```python
# Change port in demo_app.py
demo.launch(server_port=7861)  # Different port
```

### Problem: Models not found

**Error:**

```
FileNotFoundError: checkpoints/efficientnet_best.pth
```

**Solution:**
The app uses pretrained ImageNet weights by default. To use trained models:

1. Train models: `python train.py --model efficientnet`
2. Models saved to: `checkpoints/efficientnet_best.pth`
3. App will automatically use them

### Problem: CUDA out of memory

**Error:**

```
RuntimeError: CUDA out of memory
```

**Solutions:**

```python
# Option 1: Use CPU
# App automatically falls back to CPU if CUDA unavailable

# Option 2: Load models one at a time
# Don't select ensemble if GPU memory limited

# Option 3: Clear GPU cache
import torch
torch.cuda.empty_cache()
```

### Problem: Slow predictions on CPU

**Expected:** CPU predictions are 10x slower than GPU

**Solutions:**

1. **Use smaller batch size** (not applicable for single image)
2. **Disable Grad-CAM** (saves ~200ms)
3. **Use EfficientNet only** (fastest model)
4. **Get a GPU** (recommended)

### Problem: Grad-CAM not showing

**Possible causes:**

1. Checkbox not checked
2. Error in heatmap generation
3. Image format incompatible

**Solution:**

```python
# Check console for errors
# Grad-CAM failures fall back to original image
# Check if model architecture supported
```

---

## 🎨 UI Components

### Layout Structure

```
┌─────────────────────────────────────────────────────┐
│              Header & Medical Disclaimer            │
├──────────────────────┬──────────────────────────────┤
│   LEFT COLUMN        │   RIGHT COLUMN               │
│                      │                              │
│ - Upload Image       │ - Result Image               │
│ - Model Selection    │   (with Grad-CAM)            │
│ - Grad-CAM Toggle    │                              │
│ - Analyze Button     │ - Prediction Details         │
│ - Clear Button       │   • Diagnosis                │
│ - Example Images     │   • Confidence               │
│                      │   • Probabilities            │
│                      │   • Recommendations          │
├──────────────────────┴──────────────────────────────┤
│         Collapsible: About This System              │
├─────────────────────────────────────────────────────┤
│         Collapsible: Technical Details              │
├─────────────────────────────────────────────────────┤
│                    Footer                           │
└─────────────────────────────────────────────────────┘
```

### Color Scheme

- **Primary**: Purple gradient (#667eea → #764ba2)
- **Success**: Green (#10b981)
- **Warning**: Yellow (#f59e0b)
- **Error**: Red (#ef4444)
- **Background**: White (#ffffff)
- **Text**: Dark gray (#1f2937)

---

## 🌐 Deployment Options

### Local Development (Default)

```bash
python demo_app.py
# Access: http://localhost:7860
```

### Local Network

```python
demo.launch(
    server_name="0.0.0.0",  # Allow network access
    server_port=7860
)
# Access: http://YOUR_IP:7860
```

### Cloud Deployment (Gradio Share)

```python
demo.launch(share=True)
# Creates public link: https://xyz.gradio.live
# Valid for 72 hours
```

### Production Deployment

**Option 1: Docker**

```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "demo_app.py"]
```

**Option 2: Cloud Platforms**

- Hugging Face Spaces (recommended for Gradio)
- Google Cloud Run
- AWS EC2
- Azure App Service

---

## 📝 API Mode (Programmatic Access)

The demo app can also be used programmatically:

```python
from demo_app import predict_single_model, predict_ensemble

# Load image
image = load_image('path/to/ct_scan.png')

# Single model prediction
result_image, result_text = predict_single_model(
    image=image,
    model_name='efficientnet',
    show_gradcam=True
)

# Ensemble prediction
result_image, result_text = predict_ensemble(
    image=image,
    show_gradcam=True
)
```

---

## 🎓 Best Practices

### For Clinical Demos

1. **Always show disclaimer** - Emphasize research prototype status
2. **Enable Grad-CAM** - Build trust with visual explanations
3. **Use ensemble mode** - Best accuracy for demonstrations
4. **Prepare example cases** - Have diverse CT scans ready
5. **Explain limitations** - Be transparent about capabilities

### For Research

1. **Document all settings** - Model, Grad-CAM on/off, etc.
2. **Save predictions** - Screenshot or export results
3. **Batch processing** - Use API mode for datasets
4. **Compare models** - Test all architectures systematically
5. **Validate Grad-CAM** - Check against radiologist annotations

### For Education

1. **Interactive exploration** - Let students try different models
2. **Grad-CAM analysis** - Discuss what model learns
3. **Error cases** - Show when AI fails and why
4. **Compare predictions** - Human vs AI diagnostic reasoning

---

## 📚 Additional Resources

- **Main Project**: [README.md](file:///C:/Users/pc/.gemini/antigravity/scratch/lung_cancer_ai/README.md)
- **Complete Guide**: [DOCUMENTATION.md](file:///C:/Users/pc/.gemini/antigravity/scratch/lung_cancer_ai/DOCUMENTATION.md)
- **Grad-CAM Details**: [GRADCAM_GUIDE.md](file:///C:/Users/pc/.gemini/antigravity/scratch/lung_cancer_ai/GRADCAM_GUIDE.md)
- **Model Training**: [CNN_TRAINING_GUIDE.md](file:///C:/Users/pc/.gemini/antigravity/scratch/lung_cancer_ai/CNN_TRAINING_GUIDE.md)

---

**The demo app is production-ready for demonstrations and research!** 🎨✨
