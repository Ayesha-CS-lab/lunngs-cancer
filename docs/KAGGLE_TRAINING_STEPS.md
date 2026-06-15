# Kaggle Training Steps — Lung Cancer AI

Complete step-by-step guide for training on Kaggle.
Update this file whenever something changes.

---

## Setup (Do Once Per Notebook Session)

### Step 1 — Copy code to working directory
```python
import os, shutil, sys

src = "/kaggle/input/datasets/ayysha/lung-cancer-code"
dst = "/kaggle/working"

for item in os.listdir(src):
    if item == ".venv":
        continue
    s = os.path.join(src, item)
    d = os.path.join(dst, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

os.chdir("/kaggle/working")
print("Done:", os.listdir("."))
```

### Step 2 — Install packages
```python
# timm is pre-installed on Kaggle (used by EfficientNet); no install needed for it
!pip install -q albumentations xgboost joblib
```

---

## Dataset 1: IQ-OTH/NCCD

> Add dataset from right panel → Add Input → search "iqothnccd" → add adityamahimkar's dataset

### Step 3 — Prepare IQ-OTH/NCCD dataset

> NOTE: Folder names in this dataset are "Normal cases", "Bengin cases" (typo), "Malignant cases"
> The prepare_kaggle_dataset.py script does NOT handle these names — use this direct code instead.

```python
import os, shutil, random
from tqdm import tqdm

iq_path = "/kaggle/input/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset"
base = "data/kaggle_lung_cancer"

for split in ["train", "val", "test"]:
    for cls in ["no_cancer", "cancer"]:
        os.makedirs(f"{base}/{split}/{cls}", exist_ok=True)

all_images = []
for folder, label in [("Normal cases", "no_cancer"),
                       ("Bengin cases", "no_cancer"),
                       ("Malignant cases", "cancer")]:
    folder_path = os.path.join(iq_path, folder)
    if os.path.exists(folder_path):
        imgs = [f for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"{folder}: {len(imgs)} images → {label}")
        for img in imgs:
            all_images.append((os.path.join(folder_path, img), label, folder.split()[0] + "_" + img))

random.seed(42)
cancer    = [(p, l, n) for p, l, n in all_images if l == "cancer"]
no_cancer = [(p, l, n) for p, l, n in all_images if l == "no_cancer"]

def split_and_copy(items, label):
    random.shuffle(items)
    n = len(items)
    t = int(n * 0.70)
    v = int(n * 0.15)
    splits = {"train": items[:t], "val": items[t:t+v], "test": items[t+v:]}
    for split_name, split_items in splits.items():
        for src, _, name in tqdm(split_items, desc=f"{split_name}/{label}"):
            shutil.copy2(src, f"{base}/{split_name}/{label}/{name}")

split_and_copy(cancer,    "cancer")
split_and_copy(no_cancer, "no_cancer")

# Expected output:
# train: cancer=392, no_cancer=375  (total 767)
# val:   cancer=84,  no_cancer=80   (total 164)
# test:  cancer=85,  no_cancer=81   (total 166)
```

### Step 4 — GAN augmentation (~25 min)

> Trains GAN on real CT data, generates synthetic images for minority class,
> saves them into data/kaggle_lung_cancer/train/no_cancer/

```python
!python run_gan_augmentation.py --epochs 150 --batch-size 32
```

### Step 5 — Train EfficientNet-B0 (~35 min)
```python
!python train.py --model efficientnet --epochs 50 --data-dir data/kaggle_lung_cancer
```

### Step 6 — Train DenseNet-121 (~40 min)
```python
!python train.py --model densenet --epochs 50 --data-dir data/kaggle_lung_cancer
```

### Step 7 — Train ResNet-50 (~40 min)
```python
!python train.py --model resnet --epochs 50 --data-dir data/kaggle_lung_cancer
```

### Step 8 — Verify checkpoints saved
```python
import os
for f in ["efficientnet_finetuned_best.pth",
          "densenet_finetuned_best.pth",
          "resnet_finetuned_best.pth"]:
    status = "✓" if os.path.exists(f"checkpoints/{f}") else "✗ MISSING"
    print(f"{status}  {f}")
```

### Step 9 — Train ensemble (~5 min)

> Meta-learner trains on VALIDATION set predictions (not training set).
> This is the fix for the previous 100% data leakage bug.

```python
!python train_ensemble.py
```

### Step 10 — Generate Grad-CAM heatmaps
```python
!python generate_gradcam.py
```

---

## Dataset 2: LIDC-IDRI

> Add LIDC-IDRI dataset from right panel → Add Input → search "lidc idri"

### Step 11 — Find LIDC-IDRI path
```python
import os
for root, dirs, files in os.walk("/kaggle/input"):
    level = root.replace("/kaggle/input", "").count(os.sep)
    if level > 4:
        continue
    print("  " * level + os.path.basename(root) + "/")
```

### Step 12 — Prepare LIDC-IDRI dataset

> Update --source-dir with the actual path found in Step 11

```python
!python prepare_lidc_dataset.py --source-dir /kaggle/input/datasets/<lidc-path-here>
```

### Step 13 — GAN augmentation for LIDC-IDRI
```python
!python run_gan_augmentation.py --data-dir data/lidc_idri/train --epochs 150 --batch-size 32
```

### Step 14 — Train all 3 models on LIDC-IDRI
```python
!python train.py --model efficientnet --epochs 50 \
    --data-dir data/lidc_idri \
    --checkpoint-dir checkpoints/lidc_idri \
    --output-dir outputs/lidc_idri

!python train.py --model densenet --epochs 50 \
    --data-dir data/lidc_idri \
    --checkpoint-dir checkpoints/lidc_idri \
    --output-dir outputs/lidc_idri

!python train.py --model resnet --epochs 50 \
    --data-dir data/lidc_idri \
    --checkpoint-dir checkpoints/lidc_idri \
    --output-dir outputs/lidc_idri
```

### Step 15 — Train ensemble on LIDC-IDRI
```python
!python train_ensemble.py \
    --val-dir  data/lidc_idri/val \
    --test-dir data/lidc_idri/test \
    --checkpoint-dir checkpoints/lidc_idri \
    --output-dir outputs/lidc_idri
```

---

## Final Steps (Both Datasets Done)

### Step 16 — Compare both datasets
```python
!python compare_datasets.py
```

### Step 17 — Save all outputs
```python
import shutil
shutil.make_archive("/kaggle/working/all_outputs",     "zip", "outputs")
shutil.make_archive("/kaggle/working/all_checkpoints", "zip", "checkpoints")
print("Download these from the Output panel on the right.")
```

---

## Expected Results

| Dataset     | Expected Accuracy | Notes |
|-------------|------------------|-------|
| IQ-OTH/NCCD | 92–97%          | Clean structured dataset |
| LIDC-IDRI   | 85–93%          | Larger, more diverse, harder |

> The gap between datasets is a valid research finding — it shows
> generalisation challenge across different data sources.

---

## Known Issues & Fixes Applied

| Issue | Fix |
|-------|-----|
| 100% accuracy (data leakage) | Meta-learner now trains on val set, not training set |
| GAN not integrated | `run_gan_augmentation.py` trains on real CT data |
| IQ-OTH/NCCD folder names have spaces | Use direct copy code in Step 3, not prepare_kaggle_dataset.py |
| Kaggle zip rejects backslashes | Use `git archive` to create zip on Windows |
| .vscode malware injection | All .vscode files removed, blocked in .gitignore |
| `efficientnet_pytorch` not on Kaggle | Switched to `timm` (pre-installed); `base_models.py` updated |

---

## Time Estimate (T4 GPU)

| Step | Time |
|------|------|
| GAN training ×2 datasets | ~50 min |
| 3 models × IQ-OTH/NCCD  | ~2 hrs  |
| 3 models × LIDC-IDRI     | ~2.5 hrs |
| Ensemble + compare       | ~15 min  |
| **Total**                | **~5–6 hrs** |
