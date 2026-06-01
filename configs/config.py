"""
Configuration file for lung cancer detection model.
"""
import torch

# Paths
DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
SPLITS_DIR = f"{DATA_DIR}/splits"
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data parameters
IMAGE_SIZE = 224
NUM_CLASSES = 2  # Binary: Cancer vs No Cancer
BATCH_SIZE = 32
NUM_WORKERS = 4

# Training parameters
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 10  # Early stopping

# GAN parameters
GAN_ENABLED = True
GAN_LATENT_DIM = 100
GAN_EPOCHS = 100
GAN_LR = 0.0002

# Ensemble parameters
N_FOLDS = 5
ENSEMBLE_MODELS = ["efficientnet", "densenet", "resnet"]
META_LEARNER = "xgboost"  # Options: "logistic", "random_forest", "xgboost"

# Reproducibility
SEED = 42

# HU Windowing for CT scans (Lung window)
HU_WINDOW_CENTER = -600
HU_WINDOW_WIDTH = 1500

# Class weights for imbalance
CLASS_WEIGHTS = [1.0, 2.0]  # [No Cancer, Cancer]
