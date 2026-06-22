"""
Central configuration for the Deepfake Detector.
"""

import os
import tensorflow as tf
import numpy as np
import random

# ==========================================================
# Paths
# ==========================================================

TRAIN_DIR = "data/Train"
TEST_DIR = "data/Test"

OUTPUT_DIR = "outputs"

MODEL_PATH = os.path.join(OUTPUT_DIR, "deepfake_detector.keras")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "deepfake_detector_best.keras")

# ==========================================================
# Dataset
# ==========================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32          

VAL_SPLIT = 0.10         
SEED = 42

# ==========================================================
# Training
# ==========================================================

# Phase 1: train classification head
INITIAL_EPOCHS = 10
INITIAL_LR = 1e-3

# Phase 2: fine-tune backbone
FINE_TUNE_EPOCHS = 15
FINE_TUNE_LR = 1e-5

# Number of MobileNetV3 layers to unfreeze
UNFREEZE_LAST_N_LAYERS = 50

EARLY_STOPPING_PATIENCE = 5

# ==========================================================
# Reproducibility
# ==========================================================

def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
