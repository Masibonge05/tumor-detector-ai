"""
Central configuration for the Brain Tumor CNN project.

Keeping all hyperparameters and paths in one place makes the system easier
to maintain and reproduce — a key requirement for research-grade ML code.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
CM_DIR = OUTPUTS_DIR / "confusion_matrix"
PRED_DIR = OUTPUTS_DIR / "predictions"

MODELS_DIR = PROJECT_ROOT / "saved_models"
BEST_MODEL_PATH = MODELS_DIR / "brain_tumor_cnn_best.keras"
FINAL_MODEL_PATH = MODELS_DIR / "brain_tumor_cnn_final.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

REPORTS_DIR = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
# NOTE: Order MUST match the alphabetical order produced by
# tf.keras.utils.image_dataset_from_directory, which is what we use.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# Human-readable names for the dashboard
CLASS_DISPLAY = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

# ---------------------------------------------------------------------------
# Image / training hyperparameters
# ---------------------------------------------------------------------------
# Image / training hyperparameters - increased for better accuracy
# OPTIMIZATION: Increased from 128×128 to 224×224 for richer spatial features.
# Despite higher computational cost, provides better diagnostic accuracy.
IMG_SIZE = (224, 224)  # H, W — increased for better accuracy
IMG_CHANNELS = 3
BATCH_SIZE = 16  # Reduced from 32 to fit larger images in memory
# OPTIMIZATION: Increased from 10 to 50 epochs with early stopping.
# Allows deeper learning with larger model capacity.
EPOCHS = 50
# OPTIMIZATION: Adjusted to 1e-3 for better convergence with larger model.
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15  # fraction of train used for validation
SEED = 42

# Data augmentation strength - increased for better generalization
# OPTIMIZATION: Increased aggressiveness to improve robustness to variations
# in MRI positioning and intensity, critical for medical imaging.
AUG_ROTATION = 0.15  # Increased from 0.08 (±15% of 2π)
AUG_ZOOM = 0.15  # Increased from 0.08
AUG_BRIGHTNESS = 0.15  # Increased from 0.08
AUG_CONTRAST = 0.15  # Increased from 0.08

# Regularization - adjusted for larger model
L2_REG = 5e-5  # Reduced from 1e-4 for larger model
DROPOUT_BLOCK = 0.3  # Increased from 0.25
DROPOUT_HEAD = 0.6  # Increased from 0.50


# Directory creation helper for output artifacts used by training and evaluation.
def ensure_dirs() -> None:
    """Create all output directories if they don't exist yet."""
    for d in (OUTPUTS_DIR, PLOTS_DIR, CM_DIR, PRED_DIR, MODELS_DIR,
              REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
