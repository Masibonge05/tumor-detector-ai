# Central configuration file - all paths and hyperparameters are defined here

from pathlib import Path

# Project folder structure
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR  = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR   = OUTPUTS_DIR / "plots"
CM_DIR      = OUTPUTS_DIR / "confusion_matrix"
PRED_DIR    = OUTPUTS_DIR / "predictions"

MODELS_DIR       = PROJECT_ROOT / "saved_models"
BEST_MODEL_PATH  = MODELS_DIR / "brain_tumor_cnn_best.keras"
FINAL_MODEL_PATH = MODELS_DIR / "brain_tumor_cnn_final.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

REPORTS_DIR = PROJECT_ROOT / "reports"

# Class names must match the alphabetical order used by image_dataset_from_directory
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# Display names shown in the Streamlit app
CLASS_DISPLAY = {
    "glioma":     "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor":    "No Tumor",
    "pituitary":  "Pituitary Tumor",
}

# Image settings
IMG_SIZE     = (224, 224)  # height and width the model expects
IMG_CHANNELS = 3           # RGB images

# Training hyperparameters
BATCH_SIZE    = 16    # number of images processed per step
EPOCHS        = 50    # max training epochs (early stopping may end sooner)
LEARNING_RATE = 1e-3  # Adam optimizer step size
VAL_SPLIT     = 0.15  # 15% of training data used for validation
SEED          = 42    # for reproducibility

# Data augmentation applied only during training
AUG_ROTATION   = 0.15
AUG_ZOOM       = 0.15
AUG_BRIGHTNESS = 0.15
AUG_CONTRAST   = 0.15

# Regularization to reduce overfitting
L2_REG        = 5e-5
DROPOUT_BLOCK = 0.3
DROPOUT_HEAD  = 0.6


def ensure_dirs() -> None:
    # Create all output folders if they do not already exist
    for d in (OUTPUTS_DIR, PLOTS_DIR, CM_DIR, PRED_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)