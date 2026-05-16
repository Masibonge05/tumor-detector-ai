from __future__ import annotations

# Prediction and Grad-CAM utilities for inference and explainability.
from pathlib import Path
from typing import Dict, List, Tuple
import json

import cv2
import numpy as np
import tensorflow as tf

from src.models.cnn import LAST_CONV_LAYER_NAME
from src.preprocessing.preprocess import load_and_preprocess_image
from src.utils import config as cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

# Cached model instance to avoid repeated disk loads during app or CLI inference.
_MODEL: tf.keras.Model | None = None


def _load_saved_class_names() -> List[str]:
    """Load saved class order metadata from disk if available."""
    if cfg.CLASS_NAMES_PATH.exists():
        try:
            class_names = json.loads(cfg.CLASS_NAMES_PATH.read_text())
            if isinstance(class_names, list) and all(
                    isinstance(x, str) for x in class_names):
                log.debug(f"Loaded saved class names: {class_names}")
                if class_names != cfg.CLASS_NAMES:
                    log.warning(
                        "Saved class order differs from current cfg.CLASS_NAMES"
                    )
                    log.warning(f"  cfg.CLASS_NAMES: {cfg.CLASS_NAMES}")
                    log.warning(f"  saved order:      {class_names}")
                return class_names
            log.warning(
                "Saved class names metadata exists but has invalid format")
        except json.JSONDecodeError as exc:
            log.warning("Failed to parse saved class names metadata: %s", exc)
    log.debug("Using cfg.CLASS_NAMES as fallback for class order")
    return cfg.CLASS_NAMES


# Lazy model loader; loads the best or final trained model on first use.
def get_model() -> tf.keras.Model:
    """Lazy-load and cache the trained model."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    path = cfg.BEST_MODEL_PATH if cfg.BEST_MODEL_PATH.exists(
    ) else cfg.FINAL_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            "No trained model available. Run `python main.py --mode train` first."
        )
    log.info("Loading trained model from %s", path)
    _MODEL = tf.keras.models.load_model(path)
    return _MODEL


def predict_image(image_or_path) -> Dict:
    """
    Predict the tumor class of an MRI image.

    Returns a dict containing:
        predicted_class : str          – e.g. "glioma"
        display_name    : str          – e.g. "Glioma Tumor"
        confidence      : float        – probability of the predicted class (0..1)
        probabilities   : Dict[str,float] – all 4 class probabilities

    DEBUG LOGGING:
    Logs raw prediction probabilities, predicted class index, and mapped class
    label to ensure consistency across the prediction pipeline.
    """
    model = get_model()
    x = load_and_preprocess_image(image_or_path)
    probs = model.predict(x, verbose=0)[0]

    class_names = _load_saved_class_names()
    idx = int(np.argmax(probs))
    pred_name = class_names[idx]

    log.debug(f"Raw model output (probabilities): {probs}")
    log.debug(f"Predicted class index: {idx}")
    log.debug(f"Class name at index {idx}: {pred_name}")
    log.info(
        f"Prediction: class_idx={idx}, class_name='{pred_name}', "
        f"confidence={probs[idx]:.4f}, probs={dict(zip(class_names, probs))}")

    return {
        "predicted_class": pred_name,
        "display_name": cfg.CLASS_DISPLAY[pred_name],
        "confidence": float(probs[idx]),
        "probabilities": {
            class_names[i]: float(p)
            for i, p in enumerate(probs)
        },
    }


def _make_gradcam_heatmap(
    img_tensor: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str = LAST_CONV_LAYER_NAME,
    pred_index: int | None = None,
) -> np.ndarray:
    """
    Compute the Grad-CAM heatmap for `img_tensor` (shape (1, H, W, 3)).

    DEBUG LOGGING:
    Logs the predicted index used for Grad-CAM computation and the gradient
    statistics to ensure gradients are flowing correctly for the target class.
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor, training=False)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        else:
            pred_index = tf.cast(pred_index, tf.int64)
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)

    # DEBUG: Log gradient statistics
    if grads is not None:
        log.debug(f"Grad-CAM gradients shape: {grads.shape}")
        log.debug(
            f"Gradient statistics - mean: {tf.reduce_mean(grads):.6f}, "
            f"max: {tf.reduce_max(grads):.6f}, min: {tf.reduce_min(grads):.6f}"
        )
    else:
        log.warning(f"Grad-CAM gradients are None! pred_index={pred_index}")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


# Compute a Grad-CAM overlay image alongside the prediction details.
def gradcam_overlay(
    image_or_path,
    alpha: float = 0.4,
) -> Tuple[np.ndarray, Dict]:
    """
    Returns (overlay_rgb_uint8, prediction_dict).
    The overlay is the original (cropped+resized) MRI with a JET colormap
    Grad-CAM heatmap blended on top.

    DEBUG LOGGING:
    Logs the Grad-CAM computation class index to verify it matches the
    predicted class from the model.
    """
    model = get_model()
    x = load_and_preprocess_image(image_or_path)
    probs = model.predict(x, verbose=0)[0]
    class_names = _load_saved_class_names()
    idx = int(np.argmax(probs))

    log.debug(
        f"Grad-CAM computing for class index: {idx} ('{class_names[idx]}')")
    log.debug(f"Grad-CAM class probabilities: {dict(zip(class_names, probs))}")

    # Generate Grad-CAM heatmap and blend it over the input image.
    heatmap = _make_gradcam_heatmap(x, model, pred_index=idx)
    heatmap = cv2.resize(heatmap, cfg.IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    base = (x[0] * 255).astype("uint8")
    overlay = cv2.addWeighted(base, 1 - alpha, colored, alpha, 0)

    pred_name = class_names[idx]
    info = {
        "predicted_class": pred_name,
        "display_name": cfg.CLASS_DISPLAY[pred_name],
        "confidence": float(probs[idx]),
        "probabilities": {
            class_names[i]: float(p)
            for i, p in enumerate(probs)
        },
    }
    return overlay, info
