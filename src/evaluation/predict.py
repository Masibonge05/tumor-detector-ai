# Prediction and Grad-CAM utilities for running inference on MRI images

import json
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf

from src.models.cnn import LAST_CONV_LAYER_NAME
from src.preprocessing.preprocess import load_and_preprocess_image
from src.utils import config as cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

# Global variable to store the loaded model so we don't reload it every time
_MODEL: tf.keras.Model | None = None


def _load_saved_class_names() -> List[str]:
    # Load the class names that were saved during training
    if cfg.CLASS_NAMES_PATH.exists():
        try:
            class_names = json.loads(cfg.CLASS_NAMES_PATH.read_text())
            if isinstance(class_names, list) and all(isinstance(x, str) for x in class_names):
                return class_names
        except json.JSONDecodeError:
            pass
    # Fall back to the class names defined in config
    return cfg.CLASS_NAMES


def get_model() -> tf.keras.Model:
    # Load the trained model from disk only once, then reuse it
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    path = cfg.BEST_MODEL_PATH if cfg.BEST_MODEL_PATH.exists() else cfg.FINAL_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            "No trained model found. Run `python main.py --mode train` first."
        )
    log.info("Loading model from %s", path)
    _MODEL = tf.keras.models.load_model(path)
    return _MODEL


def predict_image(image_or_path) -> Dict:
    # Run the model on an image and return the predicted class and probabilities
    model = get_model()
    x = load_and_preprocess_image(image_or_path)
    probs = model.predict(x, verbose=0)[0]

    class_names = _load_saved_class_names()
    idx = int(np.argmax(probs))
    pred_name = class_names[idx]

    return {
        "predicted_class": pred_name,
        "display_name": cfg.CLASS_DISPLAY[pred_name],
        "confidence": float(probs[idx]),
        "probabilities": {class_names[i]: float(p) for i, p in enumerate(probs)},
    }


def _make_gradcam_heatmap(
    img_tensor: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str = LAST_CONV_LAYER_NAME,
    pred_index: int | None = None,
) -> np.ndarray:
    # Build a sub-model that outputs both the last conv layer and the predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    # Record gradients of the predicted class with respect to the conv layer output
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor, training=False)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Compute gradients and pool them across the spatial dimensions
    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv output by the pooled gradients to get the heatmap
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap to range [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def gradcam_overlay(image_or_path, alpha: float = 0.4) -> Tuple[np.ndarray, Dict]:
    # Generate a Grad-CAM heatmap overlaid on the input MRI image
    model = get_model()
    x = load_and_preprocess_image(image_or_path)
    probs = model.predict(x, verbose=0)[0]
    class_names = _load_saved_class_names()
    idx = int(np.argmax(probs))

    # Generate the heatmap for the predicted class
    heatmap = _make_gradcam_heatmap(x, model, pred_index=idx)

    # Resize heatmap to match the input image size
    heatmap = cv2.resize(heatmap, cfg.IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Apply a colour map to make the heatmap visible
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # Blend the heatmap with the original image
    base = (x[0] * 255).astype("uint8")
    overlay = cv2.addWeighted(base, 1 - alpha, colored, alpha, 0)

    pred_name = class_names[idx]
    info = {
        "predicted_class": pred_name,
        "display_name": cfg.CLASS_DISPLAY[pred_name],
        "confidence": float(probs[idx]),
        "probabilities": {class_names[i]: float(p) for i, p in enumerate(probs)},
    }
    return overlay, info