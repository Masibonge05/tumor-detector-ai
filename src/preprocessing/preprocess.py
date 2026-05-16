from __future__ import annotations

# Data preprocessing utilities for MRI image loading, normalization, and augmentation.
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.utils import config as cfg
from src.utils.logger import get_logger

log = get_logger(__name__)
# TensorFlow autotune and CPU parallelism settings for dataset pipelines.
AUTOTUNE = tf.data.AUTOTUNE

NUM_PARALLEL_CALLS_CPU = 2


def crop_mri_margin(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Remove the black border around an MRI scan using a simple threshold +
    largest-contour bounding box. Improves the effective resolution the
    CNN sees because more pixels actually contain brain tissue.

    Falls back to the original image if no significant contour is found.
    """
    gray = cv2.cvtColor(image,
                        cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w < 20 or h < 20:
        return image
    return image[y:y + h, x:x + w]


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range. Extracted utility function to
    reduce code duplication across load_and_preprocess_image and
    build_datasets pipeline. Improves maintainability.
    """
    return image.astype("float32") / 255.0


def _resize_image(image: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size using INTER_AREA for downsampling.
    Extracted utility to centralize resize logic.
    """
    return cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)


def load_and_preprocess_image(
    path_or_array,
    img_size: Tuple[int, int] = cfg.IMG_SIZE,
    crop_margin: bool = True,
) -> np.ndarray:
    """
    Load an MRI from disk (or accept a numpy/PIL array) and return a
    float32 tensor of shape (1, H, W, 3) ready for `model.predict`.
    """
    if isinstance(path_or_array, (str, Path)):
        img = cv2.imread(str(path_or_array))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path_or_array}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.asarray(path_or_array)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    if crop_margin:
        img = crop_mri_margin(img)

    img = _resize_image(img, img_size)
    img = _normalize_image(img)
    return np.expand_dims(img, axis=0)


def _build_augmentation() -> tf.keras.Sequential:
    """
    Augmentation pipeline applied ONLY to the training split.

    OPTIMIZATION: Using TensorFlow's Keras augmentation layers (not numpy-based)
    ensures operations run on GPU if available, or are JIT-compiled on CPU.
    These are more efficient than stateless augmentation calls.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(cfg.AUG_ROTATION),
            layers.RandomZoom(cfg.AUG_ZOOM),
            layers.RandomBrightness(cfg.AUG_BRIGHTNESS,
                                    value_range=(0.0, 1.0)),
            layers.RandomContrast(cfg.AUG_CONTRAST),
        ],
        name="data_augmentation",
    )


def _apply_normalization_and_augmentation(images,
                                          labels,
                                          augment_fn,
                                          rescale_fn,
                                          training: bool = False):
    """
    Shared normalization + augmentation logic extracted to reduce
    code duplication. Supports both training and inference modes.

    OPTIMIZATION: Combines rescaling + augmentation in a single map
    to avoid intermediate tensor copies. Critical for CPU memory efficiency.
    """
    images = rescale_fn(images)
    if training:
        images = augment_fn(images, training=True)
    return images, labels


def build_datasets(
    train_dir: Path = cfg.TRAIN_DIR,
    test_dir: Path = cfg.TEST_DIR,
    img_size: Tuple[int, int] = cfg.IMG_SIZE,
    batch_size: int = cfg.BATCH_SIZE,
    val_split: float = cfg.VAL_SPLIT,
    seed: int = cfg.SEED,
    class_names: list[str] | None = None,
):

    log.info("Loading datasets from %s", train_dir)
    class_names = class_names or cfg.CLASS_NAMES

    # Create training, validation, and test datasets from directory structure.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        class_names=class_names,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        class_names=class_names,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
        class_names=class_names,
    )

    class_names = train_ds.class_names
    log.info("Detected classes: %s", class_names)

    rescale = layers.Rescaling(1.0 / 255.0)
    augment = _build_augmentation()

    # Apply normalization and augmentation to the training dataset.
    train_ds = train_ds.map(
        lambda x, y: _apply_normalization_and_augmentation(
            x, y, augment, rescale, training=True),
        num_parallel_calls=NUM_PARALLEL_CALLS_CPU,
    ).prefetch(buffer_size=AUTOTUNE)

    val_ds = val_ds.map(
        lambda x, y: _apply_normalization_and_augmentation(
            x, y, augment, rescale, training=False),
        num_parallel_calls=NUM_PARALLEL_CALLS_CPU,
    ).cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = test_ds.map(
        lambda x, y: _apply_normalization_and_augmentation(
            x, y, augment, rescale, training=False),
        num_parallel_calls=NUM_PARALLEL_CALLS_CPU,
    ).cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
