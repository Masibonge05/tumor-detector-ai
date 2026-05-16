# Data preprocessing - loads, crops, resizes, normalizes and augments MRI images

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.utils import config as cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

# Let TensorFlow decide the best number of parallel calls automatically
AUTOTUNE = tf.data.AUTOTUNE


def crop_mri_margin(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    # Remove the black border around the MRI scan to focus on brain tissue
    gray = cv2.cvtColor(image,
                        cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    # Find the largest contour which represents the brain area
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # If the detected region is too small, return the original image
    if w < 20 or h < 20:
        return image
    return image[y:y + h, x:x + w]


def load_and_preprocess_image(
    path_or_array,
    img_size: Tuple[int, int] = cfg.IMG_SIZE,
    crop_margin: bool = True,
) -> np.ndarray:
    # Load an image from disk or accept a numpy array, then prepare it for the model
    if isinstance(path_or_array, (str, Path)):
        img = cv2.imread(str(path_or_array))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path_or_array}")
        # Convert from BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.asarray(path_or_array)
        # Handle grayscale images
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # Handle images with an alpha channel
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    if crop_margin:
        img = crop_mri_margin(img)

    # Resize to the target input size and normalize pixel values to [0, 1]
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0

    # Add batch dimension so shape becomes (1, H, W, 3)
    return np.expand_dims(img, axis=0)


def _build_augmentation() -> tf.keras.Sequential:
    # Define random augmentations applied only during training
    # This helps the model generalize by seeing slightly different versions of each image
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


def build_datasets(
    train_dir: Path = cfg.TRAIN_DIR,
    test_dir: Path = cfg.TEST_DIR,
    img_size: Tuple[int, int] = cfg.IMG_SIZE,
    batch_size: int = cfg.BATCH_SIZE,
    val_split: float = cfg.VAL_SPLIT,
    seed: int = cfg.SEED,
    class_names: list = None,
):
    log.info("Loading datasets from %s", train_dir)
    class_names = class_names or cfg.CLASS_NAMES

    # Load training images (80% of the training folder)
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

    # Load validation images (20% of the training folder)
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

    # Load test images from the separate test folder
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
        class_names=class_names,
    )

    class_names = train_ds.class_names
    log.info("Classes found: %s", class_names)

    # Rescaling layer to normalize pixel values from [0, 255] to [0, 1]
    rescale = layers.Rescaling(1.0 / 255.0)
    augment = _build_augmentation()

    # Apply augmentation and normalization to training data
    train_ds = train_ds.map(
        lambda x, y: (augment(rescale(x), training=True), y),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(buffer_size=AUTOTUNE)

    # Only normalize validation and test data (no augmentation)
    val_ds = val_ds.map(
        lambda x, y: (rescale(x), y),
        num_parallel_calls=AUTOTUNE,
    ).cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = test_ds.map(
        lambda x, y: (rescale(x), y),
        num_parallel_calls=AUTOTUNE,
    ).cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
