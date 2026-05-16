"""GPU detection / configuration helpers."""

import tensorflow as tf
from .logger import get_logger

log = get_logger(__name__)


# GPU detection and memory growth helper used before training.
def configure_gpu() -> bool:
    """
    Detect available GPUs and enable memory growth so TF doesn't allocate
    the entire device up-front. Returns True if a GPU is available.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        log.info("No GPU detected — training will run on CPU.")
        return False

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:  # already initialized
            log.warning("Could not set memory growth: %s", exc)

    log.info("Detected %d GPU device(s): %s", len(gpus),
             [g.name for g in gpus])
    return True
