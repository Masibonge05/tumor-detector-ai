# GPU configuration - detects available GPU and sets up memory usage

import tensorflow as tf
from .logger import get_logger

log = get_logger(__name__)


def configure_gpu() -> bool:
    # Check if a GPU is available
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        log.info("No GPU detected - training will run on CPU.")
        return False

    # Enable memory growth so TensorFlow only uses as much GPU memory as needed
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            log.warning("Could not set memory growth: %s", exc)

    log.info("Detected %d GPU(s): %s", len(gpus), [g.name for g in gpus])
    return True