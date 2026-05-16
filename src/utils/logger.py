"""Lightweight colorized logger used throughout the project."""

import logging
import sys


# Configure and return a reusable console logger for the project.
def get_logger(name: str = "brain_tumor_cnn") -> logging.Logger:
    """Return a configured logger. Safe to call multiple times."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
