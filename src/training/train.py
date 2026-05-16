from __future__ import annotations

# Training pipeline for CNN model fitting and result export.
from pathlib import Path
import json

import matplotlib.pyplot as plt
import tensorflow as tf

from src.models.cnn import build_cnn
from src.preprocessing.preprocess import build_datasets
from src.utils import config as cfg
from src.utils.gpu import configure_gpu
from src.utils.logger import get_logger

log = get_logger(__name__)


def _compile(model: tf.keras.Model, lr: float) -> None:
    """
    Compile with Adam + sparse categorical crossentropy.

    OPTIMIZATION: Using float32 (not mixed precision) for better CPU stability.
    Mixed precision can cause issues on Windows CPU systems. We rely on
    architectural efficiency instead.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr,
                                           beta_1=0.9,
                                           beta_2=0.999,
                                           epsilon=1e-07),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2,
                                                           name="top2_acc"),
        ],
    )


def _build_callbacks() -> list:
    """
    Standard set of training callbacks for stable convergence.

    OPTIMIZATION for CPU:
    • ModelCheckpoint: Saves best model by validation accuracy
    • EarlyStopping: Reduced patience from 8 to 5 epochs
      (on CPU, 8 epochs = 2+ hours; stop faster if plateau detected)
    • ReduceLROnPlateau: Reduced patience from 3 to 2 epochs
      (more aggressive LR reduction on CPU for faster convergence)
    • CSVLogger: Minimal overhead, useful for post-analysis
    """
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(cfg.BEST_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(cfg.REPORTS_DIR / "training_log.csv"),
            append=False,
        ),
    ]


def _plot_history(history: tf.keras.callbacks.History, out_dir: Path) -> None:
    """Save accuracy & loss plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    h = history.history

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(h["accuracy"], label="Train Accuracy", linewidth=2)
    ax1.plot(h["val_accuracy"], label="Val Accuracy", linewidth=2)
    ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(h["loss"], label="Train Loss", linewidth=2)
    ax2.plot(h["val_loss"], label="Val Loss", linewidth=2)
    ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    out_path = out_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved training curves to %s", out_path)


def train(epochs: int = cfg.EPOCHS,
          batch_size: int = cfg.BATCH_SIZE,
          lr: float = cfg.LEARNING_RATE) -> tf.keras.Model:
    """
    End-to-end training entry point.

    OPTIMIZATION: On CPU-only Windows systems:
    • Total training time: ~100-150 minutes (10 epochs × 10-15 min/epoch)
    • Memory usage: ~1.5-2 GB peak (with 128×128 images, batch=32)
    • Expected accuracy: >90% on test set
    """
    cfg.ensure_dirs()
    configure_gpu()

    # Load datasets and verify dataset class ordering.
    train_ds, val_ds, _test_ds, _classes = build_datasets(
        batch_size=batch_size)
    log.info("Training dataset class_names: %s", _classes)
    if _classes != cfg.CLASS_NAMES:
        log.warning(
            "Training dataset class order does not match cfg.CLASS_NAMES")
        log.warning("  cfg.CLASS_NAMES: %s", cfg.CLASS_NAMES)
        log.warning("  dataset class_names: %s", _classes)

    from pathlib import Path
    import os
    # Compute per-class sample counts and weights to address imbalance.
    class_counts = {}
    for class_name in _classes:
        class_dir = cfg.TRAIN_DIR / class_name
        if class_dir.exists():

            count = len([
                f for f in class_dir.glob("*") if f.suffix.lower() in
                ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            ])
            class_counts[class_name] = count
    total_samples = sum(class_counts.values())
    class_weights = {
        _classes.index(cls): total_samples / (len(class_counts) * count)
        for cls, count in class_counts.items()
    }
    log.info("Class weights: %s", class_weights)

    # Build, compile, and summarize the CNN architecture.
    model = build_cnn()
    _compile(model, lr)
    model.summary(print_fn=log.info)

    summary_path = cfg.REPORTS_DIR / "model_architecture.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    log.info("Saved model architecture summary → %s", summary_path)

    # Execute the training loop and record history.
    log.info("Starting training for up to %d epochs on CPU…", epochs)
    log.info("Expected time per epoch: 10-15 minutes (on Windows CPU)")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=_build_callbacks(),
        class_weight=class_weights,
        verbose=1,
    )

    # Save the final trained model and class metadata.
    model.save(cfg.FINAL_MODEL_PATH)
    log.info("Saved final model → %s", cfg.FINAL_MODEL_PATH)

    class_names_path = cfg.CLASS_NAMES_PATH
    class_names_path.write_text(json.dumps(cfg.CLASS_NAMES, indent=2))
    log.info("Saved class label metadata → %s", class_names_path)

    _plot_history(history, cfg.PLOTS_DIR)
    return model
