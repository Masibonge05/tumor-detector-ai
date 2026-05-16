# Training pipeline for the brain tumor CNN model

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
    # Compile the model using Adam optimizer and cross-entropy loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def _build_callbacks() -> list:
    # Callbacks control what happens during training at the end of each epoch
    return [
        # Save the best model based on validation accuracy
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(cfg.BEST_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Stop training early if validation loss stops improving
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce learning rate when training gets stuck
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        # Save training history to a CSV file
        tf.keras.callbacks.CSVLogger(
            str(cfg.REPORTS_DIR / "training_log.csv"),
            append=False,
        ),
    ]


def _plot_history(history: tf.keras.callbacks.History, out_dir: Path) -> None:
    # Plot and save the accuracy and loss curves after training
    out_dir.mkdir(parents=True, exist_ok=True)
    h = history.history

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(h["accuracy"], label="Train Accuracy")
    ax1.plot(h["val_accuracy"], label="Val Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    # Loss plot
    ax2.plot(h["loss"], label="Train Loss")
    ax2.plot(h["val_loss"], label="Val Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    fig.tight_layout()
    out_path = out_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved training curves to %s", out_path)


def train(epochs: int = cfg.EPOCHS,
          batch_size: int = cfg.BATCH_SIZE,
          lr: float = cfg.LEARNING_RATE) -> tf.keras.Model:
    # Main training function
    cfg.ensure_dirs()
    configure_gpu()

    # Load the training and validation datasets
    train_ds, val_ds, _test_ds, _classes = build_datasets(batch_size=batch_size)
    log.info("Class names from dataset: %s", _classes)

    # Calculate class weights to handle any class imbalance
    class_counts = {}
    for class_name in _classes:
        class_dir = cfg.TRAIN_DIR / class_name
        if class_dir.exists():
            count = len([
                f for f in class_dir.glob("*")
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
            ])
            class_counts[class_name] = count

    total_samples = sum(class_counts.values())
    class_weights = {
        _classes.index(cls): total_samples / (len(class_counts) * count)
        for cls, count in class_counts.items()
    }
    log.info("Class weights: %s", class_weights)

    # Build and compile the CNN
    model = build_cnn()
    _compile(model, lr)
    model.summary(print_fn=log.info)

    # Train the model
    log.info("Starting training for up to %d epochs...", epochs)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=_build_callbacks(),
        class_weight=class_weights,
        verbose=1,
    )

    # Save the final model
    model.save(cfg.FINAL_MODEL_PATH)
    log.info("Saved final model to %s", cfg.FINAL_MODEL_PATH)

    # Save class names to a file for use during prediction
    cfg.CLASS_NAMES_PATH.write_text(json.dumps(cfg.CLASS_NAMES, indent=2))
    log.info("Saved class names to %s", cfg.CLASS_NAMES_PATH)

    # Plot and save training curves
    _plot_history(history, cfg.PLOTS_DIR)
    return model