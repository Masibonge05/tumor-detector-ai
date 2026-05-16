# Evaluation pipeline - runs the model on the test set and saves results

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.evaluation.predict import _load_saved_class_names
from src.preprocessing.preprocess import build_datasets
from src.utils import config as cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


def _load_model() -> tf.keras.Model:
    # Load the best saved model, or fall back to the final model
    path = cfg.BEST_MODEL_PATH if cfg.BEST_MODEL_PATH.exists() else cfg.FINAL_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found. Run training first. "
            f"Looked at: {cfg.BEST_MODEL_PATH} and {cfg.FINAL_MODEL_PATH}"
        )
    log.info("Loading model: %s", path)
    return tf.keras.models.load_model(path)


def _plot_confusion_matrix(cm: np.ndarray, class_names, out_path: Path) -> None:
    # Plot and save the confusion matrix as a heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved confusion matrix to %s", out_path)


def evaluate() -> dict:
    # Run evaluation on the test set and return metrics
    cfg.ensure_dirs()
    model = _load_model()

    # Load class names saved during training and build the test dataset
    saved_class_names = _load_saved_class_names()
    _train_ds, _val_ds, test_ds, class_names = build_datasets(
        class_names=saved_class_names
    )

    # Collect predictions for all test batches
    y_true, y_pred, y_prob = [], [], []

    for images, labels in test_ds:
        # Get model output probabilities for each image
        probs = model.predict(images, verbose=0)
        y_prob.append(probs)

        # Take the class with the highest probability as the prediction
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(labels.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.concatenate(y_prob, axis=0)

    log.info("Evaluated %d test samples", len(y_true))

    # Calculate accuracy, precision, recall and F1 score
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    log.info("Test metrics: %s", metrics)

    # Generate and save the full classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    report_path = cfg.REPORTS_DIR / "classification_report.txt"
    report_path.write_text(report)
    log.info("Saved classification report to %s", report_path)

    # Save metrics to a JSON file (used by the Streamlit app)
    metrics_path = cfg.REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Plot and save the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(cm, class_names, cfg.CM_DIR / "confusion_matrix.png")

    # Save raw prediction arrays for any further analysis
    np.savez(cfg.REPORTS_DIR / "predictions.npz",
             y_true=y_true, y_pred=y_pred, y_prob=y_prob)

    return metrics