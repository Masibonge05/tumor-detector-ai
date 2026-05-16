from __future__ import annotations

# Evaluation pipeline for test metrics, reports, and confusion matrix output.
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
    path = cfg.BEST_MODEL_PATH if cfg.BEST_MODEL_PATH.exists(
    ) else cfg.FINAL_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found. Train one first with `python main.py --mode train`. "
            f"Looked at: {cfg.BEST_MODEL_PATH} and {cfg.FINAL_MODEL_PATH}")
    log.info("Loading model: %s", path)
    return tf.keras.models.load_model(path)


def _plot_confusion_matrix(cm: np.ndarray, class_names,
                           out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved confusion matrix → %s", out_path)


def evaluate() -> dict:
    """
    Run evaluation on the held-out test set; returns a metrics dict.

    DEBUG LOGGING:
    Logs class names, predicted indices, and probability distributions to
    verify correct class label mapping throughout evaluation.
    """
    cfg.ensure_dirs()
    model = _load_model()

    # Load the model and datasets, preserving saved class ordering.
    saved_class_names = _load_saved_class_names()
    _train_ds, _val_ds, test_ds, class_names = build_datasets(
        class_names=saved_class_names)

    log.info(f"Configuration CLASS_NAMES: {cfg.CLASS_NAMES}")
    log.info(f"Dataset class_names:       {class_names}")
    log.info(f"Saved class_names metadata: {saved_class_names}")

    if class_names != cfg.CLASS_NAMES:
        log.warning("Class name mismatch between config and dataset!")
        log.warning(f"  Config:  {cfg.CLASS_NAMES}")
        log.warning(f"  Dataset: {class_names}")
    if saved_class_names != cfg.CLASS_NAMES:
        log.warning("Class name mismatch between config and saved metadata!")
        log.warning(f"  Config:  {cfg.CLASS_NAMES}")
        log.warning(f"  Saved:   {saved_class_names}")

    y_true, y_pred, y_prob = [], [], []
    batch_count = 0

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_prob.append(probs)

        batch_preds = np.argmax(probs, axis=1).tolist()
        y_pred.extend(batch_preds)

        batch_labels = labels.numpy().tolist()
        y_true.extend(batch_labels)

        batch_count += 1
        if batch_count <= 3:
            log.debug(
                f"Batch {batch_count}: true_labels={batch_labels}, pred_indices={batch_preds}"
            )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.concatenate(y_prob, axis=0)

    log.info(f"Evaluation on {len(y_true)} test samples")
    log.info(f"Unique true labels: {np.unique(y_true)}")
    log.info(f"Unique pred labels: {np.unique(y_pred)}")

    # ----- Metrics -----
    # Compute aggregate evaluation metrics from test predictions.
    metrics = {
        "accuracy":
        float(accuracy_score(y_true, y_pred)),
        "precision_macro":
        float(precision_score(y_true, y_pred, average="macro",
                              zero_division=0)),
        "recall_macro":
        float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro":
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    log.info("Test metrics: %s", metrics)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    log.info("\n%s", report)

    # Save the text report, metrics, confusion matrix, and prediction artifacts.
    report_path = cfg.REPORTS_DIR / "classification_report.txt"
    report_path.write_text(report)
    log.info("Saved classification report → %s", report_path)

    metrics_path = cfg.REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(y_true, y_pred)
    log.info(f"Confusion matrix:\n{cm}")
    _plot_confusion_matrix(cm, class_names,
                           cfg.CM_DIR / "confusion_matrix.png")

    # Persist raw prediction arrays for later analysis or debugging.
    np.savez(cfg.REPORTS_DIR / "predictions.npz",
             y_true=y_true,
             y_pred=y_pred,
             y_prob=y_prob)

    return metrics
