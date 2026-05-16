from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

# Ensure the app can import project modules even when executed from the app folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import config as cfg

# Load saved class labels if they exist, otherwise fall back to config.
def _load_saved_class_names() -> list[str]:
    if cfg.CLASS_NAMES_PATH.exists():
        try:
            data = json.loads(cfg.CLASS_NAMES_PATH.read_text())
            if isinstance(data, list) and all(
                    isinstance(x, str) for x in data):
                return data
        except json.JSONDecodeError:
            pass
    return cfg.CLASS_NAMES


# Configure the Streamlit page metadata and layout.
st.set_page_config(
    page_title="NeuroScan AI – Brain Tumor MRI Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    _UJ_LOGO_BASE64 = (
        "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"
    )
    logo_bytes = base64.b64decode(_UJ_LOGO_BASE64)
    logo_image = Image.open(io.BytesIO(logo_bytes))
    logo_image = logo_image.resize((180, 180), Image.LANCZOS)
    st.image(logo_image, width=180)
    st.markdown("## NeuroScan AI")
    st.caption(
        "Signal Processing 4A (SIGEEA4) • Deep Learning for Medical Imaging")
    st.caption("**Name:** SHABALALA MN")
    st.caption("**Student Number:** 219024007")
    # Navigation menu for the different app sections.
    page = st.radio(
        "Navigation",
        ["Overview", "Prediction", "Performance"],
    )
    st.markdown("---")


# Lazy-load the predictor functions and cache the model for the app session.
@st.cache_resource(show_spinner="Loading trained CNN model…")
def _load_predictor():
    from src.evaluation.predict import predict_image, gradcam_overlay, get_model
    _ = get_model()  # force load + cache
    return predict_image, gradcam_overlay


# Main overview dashboard content showing project details.
if page == "Overview":
    st.title("Brain Tumor MRI Classification")
    st.subheader(
        "Convolutional Neural Networks for Automated Diagnostic Triage")

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in [
        (c1, "Tumor Classes", "4"),
        (c2, "Input Resolution", f"{cfg.IMG_SIZE[0]}×{cfg.IMG_SIZE[1]}"),
        (c3, "Architecture", "Custom Deep CNN"),
        (c4, "Explainability", "Grad-CAM"),
    ]:
        with col:
            if label == "Architecture":
                st.markdown(f"**{label}**")
                st.write(value)
            else:
                st.metric(label, value)

    st.markdown("### Project Summary")
    st.markdown("""
        NeuroScan AI is a research prototype that demonstrates how a carefully
        designed Convolutional Neural Network can classify brain MRI scans into
        four diagnostic categories: Glioma, Meningioma, Pituitary
        Tumor, and No Tumor, to assist radiologists with rapid triage.

        The system combines:
        * a modular preprocessing pipeline (skull/margin cropping, normalization, augmentation),
        * a deep CNN with Batch Normalization, Dropout, and L2 regularization,
        * a complete training pipeline with checkpointing and learning-rate scheduling,
        * research-quality evaluation (precision, recall, F1, confusion matrix),
        * Grad-CAM heatmaps for explainable AI.
        """)

    st.info("Use the sidebar to upload an MRI scan and run a live prediction.")

# Prediction page where users upload images and view CNN output.
elif page == "Prediction":
    st.title("MRI Prediction")
    st.caption(
        "Upload an MRI scan and the CNN will predict the tumor class with confidence scores and a Grad-CAM heatmap."
    )

    uploaded = st.file_uploader(
        "Upload MRI image (JPG / PNG)",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if uploaded is None:
        st.warning("Please upload an MRI image to start.")
    else:
        try:
            predict_image, gradcam_overlay = _load_predictor()
        except FileNotFoundError as exc:
            st.error(f" {exc}")
            st.stop()

        image = Image.open(uploaded).convert("RGB")
        np_img = np.array(image)

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("#### Uploaded MRI")
            st.image(image, use_container_width=True)

        with st.spinner("Running CNN inference…"):
            try:
                overlay, info = gradcam_overlay(np_img)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

        with col_right:
            st.markdown("#### Grad-CAM Explainability")
            st.image(overlay,
                     use_container_width=True,
                     caption="Regions the CNN focused on for its decision")

        predicted_label = info["display_name"]
        predicted_prob = info["confidence"]
        if info["predicted_class"] == "notumor":
            st.success(f"**Predicted class:** {predicted_label}  \n"
                       f"**CNN confidence:** {predicted_prob*100:.2f}%")
        else:
            st.warning(f"**Predicted class:** {predicted_label}  \n"
                       f"**CNN confidence:** {predicted_prob*100:.2f}%")

        probabilities = {
            class_name: float(info["probabilities"].get(class_name, 0.0))
            for class_name in cfg.CLASS_NAMES
        }

        st.markdown("#### Probability Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        classes = [cfg.CLASS_DISPLAY[c] for c in cfg.CLASS_NAMES]
        probs = [probabilities[c] for c in cfg.CLASS_NAMES]
        bars = ax.bar(classes, probs, color='blue')
        ax.set_ylabel('Probability')
        ax.set_title('Class Probabilities')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        plt.xticks(rotation=45, ha='right')
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    height + 0.01,
                    f'{prob:.1%}',
                    ha='center',
                    va='bottom')
        st.pyplot(fig)

        report_lines = [
            "Brain Tumor MRI Prediction Report",
            "------------------------------",
            f"Predicted class: {predicted_label}",
            f"Model confidence: {predicted_prob*100:.2f}%",
            "",
            "Class probabilities:",
        ]
        for class_name in cfg.CLASS_NAMES:
            report_lines.append(
                f"- {cfg.CLASS_DISPLAY[class_name]}: {probabilities[class_name]*100:.2f}%"
            )

        report_text = "\n".join(report_lines)
        st.download_button(
            label="Download prediction report",
            data=report_text,
            file_name="prediction_report.txt",
            mime="text/plain",
        )

# Performance page displaying evaluation metrics and artifacts.
elif page == "Performance":
    st.title("Model Performance")

    metrics_path = cfg.REPORTS_DIR / "metrics.json"
    cm_path = cfg.CM_DIR / "confusion_matrix.png"
    curves_path = cfg.PLOTS_DIR / "training_curves.png"
    report_path = cfg.REPORTS_DIR / "classification_report.txt"

    if metrics_path.exists():
        import json
        metrics = json.loads(metrics_path.read_text())
        c1, c2, c3, c4 = st.columns(4)
        for col, key, label in [
            (c1, "accuracy", "Accuracy"),
            (c2, "precision_macro", "Precision (macro)"),
            (c3, "recall_macro", "Recall (macro)"),
            (c4, "f1_macro", "F1-Score (macro)"),
        ]:
            with col:
                st.metric(label, f"{metrics[key]*100:.2f}%")
    else:
        st.info("Run `python main.py --mode evaluate` to generate metrics.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Training Curves")
        if curves_path.exists():
            st.image(str(curves_path), use_container_width=True)
        else:
            st.caption("No training curves yet — run training first.")
    with col2:
        st.markdown("#### Confusion Matrix")
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.caption("No confusion matrix yet — run evaluation first.")

    if report_path.exists():
        st.markdown("#### Classification Report")
        st.code(report_path.read_text(), language="text")
