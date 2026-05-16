# Brain Tumor MRI Classification using Convolutional Neural Networks

**Student Name:** SHABALALA MN   
**Course:** Signal Processing 4A (SIGEEA4)  
**Project:** Brain Tumor Detection from MRI Scans

An academic project that classifies brain MRI scans into four diagnostic categories using a custom convolutional neural network, with a complete training pipeline, evaluation artifacts, and an interactive Streamlit dashboard.

## Project Structure

- `main.py` — primary CLI entry point for training, evaluation, and single-image prediction
- `app/streamlit_app.py` — Streamlit dashboard for MRI upload, prediction, Grad-CAM explainability, and performance review
- `src/models/` — CNN architecture definition
- `src/preprocessing/` — preprocessing, augmentation, and dataset pipeline
- `src/training/` — training workflow, callbacks, and plotting
- `src/evaluation/` — evaluation metrics, classification report, confusion matrix, and prediction utilities
- `src/utils/` — configuration, logging, and GPU detection

## Installation

```bash
cd brain_tumor_cnn
pip install -r requirements.txt
```

## Usage

### Train the model
```bash
python main.py --mode train --epochs 50 --batch-size 16
```

### Evaluate the model
```bash
python main.py --mode evaluate
```
```

### Launch the Streamlit app
```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.


