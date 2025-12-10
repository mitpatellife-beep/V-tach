Project Overview

This project implements and evaluates two supervised machine-learning models for detecting Ventricular Tachycardia (VT) from single-lead ECG waveforms:

Logistic Regression (classical baseline)

1D Convolutional Neural Network (deep learning baseline)

The project demonstrates the challenges of rare-event detection under extreme class imbalance (10 VT vs 990 non-VT).
All code runs end-to-end in Jupyter or Google Colab.

1. Dataset

Name: ECG Signals
Source: Kaggle – radwakandeel/ecg-signals
URL: https://www.kaggle.com/datasets/radwakandeel/ecg-signals

Format: .mat files grouped into 17 arrhythmia classes (e.g., 1 NSR, 10 VT, 4 AFIB, etc.)
Waveform length: 3,600 samples per ECG segment

Task framing:

Positive class (1): VT recordings from folder 10 VT

Negative class (0): All other 16 rhythm classes

Imbalance: 10 VT vs 990 non-VT

No PHI included; data is de-identified.

2. Repository Structure
project-root/
│
├── vtach_ecg_main.ipynb     # Main notebook: EDA, preprocessing, LR, CNN, evaluation
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── (Generated during runtime)
     ├── data/               # Kaggle dataset downloaded by kagglehub
     ├── figures/            # Plots such as imbalance charts and CNN learning curves

3. Environment & Dependencies

Recommended environment:

Python 3.9 or 3.10

GPU optional (CPU works but CNN will run slower)

Major libraries:

numpy
pandas
scipy
scikit-learn
matplotlib
torch
neurokit2            # optional (used for HR-based feature extraction)
kagglehub


Install with:

pip install -r requirements.txt

4. Reproducibility Notes

The code sets:

numpy, random, and torch seeds

CUDA deterministic flags (if GPU available)

These ensure that experiments are repeatable across runs.

5. How to Run (Google Colab Recommended)
Step 1 — Open the Notebook

Upload vtach_ecg_main.ipynb to Google Colab.

Step 2 — Install Dependencies

Run the first cell:

!pip install -r requirements.txt

Step 3 — Authenticate Kaggle

Either:

Upload your kaggle.json API key, or

Approve KaggleHub access

Step 4 — Run All Cells in Order

The notebook performs:

Download dataset

kagglehub.dataset_download("radwakandeel/ecg-signals")


Load .mat ECG signals into a pandas DataFrame (shape: 1000 × 3601)

EDA

Class imbalance bar chart

Sample NSR and VT waveforms

Preprocessing

Stratified train/test split (70/30 or 80/20)

Standardization for LR and CNN

Train models

Logistic Regression with 5-fold CV

CNN with weighted BCE loss, dropout, and internal validation set

Evaluate

Sensitivity, specificity, false-positive rate

Precision

ROC-AUC and AUCPR (important for rare events)

Outputs

Training/validation loss curves

EDA plots

Metrics printed to console

6. Expected Outputs

The notebook generates:

Class imbalance plot

NSR and VT waveform examples

CNN overfitting curves

Logistic regression CV metrics

Final test metrics for both models
