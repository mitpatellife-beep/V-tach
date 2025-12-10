📄 README — VTach Detection from Single-Lead ECG
Project Overview

This project implements and evaluates two supervised machine-learning models for detecting Ventricular Tachycardia (VT) from single-lead ECG waveforms:

Logistic Regression (classical baseline)

1D Convolutional Neural Network (deep learning baseline)

The goal is to demonstrate the challenges of rare-event detection under extreme imbalance (10 VT vs 990 non-VT).
All code runs end-to-end in a Jupyter notebook or Google Colab.

1. Dataset

Name: ECG Signals
Source: Kaggle — radwakandeel/ecg-signals
URL: https://www.kaggle.com/datasets/radwakandeel/ecg-signals

Format:
.mat ECG files grouped into 17 arrhythmia folders (e.g., 1 NSR, 10 VT, 4 AFIB, etc.)
Waveform length: 3,600 samples per ECG segment

Task framing:

Positive class (1): VT recordings from folder 10 VT

Negative class (0): All other rhythm classes

Imbalance: 10 VT vs 990 non-VT

Data is fully de-identified; no PHI or subject identifiers included.

2. Repository Structure
project-root/
│
├── vtach_ecg_main.ipynb     # Main notebook: EDA, preprocessing, LR, CNN, evaluation
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── (Generated during runtime)
     ├── data/               # Kaggle dataset downloaded by kagglehub
     ├── figures/            # Plots (EDA + CNN learning curves)

3. Environment & Dependencies

Recommended environment:

Python 3.9 or 3.10

CPU or GPU (GPU optional but speeds up CNN training)

Major libraries:

numpy
pandas
scipy
scikit-learn
matplotlib
torch
neurokit2            # optional (used for HR-based feature extraction)
kagglehub


Install dependencies:

pip install -r requirements.txt

4. Reproducibility Notes (Deterministic Seeds — REQUIRED)

This project enables full deterministic reproducibility by setting seeds for all major frameworks:

Python random

NumPy

PyTorch (CPU)

PyTorch CUDA (with deterministic backend settings)

These seeds are initialized at the top of the notebook in the “Reproducibility” section so that all model training runs are repeatable across environments.

5. How to Run (Google Colab Recommended)
Step 1 — Open the Notebook

Upload vtach_ecg_main.ipynb to Google Colab.

Step 2 — Install Dependencies

Run the first cell:

!pip install -r requirements.txt

Step 3 — Authenticate Kaggle

You may either:

Upload your kaggle.json API key in Colab, or

Allow kagglehub to access your Kaggle account interactively

Step 4 — Run All Cells (Top-to-Bottom)

The notebook performs the full pipeline:

1. Download dataset
kagglehub.dataset_download("radwakandeel/ecg-signals")

2. Load data

Load .mat ECG signals and convert them into a DataFrame
(shape: 1000 rows × 3601 columns).

3. EDA

Class imbalance bar plot

Example NSR and VT waveform plots

4. Preprocessing

Stratified train/test split (70/30 and 80/20)

Standardization (train fit, test transform)

5. Train Models

Logistic Regression

5-fold stratified cross-validation

Class-balanced training

1D CNN

Internal train/validation split

Weighted BCE loss

Dropout and class weighting

Multiple hyperparameter configurations

6. Evaluate

Metrics printed to console:

Sensitivity

Specificity

False Positive Rate

Precision

ROC-AUC

AUCPR (critical for imbalance)

7. Outputs

Notebook automatically generates:

EDA plots

CNN training/validation loss curves

Logistic regression and CNN test metrics

6. Expected Runtime & Hardware Notes

CPU: Full run completes in ~3–5 minutes

GPU: CNN training reduces to <1 minute (e.g., Colab T4)

Memory usage: <1 GB

The entire pipeline is lightweight and suitable for laptops or Colab free tier.

7. Citation

If using this project in academic work, please cite the Kaggle dataset and toolkits (NumPy, Pandas, SciPy, PyTorch, scikit-learn, Matplotlib, NeuroKit2, KaggleHub).
