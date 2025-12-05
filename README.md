# V-Tach Detection from Single-Lead ECG

This project implements and evaluates two supervised models for detecting Ventricular Tachycardia (V-tach) from single-lead ECG waveforms:

1. Logistic Regression (classical baseline, from the course)
2. 1D Convolutional Neural Network (deep model, from CNN lectures on ECG)

The work is based on a public arrhythmia dataset from Kaggle and is designed to run end-to-end in a Jupyter notebook or in Google Colab.

---

## 1. Dataset

- **Name:** ECG Signals
- **Source:** Kaggle – `radwakandeel/ecg-signals`
- **URL:** https://www.kaggle.com/datasets/radwakandeel/ecg-signals
- **Format:** `.mat` files organized in 17 class folders (e.g., `1 NSR`, `10 VT`, `4 AFIB`, etc.)
- **Waveform length:** 3,600 samples per recording
- **Task framing in this project:**
  - Positive class (1): V-tach (`"10 VT"` folder)
  - Negative class (0): all other 16 rhythm classes
- **Imbalance:** 10 V-tach vs 990 non-V-tach

No PHI or subject identifiers are included; data is fully de-identified.

---

## 2. How to Run (Google Colab)

1. Open **Google Colab** and upload `vtach_ecg_main.ipynb` (from `notebooks/`).
2. In the first cell, install dependencies from `requirements.txt`:

   ```python
   !pip install -r requirements.txt
Make sure you have a Kaggle API key (kaggle.json) configured in Colab, or allow kagglehub to access your Kaggle account.

Run all cells in order:

Download dataset with kagglehub.dataset_download("radwakandeel/ecg-signals")

Load .mat ECG signals and build a DataFrame

Perform EDA:

Class imbalance bar plot

Example V-tach waveform plot (and optional non-V-tach plot)

Preprocess:

Train/test stratified split

Standardization (fit on train, apply to test)

Train models:

Logistic Regression with 5-fold stratified cross-validation on the training set

1D CNN with internal train/validation split and weighted BCE loss

Evaluate on the held-out test set using sensitivity, specificity, FPR, precision, ROC-AUC

The notebook will print metrics and display the EDA plots and CNN training curves.
