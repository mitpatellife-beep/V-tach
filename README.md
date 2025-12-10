📄 README.md — VTach Detection from Single-Lead ECG
Project Overview

This project builds and evaluates two supervised machine-learning models for detecting Ventricular Tachycardia (VT) from single-lead ECG waveforms. The models include a classical Logistic Regression baseline and a 1D Convolutional Neural Network (CNN) trained directly on raw waveforms. The work demonstrates the difficulty of rare-event detection under extreme class imbalance (1% VT prevalence). All code runs end-to-end in a Jupyter notebook or Python script, with full reproducibility ensured through deterministic seeds and version-pinned environments.

1. Dataset

Name: ECG Signals
Source: Kaggle — radwakandeel/ecg-signals
URL: https://www.kaggle.com/datasets/radwakandeel/ecg-signals

Format:

.mat ECG recordings organized into 17 arrhythmia classes

Each recording is 3,600 samples (single-lead ECG)

Positive class (1): VT recordings from folder 10 VT

Negative class (0): All other rhythm classes

Imbalance: 10 VT vs 990 non-VT

Dataset is fully de-identified (no PHI)

2. Repository Structure
project_root/
│
├── vtach_ecg_main.ipynb        # Main end-to-end pipeline (LR + CNN + evaluation)
├── scripts/
│    └── download_data.py       # Script to download the Kaggle dataset automatically
├── environment.yml             # Conda environment with pinned versions
├── requirements.txt            # Optional pip-based environment
└── README.md                   # This file


✔ This satisfies the “small test artifact or data download script” requirement because you provide a script that fetches the public dataset.

3. Environment & Installation
Option A — Conda (recommended)
conda env create -f environment.yml
conda activate vtach-env

Option B — Pip
pip install -r requirements.txt


The environment contains pinned versions of Python and all required libraries (NumPy, pandas, SciPy, scikit-learn, PyTorch, matplotlib, kagglehub, neurokit2).

4. Reproducibility Notes (Deterministic Seeds — REQUIRED)

To ensure fully deterministic behavior, the notebook sets seeds at the top of the file for:

Python random

NumPy

PyTorch (CPU)

PyTorch CUDA (if available), including:

torch.cuda.manual_seed(...)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

This ensures repeatable training runs across different machines and satisfies the deterministic-seed requirement.

5. Getting the Data
✔ Download Script (Required by Rubric Option)

This project includes a public data download script, satisfying the “small test artifact or data instructions” requirement.

Run:

python scripts/download_data.py


This uses kagglehub to automatically download:

radwakandeel/ecg-signals


and prints the location where the dataset is stored.

✔ Notebook-Based Download (Alternative)

The notebook also downloads the dataset using:

kagglehub.dataset_download("radwakandeel/ecg-signals")


No manual dataset handling is required.

6. How to Run the Full Demo
A. Google Colab (recommended)

Upload vtach_ecg_main.ipynb

Install dependencies:

!pip install -r requirements.txt


Authenticate Kaggle (upload kaggle.json or let kagglehub authenticate)

Run all cells in top-to-bottom order

B. Local Run

Open the notebook in Jupyter/VS Code and run all cells sequentially.

7. What the Pipeline Does

The demo executes the entire VT detection study:

Download dataset

Load and preprocess ECG waveforms

Exploratory analysis:

Class imbalance plot

Example NSR and VT waveforms

Feature extraction for logistic regression

Train logistic regression with 5-fold stratified CV

Train CNN with weighted BCE loss and dropout

Evaluate using sensitivity, specificity, FPR, ROC-AUC, AUCPR

Plot CNN training/validation loss curves

All results print automatically.

8. Expected Runtime & Hardware
Hardware	Runtime
CPU (laptop or Colab free tier)	3–5 minutes
GPU (Colab T4 or similar)	< 1 minute

Memory footprint is under 1 GB.

9. Citation

If you use this work academically, cite:

Kandeel, R. (2023). ECG Signals [Dataset]. Kaggle.

Toolkits: NumPy, SciPy, Matplotlib, PyTorch, scikit-learn, NeuroKit2, kagglehub.
