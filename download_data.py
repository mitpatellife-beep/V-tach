
"""
download_data.py
Utility script to download the ECG dataset for the VTach project.

This script uses kagglehub to automatically fetch the arrhythmia dataset and
place it in the local working directory. Running this script ensures users
can reproduce the full pipeline without manually downloading files.
"""

import kagglehub

def download_ecg_dataset():
    """
    Downloads the Kaggle ECG dataset via kagglehub and prints the local path.
    """
    path = kagglehub.dataset_download("radwakandeel/ecg-signals")
    print("Dataset downloaded to:", path)

if __name__ == "__main__":
    download_ecg_dataset()
