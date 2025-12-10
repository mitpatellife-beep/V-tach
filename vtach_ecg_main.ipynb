# ============================================
# Ventricular Tachycardia Detection Project
# Logistic Regression (features) + 1D CNN
# ============================================

# ---------- Imports ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random
import kagglehub
import scipy.io

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    average_precision_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Optional NeuroKit2 for feature extraction
try:
    import neurokit2 as nk
    HAVE_NK = True
    print("NeuroKit2 available: using physiological features.")
except ImportError:
    HAVE_NK = False
    print("NeuroKit2 NOT available: falling back to simple waveform statistics only.")

# ---------- Reproducibility ----------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================
# 1. Load dataset (folders by class)
# ============================================

path = kagglehub.dataset_download("radwakandeel/ecg-signals")
print("Dataset path:", path)

# In this dataset, the .mat files are under "New folder/"
root_dir = Path(path) / "New folder"

data_rows = []
for class_dir in sorted(root_dir.iterdir()):
    if not class_dir.is_dir():
        continue
    label = class_dir.name
    print("Processing:", label)
    mat_files = list(class_dir.glob("*.mat"))
    for mat_file in mat_files:
        mat_data = scipy.io.loadmat(mat_file)
        if "val" not in mat_data:
            print(f"  Warning: 'val' key not found in {mat_file}, skipping.")
            continue
        signal = mat_data["val"].flatten()
        row = {f"s{i}": v for i, v in enumerate(signal)}
        row["label"] = label
        data_rows.append(row)

df = pd.DataFrame(data_rows)
print("\nFinal dataframe shape:", df.shape)
print("Label distribution:\n", df["label"].value_counts())

# Map V-tach vs non-V-tach
VTACH_LABEL = "10 VT"
df["is_vtach"] = (df["label"] == VTACH_LABEL).astype(int)
print("\nBinary distribution (0=non-VT, 1=VT):\n", df["is_vtach"].value_counts())

signal_cols = [c for c in df.columns if c.startswith("s")]
print("\nNumber of signal samples per recording:", len(signal_cols))

# ============================================
# 2. EDA plots: class imbalance, example waveforms
# ============================================

# --- Plot 1: Class imbalance ---
plt.figure(figsize=(4, 3))
df["is_vtach"].value_counts().rename({0: "Non-VT", 1: "VT"}).plot(kind="bar")
plt.title("Class Imbalance (Non-VT vs VT)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --- Plot 2: Example waveforms (NSR vs VT if present) ---
def plot_example_waveform(df, label_name, title):
    subset = df[df["label"] == label_name]
    if subset.empty:
        print(f"No examples found for label: {label_name}")
        return
    ex = subset.iloc[0][signal_cols].values
    plt.figure(figsize=(8, 3))
    plt.plot(ex)
    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude (a.u.)")
    plt.tight_layout()
    plt.show()

plot_example_waveform(df, "1 NSR", "Example Normal Sinus Rhythm (NSR)")
plot_example_waveform(df, "10 VT", "Example Ventricular Tachycardia (VT)")

# ============================================
# 3. Feature extraction for Logistic Regression
# ============================================

# Assumed sampling rate (approximate for NK; not strictly given)
FS = 360

def extract_features_neurokit(signal_1d, fs=FS):
    """
    Extract simple ECG features using NeuroKit2 (HR mean, HR std).
    Wrapped in try/except to be robust to processing errors.
    """
    feat = {}
    try:
        signals, info = nk.ecg_process(signal_1d, sampling_rate=fs)
        hr = signals["ECG_Rate"].values
        feat["hr_mean"] = float(np.nanmean(hr))
        feat["hr_std"] = float(np.nanstd(hr))
    except Exception:
        feat["hr_mean"] = np.nan
        feat["hr_std"] = np.nan
    return feat

def extract_features_simple(signal_1d):
    """Simple waveform statistics (always used)."""
    return {
        "sig_mean": float(np.mean(signal_1d)),
        "sig_std": float(np.std(signal_1d)),
        "sig_min": float(np.min(signal_1d)),
        "sig_max": float(np.max(signal_1d)),
    }

def build_feature_matrix(df):
    """
    Build feature matrix for logistic regression:
    - NeuroKit-based features (if available)
    - Simple statistics
    """
    feats = []
    for _, row in df.iterrows():
        sig = row[signal_cols].values.astype(float)
        feat_dict = {}
        if HAVE_NK:
            feat_dict.update(extract_features_neurokit(sig))
        feat_dict.update(extract_features_simple(sig))
        feat_dict["is_vtach"] = row["is_vtach"]
        feats.append(feat_dict)

    feat_df = pd.DataFrame(feats)
    # Drop all-NaN columns (if any)
    feat_df = feat_df.dropna(axis=1, how="all")
    # Simple numerical imputation
    feat_df = feat_df.fillna(feat_df.mean(numeric_only=True))
    return feat_df

feat_df = build_feature_matrix(df)
print("\nFeature columns for logistic regression:\n", feat_df.columns.tolist())

X_feat = feat_df.drop(columns=["is_vtach"]).values
y = feat_df["is_vtach"].values

# For CNN we use raw waveform
X_wave = df[signal_cols].values

# ============================================
# 4. Metric helper
# ============================================

def evaluate_binary(y_true, y_prob, split_name="Test"):
    """
    Compute confusion matrix, Sensitivity, Specificity, FPR,
    Precision, ROC-AUC, and AUCPR.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
    else:
        roc_auc, auc_pr = np.nan, np.nan

    print(f"\n=== {split_name} Metrics ===")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"Sensitivity (Recall): {sensitivity:.3f}")
    print(f"Specificity:         {specificity:.3f}")
    print(f"False Positive Rate: {fpr:.3f}")
    print(f"Precision:           {precision:.3f}")
    print(f"ROC-AUC:             {roc_auc:.3f}")
    print(f"AUCPR:               {auc_pr:.3f}")

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "fpr": fpr,
        "precision": precision,
        "roc_auc": roc_auc,
        "auc_pr": auc_pr,
    }

# ============================================
# 5. Logistic Regression Experiments (70/30 & 80/20)
# ============================================

def run_logreg_experiment(test_size):
    print("\n" + "=" * 60)
    print(f"LOGISTIC REGRESSION — test_size={test_size:.2f}")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5-fold CV on training data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_metrics = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train), start=1):
        X_tr, X_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        logreg = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )
        logreg.fit(X_tr, y_tr)
        y_val_prob = logreg.predict_proba(X_val)[:, 1]
        print(f"\nFold {fold}")
        m = evaluate_binary(y_val, y_val_prob, split_name=f"CV Fold {fold}")
        cv_metrics.append(m)

    # Mean CV metrics
    print("\n=== Mean CV metrics across 5 folds ===")
    mean_cv = {k: np.mean([d[k] for d in cv_metrics]) for k in cv_metrics[0].keys()}
    for k, v in mean_cv.items():
        print(f"{k}: {v:.3f}")

    # Final train on full training set, test
    final_logreg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    final_logreg.fit(X_train_scaled, y_train)
    y_test_prob = final_logreg.predict_proba(X_test_scaled)[:, 1]
    test_metrics = evaluate_binary(y_test, y_test_prob, split_name="Test")

    return mean_cv, test_metrics

cv_70, test_70 = run_logreg_experiment(test_size=0.30)
cv_80, test_80 = run_logreg_experiment(test_size=0.20)

print("\nSummary — Logistic Regression")
print("70/30 split — Test:", test_70)
print("80/20 split — Test:", test_80)

# ============================================
# 6. CNN with DROPOUT
# ============================================

class CNN1D_Dropout(nn.Module):
    def __init__(self, input_len=3600):
        super().__init__()
        self.conv1 = nn.Conv1d(1,16,5,padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16,32,5,padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)

        reduced = input_len//4
        self.fc1 = nn.Linear(32*reduced,128)
        self.drop = nn.Dropout(0.50)
        self.out = nn.Linear(128,1)

    def forward(self,x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = self.drop(torch.relu(self.fc1(x)))
        return self.out(x)

# ============================================
# 6. CNN Model Definition
# ============================================

class CNN1DClassifier(nn.Module):
    def __init__(self, input_length=3600):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)

        length_after_pool = input_length // 4  # two poolings by factor 2
        self.fc = nn.Linear(32 * length_after_pool, 1)

    def forward(self, x):
        # x: (batch, 1, 3600)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # logits

# ============================================
# 7. CNN Experiments
# ============================================

def run_cnn_experiment(test_size=0.30, batch_size=32, lr=1e-3, num_epochs=20):
    print("\n" + "=" * 60)
    print(f"CNN — test_size={test_size:.2f}, batch_size={batch_size}, lr={lr}")
    print("=" * 60)

    # Train/test split for waveforms
    X_train, X_test, y_train, y_test = train_test_split(
        X_wave, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    # Standardize per-sample index
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Inner train/val split (80/20 of training)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_STATE
    )

    # Create tensor datasets
    def to_tensor_dataset(X_np, y_np):
        X_t = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1)  # (batch, 1, 3600)
        y_t = torch.tensor(y_np, dtype=torch.float32)
        return TensorDataset(X_t, y_t)

    train_ds = to_tensor_dataset(X_tr, y_tr)
    val_ds = to_tensor_dataset(X_val, y_val)
    test_ds = to_tensor_dataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Compute pos_weight
    n_pos = np.sum(y_tr == 1)
    n_neg = np.sum(y_tr == 0)
    pos_weight_value = n_neg / n_pos if n_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    print("CNN pos_weight (N_neg / N_pos):", pos_weight_value)

    model = CNN1DClassifier(input_length=X_wave.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    # ----- Training loop -----
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch).squeeze(1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * X_batch.size(0)

        train_loss = running_train_loss / len(train_loader.dataset)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(X_batch).squeeze(1)
                loss = criterion(logits, y_batch)
                running_val_loss += loss.item() * X_batch.size(0)

        val_loss = running_val_loss / len(val_loader.dataset)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:02d}/{num_epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")

    # Plot loss curves
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Train/Val Loss (test={int((1 - test_size) * 100)}%, bs={batch_size}, lr={lr})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Evaluate on val and test
    def get_probs(dl, split_name):
        model.eval()
        all_probs, all_y = [], []
        with torch.no_grad():
            for X_batch, y_batch in dl:
                X_batch = X_batch.to(device)
                logits = model(X_batch).squeeze(1)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_y.append(y_batch.numpy())
        y_all = np.concatenate(all_y)
        p_all = np.concatenate(all_probs)
        metrics = evaluate_binary(y_all, p_all, split_name=split_name)
        return metrics

    val_metrics = get_probs(val_loader, "CNN Validation")
    test_metrics = get_probs(test_loader, "CNN Test")

    return val_metrics, test_metrics


# Run CNN experiments for both splits & multiple hyperparameters
cnn_results = {}
for test_size in [0.30, 0.20]:         # 70/30 and 80/20
    for batch_size in [16, 32]:
        for lr in [1e-3, 3e-4]:
            key = f"test{int((1 - test_size) * 100)}_bs{batch_size}_lr{lr}"
            val_m, test_m = run_cnn_experiment(
                test_size=test_size,
                batch_size=batch_size,
                lr=lr,
                num_epochs=20,    # you can reduce for speed
            )
            cnn_results[key] = {"val": val_m, "test": test_m}

print("\nSummary of CNN experiments:")
for key, res in cnn_results.items():
    print(f"\nConfig: {key}")
    print("  Val ROC-AUC:", f"{res['val']['roc_auc']:.3f}", "Val AUCPR:", f"{res['val']['auc_pr']:.3f}")
    print("  Test ROC-AUC:", f"{res['test']['roc_auc']:.3f}", "Test AUCPR:", f"{res['test']['auc_pr']:.3f}")

print("\nAll experiments complete.")
