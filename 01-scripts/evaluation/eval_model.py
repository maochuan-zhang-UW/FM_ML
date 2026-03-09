import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')   # CPU-only; Metal plugin crashes on predict
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

# ---------------------------------------------------------------------
# Normalize function (same as training)
# ---------------------------------------------------------------------
def norm(X):
    max_val = np.max(abs(X), axis=1, keepdims=True)
    max_val[max_val == 0] = 1
    return X / max_val


def plot_confusion_matrix(cm, classes, title, fname):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted Label",
        ylabel="True Label",
        title=title,
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# =============================================================================
#                   LOAD TRAINED MODEL
# =============================================================================
ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(ROOT, "06-models", "PolarPicker_h5_20260308_154910.keras")
FIGS_DIR   = os.path.join(ROOT, "03-figs")
os.makedirs(FIGS_DIR, exist_ok=True)

model = keras.models.load_model(model_path, safe_mode=False)
print("Loaded trained model:", model_path)

# =============================================================================
#                 SETTINGS: DATA DIR & STATIONS
# =============================================================================
VAL_H5   = os.path.join(ROOT, "02-data", "K_aug", "TMSF_Val_003", "val_dataset.h5")
STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]
CLASSES  = ["Negative", "Positive"]

all_true = []
all_pred = []

print("\n================ EVALUATION PER STATION ================\n")

with h5py.File(VAL_H5, "r") as f:
    for sta in STATIONS:
        if sta not in f:
            print(f"  {sta}: not found in HDF5, skipping.")
            continue

        X = f[f"{sta}/waveforms"][:]    # (N, 64, 1)  float32
        y = f[f"{sta}/polarities"][:]   # (N,)        int32  (0 or 1)

        Xn = norm(X)

        y_raw = model.predict(Xn, batch_size=256, verbose=0)
        y_pred_prob = y_raw[1] if isinstance(y_raw, (list, tuple)) else y_raw
        y_pred = np.argmax(y_pred_prob, axis=1)

        acc = accuracy_score(y, y_pred)
        print(f"  {sta}:  n={len(y):5d}  accuracy={acc:.4f} ({acc*100:.2f}%)")

        # Per-station confusion matrix
        cm_sta = confusion_matrix(y, y_pred)
        plot_confusion_matrix(
            cm_sta, CLASSES,
            title=f"Confusion Matrix — {sta}  (acc={acc*100:.2f}%)",
            fname=os.path.join(FIGS_DIR, f"confmat_eval_sigma001_{sta}.png"),
        )

        all_true.append(y)
        all_pred.append(y_pred)

# =============================================================================
#                     GLOBAL METRICS (ALL STATIONS)
# =============================================================================
if all_true:
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    acc_g = accuracy_score(all_true, all_pred)
    prec_g, rec_g, f1_g, _ = precision_recall_fscore_support(
        all_true, all_pred, average="binary", pos_label=1, zero_division=0
    )

    print("\n================ GLOBAL (ALL STATIONS) ================\n")
    print(f"  Total samples : {len(all_true)}")
    print(f"  Accuracy      : {acc_g:.4f} ({acc_g*100:.2f}%)")
    print(f"  Precision     : {prec_g:.4f}")
    print(f"  Recall        : {rec_g:.4f}")
    print(f"  F1 score      : {f1_g:.4f}")

    print("\nClassification report:")
    print(classification_report(all_true, all_pred, target_names=CLASSES, digits=4))

    # Global confusion matrix
    cm_all = confusion_matrix(all_true, all_pred)
    print("Confusion Matrix:\n", cm_all)
    plot_confusion_matrix(
        cm_all, CLASSES,
        title=f"Confusion Matrix — All Stations  (acc={acc_g*100:.2f}%)",
        fname=os.path.join(FIGS_DIR, "confmat_eval_sigma001_all_stations.png"),
    )

print("\nDone.")
