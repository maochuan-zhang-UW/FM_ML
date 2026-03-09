"""
train_axialpolcap_h5.py
========================
Train the AxialPolCap P-wave polarity classifier on HDF5 datasets.

Input datasets (produced by the data preparation scripts)
----------------------------------------------------------
  TMSF_Tra_002/train_dataset.h5   ~150 000 augmented waveforms
      Built by 01_build_training_dataset_h5.py from A_wave_train.h5.
      Augmentation: noise addition (lognormal SNR), waveform flip, and
      cubic-spline time shift (sigma=0.02 s). Perfectly balanced 50/50
      positive/negative polarity.

  TMSF_Val_002/val_dataset.h5     ~10 000 augmented waveforms
      Built by 02_build_eval_dataset_h5.py from A_wave_val.h5 (the 20%
      held-out split of the original cleaned dataset). Same augmentation
      pipeline but drawn from DIFFERENT real events than the train set.

Data split inside this script
------------------------------
The 150 000 training samples are divided 80 / 10 / 10:
  X_train  (80 %, ~120 000)  — gradient updates only
  X_mon    (10 %,  ~15 000)  — monitoring: drives EarlyStopping & ReduceLROnPlateau
  X_test   (10 %,  ~15 000)  — internal test: evaluated after training

The 10 000 val samples (X_val) are never seen during training or monitoring.
They are evaluated at the very end as a fully independent benchmark.

Model architecture  (AxialPolCap / PolarPicker)
-----------------------------------------------
Input  : (64, 1)  — 64-sample waveform window at 100 Hz (±0.32 s around P-arrival)
Encoder: Conv1D(32,32) → Dropout → BN → MaxPool(2)
         Conv1D(8,16)  → BN → MaxPool(2)   →  latent shape (16, 8)
Decoder: Conv1D → BN → UpSample → Conv1D → BN → UpSample → Conv1D(1,64)
         Reconstructs the input waveform (MSE loss, weight=1)
Classifier: Flatten → Dense(2, softmax)
         Predicts polarity probability [P(neg), P(pos)] (Huber loss, weight=200)

The heavy classifier weight (200×) forces the shared encoder to learn
polarity-discriminative features rather than pure reconstruction features.
Prediction: model.predict(X)[1]  →  shape (N, 2), argmax gives 0/1 label.

Outputs saved
-------------
  06-models/PolarPicker_h5_{timestamp}.keras   best checkpoint (lowest val_loss on X_mon)
  06-models/history/history_h5_{timestamp}.json  training curves + metadata
  03-figs/dashboard_h5_{timestamp}.png      4-panel training dashboard (loss / acc / MSE / LR)
  03-figs/report_h5_test_{timestamp}.txt    classification report — internal test
  03-figs/confmat_h5_test_{timestamp}.png
  03-figs/roc_h5_test_{timestamp}.png
  03-figs/pr_curve_h5_test_{timestamp}.png  Precision-Recall curve — internal test
  03-figs/conf_hist_h5_test_{timestamp}.png confidence histogram  — internal test
  03-figs/report_h5_val_{timestamp}.txt     classification report — separate val
  03-figs/confmat_h5_val_{timestamp}.png
  03-figs/roc_h5_val_{timestamp}.png
  03-figs/pr_curve_h5_val_{timestamp}.png   Precision-Recall curve — separate val
  03-figs/conf_hist_h5_val_{timestamp}.png  confidence histogram  — separate val
  03-figs/station_acc_h5_{timestamp}.png    per-station accuracy bar chart (val set)
  03-figs/reconstruction_h5_{timestamp}.png 8 waveform reconstruction examples

Run from any directory:
    conda activate tf_macos
    python /path/to/01-scripts/training/train_axialpolcap_h5.py
"""

import os
import json
import time
from datetime import datetime

import numpy as np
import h5py
import tensorflow as tf  # noqa: F401 – kept for TF backend initialisation
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, BatchNormalization,
    MaxPooling1D, Flatten, UpSampling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_H5   = os.path.join(ROOT, "02-data", "K_aug", "TMSF_Tra_002", "train_dataset.h5")
VAL_H5     = os.path.join(ROOT, "02-data", "K_aug", "TMSF_Val_002",  "val_dataset.h5")
MODELS_DIR = os.path.join(ROOT, "06-models")
HIST_DIR   = os.path.join(ROOT, "06-models", "history")
FIGS_DIR   = os.path.join(ROOT, "03-figs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm(X):
    """Max-normalize along the time axis (safety pass; data already normalized)."""
    max_val = np.max(np.abs(X), axis=1, keepdims=True)
    max_val[max_val == 0] = 1
    return X / max_val


def load_h5_all(h5_path):
    """Load /all/waveforms and /all/polarities from an HDF5 dataset file."""
    with h5py.File(h5_path, "r") as f:
        X = f["all/waveforms"][:]    # (N, 64, 1) float32
        y = f["all/polarities"][:]   # (N,)       int32  (0 or 1)
    print(f"  Loaded {len(X):7d} samples from {os.path.basename(h5_path)}  "
          f"(pos={int(y.sum())}, neg={int((y==0).sum())})")
    return X, y


def plot_confusion_matrix(cm, classes, fname):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# LR logger callback
# ---------------------------------------------------------------------------

class LRLogger(keras.callbacks.Callback):
    """Record the learning rate at the end of every epoch."""
    def __init__(self):
        super().__init__()
        self.lrs = []

    def on_epoch_end(self, epoch, logs=None):
        lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        self.lrs.append((epoch + 1, lr))
        if logs is not None:
            logs['lr'] = lr


# ---------------------------------------------------------------------------
# Timeout callback
# ---------------------------------------------------------------------------

class WallClockTimeout(keras.callbacks.Callback):
    """Stop training if a single batch exceeds max_batch_sec.

    Parameters
    ----------
    max_batch_sec : float
        Maximum seconds allowed for one batch (default 300 = 5 min).
        Catches the memory-leak slowdown where individual steps balloon.
    """

    def __init__(self, max_batch_sec=300):
        super().__init__()
        self.max_batch_sec = max_batch_sec
        self._batch_start  = None

    def on_train_batch_begin(self, batch, logs=None):  # noqa: ARG002
        self._batch_start = time.time()

    def on_train_batch_end(self, batch, logs=None):  # noqa: ARG002
        batch_elapsed = time.time() - self._batch_start
        if batch_elapsed > self.max_batch_sec:
            print(f"\n[WallClockTimeout] Batch {batch} took {batch_elapsed:.0f}s "
                  f"(limit {self.max_batch_sec}s). Stopping training.")
            self.model.stop_training = True


# ---------------------------------------------------------------------------
# Extra plotting helpers
# ---------------------------------------------------------------------------

def plot_training_dashboard(history_dict, lr_history, fname):
    """
    4-panel figure:
      top-left  : total loss (train vs monitor)
      top-right : classifier accuracy (train vs monitor)
      bot-left  : decoder MSE (train vs monitor)
      bot-right : learning-rate schedule
    """
    epochs = range(1, len(history_dict['loss']) + 1)

    # Find metric keys robustly by substring
    def find_key(d, substr, exclude='val'):
        return next((k for k in d if substr in k and exclude not in k), None)

    acc_key     = find_key(history_dict, 'acc')
    val_acc_key = f'val_{acc_key}' if acc_key else None
    mse_key     = find_key(history_dict, 'mse')
    val_mse_key = f'val_{mse_key}' if mse_key else None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Total loss
    axes[0, 0].plot(epochs, history_dict['loss'],     label='Train')
    axes[0, 0].plot(epochs, history_dict['val_loss'], label='Monitor')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    # Classifier accuracy
    if acc_key and val_acc_key and val_acc_key in history_dict:
        axes[0, 1].plot(epochs, history_dict[acc_key],     label='Train')
        axes[0, 1].plot(epochs, history_dict[val_acc_key], label='Monitor')
    axes[0, 1].set_title('Classifier Accuracy')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    # Decoder MSE
    if mse_key and val_mse_key and val_mse_key in history_dict:
        axes[1, 0].plot(epochs, history_dict[mse_key],     label='Train')
        axes[1, 0].plot(epochs, history_dict[val_mse_key], label='Monitor')
    axes[1, 0].set_title('Decoder MSE')
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('MSE')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    # Learning rate  (lr_history is a list of (epoch, lr) tuples)
    ep_nums, lr_vals = zip(*lr_history) if lr_history else ([], [])
    axes[1, 1].plot(ep_nums, lr_vals, color='darkorange')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('LR')
    axes[1, 1].set_yscale('log'); axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('Training Dashboard', fontsize=14)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_pr_curve(y_true, y_prob, label, fname):
    """Precision-Recall curve with average precision score."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'AP = {ap:.3f}')
    plt.axhline(y_true.mean(), color='gray', linestyle='--',
                label=f'Baseline (prevalence={y_true.mean():.2f})')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'Precision-Recall — {label}')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_confidence_histogram(y_true, y_prob, label, fname):
    """
    Histogram of P(positive) split by correct / incorrect predictions.
    Helps diagnose overconfidence or underconfidence.
    """
    pred = (y_prob >= 0.5).astype(int)
    correct   = y_prob[pred == y_true]
    incorrect = y_prob[pred != y_true]

    plt.figure(figsize=(7, 4))
    bins = np.linspace(0, 1, 26)
    plt.hist(correct,   bins=bins, alpha=0.6, color='steelblue', label='Correct')
    plt.hist(incorrect, bins=bins, alpha=0.6, color='tomato',    label='Incorrect')
    plt.axvline(0.5, color='k', linestyle='--', linewidth=0.8)
    plt.xlabel('P(positive)'); plt.ylabel('Count')
    plt.title(f'Confidence Histogram — {label}')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_per_station_accuracy(model, val_h5_path, fname):
    """
    Bar chart of per-station accuracy on the separate validation set.
    Loads each station group individually from the HDF5 file.
    """
    stations, accs, counts = [], [], []
    with h5py.File(val_h5_path, 'r') as f:
        for sta in ['AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1']:
            if sta not in f:
                continue
            X = norm(f[f'{sta}/waveforms'][:])
            y = f[f'{sta}/polarities'][:]
            if len(X) == 0:
                continue
            y_pred = model.predict(X, batch_size=256, verbose=0)[1]
            acc = float((np.argmax(y_pred, axis=1) == y).mean())
            stations.append(sta)
            accs.append(acc)
            counts.append(len(X))

    colors = ['steelblue' if a >= 0.8 else 'orange' if a >= 0.65 else 'tomato'
              for a in accs]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(stations, accs, color=colors, edgecolor='k', linewidth=0.6)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Chance')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Station Accuracy — Separate Validation Set')
    ax.legend()
    for bar, acc, n in zip(bars, accs, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{acc:.2f}\n(n={n})', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_reconstructions(model, X_sample, y_sample, fname, n=8):
    """
    Show n waveforms: input (blue) vs decoder reconstruction (orange).
    Title shows true polarity label.
    """
    idx   = np.random.choice(len(X_sample), size=n, replace=False)
    X_sub = X_sample[idx]
    y_sub = y_sample[idx]

    recon = model.predict(X_sub, verbose=0)[0]   # decoder output

    fig, axes = plt.subplots(2, n // 2, figsize=(14, 5))
    axes = axes.flatten()
    t = np.arange(64)
    labels = {0: 'Neg', 1: 'Pos'}

    for i, ax in enumerate(axes):
        ax.plot(t, X_sub[i, :, 0], color='steelblue', lw=1.2, label='Input')
        ax.plot(t, recon[i, :, 0], color='darkorange', lw=1.2,
                linestyle='--', label='Recon')
        ax.set_title(f'True: {labels[y_sub[i]]}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle('Waveform Reconstruction Examples (Input vs Decoder)', fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_polarPicker(drop_rate=0.3, learn_rate=0.001):
    encoder = Sequential([
        Input(shape=(64, 1)),
        Conv1D(32, 32, activation='relu', padding='same'),
        Dropout(drop_rate),
        BatchNormalization(),
        MaxPooling1D(2, padding='same'),

        Conv1D(8, 16, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2, padding='same'),
    ])

    decoder = Sequential([
        Input(shape=(16, 8)),
        Conv1D(8, 16, activation='tanh', padding='same'),
        BatchNormalization(),
        UpSampling1D(2),

        Conv1D(32, 32, activation='relu', padding='same'),
        BatchNormalization(),
        UpSampling1D(2),

        Conv1D(1, 64, padding='same', activation='tanh'),
    ])

    classifier = Sequential([
        Input(shape=(16, 8)),
        Flatten(),
        Dense(2, activation='softmax')
    ])

    X_in = Input(shape=(64, 1))
    enc  = encoder(X_in)
    dec  = decoder(enc)
    p    = classifier(enc)

    model = Model(inputs=X_in, outputs=[dec, p])

    hub = keras.losses.Huber(delta=0.5)
    model.compile(
        optimizer=keras.optimizers.Adam(learn_rate),
        loss=['mse', hub],
        loss_weights=[1, 200],
        metrics=['mse', 'acc']
    )
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for d in [MODELS_DIR, HIST_DIR, FIGS_DIR]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("train_axialpolcap_h5.py")
    print("=" * 60)

    # --- Load data ---
    print("\nLoading training data ...")
    X_all, y_all = load_h5_all(TRAIN_H5)

    # Split train data 80 / 10 / 10  (stratified by polarity)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
    )
    X_mon, X_test, y_mon, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    print(f"  Train : {X_train.shape}  "
          f"(pos={int(y_train.sum())}, neg={int((y_train==0).sum())})")
    print(f"  Monitor: {X_mon.shape}  "
          f"(pos={int(y_mon.sum())}, neg={int((y_mon==0).sum())})")
    print(f"  Test  : {X_test.shape}  "
          f"(pos={int(y_test.sum())}, neg={int((y_test==0).sum())})")

    print("\nLoading separate validation data ...")
    X_val, y_val = load_h5_all(VAL_H5)

    # Normalize (data already max-normalized; this is a safety pass)
    X_train = norm(X_train)
    X_mon   = norm(X_mon)
    X_test  = norm(X_test)
    X_val   = norm(X_val)

    # One-hot labels for the classifier output
    y_train_cat = to_categorical(y_train, 2)
    y_mon_cat   = to_categorical(y_mon,   2)
    y_test_cat  = to_categorical(y_test,  2)
    y_val_cat   = to_categorical(y_val,   2)

    print(f"\nTrain: {X_train.shape} | Monitor: {X_mon.shape} | "
          f"Test: {X_test.shape} | Val (separate): {X_val.shape}")

    # --- Build model ---
    model = build_polarPicker()
    model.summary()

    # --- Callbacks ---
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"PolarPicker_h5_{timestamp}.keras")

    lr_logger = LRLogger()

    callbacks = [
        # ReduceLR fires first (patience=3), then EarlyStopping (patience=10)
        # so the LR gets a chance to drop before training stops
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            model_path, save_best_only=True
        ),
        lr_logger,
        WallClockTimeout(max_batch_sec=300),
    ]

    # --- Train ---
    # X_mon is the 10% monitoring split from train data (for early stopping / LR)
    # X_val is the separate held-out validation set — evaluated AFTER training only
    print("\nTraining ...")
    history = None
    try:
        history = model.fit(
            X_train,
            [X_train, y_train_cat],
            validation_data=(X_mon, [X_mon, y_mon_cat]),
            epochs=100,
            batch_size=256,
            callbacks=callbacks,
            verbose=1
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted. Loading best saved checkpoint ...")
        model = keras.models.load_model(model_path)

    print(f"\nBest model saved: {model_path}")

    # --- Save history (skipped if interrupted before first epoch) ---
    if history is not None:
        history_dict = {k: [float(v) for v in vals]
                        for k, vals in history.history.items()}

        experiment_log = {
            "history":        history_dict,
            "train_h5":       TRAIN_H5,
            "val_h5":         VAL_H5,
            "learning_rate":  0.001,
            "batch_size":     256,
            "dropout_rate":   0.3,
            "loss_weights":   [1, 200],
            "epochs_trained": len(history_dict["loss"])
        }

        history_path = os.path.join(HIST_DIR, f"history_h5_{timestamp}.json")
        with open(history_path, "w") as f:
            json.dump(experiment_log, f, indent=4)
        print(f"Training history saved: {history_path}")

        # --- Training dashboard (loss / accuracy / MSE / LR) ---
        plot_training_dashboard(
            history_dict, lr_logger.lrs,
            os.path.join(FIGS_DIR, f"dashboard_h5_{timestamp}.png")
        )
        print(f"Training dashboard saved.")
    else:
        print("Training was interrupted — history/dashboard skipped.")

    def evaluate_split(X, y, label, suffix):
        """Run prediction, save report + confmat + ROC + PR + confidence hist."""
        print(f"\nEvaluating on {label} ...")
        y_pred_raw = model.predict(X, batch_size=256)
        y_pred     = y_pred_raw[1]          # (N, 2) probabilities
        y_prob_pos = y_pred[:, 1]           # P(positive)
        pred_idx   = np.argmax(y_pred, axis=1)

        classes = ["Negative", "Positive"]

        # Classification report
        report = classification_report(y, pred_idx, target_names=classes, digits=4)
        with open(os.path.join(FIGS_DIR,
                  f"report_h5_{suffix}_{timestamp}.txt"), "w") as f:
            f.write(f"=== {label} ===\n" + report)
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y, pred_idx)
        plot_confusion_matrix(
            cm, classes,
            os.path.join(FIGS_DIR, f"confmat_h5_{suffix}_{timestamp}.png")
        )

        # ROC curve
        fpr, tpr, _ = roc_curve(y, y_prob_pos)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC — {label}"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGS_DIR,
                    f"roc_h5_{suffix}_{timestamp}.png"), dpi=150)
        plt.close()

        # Precision-Recall curve
        plot_pr_curve(
            y, y_prob_pos, label,
            os.path.join(FIGS_DIR, f"pr_curve_h5_{suffix}_{timestamp}.png")
        )

        # Confidence histogram
        plot_confidence_histogram(
            y, y_prob_pos, label,
            os.path.join(FIGS_DIR, f"conf_hist_h5_{suffix}_{timestamp}.png")
        )

        print(f"  ROC-AUC: {roc_auc:.4f}")
        return roc_auc

    # Internal test split (10% of train data — same augmentation distribution)
    auc_test = evaluate_split(X_test, y_test, "Internal test set (10% of train)", "test")

    # Separate validation set (independent real waveforms from A_wave_val.h5)
    auc_val  = evaluate_split(X_val, y_val, "Separate val (A_wave_val.h5)", "val")

    # Per-station accuracy bar chart on separate val set
    plot_per_station_accuracy(
        model, VAL_H5,
        os.path.join(FIGS_DIR, f"station_acc_h5_{timestamp}.png")
    )
    print("Per-station accuracy chart saved.")

    # Waveform reconstruction examples (8 random samples from test set)
    plot_reconstructions(
        model, X_test, y_test,
        os.path.join(FIGS_DIR, f"reconstruction_h5_{timestamp}.png")
    )
    print("Reconstruction examples saved.")

    print(f"\nSummary:")
    print(f"  Internal test AUC : {auc_test:.4f}")
    print(f"  Separate val  AUC : {auc_val:.4f}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
