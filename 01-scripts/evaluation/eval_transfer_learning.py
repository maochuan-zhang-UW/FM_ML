# 11_eval_finetuned_per_station.py
import os
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)  # fix protobuf crash on macOS
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
import csv

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
DATA_DIR = "/Users/mcZhang/Documents/GitHub/FM_ML/02-data/K_aug/TMSF_Val_001"
# Use the fine-tuned model you just trained (saved in your previous script).
MODEL_PATH = "/Users/mcZhang/Documents/GitHub/FM_ML/06-models/PolarCAP_finetuned_TMSF.h5"

#MODEL_PATH = "/Users/mcZhang/Documents/GitHub/FM_ML/06-models/PolarCAP_finetuned_STEP010.h5" #higher SNR data with fine-tune
STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]
BATCH_SIZE = 512
OUT_CSV = os.path.join(DATA_DIR, "finetuned_metrics_per_station.csv")

# Force CPU (avoids rare Metal/protobuf crashes on macOS)
try:
    tf.config.set_visible_devices([], "GPU")
    print("✅ Running on CPU (GPU disabled).")
except Exception as e:
    print("Note: could not change visible devices:", e)

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
def norm(X, eps=1e-8):
    """Per-trace max-abs normalization for (N,64,1)."""
    m = np.max(np.abs(X), axis=1, keepdims=True)
    m = np.maximum(m, eps)
    return X / m

def load_station(st):
    Xp = os.path.join(DATA_DIR, f"timeseries_{st}.npy")
    Yp = os.path.join(DATA_DIR, f"polarities_{st}.npy")
    assert os.path.exists(Xp), f"Missing: {Xp}"
    assert os.path.exists(Yp), f"Missing: {Yp}"
    X = np.load(Xp)
    y = np.load(Yp).astype(int)
    return X, y

def get_probs(model, Xn):
    """Return class probs (N,2) with probs[:,1] = Positive."""
    with tf.device("/CPU:0"):
        out = model.predict(Xn, batch_size=BATCH_SIZE, verbose=0)
    if isinstance(out, (list, tuple)):
        P = out[1]  # [decoder, classifier]
    else:
        P = out
    if P.ndim == 1:  # sigmoid fallback (unlikely for this model)
        P = np.stack([1.0 - P, P], axis=1)
    return P

def eval_metrics(y_true, probs):
    y_hat = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, y_hat)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs[:,1])
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_hat)
    tn, fp, fn, tp = cm.ravel()
    return acc, prec, rec, f1, auc, (tn, fp, fn, tp), classification_report(y_true, y_hat, target_names=["Negative","Positive"])

# -----------------------------------------------------------
# Load fine-tuned model
# -----------------------------------------------------------
assert os.path.exists(MODEL_PATH), f"❌ Model file not found: {MODEL_PATH}"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Loaded fine-tuned model:", MODEL_PATH)

# -----------------------------------------------------------
# Evaluate per station
# -----------------------------------------------------------
rows = []
all_X, all_y = [], []

for st in STATIONS:
    X, y = load_station(st)
    print(f"\n—— Evaluating {st}: X={X.shape}, y={y.shape}, balance={dict(zip(*np.unique(y, return_counts=True)))}")

    Xn = norm(X)
    probs = get_probs(model, Xn)

    acc, prec, rec, f1, auc, (tn, fp, fn, tp), report = eval_metrics(y, probs)

    print(f"{st}  N={len(y)}  Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    print(report)

    rows.append([st, len(y), acc, prec, rec, f1, auc, tn, fp, fn, tp])
    all_X.append(X)
    all_y.append(y)

# -----------------------------------------------------------
# Evaluate ALL combined
# -----------------------------------------------------------
X_all = norm(np.concatenate(all_X, axis=0))
y_all = np.concatenate(all_y, axis=0)
probs_all = get_probs(model, X_all)
acc, prec, rec, f1, auc, (tn, fp, fn, tp), report = eval_metrics(y_all, probs_all)

print("\n==== Combined (ALL stations) ====")
print(f"N={len(y_all)}  Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
print(report)

rows.append(["ALL", len(y_all), acc, prec, rec, f1, auc, tn, fp, fn, tp])

# -----------------------------------------------------------
# Save CSV summary
# -----------------------------------------------------------
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["station","N","accuracy","precision","recall","f1","auc","tn","fp","fn","tp"])
    w.writerows(rows)

print(f"\n💾 Saved per-station metrics to: {OUT_CSV}")
