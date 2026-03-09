import os
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)  # fix protobuf crash on macOS
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
data_dir = "/Users/mcZhang/Documents/GitHub/FM_ML/02-data"
model_path = "/Users/mcZhang/Documents/GitHub/FM_ML/06-models/PolarCAP.h5"  # <- adjust if needed

# -----------------------------
# Load data
# -----------------------------
X_test = np.load(os.path.join(data_dir, "test_all_timeseries.npy"))
Y_test = np.load(os.path.join(data_dir, "test_all_polarities.npy"))
stations_path = os.path.join(data_dir, "test_all_station_ids.npy")
stations = np.load(stations_path) if os.path.exists(stations_path) else None

print("✅ Loaded test data:")
print("X_test:", X_test.shape, "Y_test:", Y_test.shape)
print("Class balance:", dict(zip(*np.unique(Y_test, return_counts=True))))


# -----------------------------
# Normalization (safer epsilon)
# -----------------------------
def norm(X, eps=1e-8):
    maxi = np.max(np.abs(X), axis=1, keepdims=True)
    maxi = np.maximum(maxi, eps)
    return X / maxi

Xn = norm(X_test)

# -----------------------------
# Load model (compat fix)
# -----------------------------
# Keras 3 may fail to deserialize legacy metrics; bypass with compile=False
assert os.path.exists(model_path), f"❌ Model file not found: {model_path}"
model = tf.keras.models.load_model(model_path, compile=False)
print("✅ Model loaded (compile=False).")

# -----------------------------
# Predict
# -----------------------------
# Your model has two outputs: [decoder_reconstruction, class_probs]
y_pred_raw = model.predict(Xn, batch_size=512, verbose=1)
if isinstance(y_pred_raw, list) or isinstance(y_pred_raw, tuple):
    probs = y_pred_raw[1]  # (N,2) softmax
else:
    probs = y_pred_raw     # in case the model was saved with only the classifier head
    if probs.ndim == 1:
        # make it (N,2) if it's a single sigmoid output (unlikely for this model)
        probs = np.stack([1 - probs, probs], axis=1)

y_hat = np.argmax(probs, axis=1)

# -----------------------------
# Global metrics
# -----------------------------
acc = accuracy_score(Y_test, y_hat)
prec, rec, f1, _ = precision_recall_fscore_support(Y_test, y_hat, average="binary")
cm = confusion_matrix(Y_test, y_hat)

print(f"\n🔎 Overall Accuracy: {acc:.4f}")
print(f"Precision (Positive): {prec:.4f}")
print(f"Recall (Positive):    {rec:.4f}")
print(f"F1 (Positive):        {f1:.4f}")

print("\n📊 Classification report:\n",
      classification_report(Y_test, y_hat, target_names=["Negative", "Positive"]))

# -----------------------------
# Confusion matrix plot
# -----------------------------
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest')
plt.title(f"Confusion Matrix (Acc={acc:.3f})")
plt.colorbar()
tick_labels = ["Negative", "Positive"]
plt.xticks([0,1], [f"Pred {t}" for t in tick_labels])
plt.yticks([0,1], [f"True {t}" for t in tick_labels])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.tight_layout()
plt.show()

# -----------------------------
# Per-station breakdown (optional)
# -----------------------------
if stations is not None:
    print("\n🏷️ Per-station metrics:")
    uniq = np.unique(stations)
    rows = []
    for st in uniq:
        idx = (stations == st)
        if idx.sum() == 0:
            continue
        y_true_s = Y_test[idx]
        y_hat_s = y_hat[idx]
        acc_s = accuracy_score(y_true_s, y_hat_s)
        prec_s, rec_s, f1_s, _ = precision_recall_fscore_support(y_true_s, y_hat_s, average="binary")
        rows.append((st, idx.sum(), acc_s, prec_s, rec_s, f1_s))
        print(f"{st:>4s} | N={idx.sum():5d} | Acc={acc_s:.3f} | P={prec_s:.3f} R={rec_s:.3f} F1={f1_s:.3f}")

    # Save a CSV summary if you like
    out_csv = os.path.join(data_dir, "polarcap_per_station_metrics.csv")
    with open(out_csv, "w") as f:
        f.write("station,N,accuracy,precision,recall,f1\n")
        for st, n, a, p, r, f1s in rows:
            f.write(f"{st},{n},{a:.6f},{p:.6f},{r:.6f},{f1s:.6f}\n")
    print(f"\n💾 Saved per-station summary to: {out_csv}")

# -----------------------------
# (Optional) threshold tuning
# -----------------------------
# If you ever want to tune a decision threshold instead of argmax, you can
# use probs[:,1] (probability of Positive) and sweep thresholds to maximize F1.
