import os
import csv
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)
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


# =============================================================================
#                   PATHS & BASIC SETTINGS
# =============================================================================

# Folder containing LOSO models
model_dir = "/Users/mcZhang/Documents/GitHub/FM_ML/06-models/LOSO_010"
#model_dir = "/Users/mcZhang/Documents/GitHub/FM_ML/06-models"

# Folder containing npy data
data_dir = "/Users/mcZhang/Documents/GitHub/FM_ML/02-data/K_aug/Val"

# Station list (and model tags)
stations = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]

classes = ["Negative", "Positive"]

# Output CSV
out_csv = os.path.join(model_dir, "LOSO_results_1to1.csv")

# Build expected model filenames in order
model_files = [f"PolarPicker_LOSO_{sta}.keras" for sta in stations]
#model_files = [f"model_{sta}.keras" for sta in stations]

print("Planned model–dataset pairs:")
for sta, mf in zip(stations, model_files):
    print(f"  Model {mf}  ->  station {sta}")

# To store all results (for CSV)
results = []

print("\n================ LOSO 1:1 EVALUATION ================\n")

for sta, model_file in zip(stations, model_files):
    model_path = os.path.join(model_dir, model_file)

    if not os.path.exists(model_path):
        print(f"⚠️  Model file missing: {model_path}, skipping.")
        continue

    print(f"\n############################################")
    print(f"## Evaluating model: {model_file} on station {sta}")
    print("############################################\n")

    # Load model (allow Lambda deserialization)
    model = keras.models.load_model(model_path, safe_mode=False)
    print("✅ Loaded model:", model_path)

    # Corresponding dataset for this station
    X_file = os.path.join(data_dir, f"timeseries_{sta}.npy")
    y_file = os.path.join(data_dir, f"polarities_{sta}.npy")

    if not (os.path.exists(X_file) and os.path.exists(y_file)):
        print(f"⚠️  Missing data files for station '{sta}' ({X_file} or {y_file}), skipping.")
        continue

    X = np.load(X_file)
    y = np.load(y_file)

    print(f"Data shape for station {sta}:", X.shape, y.shape)

    # Normalize (keeps consistency with training)
    Xn = norm(X)

    # Model prediction
    y_raw = model.predict(Xn, verbose=0)

    # Handle single-output vs multi-output model
    if isinstance(y_raw, (list, tuple)):
        # assume [decoder_out, classifier_out]
        y_pred_prob = y_raw[1]
    else:
        y_pred_prob = y_raw

    y_pred = np.argmax(y_pred_prob, axis=1)

    # Metrics
    acc = accuracy_score(y, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", pos_label=1, zero_division=0
    )

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 score : {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y, y_pred, target_names=classes, digits=4))

    cm = confusion_matrix(y, y_pred)
    print("Confusion matrix:\n", cm)

    # Store row for CSV
    results.append({
        "model": sta,            # model tag == station
        "station": sta,
        "n_samples": int(len(y)),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    })

# =============================================================================
#                     WRITE CSV SUMMARY
# =============================================================================
fieldnames = ["model", "station", "n_samples",
              "accuracy", "precision", "recall", "f1"]

with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print("\n===========================================")
print("LOSO 1:1 evaluation finished.")
print("Results saved to:", out_csv)
print("===========================================\n")

# Optional: pretty print a compact summary table
print("Model\tStation\tN\tAcc\tPrec\tRec\tF1")
for r in results:
    print(
        f"{r['model']}\t{r['station']}\t{r['n_samples']}\t"
        f"{r['accuracy']:.4f}\t{r['precision']:.4f}\t"
        f"{r['recall']:.4f}\t{r['f1']:.4f}"
    )
