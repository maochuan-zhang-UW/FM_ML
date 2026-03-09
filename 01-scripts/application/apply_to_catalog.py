import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)

from tensorflow import keras
from scipy.io import loadmat, savemat
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# Normalize (same as training)
# ============================================================
def norm(X):
    max_val = np.max(np.abs(X), axis=1, keepdims=True)
    max_val[max_val == 0] = 1
    return X / max_val

# ============================================================
# LOAD MODEL
# ============================================================
model_path = "./06-models/PolarPicker_unified_TS.keras"
model = keras.models.load_model(model_path)
print("✅ Loaded model:", model_path)

# ============================================================
# LOAD MATLAB DATA
# ============================================================
mat_file = "/Users/mcZhang/Documents/GitHub/FM_ML/02-data/A_wave_2015_2022CC.mat"
data = loadmat(mat_file, struct_as_record=False, squeeze_me=True)
Felix = data["Felix"]
Felix = Felix[14999:]


if not isinstance(Felix, np.ndarray):
    Felix = np.array([Felix])

stations = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]
classes = ["Negative (-1)", "Positive (1)"]

all_true = []
all_pred = []
skipped_no_polarity = 0

Felix_new = []

print(f"\n🔍 Processing {len(Felix)} events...\n")

# ============================================================
# PROCESS EVENTS
# ============================================================
for i, event in enumerate(Felix):

    waveforms = []
    ground_truth = []
    used_stations = []

    # Copy original event fields
    event_dict = {field: getattr(event, field) for field in event._fieldnames}

    for sta in stations:
        w_key = f"W_{sta}"
        po_key = f"Po_{sta}"

        if not hasattr(event, w_key) or not hasattr(event, po_key):
            continue

        waveform = getattr(event, w_key)
        if waveform is None:
            continue

        waveform = np.asarray(waveform).flatten()
        if waveform.size != 64:
            continue

        polarity_gt = int(np.asarray(getattr(event, po_key)).flatten()[0])

        waveforms.append(waveform)
        ground_truth.append(polarity_gt)
        used_stations.append(sta)

    if len(waveforms) == 0:
        continue

    X = np.array(waveforms, dtype=np.float32)
    Xn = norm(X)

    # ========================================================
    # Predict
    # ========================================================
    y_raw = model.predict(Xn, verbose=0)
    y_prob = y_raw[1] if isinstance(y_raw, (list, tuple)) else y_raw

    y_bin = np.argmax(y_prob, axis=1)
    y_pred = np.where(y_bin == 0, -1, 1)

    # ========================================================
    # Confidence & Entropy
    # ========================================================
    confidence = np.max(y_prob, axis=1)
    entropy = -np.sum(y_prob * np.log(y_prob + 1e-12), axis=1)

    # ========================================================
    # Write back to Po field
    # Format: [GT, Prediction, Confidence, Entropy]
    # ========================================================
    for j, sta in enumerate(used_stations):
        po_key = f"Po_{sta}"

        event_dict[po_key] = np.array(
            [
                ground_truth[j],     # Ground truth polarity
                y_pred[j],           # DL prediction
                confidence[j],       # 3rd = confidence
                entropy[j]           # 4th = entropy
            ],
            dtype=np.float32
        )

        if ground_truth[j] != 0:
            all_true.append(ground_truth[j])
            all_pred.append(y_pred[j])
        else:
            skipped_no_polarity += 1

    Felix_new.append(event_dict)

    if (i + 1) % 1000 == 0:
        print(f"Processed {i+1}/{len(Felix)} events")

# ============================================================
# METRICS
# ============================================================
all_true = np.array(all_true)
all_pred = np.array(all_pred)

print(f"\n⚠️ Skipped {skipped_no_polarity} zero-polarity picks")

if len(all_true) > 0:
    acc = accuracy_score(all_true, all_pred)
    print(f"\nOverall Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(all_true, all_pred,
                                labels=[-1, 1],
                                target_names=classes,
                                digits=4,
                                zero_division=0))

    cm = confusion_matrix(all_true, all_pred, labels=[-1, 1])
    print("\nConfusion Matrix:")
    print(cm)
else:
    print("❌ No valid polarities found.")

# ============================================================
# SAVE
# ============================================================
output_file = (
    "/Users/mcZhang/Documents/GitHub/FM_ML/02-data/"
    "A_wave_2015_2022CC_with_predictions_Final_confidiencev2.mat"
)

savemat(output_file, {"Felix": Felix_new}, format="5", do_compression=True)

print(f"\n✅ Saved {len(Felix_new)} events to:")
print(output_file)

print("\nPo field format now = [GT, Prediction, Confidence, Entropy]")
