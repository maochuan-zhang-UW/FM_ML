# 10_finetune_polarcap_head.py
import os
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)  # fix protobuf crash on macOS
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------
# Paths & config
# -----------------------------
DATA_DIR = "/Users/mcZhang/Documents/GitHub/FM_ML/02-data/K_aug/TMSF_Tra_001"
X_PATH = os.path.join(DATA_DIR, "train_timeseries_all.npy")
Y_PATH = os.path.join(DATA_DIR, "train_polarities_all.npy")

MODEL_PATH = "/Users/mcZhang/Documents/GitHub/FM_ML/06-models/PolarCAP.h5"  # <- adjust if needed
#OUT_PATH = os.path.join(MODEL_PATH, "PolarCAP_finetuned_STEP.h5")  # legacy H5 to match your setup
OUT_DIR = "/Users/mcZhang/Documents/GitHub/FM_ML/06-models"
OUT_PATH = os.path.join(OUT_DIR, "PolarCAP_finetuned_TMSF_001.h5")

BATCH = 512
EPOCHS = 40
VAL_SIZE = 0.2          # keep some data to monitor overfitting
HELDOUT_SIZE = 0.0      # set to e.g. 0.1 if you want a final untouched set
SEED = 1337

# Run on CPU (avoids TF-metal protobuf crash on some macOS setups)
try:
    tf.config.set_visible_devices([], "GPU")
    print("✅ Running on CPU (GPU disabled).")
except Exception:
    pass

# -----------------------------
# Load data
# -----------------------------
X = np.load(X_PATH)   # (N, 64, 1)
y = np.load(Y_PATH).astype(int)  # (N,)

print(f"Loaded X={X.shape}, y={y.shape}. Class balance:",
      dict(zip(*np.unique(y, return_counts=True))))

# One-hot labels for CE
y_oh = keras.utils.to_categorical(y, num_classes=2)

# -----------------------------
# Train/Val/(optional Heldout) split
# -----------------------------
if HELDOUT_SIZE > 0:
    X_tmp, X_held, y_oh_tmp, y_oh_held, y_tmp, y_held = train_test_split(
        X, y_oh, y, test_size=HELDOUT_SIZE, stratify=y, random_state=SEED
    )
else:
    X_tmp, y_oh_tmp, y_tmp = X, y_oh, y
    X_held = y_oh_held = y_held = None

X_tr, X_va, ytr_oh, yva_oh, ytr, yva = train_test_split(
    X_tmp, y_oh_tmp, y_tmp, test_size=VAL_SIZE, stratify=y_tmp, random_state=SEED
)

print(f"Split: train={X_tr.shape[0]}, val={X_va.shape[0]}",
      (f", heldout={X_held.shape[0]}" if X_held is not None else ""))

# -----------------------------
# Normalization (per-trace max-abs)
# -----------------------------
def norm(Xin, eps=1e-8):
    m = np.max(np.abs(Xin), axis=1, keepdims=True)
    m = np.maximum(m, eps)
    return Xin / m

X_tr = norm(X_tr)
X_va = norm(X_va)
if X_held is not None:
    X_held = norm(X_held)

# -----------------------------
# Load base model
# -----------------------------
assert os.path.exists(MODEL_PATH), f"❌ Model not found: {MODEL_PATH}"
base = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Loaded base model (compile=False).")

# -----------------------------
# Build a new classifier head on top of the latent
# -----------------------------
# We will:
#  - keep the original decoder output (to preserve graph), but give it 0 weight
#  - replace the classifier head with a fresh Dense(2, softmax)

# Find the Flatten layer that feeds the old classifier
flatten_layer = None
for lyr in base.layers:
    if isinstance(lyr, layers.Flatten):
        flatten_layer = lyr
        break
if flatten_layer is None:
    raise RuntimeError("Could not find Flatten layer in the loaded model.")

latent = flatten_layer.output  # this was enc flattened (shape ~ 128)

# New classifier head
new_cls = layers.Dense(2, activation="softmax", name="cls_new")(latent)

# Keep original decoder as output[0] (usually something like 'conv1d' with 1 channel)
dec_out = base.outputs[0] if isinstance(base.outputs, (list, tuple)) else None
if dec_out is None:
    # If your saved model has only classifier, create a 2nd output stub:
    inputs = base.input
    model = Model(inputs, new_cls)
    outputs = [new_cls]
else:
    inputs = base.input
    model = Model(inputs, [dec_out, new_cls])
    outputs = [dec_out, new_cls]

# Freeze all layers except our new head
for lyr in model.layers:
    lyr.trainable = False
# The new Dense head is the last layer; make it trainable
model.get_layer("cls_new").trainable = True

# Compile: zero weight for decoder, CE for classifier
losses = []
loss_weights = []
metrics = []

if len(outputs) == 2:
    losses = ["mse", "categorical_crossentropy"]
    loss_weights = [0.0, 1.0]  # ignore decoder loss, train classifier only
    metrics = [[], [keras.metrics.CategoricalAccuracy(name="acc")]]
    y_tr_targets = [X_tr, ytr_oh]
    y_va_targets = [X_va, yva_oh]
else:
    losses = ["categorical_crossentropy"]
    loss_weights = [1.0]
    metrics = [[keras.metrics.CategoricalAccuracy(name="acc")]]
    y_tr_targets = [ytr_oh]
    y_va_targets = [yva_oh]

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss=losses,
    loss_weights=loss_weights,
    metrics=metrics,
)

model.summary()

# -----------------------------
# Train (only the new head)
# -----------------------------
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_cls_new_acc" if len(outputs)==2 else "val_acc",
        mode="max",                      # 👈 tell it to maximize accuracy
        patience=5, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
    )
]

history = model.fit(
    X_tr, y_tr_targets,
    validation_data=(X_va, y_va_targets),
    epochs=EPOCHS,
    batch_size=BATCH,
    verbose=2,
    callbacks=callbacks
)

# -----------------------------
# Evaluate on val (and optional heldout)
# -----------------------------
def get_probs(m, X):
    out = m.predict(X, batch_size=BATCH, verbose=0)
    if isinstance(out, (list, tuple)):
        probs = out[1]
    else:
        probs = out
    if probs.ndim == 1:
        probs = np.stack([1-probs, probs], axis=1)
    return probs

probs_val = get_probs(model, X_va)
yhat_val = np.argmax(probs_val, axis=1)
print("\nValidation:")
print("  Acc:", accuracy_score(yva, yhat_val))
print(classification_report(yva, yhat_val, target_names=["Negative","Positive"]))

if X_held is not None:
    probs_held = get_probs(model, X_held)
    yhat_held = np.argmax(probs_held, axis=1)
    print("\nHELD-OUT (never used in training/val):")
    print("  Acc:", accuracy_score(y_held, yhat_held))
    print(classification_report(y_held, yhat_held, target_names=["Negative","Positive"]))

# -----------------------------
# Save fine-tuned model
# -----------------------------
model.save(OUT_PATH)  # saves as H5 because of the .h5 suffix
print(f"\n💾 Saved fine-tuned model to: {OUT_PATH}")
