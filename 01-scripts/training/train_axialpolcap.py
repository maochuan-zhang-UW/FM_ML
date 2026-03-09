import os
import json
import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, BatchNormalization,
    MaxPooling1D, Flatten, UpSampling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

tf.config.run_functions_eagerly(True)

# =============================================================================
# Helper: normalize waveform
# =============================================================================
def norm(X):
    max_val = np.max(abs(X), axis=1, keepdims=True)
    max_val[max_val == 0] = 1
    return X / max_val


# =============================================================================
# Build Model
# =============================================================================
def build_polarPicker(drop_rate=0.3, learn_rate=0.001):

    # Encoder
    encoder = Sequential([
        Input(shape=(64,1)),
        Conv1D(32, 32, activation='relu', padding='same'),
        Dropout(drop_rate),
        BatchNormalization(),
        MaxPooling1D(2, padding='same'),

        Conv1D(8, 16, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2, padding='same'),
    ])

    # Decoder
    decoder = Sequential([
        Input(shape=(16,8)),
        Conv1D(8, 16, activation='tanh', padding='same'),
        BatchNormalization(),
        UpSampling1D(2),

        Conv1D(32, 32, activation='relu', padding='same'),
        BatchNormalization(),
        UpSampling1D(2),

        Conv1D(1, 64, padding='same', activation='tanh'),
    ])

    # Classifier
    classifier = Sequential([
        Input(shape=(16,8)),
        Flatten(),
        Dense(2, activation='softmax')
    ])

    X = Input(shape=(64,1))
    enc = encoder(X)
    dec = decoder(enc)
    p = classifier(enc)

    model = Model(inputs=X, outputs=[dec, p])

    hub = keras.losses.Huber(delta=0.5)

    model.compile(
        optimizer=keras.optimizers.Adam(learn_rate),
        loss=['mse', hub],
        loss_weights=[1, 200],
        metrics=['mse', 'acc']
    )

    return model


# =============================================================================
# Plot confusion matrix
# =============================================================================
def plot_confusion_matrix(cm, classes, fname):
    fig, ax = plt.subplots(figsize=(5,5))
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

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i,j], ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


# =============================================================================
# LOAD DATA
# =============================================================================
data_dir = "/Users/mcZhang/Documents/GitHub/FM_ML/02-data/K_aug/TMSF_Tra_001"

X_all = np.load(os.path.join(data_dir, "timeseries_all.npy"))
y_all = np.load(os.path.join(data_dir, "polarities_all.npy"))

print("Loaded dataset:", X_all.shape, y_all.shape)

# =============================================================================
# TRAIN / VAL / TEST SPLIT
# =============================================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all,
    test_size=0.20,
    random_state=42,
    stratify=y_all
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

# Normalize
X_train = norm(X_train)
X_val   = norm(X_val)
X_test  = norm(X_test)

# One-hot
y_train_cat = to_categorical(y_train, 2)
y_val_cat   = to_categorical(y_val, 2)
y_test_cat  = to_categorical(y_test, 2)

# =============================================================================
# TRAIN MODEL
# =============================================================================
model = build_polarPicker()

os.makedirs("./06-models", exist_ok=True)
os.makedirs("./06-models/history", exist_ok=True)
os.makedirs("./03-figs", exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"./06-models/PolarPicker_{timestamp}.keras"

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=8,
    min_lr=1e-6
)

ckpt = keras.callbacks.ModelCheckpoint(
    model_path,
    save_best_only=True
)

history = model.fit(
    X_train,
    [X_train, y_train_cat],
    validation_data=(X_val, [X_val, y_val_cat]),
    epochs=40,
    batch_size=256,
    callbacks=[early_stop, reduce_lr, ckpt],
    verbose=1
)

print("✅ Best model saved at:", model_path)

# =============================================================================
# SAVE HISTORY + METADATA
# =============================================================================
history_dict = {k: [float(v) for v in val] for k, val in history.history.items()}

experiment_log = {
    "history": history_dict,
    "learning_rate": 0.001,
    "batch_size": 256,
    "dropout_rate": 0.3,
    "loss_weights": [1, 200],
    "epochs_trained": len(history_dict["loss"])
}

history_path = f"./06-models/history/history_{timestamp}.json"

with open(history_path, "w") as f:
    json.dump(experiment_log, f, indent=4)

print("✅ Training history saved at:", history_path)

# =============================================================================
# PLOT TRAINING CURVES
# =============================================================================
plt.figure()
plt.plot(history_dict['loss'])
plt.plot(history_dict['val_loss'])
plt.legend(['Train Loss','Val Loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"./03-figs/loss_curve_{timestamp}.png")
plt.close()

if 'classifier_acc' in history_dict:
    plt.figure()
    plt.plot(history_dict['classifier_acc'])
    plt.plot(history_dict['val_classifier_acc'])
    plt.legend(['Train Acc','Val Acc'])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f"./03-figs/accuracy_curve_{timestamp}.png")
    plt.close()

# =============================================================================
# TEST EVALUATION
# =============================================================================
y_pred = model.predict(X_test)[1]
pred_idx = np.argmax(y_pred, axis=1)
true_idx = y_test

classes = ["Negative", "Positive"]

# Classification report
report = classification_report(true_idx, pred_idx, target_names=classes, digits=4)

with open(f"./03-figs/classification_report_{timestamp}.txt", "w") as f:
    f.write(report)

print(report)

# Confusion matrix
cm = confusion_matrix(true_idx, pred_idx)
plot_confusion_matrix(cm, classes,
                      f"./03-figs/confmat_{timestamp}.png")

# ROC curve
fpr, tpr, _ = roc_curve(true_idx, y_pred[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig(f"./03-figs/roc_curve_{timestamp}.png")
plt.close()

print("🎉 Training + logging complete.")
