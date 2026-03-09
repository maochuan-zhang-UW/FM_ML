import os
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)

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
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score
)

import matplotlib.pyplot as plt
import pandas as pd


# ================================================================
# Helper: Normalize waveform
# ================================================================
def norm(X):
    max_val = np.max(abs(X), axis=1, keepdims=True)
    max_val[max_val == 0] = 1
    return X / max_val


# ================================================================
# Build AE + classifier with named outputs
# ================================================================
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
    ], name="encoder")

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
    ], name="decoder_module")

    # Classifier head
    classifier = Sequential([
        Input(shape=(16,8)),
        Flatten(),
        Dense(2, activation='softmax')
    ], name="classifier_module")

    # Connect model
    X = Input(shape=(64,1))
    enc = encoder(X)

    dec = decoder(enc)
    dec = keras.layers.Lambda(lambda x: x, name="decoder")(dec)

    p = classifier(enc)
    p = keras.layers.Lambda(lambda x: x, name="classifier")(p)

    model = Model(inputs=X, outputs=[dec, p], name="PolarPicker")

    hub = keras.losses.Huber(delta=0.5)

    model.compile(
        optimizer=keras.optimizers.Adam(learn_rate),
        loss={'decoder': 'mse', 'classifier': hub},
        loss_weights={'decoder': 1, 'classifier': 200},
        metrics={'decoder': ['mse'], 'classifier': ['acc']}
    )

    return model


# ================================================================
# Plot training curves
# ================================================================
def save_training_curves(history, station):
    plt.figure(figsize=(7,5))
    plt.plot(history.history['classifier_acc'], label='Train Acc')
    plt.plot(history.history['val_classifier_acc'], label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training Accuracy Curve (LOSO {station})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"./03-figs/LOSO/Curve_LOSO_{station}.png", dpi=150)
    plt.close()


# ================================================================
# Confusion matrix plot helper
# ================================================================
def plot_confusion_matrix(cm, classes, fname):
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted"
    )

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i,j],
                    ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


# ================================================================
# LOSO Training
# ================================================================
data_dir = "/Users/mcZhang/Documents/GitHub/FM_ML/02-data/K_aug/STEP010"
#stations = ['AS1','AS2','CC1','EC1','EC2','EC3','ID1']
stations = ['EC2','EC3','ID1']

os.makedirs("./06-models/LOSO_010", exist_ok=True)
os.makedirs("./03-figs/LOSO_010", exist_ok=True)

results = []   # Store metrics into CSV


for test_station in stations:

    print(f"\n==============================================")
    print(f"🚀 LOSO Training: Leave out {test_station}")
    print(f"==============================================\n")

    # ------------------------------------------------------------
    # Load test data for this station
    # ------------------------------------------------------------
    X_test = np.load(os.path.join(data_dir, f"test_timeseries_{test_station}.npy"))
    y_test = np.load(os.path.join(data_dir, f"test_polarities_{test_station}.npy"))

    # ------------------------------------------------------------
    # Load training/validation data from other stations
    # ------------------------------------------------------------
    X_train_list, y_train_list = [], []
    X_val_list,   y_val_list   = [], []

    for st in stations:
        if st == test_station:
            continue

        X_train_list.append(np.load(os.path.join(data_dir, f"train_timeseries_{st}.npy")))
        y_train_list.append(np.load(os.path.join(data_dir, f"train_polarities_{st}.npy")))

        X_val_list.append(np.load(os.path.join(data_dir, f"val_timeseries_{st}.npy")))
        y_val_list.append(np.load(os.path.join(data_dir, f"val_polarities_{st}.npy")))

    X_train = np.concatenate(X_train_list, axis=0)
    X_val   = np.concatenate(X_val_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_val   = np.concatenate(y_val_list, axis=0)

    print(f"Training shape: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------
    X_train = norm(X_train)
    X_val   = norm(X_val)
    X_test  = norm(X_test)

    y_train_cat = to_categorical(y_train, 2)
    y_val_cat   = to_categorical(y_val, 2)

    # ------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------
    model = build_polarPicker()

    ckpt_path = f"./06-models/LOSO_010/PolarPicker_LOSO_{test_station}.keras"

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6),
        keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True)
    ]

    history = model.fit(
        X_train, {"decoder": X_train, "classifier": y_train_cat},
        validation_data=(X_val, {"decoder": X_val, "classifier": y_val_cat}),
        epochs=40,
        batch_size=256,
        callbacks=callbacks,
        verbose=1
    )

    save_training_curves(history, test_station)

    # ------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------
    y_pred_prob = model.predict(X_test)[1]
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob[:,1])

    results.append([test_station, acc, prec, rec, f1, auc])

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm, ["Neg","Pos"],
        f"./03-figs/LOSO/ConfMat_LOSO010_{test_station}.png"
    )

    print(f"Finished {test_station}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")


# ================================================================
# Store results into CSV
# ================================================================
df = pd.DataFrame(
    results,
    columns=["Station","Accuracy","Precision","Recall","F1","AUC"]
)
df.to_csv("./06-models/LOSO010/LOSO_results.csv", index=False)

print("\n🎉 ALL DONE! Results saved to:")
print("📄 ./06-models/LOSO010/LOSO_results.csv")
