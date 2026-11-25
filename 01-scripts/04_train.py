"""Step 4: train one model with one of the four training strategies and
evaluate it per station on the common test sets (Figs 4-7, S3).

Strategies
  scratch   : AxialPolCap architecture, random init, MSE + Huber(0.5) loss 1:200
  probe     : PolarCAP pretrained, encoder frozen, new Dense(2) head trained
  finetune  : PolarCAP pretrained, all layers trainable, new Dense(2) head
  freezecls : PolarCAP pretrained, own softmax head frozen, encoder retrained

Protocol (identical for every strategy)
  train on <data>/train_*_all.npy, early-stop on <data>/val_*_all.npy,
  max 40 epochs, restore best weights.
  Per-strategy optimizer settings follow the original recipes
  (train_axialpolcap.py for scratch, transfer_learning.py for PolarCAP).

Evaluation sets (per station, no split prefix)
  Val          : common test set of Figs 4-5 (original SNR, no time shift)
  TMSF_Val_001 : sigma = 0.01 s time-shifted test set (Fig. 6)
  TMSF_Val     : sigma = 0.02 s time-shifted test set (Fig. 6)

Usage (from FM_ML/):
  python 01-scripts/04_train.py <strategy> <data_dir_name> [seed] [holdout]
  e.g. ... scratch TMSF_Tra_000
       ... scratch TMSF_Tra_000 0 ID1    (LOSO: train on the other six stations)
"""
import json, os, sys, time
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential

strategy, data_name = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
holdout = sys.argv[4] if len(sys.argv) > 4 else None   # LOSO: station left out
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRA = f"{ROOT}/02-data/K_aug/{data_name}"
# Test sets (Val, TMSF_Val_001, TMSF_Val) built by 03_build_datasets.py --mode test
EVAL_ROOT = os.environ.get("AXIAL_EVAL_ROOT", f"{ROOT}/02-data/K_aug")
EVAL_SETS = ["Val", "TMSF_Val_001", "TMSF_Val"]
# Pretrained PolarCAP weights (Chakraborty et al. 2022), not redistributed here
PCAP = os.environ.get("POLARCAP_H5", f"{ROOT}/06-models/PolarCAP.h5")
OUT = f"{ROOT}/06-models/retrain_20261122"
os.makedirs(OUT, exist_ok=True)
tag = f"{data_name}_{strategy}_s{seed}" + (f"_loso{holdout}" if holdout else "")
STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]
keras.utils.set_random_seed(seed)


def norm(X):
    X = X.astype("float32")
    if X.ndim == 2:
        X = X[..., None]
    m = np.max(np.abs(X), axis=1, keepdims=True)
    m[m == 0] = 1
    return X / m


oh = lambda y: keras.utils.to_categorical(y, 2)
def load_split(sp):
    if holdout is None:
        return (norm(np.load(f"{TRA}/{sp}_timeseries_all.npy")),
                np.load(f"{TRA}/{sp}_polarities_all.npy").astype(int).ravel())
    keep = [s for s in STATIONS if s != holdout]
    X = np.concatenate([norm(np.load(f"{TRA}/{sp}_timeseries_{s}.npy")) for s in keep])
    y = np.concatenate([np.load(f"{TRA}/{sp}_polarities_{s}.npy").astype(int).ravel()
                        for s in keep])
    return X, y


Xtr, ytr = load_split("train")
Xva, yva = load_split("val")
print(f"[{tag}] train {len(ytr)} val {len(yva)}", flush=True)


def build_scratch(drop_rate=0.3, lr=1e-3):
    enc = Sequential([
        layers.Input((64, 1)),
        layers.Conv1D(32, 32, activation="relu", padding="same"),
        layers.Dropout(drop_rate), layers.BatchNormalization(),
        layers.MaxPooling1D(2, padding="same"),
        layers.Conv1D(8, 16, activation="relu", padding="same"),
        layers.BatchNormalization(), layers.MaxPooling1D(2, padding="same")])
    dec = Sequential([
        layers.Input((16, 8)),
        layers.Conv1D(8, 16, activation="tanh", padding="same"),
        layers.BatchNormalization(), layers.UpSampling1D(2),
        layers.Conv1D(32, 32, activation="relu", padding="same"),
        layers.BatchNormalization(), layers.UpSampling1D(2),
        layers.Conv1D(1, 64, padding="same", activation="tanh")])
    cls = Sequential([layers.Input((16, 8)), layers.Flatten(),
                      layers.Dense(2, activation="softmax")])
    x = layers.Input((64, 1))
    e = enc(x)
    m = Model(x, [dec(e), cls(e)])
    m.compile(optimizer=keras.optimizers.Adam(lr),
              loss=["mse", keras.losses.Huber(delta=0.5)],
              loss_weights=[1, 200], metrics=[[], ["accuracy"]])
    return m


def build_polarcap(mode, lr):
    base = keras.models.load_model(PCAP, compile=False)
    if mode == "freezecls":
        head = next(l for l in base.layers if isinstance(l, layers.Dense))
        m = Model(base.input, [base.outputs[0], base.outputs[1]])
        for l in m.layers:
            l.trainable = l is not head
    else:
        flat = next(l for l in base.layers if isinstance(l, layers.Flatten))
        new = layers.Dense(2, activation="softmax", name="cls_new")(flat.output)
        m = Model(base.input, [base.outputs[0], new])
        for l in m.layers:
            l.trainable = (mode == "finetune")
        m.get_layer("cls_new").trainable = True
    m.compile(optimizer=keras.optimizers.Adam(lr),
              loss=["mse", "categorical_crossentropy"], loss_weights=[0.0, 1.0],
              metrics=[[], ["accuracy"]])
    return m


if strategy == "scratch":
    model, batch = build_scratch(), 256
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1,
                                             patience=8, min_lr=1e-6)]
else:
    lr = {"probe": 3e-4, "finetune": 1e-4, "freezecls": 1e-4}[strategy]
    model, batch = build_polarcap(strategy, lr), 512
    cbs = [keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=2, min_lr=1e-6)]
    # early stopping on validation polarity accuracy (as in transfer_learning.py)


class ValAcc(keras.callbacks.Callback):
    """Per-epoch validation polarity accuracy; restore best weights at the end."""
    def __init__(self, patience):
        super().__init__(); self.patience = patience
        self.best, self.wait, self.best_w, self.hist = -1, 0, None, []

    def on_epoch_end(self, ep, logs=None):
        p = self.model.predict(Xva, batch_size=4096, verbose=0)[1]
        a = float((p.argmax(1) == yva).mean()); self.hist.append(a)
        print(f"[{tag}] ep{ep+1:02d} val_acc={a:.4f}", flush=True)
        if a > self.best:
            self.best, self.wait, self.best_w = a, 0, self.model.get_weights()
        else:
            self.wait += 1
            if strategy != "scratch" and self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if strategy != "scratch" and self.best_w is not None:
            self.model.set_weights(self.best_w)


va_cb = ValAcc(patience=5)
t0 = time.time()
model.fit(Xtr, [Xtr, oh(ytr)], validation_data=(Xva, [Xva, oh(yva)]),
          epochs=40, batch_size=batch, shuffle=True, verbose=0,
          callbacks=cbs + [va_cb])
train_s = time.time() - t0
model.save(f"{OUT}/{tag}.keras")

res = {"strategy": strategy, "data": data_name, "seed": seed, "holdout": holdout,
       "epochs_run": len(va_cb.hist), "val_acc_hist": va_cb.hist,
       "n_trainable": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
       "train_seconds": train_s, "eval": {}}
for es in EVAL_SETS:
    accs, n_tot, n_ok = {}, 0, 0
    for st in STATIONS:
        X = norm(np.load(f"{EVAL_ROOT}/{es}/timeseries_{st}.npy"))
        y = np.load(f"{EVAL_ROOT}/{es}/polarities_{st}.npy").astype(int).ravel()
        if y.min() < 0:
            y = (y > 0).astype(int)
        p = model.predict(X, batch_size=4096, verbose=0)[1].argmax(1)
        accs[st] = float((p == y).mean()); n_tot += len(y); n_ok += int((p == y).sum())
    accs["station_mean"] = float(np.mean([accs[s] for s in STATIONS]))
    accs["pooled"] = n_ok / n_tot
    res["eval"][es] = accs
    print(f"[{tag}] {es}: " + " ".join(f"{s}={accs[s]*100:.2f}" for s in STATIONS)
          + f" mean={accs['station_mean']*100:.2f}", flush=True)
json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=1)
print(f"done {tag} {train_s:.0f}s epochs={len(va_cb.hist)}", flush=True)
