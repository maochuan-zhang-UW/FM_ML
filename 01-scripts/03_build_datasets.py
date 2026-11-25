"""Step 3: crop, time-shift, normalize and split augmented waveforms into the
NumPy datasets used for training and testing.

For every waveform (200 samples, P pick at sample 100):
  1. draw an integer P-arrival shift ~ N(0, sigma) samples, rounded and
     clipped to +/-4 samples (1 sample = 0.01 s); sigma = 0 means no shift,
  2. crop the 64 samples centred on the shifted pick,
  3. max-normalize.

--mode trainvaltest : split each station 80/10/10 (stratified on polarity,
                      random_state 42) and write
                      {train,val,test}_{timeseries,polarities}_{STA|all}.npy
--mode test         : no split; write {timeseries,polarities}_{STA|all}.npy

Polarity labels: 0 = negative, 1 = positive (from the manual polarity).

Datasets of the paper (directory names as used by 04_train.py)
  TMSF_Tra_000  train_org.h5,     trainvaltest, sigma 0   Figs 4, 5, 6
  TMSF_Tra_001  train_org.h5,     trainvaltest, sigma 1   Fig. 7, final model
  TMSF_Tra_002s train_org.h5,     trainvaltest, sigma 2   Fig. 7
  STEP          train_step.h5,    trainvaltest, sigma 0   Fig. 5 (high SNR)
  STEP010       train_step010.h5, trainvaltest, sigma 0   Fig. 5 (low SNR)
  Val           test.h5,          test,         sigma 0   common test set
  TMSF_Val_001  test.h5,          test,         sigma 1   0.01 s shifted test set
  TMSF_Val      test.h5,          test,         sigma 2   0.02 s shifted test set

--input also accepts a directory of the archived MATLAB files
({STA}{suffix}, e.g. AS1_T_add.mat), which rebuilds the paper's datasets from
the archived augmentation; with sigma 0 the result is identical to them.

Usage (from the repository root):
  python 01-scripts/03_build_datasets.py --input 02-data/augmented/train_org.h5 \
      --mode trainvaltest --sigma-samples 1 --seed 1 --out 02-data/K_aug/TMSF_Tra_001
"""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]
MAX_SHIFT_SAMPLES = 4
HALF = 32


def load_station(src, st, suffix):
    """Return (waveforms (N, 200), manual polarity (N,) in -1/+1)."""
    if Path(src).is_dir():
        mat = loadmat(os.path.join(src, st + suffix))
        data = mat[[k for k in mat if not k.startswith("__")][0]].squeeze()
        w = np.stack([np.asarray(d[f"W_{st}"], float).ravel() for d in data])
        y = np.array([int(np.asarray(d[f"Man_{st}"]).squeeze()) for d in data])
        return w, y
    with h5py.File(src, "r") as h5:
        return h5[f"{st}/waveform"][()], h5[f"{st}/manual_polarity"][()].astype(int)


def crop(w, sigma, rng):
    n = len(w)
    if sigma > 0:
        s = np.clip(np.round(rng.normal(0.0, sigma, n)), -MAX_SHIFT_SAMPLES, MAX_SHIFT_SAMPLES).astype(int)
    else:
        s = np.zeros(n, int)
    c = w.shape[1] // 2 + s
    idx = c[:, None] + np.arange(-HALF, HALF)[None, :]
    x = np.take_along_axis(w, idx, axis=1)
    m = np.abs(x).max(axis=1, keepdims=True)
    m[m == 0] = 1.0
    return (x / m).astype(np.float32)[..., None], s


def save(out, name, X, y):
    np.save(os.path.join(out, f"{name.replace('{}', 'timeseries')}.npy"), X)
    np.save(os.path.join(out, f"{name.replace('{}', 'polarities')}.npy"), y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="augmented .h5 from 02_augment.py, or a directory of archived .mat files")
    ap.add_argument("--suffix", default="_T_add.mat", help="archived .mat only: file name after the station code")
    ap.add_argument("--mode", choices=["trainvaltest", "test"], required=True)
    ap.add_argument("--sigma-samples", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0, help="seed for the time shifts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    merged = {}
    for st in STATIONS:
        w, man = load_station(a.input, st, a.suffix)
        X, sh = crop(w, a.sigma_samples, rng)
        y = (man == 1).astype(np.int32)
        if a.mode == "test":
            parts = {"": (X, y)}
        else:
            Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=42, stratify=ytmp)
            parts = {"train_": (Xtr, ytr), "val_": (Xva, yva), "test_": (Xte, yte)}
        for p, (Xs, ys) in parts.items():
            save(a.out, f"{p}{{}}_{st}", Xs, ys)
            merged.setdefault(p, []).append((Xs, ys))
        print(f"{st}: {len(y)} waveforms -> " + "/".join(str(len(v[1])) for v in parts.values())
              + f"; shift std {sh.std():.2f} samples")
    for p, lst in merged.items():
        Xa, ya = np.vstack([x for x, _ in lst]), np.hstack([y for _, y in lst])
        if p:   # training sets are shuffled after merging; test sets keep station order
            Xa, ya = shuffle(Xa, ya, random_state=42)
        save(a.out, f"{p}{{}}_all", Xa, ya)
        print(f"{p or 'test '}all: {len(ya)} waveforms")


if __name__ == "__main__":
    main()
