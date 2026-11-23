"""Convert augmented MATLAB waveform sets to the NPY training sets of Figs 4-7.

Generalizes augmentation/L4_saveNPY_TMSF_001tran.py, which built TMSF_Tra_001.
Each per-station struct file holds augmented 200-sample waveforms (W_{STA}) and
their manual polarity (Man_{STA}); for every waveform this script

  1. draws an integer P-arrival shift ~ N(0, sigma_samples), clipped to +/-4
     samples (1 sample = 0.01 s at 100 Hz),
  2. crops a 64-sample window centred on the shifted arrival,
  3. max-normalizes it,

then splits each station 80/10/10 (train/val/test, stratified on polarity,
random_state=42, so every training set shares the same split and labels) and
writes per-station and merged files:
  {train,val,test}_timeseries_{STA|all}.npy, {train,val,test}_polarities_{STA|all}.npy

Training sets used in the paper (all from the same templates and split):
  TMSF_Tra_000  : Org/*_T_add.mat,      sigma 0  (Figs 4, 5 original SNR, 6)
  TMSF_Tra_001  : Org/*_T_add.mat,      sigma 1  (Fig. 7, 0.01 s; final model)
  TMSF_Tra_002s : Org/*_T_add.mat,      sigma 2  (Fig. 7, 0.02 s)
  STEP          : STEP/*_T_add_ST.mat,  sigma 0  (Fig. 5, high SNR 10-30 dB)
  STEP010       : STEP010/*_T_add_ST.mat, sigma 0 (Fig. 5, low SNR 0-10 dB)
The augmented .mat inputs come from augmentation/K_augument_mimic_org.m,
K_augument_mimic_step.m and K_augument_mimic_step_0_10.m.

Example (from the repository root):
  python 01-scripts/data_preparation/05_build_timeshift_training_sets.py \
      --input-dir 02-data/K_aug/Org --suffix _T_add.mat \
      --sigma-samples 2 --seed 2 --out 02-data/K_aug/TMSF_Tra_002s
"""
import argparse
import os

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]
MAX_SHIFT_SAMPLES = 4


def build_station(path, station, sigma, rng):
    mat = loadmat(path)
    keys = [k for k in mat if not k.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"unexpected variables in {path}: {keys}")
    data = mat[keys[0]].squeeze()
    X, y, shifts = [], [], []
    for item in data:
        w = item[f"W_{station}"].squeeze()
        s = int(np.round(rng.normal(0.0, sigma))) if sigma > 0 else 0
        s = int(np.clip(s, -MAX_SHIFT_SAMPLES, MAX_SHIFT_SAMPLES))
        c = len(w) // 2 + s
        crop = w[c - 32:c + 32]
        m = np.max(np.abs(crop))
        X.append((crop / (m if m else 1.0)).reshape(-1, 1))
        y.append(0 if int(item[f"Man_{station}"].squeeze()) == -1 else 1)
        shifts.append(s)
    return (np.asarray(X, np.float32), np.asarray(y, np.int32), np.asarray(shifts, np.int32))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--suffix", default="_T_add.mat", help="per-station file name after the station code")
    ap.add_argument("--sigma-samples", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0, help="seed for the time shifts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    merged = {sp: ([], []) for sp in ("train", "val", "test")}
    for st in STATIONS:
        X, y, sh = build_station(os.path.join(a.input_dir, st + a.suffix), st, a.sigma_samples, rng)
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=42, stratify=ytmp)
        for sp, (Xs, ys) in {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}.items():
            np.save(os.path.join(a.out, f"{sp}_timeseries_{st}.npy"), Xs)
            np.save(os.path.join(a.out, f"{sp}_polarities_{st}.npy"), ys)
            merged[sp][0].append(Xs); merged[sp][1].append(ys)
        print(f"{st}: {len(y)} waveforms -> {len(ytr)}/{len(yva)}/{len(yte)}; shift std {sh.std():.2f} samples")
    for sp, (Xs, ys) in merged.items():
        Xa, ya = shuffle(np.vstack(Xs), np.hstack(ys), random_state=42)
        np.save(os.path.join(a.out, f"{sp}_timeseries_all.npy"), Xa)
        np.save(os.path.join(a.out, f"{sp}_polarities_all.npy"), ya)
        print(f"{sp}: {len(ya)} waveforms")


if __name__ == "__main__":
    main()
