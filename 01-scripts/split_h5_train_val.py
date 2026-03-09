"""
split_h5_train_val.py
=====================
Split A_wave_dB20_cleaned.h5 into 80% augmentation (train) and 20% validation
sets, stratified by polarity per station.

Input
-----
  02-data/A_wave_dB20_cleaned.h5

Output  (02-data/K_aug/)
------------------------
  A_wave_train.h5   -- 80% of each station, polarity-stratified
  A_wave_val.h5     -- 20% of each station, polarity-stratified

HDF5 layout (same for both files)
----------------------------------
  /{STA}/waveforms   (N, 200)  float32
  /{STA}/polarities  (N,)      int8      -1 or +1
  /{STA}/event_id    (N,)      int32
  /{STA}/sp          (N,)      float32
  /{STA}/ddt         (N,)      float32
  attrs: source, fs_hz, split, split_seed, stations

Run from any directory:
    python /path/to/01-scripts/split_h5_train_val.py
"""

import os
import numpy as np
import h5py
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_H5  = os.path.join(ROOT, "02-data", "A_wave_dB20_cleaned.h5")
OUT_DIR   = os.path.join(ROOT, "02-data", "K_aug")
TRAIN_H5  = os.path.join(OUT_DIR, "A_wave_train.h5")
VAL_H5    = os.path.join(OUT_DIR, "A_wave_val.h5")

STATIONS    = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]
VAL_FRAC    = 0.20
RANDOM_SEED = 42
DATASETS    = ["waveforms", "polarities", "event_id", "sp", "ddt"]


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_station(f_in, sta):
    """
    Load one station from the source HDF5, stratified-split 80/20.
    Returns (train_dict, val_dict) where each is {dataset_name -> np.ndarray}.
    """
    grp = f_in[sta]
    data = {ds: grp[ds][:] for ds in DATASETS}
    n = len(data["waveforms"])

    # Stratify on polarity; if only one class present fall back to random
    pol = data["polarities"]
    rng = np.random.default_rng(RANDOM_SEED)

    # Shuffle the full index first so selection is order-independent
    idx = np.arange(n)
    rng.shuffle(idx)
    data = {ds: data[ds][idx] for ds in DATASETS}
    pol  = data["polarities"]    # re-read after shuffle

    # Stratify only when both classes are present
    stratify = pol if len(np.unique(pol)) > 1 else None

    idx2 = np.arange(n)
    idx_train, idx_val = train_test_split(
        idx2,
        test_size=VAL_FRAC,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )

    # Shuffle within each split so order inside the file is also random
    rng.shuffle(idx_train)
    rng.shuffle(idx_val)

    train = {ds: data[ds][idx_train] for ds in DATASETS}
    val   = {ds: data[ds][idx_val]   for ds in DATASETS}
    return train, val


def write_split(h5_path, split_data, split_label, source_path):
    """Write a dict of {sta -> {ds -> array}} to an HDF5 file."""
    with h5py.File(h5_path, "w") as f:
        f.attrs["source"]      = source_path
        f.attrs["fs_hz"]       = 100
        f.attrs["split"]       = split_label
        f.attrs["split_seed"]  = RANDOM_SEED
        f.attrs["val_frac"]    = VAL_FRAC
        f.attrs["stations"]    = STATIONS

        for sta, data in split_data.items():
            grp = f.create_group(sta)
            for ds_name, arr in data.items():
                kw = dict(compression="gzip", compression_opts=4) \
                     if arr.dtype == np.float32 else {}
                grp.create_dataset(ds_name, data=arr, **kw)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("split_h5_train_val.py")
    print("=" * 60)
    print(f"Input : {INPUT_H5}")
    print(f"Output: {OUT_DIR}/")
    print(f"Split : {int((1-VAL_FRAC)*100)}% train / {int(VAL_FRAC*100)}% val  "
          f"(stratified by polarity, seed={RANDOM_SEED})\n")

    train_all, val_all = {}, {}

    with h5py.File(INPUT_H5, "r") as f_in:
        for sta in STATIONS:
            if sta not in f_in:
                print(f"  {sta}: not found in source — skipping")
                continue

            train, val = split_station(f_in, sta)
            train_all[sta] = train
            val_all[sta]   = val

            n_tr, n_v = len(train["waveforms"]), len(val["waveforms"])
            pol_tr = train["polarities"]
            pol_v  = val["polarities"]
            print(f"  {sta}: {n_tr:4d} train  "
                  f"(pos={( pol_tr== 1).sum()}, neg={(pol_tr==-1).sum()})  |  "
                  f"{n_v:3d} val  "
                  f"(pos={( pol_v == 1).sum()}, neg={(pol_v ==-1).sum()})")

    write_split(TRAIN_H5, train_all, "train", INPUT_H5)
    write_split(VAL_H5,   val_all,   "val",   INPUT_H5)

    print(f"\nSaved:")
    print(f"  {TRAIN_H5}")
    print(f"  {VAL_H5}")
    print("\nDone.")


if __name__ == "__main__":
    main()
