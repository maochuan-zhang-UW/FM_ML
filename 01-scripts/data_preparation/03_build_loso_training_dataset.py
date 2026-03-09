"""
03_build_loso_training_dataset.py
==================================
Build the per-station augmented datasets used for LOSO (Leave-One-Station-Out)
cross-validation training of AxialPolCap.

Replaces two earlier scripts:
  - K_augument_all_step_SNRdis.m  (step-SNR noise augmentation from MATLAB)
  - L_saveNPY.py                  (center-crop, split, .npy export)

Differences from 01_build_training_dataset.py
----------------------------------------------
- Reads from Template.mat (contains variables AS1, AS2, ..., ID1) rather
  than Template_divide.mat.
- Uses the LAST 20% of each station's events (MATLAB convention: idx_start =
  ceil(0.8 * length)).  The first 80% are assumed to be the training-set
  templates used in 01_build_training_dataset.py.
- Uses a STEP SNR distribution instead of lognormal:
    10% of copies: SNR drawn uniformly from  [0, 5] dB
    80% of copies: SNR drawn uniformly from  [5, 35] dB
    10% of copies: SNR drawn uniformly from [35, 50] dB
- ID1 station gets a 3x multiplier on the number of synthetic copies
  (base_multi_trace = 20 copies per event; ID1 uses 60).
- No time shift is applied (LOSO dataset is shift-free).
- Output directory: ./02-data/K_aug/STEP010/

Run from FM_ML/ root:
    conda activate tf-2.14.0
    python scripts/data_preparation/03_build_loso_training_dataset.py
"""

import os
import math
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]

# Paths relative to FM_ML/ root
TEMPLATE_MAT   = "./02-data/K_aug/Template.mat"
NOISE_MAT      = "./02-data/H_noi/H_Noise_200.mat"
SNR_VALUES_MAT = "./02-data/H_noi/H_noise_dB20_snrValue.mat"   # loaded but unused
OUT_DIR        = "./02-data/K_aug/STEP010"

# Augmentation
BASE_MULTI_TRACE  = 20   # noisy copies per event (ID1 gets 3x this value)
ID1_MULTIPLIER    = 3
RANDOM_SEED       = 42

# Step-SNR distribution
#   Bucket weights (must sum to 1.0) and SNR ranges [low, high] dB
SNR_BUCKETS = [
    (0.10, 0.0,  5.0),   # 10% in [0, 5] dB
    (0.80, 5.0, 35.0),   # 80% in [5, 35] dB
    (0.10, 35.0, 50.0),  # 10% in [35, 50] dB
]

# Waveform geometry
N_SAMPLES_RAW = 200
CROP_CENTER   = 100
CROP_HALF     = 32
CROP_START    = CROP_CENTER - CROP_HALF   # 68
CROP_END      = CROP_CENTER + CROP_HALF   # 132

# Split fractions
VAL_FRAC  = 0.10
TEST_FRAC = 0.10
SPLIT_NAMES = ["train", "val", "test"]

# ---------------------------------------------------------------------------
# Optional tqdm progress bar
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm
    def _progress(iterable, desc="", total=None):
        return tqdm(iterable, desc=desc, total=total, ncols=80, leave=False)
except ImportError:
    def _progress(iterable, desc="", total=None):
        return iterable


# ---------------------------------------------------------------------------
# Signal-processing helpers
# ---------------------------------------------------------------------------

def _rms(x):
    return np.sqrt(np.mean(x ** 2))


def scale_noise_to_snr(signal_wf, noise_wf, target_snr_db):
    """
    Return noise_wf scaled so that adding it to signal_wf yields ~target_snr_db dB.

    SNR windows (200-Hz):
      signal : samples 80-159
      noise  : samples  0-79 (pre-signal reference level)
    """
    rms_sig          = _rms(signal_wf[80:160])
    rms_n            = _rms(noise_wf) + 1e-12
    rms_noise_target = rms_sig / (10.0 ** (target_snr_db / 20.0))
    return (rms_noise_target / rms_n) * noise_wf


def draw_step_snr(rng):
    """
    Draw one SNR value from the step distribution defined by SNR_BUCKETS.
    """
    r = rng.random()
    cumulative = 0.0
    for weight, lo, hi in SNR_BUCKETS:
        cumulative += weight
        if r <= cumulative:
            return float(rng.uniform(lo, hi))
    return float(rng.uniform(SNR_BUCKETS[-1][1], SNR_BUCKETS[-1][2]))


def crop_and_normalize(wf):
    """
    Center-crop 200-sample waveform to 64 samples, max-normalize.
    Returns shape (64, 1), dtype float32.
    """
    segment = wf[CROP_START:CROP_END].copy()
    max_val = np.max(np.abs(segment))
    if max_val > 1e-12:
        segment = segment / max_val
    return segment.astype(np.float32).reshape(64, 1)


def polarity_to_binary(pol):
    return int(float(pol) > 0)


# ---------------------------------------------------------------------------
# Load MATLAB source files
# ---------------------------------------------------------------------------

def load_templates(path):
    """
    Load Template.mat (struct_as_record=False, squeeze_me=True).

    The file contains variables named AS1, AS2, ..., ID1 (not AS1_T etc.).
    Each variable is a struct array with W_{STA} and Man_{STA} (or Po_{STA})
    fields per event.

    Only the LAST 20% of each station's events are used (matching the MATLAB
    convention: idx_start = ceil(0.8 * length)).

    Returns
    -------
    dict : {station_name -> list of (wf_200, label_binary)}
    """
    mat = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    result = {}

    for sta in STATIONS:
        # Variable name in Template.mat is just the station name (no _T suffix)
        if sta not in mat:
            print(f"  [WARN] '{sta}' not found in {path}; skipping.")
            continue

        raw = mat[sta]
        if not hasattr(raw, "__len__"):
            raw = [raw]

        n_total   = len(raw)
        idx_start = math.ceil(0.8 * n_total)   # last 20% (MATLAB convention)
        raw       = raw[idx_start:]

        wf_field   = f"W_{sta}"
        pol_fields = [f"Man_{sta}", f"Po_{sta}"]

        records = []
        for ev in raw:
            wf = getattr(ev, wf_field, None)
            if wf is None:
                continue
            wf = np.asarray(wf, dtype=np.float64).ravel()
            if len(wf) < N_SAMPLES_RAW:
                continue

            pol = None
            for pf in pol_fields:
                if hasattr(ev, pf):
                    pol = polarity_to_binary(getattr(ev, pf))
                    break
            if pol is None:
                continue

            records.append((wf[:N_SAMPLES_RAW], pol))

        result[sta] = records
        print(f"  Loaded {len(records):5d} templates for {sta} "
              f"(last 20% of {n_total} total events)")

    return result


def load_noise(path):
    """
    Load H_Noise_200.mat.  Pools all stations' noise into one list.

    Returns list of np.ndarray, each shape (200,).
    """
    mat   = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    felix = mat.get("Felix")
    if felix is None:
        raise KeyError(f"'Felix' not found in {path}")
    if not hasattr(felix, "__len__"):
        felix = [felix]

    noise_list = []
    for ev in felix:
        for sta in STATIONS:
            wf = getattr(ev, f"W_{sta}", None)
            if wf is None:
                continue
            wf = np.asarray(wf, dtype=np.float64).ravel()
            if len(wf) >= N_SAMPLES_RAW:
                noise_list.append(wf[:N_SAMPLES_RAW])

    print(f"  Loaded {len(noise_list)} noise waveforms from {path}")
    return noise_list


# ---------------------------------------------------------------------------
# Per-station augmentation
# ---------------------------------------------------------------------------

def augment_station(sta, templates, noise_list, rng):
    """
    Build augmented dataset for one station using the step-SNR distribution.

    Parameters
    ----------
    sta       : station name (used only for n_copies multiplier logic)
    templates : list of (wf_200, label_binary)
    noise_list: list of 200-sample noise waveforms
    rng       : np.random.Generator

    Returns
    -------
    X : np.ndarray, shape (N, 64, 1), float32
    y : np.ndarray, shape (N,),       int32
    """
    n_templates = len(templates)
    if n_templates == 0:
        print(f"  [WARN] No templates for {sta}; skipping.")
        return np.empty((0, 64, 1), np.float32), np.empty((0,), np.int32)

    # Determine copies per template
    n_copies = BASE_MULTI_TRACE
    if sta == "ID1":
        n_copies *= ID1_MULTIPLIER

    n_noise = len(noise_list)
    X_list, y_list = [], []

    # --- Original templates (no time shift for LOSO dataset) ---
    for wf_raw, pol in _progress(templates, desc=f"{sta} originals", total=n_templates):
        X_list.append(crop_and_normalize(wf_raw))
        y_list.append(pol)

    # --- Noisy copies ---
    for wf_raw, pol in _progress(templates, desc=f"{sta} augment ", total=n_templates):
        for _ in range(n_copies):
            target_snr   = draw_step_snr(rng)
            noise_wf     = noise_list[rng.integers(0, n_noise)]
            scaled_noise = scale_noise_to_snr(wf_raw, noise_wf, target_snr)
            synthetic    = wf_raw + scaled_noise
            X_list.append(crop_and_normalize(synthetic))
            y_list.append(pol)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"  {sta}: {len(X)} samples  "
          f"(pos={int(y.sum())}, neg={int((y == 0).sum())})")
    return X, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("03_build_loso_training_dataset.py")
    print("=" * 60)

    # Load source data
    print(f"\nLoading templates (last 20% per station):\n  {TEMPLATE_MAT}")
    templates_all = load_templates(TEMPLATE_MAT)

    print(f"\nLoading noise:\n  {NOISE_MAT}")
    noise_list = load_noise(NOISE_MAT)

    # Containers for merged arrays
    merged = {sp: {"X": [], "y": []} for sp in SPLIT_NAMES}

    # Per-station loop
    for sta in STATIONS:
        print(f"\n{'=' * 40}")
        print(f"Station: {sta}")
        print(f"{'=' * 40}")

        if sta not in templates_all:
            print(f"  Skipping {sta} (no templates).")
            continue

        X, y = augment_station(sta, templates_all[sta], noise_list, rng)
        if len(X) == 0:
            continue

        # Stratified 80/10/10 split
        holdout = VAL_FRAC + TEST_FRAC   # 0.20
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=holdout, random_state=RANDOM_SEED, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_tmp
        )

        splits = {
            "train": (X_train, y_train),
            "val":   (X_val,   y_val),
            "test":  (X_test,  y_test),
        }

        for sp, (Xs, ys) in splits.items():
            np.save(os.path.join(OUT_DIR, f"{sp}_timeseries_{sta}.npy"), Xs)
            np.save(os.path.join(OUT_DIR, f"{sp}_polarities_{sta}.npy"), ys)
            n_pos = int(ys.sum())
            print(f"  {sp:5s}: {len(Xs):6d} samples "
                  f"(pos={n_pos}, neg={len(ys)-n_pos})")
            merged[sp]["X"].append(Xs)
            merged[sp]["y"].append(ys)

    # Merged all-station files
    print(f"\n{'=' * 40}")
    print("Saving merged all-station files ...")
    for sp in SPLIT_NAMES:
        if not merged[sp]["X"]:
            continue
        X_all = np.concatenate(merged[sp]["X"], axis=0).astype(np.float32)
        y_all = np.concatenate(merged[sp]["y"], axis=0).astype(np.int32)
        X_all, y_all = shuffle(X_all, y_all, random_state=RANDOM_SEED)
        np.save(os.path.join(OUT_DIR, f"{sp}_timeseries_all.npy"), X_all)
        np.save(os.path.join(OUT_DIR, f"{sp}_polarities_all.npy"), y_all)
        print(f"  {sp:5s}_all: {len(X_all):7d} samples "
              f"(pos={int(y_all.sum())}, neg={int((y_all==0).sum())})")

    print(f"\nDone. Output: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
