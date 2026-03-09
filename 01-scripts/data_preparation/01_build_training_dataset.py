"""
01_build_training_dataset.py
============================
Build the unified (non-LOSO) training dataset for AxialPolCap.

Replaces two earlier scripts:
  - K_augument_mimic_org.m     (lognormal-SNR noise augmentation from MATLAB)
  - L4_saveNPY_TMSF_001tran.py (center-crop, time-shift, split, .npy export)

The key change from the old pipeline is that we read directly from
Template_divide.mat instead of relying on the intermediate per-station
K_aug/Org/{STA}_T_add.mat files produced by the MATLAB script.

Pipeline
--------
For each station (AS1, AS2, CC1, EC1, EC2, EC3, ID1):
  1. Load template events from Template_divide.mat (training templates only).
  2. Keep all originals; augment with noisy copies to reach TARGET_TOTAL samples.
     - Per-copy: draw target SNR from a lognormal fit to the station's empirical
       SNR distribution, pick a random noise snippet from H_Noise_200.mat, scale
       it, and add to the template.
  3. Apply a random time shift to every waveform (original + synthetic):
       shift ~ Normal(0, sigma=2 samples at 200 Hz = 0.01 s std).
       Shift implemented via cubic-spline interpolation (no hard integer rounding).
  4. Center-crop 200 -> 64 samples  (center=100, crop[68:132]).
  5. Max-normalize each 64-sample segment independently.
  6. Convert polarity label: -1 -> 0,  +1 -> 1.
  7. Stratified 80 / 10 / 10 (train / val / test) split.
  8. Save per-station .npy files and merged all-station .npy files.

Output directory: ./02-data/K_aug/TMSF_Tra_001/

Run from FM_ML/ root:
    conda activate tf-2.14.0
    python scripts/data_preparation/01_build_training_dataset.py
"""

import os
import sys
import numpy as np
import scipy.io
from scipy.stats import lognorm
from scipy.interpolate import CubicSpline
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]

# Paths relative to FM_ML/ root
TEMPLATE_MAT   = "./02-data/K_aug/Template_divide.mat"
NOISE_MAT      = "./02-data/H_noi/H_Noise_200.mat"
SNR_VALUES_MAT = "./02-data/H_noi/H_noise_dB20_snrValue.mat"
OUT_DIR        = "./02-data/K_aug/TMSF_Tra_001"

# Augmentation
TARGET_TOTAL  = 15000   # total samples per station (templates + synthetic copies)
RANDOM_SEED   = 42

# Waveform geometry
N_SAMPLES_RAW = 200     # raw waveform length (samples, 200 Hz)
CROP_CENTER   = 100     # P-arrival sample index in 200-sample window
CROP_HALF     = 32      # half-width -> 64-sample crop
CROP_START    = CROP_CENTER - CROP_HALF   # 68
CROP_END      = CROP_CENTER + CROP_HALF   # 132

# Time-shift parameters (at 200 Hz)
#   1 sample at 100 Hz  = 2 samples at 200 Hz  = 0.01 s
SIGMA_SAMPLES = 2       # std of Normal shift distribution (200-Hz samples)
MAX_SHIFT     = 8       # hard clip on |shift| (200-Hz samples = 4 samples at 100 Hz)

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
    """Root-mean-square of a 1-D array."""
    return np.sqrt(np.mean(x ** 2))


def compute_snr_db(wf):
    """
    Compute SNR (dB) using fixed 200-Hz windows:
      signal window : samples 80-159
      pre-signal    : samples  0-79
    """
    rms_s = _rms(wf[80:160])
    rms_n = _rms(wf[0:80]) + 1e-12
    return 20.0 * np.log10(rms_s / rms_n)


def scale_noise_to_snr(signal_wf, noise_wf, target_snr_db):
    """
    Return a scaled copy of noise_wf such that adding it to signal_wf
    yields approximately target_snr_db dB SNR.

    Scale factor derivation:
      rms_noise_target = rms(signal_window) / 10^(SNR_dB/20)
      scale = rms_noise_target / rms(noise_wf)
    """
    rms_sig          = _rms(signal_wf[80:160])
    rms_n            = _rms(noise_wf) + 1e-12
    rms_noise_target = rms_sig / (10.0 ** (target_snr_db / 20.0))
    return (rms_noise_target / rms_n) * noise_wf


def apply_time_shift(wf, rng):
    """
    Apply a random sub-sample time shift to a 200-Hz waveform via cubic spline.

    Draw shift ~ Normal(0, SIGMA_SAMPLES), clip to [-MAX_SHIFT, MAX_SHIFT].
    Returns a new array of the same length (200 samples).
    """
    shift = float(np.clip(rng.normal(0.0, SIGMA_SAMPLES), -MAX_SHIFT, MAX_SHIFT))
    if abs(shift) < 1e-6:
        return wf.copy()
    t        = np.arange(N_SAMPLES_RAW) / 200.0   # time axis in seconds
    dt_sec   = shift / 200.0
    cs       = CubicSpline(t, wf, extrapolate=True)
    return cs(t + dt_sec)


def crop_and_normalize(wf):
    """
    Center-crop 200-sample waveform to 64 samples and max-normalize.
    Returns shape (64, 1), dtype float32.
    """
    segment = wf[CROP_START:CROP_END].copy()
    max_val = np.max(np.abs(segment))
    if max_val > 1e-12:
        segment = segment / max_val
    return segment.astype(np.float32).reshape(64, 1)


def polarity_to_binary(pol):
    """Convert MATLAB polarity convention (-1 / +1) to binary (0 / 1)."""
    return int(float(pol) > 0)


# ---------------------------------------------------------------------------
# Load MATLAB source files
# ---------------------------------------------------------------------------

def load_templates(path):
    """
    Load Template_divide.mat using struct_as_record=False, squeeze_me=True.

    Returns
    -------
    dict : {station_name -> list of (wf_200, label_binary)}
      wf_200 : np.ndarray shape (200,), float64
      label  : int, 0 or 1
    """
    mat = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    result = {}

    for sta in STATIONS:
        key = f"{sta}_T"
        if key not in mat:
            print(f"  [WARN] '{key}' not found in {path}; skipping {sta}.")
            continue

        raw = mat[key]
        # squeeze_me may return a single struct (0-d object) or an array of structs
        if not hasattr(raw, "__len__"):
            raw = [raw]

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
        print(f"  Loaded {len(records):5d} templates for {sta}")

    return result


def load_noise(path):
    """
    Load H_Noise_200.mat.

    The file contains a struct array 'Felix'; each entry has W_{STA} fields
    (200-Hz, 200-sample noise waveforms). All stations' noise is pooled.

    Returns
    -------
    list of np.ndarray, each shape (200,)
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


def load_snr_values(path):
    """
    Load H_noise_dB20_snrValue.mat.

    The file contains 'snrValues', a MATLAB cell array of length n_stations.
    Element i corresponds to STATIONS[i].

    Returns
    -------
    dict : {station_name -> np.ndarray of SNR values (dB)}
    """
    mat     = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    snr_raw = mat.get("snrValues")
    if snr_raw is None:
        raise KeyError(f"'snrValues' not found in {path}")

    snr_raw = np.asarray(snr_raw).ravel()
    snr_dict = {}
    for i, sta in enumerate(STATIONS):
        if i < len(snr_raw):
            vals = np.asarray(snr_raw[i], dtype=np.float64).ravel()
        else:
            vals = np.array([10.0])
        snr_dict[sta] = vals
        print(f"  SNR for {sta}: n={len(vals)}, median={np.median(vals):.1f} dB")

    return snr_dict


# ---------------------------------------------------------------------------
# Per-station augmentation
# ---------------------------------------------------------------------------

def augment_station(sta, templates, noise_list, snr_vals, rng):
    """
    Build the augmented waveform set for one station.

    Strategy
    --------
    1. Keep every template (with a random time shift).
    2. Fill remaining slots up to TARGET_TOTAL with synthetic waveforms:
       each = template + lognormal-SNR-scaled noise snippet + time shift.

    Returns
    -------
    X : np.ndarray, shape (N, 64, 1), float32
    y : np.ndarray, shape (N,),       int32
    """
    n_templates = len(templates)
    if n_templates == 0:
        print(f"  [WARN] No templates for {sta}; skipping.")
        return np.empty((0, 64, 1), np.float32), np.empty((0,), np.int32)

    # Fit lognormal to positive-SNR empirical values
    pos_snr = snr_vals[snr_vals > 0]
    if len(pos_snr) < 2:
        pos_snr = np.array([5.0, 10.0, 15.0])
    shape, loc, scale = lognorm.fit(pos_snr, floc=0)

    n_synthetic      = max(0, TARGET_TOTAL - n_templates)
    copies_per_tmpl  = int(np.ceil(n_synthetic / n_templates)) if n_templates else 0
    n_noise          = len(noise_list)

    X_list, y_list = [], []
    total_synth = 0

    # --- Original templates (time-shift only) ---
    for wf_raw, pol in _progress(templates, desc=f"{sta} originals", total=n_templates):
        wf_shifted = apply_time_shift(wf_raw, rng)
        X_list.append(crop_and_normalize(wf_shifted))
        y_list.append(pol)

    # --- Synthetic noisy copies ---
    for wf_raw, pol in _progress(templates, desc=f"{sta} augment ", total=n_templates):
        for _ in range(copies_per_tmpl):
            if total_synth >= n_synthetic:
                break
            target_snr  = float(np.clip(
                lognorm.rvs(shape, loc=loc, scale=scale, random_state=rng),
                0.5, 60.0
            ))
            noise_wf    = noise_list[rng.integers(0, n_noise)]
            scaled_noise = scale_noise_to_snr(wf_raw, noise_wf, target_snr)
            synthetic    = apply_time_shift(wf_raw + scaled_noise, rng)
            X_list.append(crop_and_normalize(synthetic))
            y_list.append(pol)
            total_synth += 1
        if total_synth >= n_synthetic:
            break

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
    print("01_build_training_dataset.py")
    print("=" * 60)

    # Load source data
    print(f"\nLoading templates:\n  {TEMPLATE_MAT}")
    templates_all = load_templates(TEMPLATE_MAT)

    print(f"\nLoading noise:\n  {NOISE_MAT}")
    noise_list = load_noise(NOISE_MAT)

    print(f"\nLoading SNR distribution:\n  {SNR_VALUES_MAT}")
    snr_dict = load_snr_values(SNR_VALUES_MAT)

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

        X, y = augment_station(
            sta,
            templates_all[sta],
            noise_list,
            snr_dict.get(sta, np.array([10.0])),
            rng,
        )
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
