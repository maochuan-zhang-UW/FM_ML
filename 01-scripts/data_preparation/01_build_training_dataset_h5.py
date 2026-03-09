"""
01_build_training_dataset_h5.py
================================
Build unified training dataset for AxialPolCap from HDF5 source files.

Replaces 01_build_training_dataset.py (which read from Template_divide.mat).

Key differences from the original pipeline
-------------------------------------------
- Source   : A_wave_train.h5  (100 Hz, 200 samples, -1 s to +1 s)
- Noise    : A_wave_noise_10000.h5  (100 Hz, 100 samples = 1 s per snippet)
             Two snippets are concatenated to match the 200-sample signal length.
- Flip aug : Every template is also negated (-wf, polarity flipped) with
             independent noise → each real event contributes 2 base waveforms.
- Time shift: Normal(0, sigma=0.02 s) at 100 Hz, clipped to ±0.08 s.
             Implemented via cubic-spline interpolation.
- Target   : ~150 000 total waveforms across all 7 stations
             (TARGET_PER_STATION = 150 000 // 7 ≈ 21 428 per station).
- Output   : single HDF5 file  TMSF_Tra_002/train_dataset.h5

Pipeline per station
--------------------
1. Load templates (wf 200-sample, pol -1/+1) from A_wave_train.h5.
2. For each template produce two base entries:
     a. original  wf  + time shift
     b. flipped  -wf  + different time shift   (polarity inverted)
3. Fill to TARGET_PER_STATION with synthetic copies:
     original copies : wf  + lognormal-SNR noise + time shift
     flip copies     : -wf + different lognormal-SNR noise + time shift
4. Center-crop 200 → 64 samples  (samples 68:132, P-arrival at sample 100).
5. Max-normalize each 64-sample segment independently.
6. Convert polarity: -1 → 0,  +1 → 1.
7. Save per-station groups + merged /all group to one HDF5 file.

Output layout
-------------
TMSF_Tra_002/train_dataset.h5
  /{STA}/waveforms   (N, 64, 1)  float32
  /{STA}/polarities  (N,)        int32    (0 or 1)
  /all/waveforms     (N_total, 64, 1)  float32   (shuffled)
  /all/polarities    (N_total,)        int32
  attrs: source_wf, source_noise, fs_hz, target_per_station, seed, stations

Run from any directory:
    python /path/to/01-scripts/data_preparation/01_build_training_dataset_h5.py
"""

import os
import numpy as np
import h5py
from scipy.stats import lognorm
from scipy.interpolate import CubicSpline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_H5   = os.path.join(ROOT, "02-data", "K_aug", "A_wave_train.h5")
NOISE_H5   = os.path.join(ROOT, "02-data", "A_wave_noise_10000.h5")
OUT_DIR    = os.path.join(ROOT, "02-data", "K_aug", "TMSF_Tra_002")
OUT_H5     = os.path.join(OUT_DIR, "train_dataset.h5")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]

TARGET_TOTAL_ALL   = 150_000
TARGET_PER_STATION = TARGET_TOTAL_ALL // len(STATIONS)   # 21 428

RANDOM_SEED = 42

# Waveform geometry (100 Hz, 200 samples = 2 s, P-arrival at sample 100)
FS            = 100
N_SAMPLES_RAW = 200
CROP_CENTER   = 100      # P-arrival sample index
CROP_HALF     = 32       # half-width → 64-sample crop
CROP_START    = CROP_CENTER - CROP_HALF   # 68
CROP_END      = CROP_CENTER + CROP_HALF   # 132

# Time-shift parameters (100 Hz)
#   sigma = 0.02 s → 2 samples at 100 Hz
SIGMA_SEC     = 0.02
SIGMA_SAMPLES = SIGMA_SEC * FS          # 2 samples
MAX_SHIFT_SEC = 0.08
MAX_SHIFT_SMP = MAX_SHIFT_SEC * FS      # 8 samples

# SNR window indices (same as 200-Hz pipeline, indices still valid at 100 Hz)
#   pre-signal noise : samples  0 – 79
#   signal           : samples 80 – 159
SNR_NOISE_WIN = (0,  80)
SNR_SIG_WIN   = (80, 160)

# SNR clip range when drawing from lognormal
SNR_MIN_DB = 0.5
SNR_MAX_DB = 60.0


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


def compute_snr_db(wf):
    """SNR in dB: signal window 80-160, noise window 0-80."""
    rms_s = _rms(wf[SNR_SIG_WIN[0]:SNR_SIG_WIN[1]])
    rms_n = _rms(wf[SNR_NOISE_WIN[0]:SNR_NOISE_WIN[1]]) + 1e-12
    return 20.0 * np.log10(rms_s / rms_n)


def scale_noise_to_snr(signal_wf, noise_wf, target_snr_db):
    """Scale noise_wf so that signal_wf + scaled_noise ≈ target_snr_db dB."""
    rms_sig          = _rms(signal_wf[SNR_SIG_WIN[0]:SNR_SIG_WIN[1]])
    rms_n            = _rms(noise_wf) + 1e-12
    rms_noise_target = rms_sig / (10.0 ** (target_snr_db / 20.0))
    return (rms_noise_target / rms_n) * noise_wf


def get_noise_200(noise_pool, rng):
    """
    Build a 200-sample noise trace by concatenating two random 100-sample
    snippets (matching source noise length to signal length).
    """
    i1, i2 = rng.integers(0, len(noise_pool), size=2)
    return np.concatenate([noise_pool[i1], noise_pool[i2]])


def apply_time_shift(wf, rng):
    """
    Apply a random time shift via cubic-spline interpolation.
    Shift drawn from Normal(0, SIGMA_SAMPLES), clipped to ±MAX_SHIFT_SMP.
    """
    shift = float(np.clip(rng.normal(0.0, SIGMA_SAMPLES),
                          -MAX_SHIFT_SMP, MAX_SHIFT_SMP))
    if abs(shift) < 1e-6:
        return wf.copy()
    t      = np.arange(N_SAMPLES_RAW) / float(FS)
    dt_sec = shift / float(FS)
    cs     = CubicSpline(t, wf, extrapolate=True)
    return cs(t + dt_sec)


def crop_and_normalize(wf):
    """Center-crop to 64 samples, max-normalize → shape (64, 1) float32."""
    segment = wf[CROP_START:CROP_END].copy()
    max_val = np.max(np.abs(segment))
    if max_val > 1e-12:
        segment = segment / max_val
    return segment.astype(np.float32).reshape(64, 1)


def pol_to_binary(pol_raw):
    """Convert raw polarity (-1 / +1) to binary (0 / 1)."""
    return 1 if int(pol_raw) > 0 else 0


# ---------------------------------------------------------------------------
# Load source data
# ---------------------------------------------------------------------------

def load_train_templates(h5_path):
    """
    Load per-station waveforms and polarities from A_wave_train.h5.

    Returns
    -------
    dict : {station -> list of (wf_200, pol_raw)}
        wf_200  : np.ndarray (200,) float64
        pol_raw : int, -1 or +1
    """
    result = {}
    with h5py.File(h5_path, "r") as f:
        for sta in STATIONS:
            if sta not in f:
                print(f"  [WARN] {sta} not found in {h5_path}")
                continue
            wf_all  = f[f"{sta}/waveforms"][:]    # (N, 200)
            pol_all = f[f"{sta}/polarities"][:]   # (N,) int8  -1 or +1
            records = [(wf_all[i], int(pol_all[i]))
                       for i in range(len(wf_all))
                       if int(pol_all[i]) != 0]   # skip no-pick (pol=0)
            result[sta] = records
            print(f"  {sta}: {len(records):4d} templates  "
                  f"(pos={(sum(p==1 for _,p in records))}, "
                  f"neg={(sum(p==-1 for _,p in records))})")
    return result


def load_noise_pool(h5_path):
    """
    Pool all stations' 100-sample noise snippets from A_wave_noise_10000.h5.

    Returns
    -------
    list of np.ndarray, each shape (100,)
    """
    pool = []
    with h5py.File(h5_path, "r") as f:
        for sta in STATIONS:
            if sta not in f:
                continue
            wf = f[f"{sta}/waveforms"][:]   # (N, 100)
            pool.extend(wf.astype(np.float64))
    print(f"  Noise pool: {len(pool)} snippets (100 samples each)")
    return pool


# ---------------------------------------------------------------------------
# Per-station augmentation
# ---------------------------------------------------------------------------

def fit_lognormal(waveforms):
    """
    Fit a lognormal distribution to the positive SNR values of a set of
    waveforms. Returns (shape, loc, scale) from scipy lognorm.fit.
    """
    snr_vals = np.array([compute_snr_db(wf) for wf in waveforms])
    pos_snr  = snr_vals[snr_vals > 0]
    if len(pos_snr) < 2:
        pos_snr = np.array([5.0, 10.0, 15.0])
    return lognorm.fit(pos_snr, floc=0)


def draw_snr(shape, loc, scale, rng):
    """Draw one SNR value from lognormal, clipped to [SNR_MIN_DB, SNR_MAX_DB]."""
    return float(np.clip(
        lognorm.rvs(shape, loc=loc, scale=scale, random_state=rng),
        SNR_MIN_DB, SNR_MAX_DB
    ))


def augment_station(sta, templates, noise_pool, rng):
    """
    Build the full augmented waveform set for one station.

    Strategy
    --------
    Base waveforms (no added noise):
      - original wf  + time shift  (pol unchanged)
      - flipped  -wf + time shift  (pol inverted)
    Synthetic waveforms (noise + time shift):
      - n_synth_orig copies of originals with random noise
      - n_synth_flip copies of flipped  with random noise (independent)

    Returns
    -------
    X : (N, 64, 1) float32
    y : (N,)       int32   (binary: 0 or 1)
    """
    n_tmpl = len(templates)
    if n_tmpl == 0:
        print(f"  [WARN] No templates for {sta}; skipping.")
        return np.empty((0, 64, 1), np.float32), np.empty((0,), np.int32)

    # Fit SNR distribution from real waveforms
    lg_shape, lg_loc, lg_scale = fit_lognormal([wf for wf, _ in templates])

    # How many synthetic samples needed (after base originals + flips)
    n_base       = 2 * n_tmpl
    n_synthetic  = max(0, TARGET_PER_STATION - n_base)
    n_synth_orig = n_synthetic // 2
    n_synth_flip = n_synthetic - n_synth_orig

    copies_per_orig = int(np.ceil(n_synth_orig / n_tmpl)) if n_tmpl else 0
    copies_per_flip = int(np.ceil(n_synth_flip / n_tmpl)) if n_tmpl else 0

    X_list, y_list = [], []
    synth_orig_count = 0
    synth_flip_count = 0

    for wf_raw, pol_raw in _progress(templates, desc=f"{sta} base  ", total=n_tmpl):
        pol_bin      = pol_to_binary(pol_raw)
        pol_bin_flip = 1 - pol_bin

        # --- Base: original with time shift ---
        X_list.append(crop_and_normalize(apply_time_shift(wf_raw, rng)))
        y_list.append(pol_bin)

        # --- Base: flipped with independent time shift ---
        X_list.append(crop_and_normalize(apply_time_shift(-wf_raw, rng)))
        y_list.append(pol_bin_flip)

    # --- Synthetic from originals ---
    for wf_raw, pol_raw in _progress(templates, desc=f"{sta} synth ", total=n_tmpl):
        pol_bin = pol_to_binary(pol_raw)
        for _ in range(copies_per_orig):
            if synth_orig_count >= n_synth_orig:
                break
            noise_200    = get_noise_200(noise_pool, rng)
            target_snr   = draw_snr(lg_shape, lg_loc, lg_scale, rng)
            scaled_noise = scale_noise_to_snr(wf_raw, noise_200, target_snr)
            synthetic    = apply_time_shift(wf_raw + scaled_noise, rng)
            X_list.append(crop_and_normalize(synthetic))
            y_list.append(pol_bin)
            synth_orig_count += 1
        if synth_orig_count >= n_synth_orig:
            break

    # --- Synthetic from flips ---
    for wf_raw, pol_raw in _progress(templates, desc=f"{sta} flip  ", total=n_tmpl):
        pol_bin_flip = 1 - pol_to_binary(pol_raw)
        wf_flip      = -wf_raw
        for _ in range(copies_per_flip):
            if synth_flip_count >= n_synth_flip:
                break
            noise_200    = get_noise_200(noise_pool, rng)
            target_snr   = draw_snr(lg_shape, lg_loc, lg_scale, rng)
            scaled_noise = scale_noise_to_snr(wf_flip, noise_200, target_snr)
            synthetic    = apply_time_shift(wf_flip + scaled_noise, rng)
            X_list.append(crop_and_normalize(synthetic))
            y_list.append(pol_bin_flip)
            synth_flip_count += 1
        if synth_flip_count >= n_synth_flip:
            break

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"  {sta}: {len(X):6d} samples  "
          f"(pos={int(y.sum())}, neg={int((y==0).sum())})")
    return X, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("01_build_training_dataset_h5.py")
    print("=" * 60)
    print(f"Source waveforms : {TRAIN_H5}")
    print(f"Source noise     : {NOISE_H5}")
    print(f"Output           : {OUT_H5}")
    print(f"Target per station: {TARGET_PER_STATION}  (total ~{TARGET_TOTAL_ALL})")
    print()

    print("Loading training templates ...")
    templates_all = load_train_templates(TRAIN_H5)

    print("\nLoading noise pool ...")
    noise_pool = load_noise_pool(NOISE_H5)

    print()
    all_X, all_y = [], []

    with h5py.File(OUT_H5, "w") as f_out:
        f_out.attrs["source_wf"]          = TRAIN_H5
        f_out.attrs["source_noise"]       = NOISE_H5
        f_out.attrs["fs_hz"]              = FS
        f_out.attrs["target_per_station"] = TARGET_PER_STATION
        f_out.attrs["seed"]               = RANDOM_SEED
        f_out.attrs["stations"]           = STATIONS

        for sta in STATIONS:
            print(f"\n{'=' * 40}")
            print(f"Station: {sta}")
            print(f"{'=' * 40}")

            if sta not in templates_all or len(templates_all[sta]) == 0:
                print(f"  Skipping {sta} (no templates).")
                continue

            X, y = augment_station(sta, templates_all[sta], noise_pool, rng)
            if len(X) == 0:
                continue

            # Shuffle station data before saving
            perm = rng.permutation(len(X))
            X, y = X[perm], y[perm]

            grp = f_out.create_group(sta)
            grp.create_dataset("waveforms",  data=X, dtype="float32",
                               compression="gzip", compression_opts=4)
            grp.create_dataset("polarities", data=y, dtype="int32")

            all_X.append(X)
            all_y.append(y)

        # Merged /all group (globally shuffled)
        print(f"\n{'=' * 40}")
        print("Saving merged /all group ...")
        X_all = np.concatenate(all_X, axis=0).astype(np.float32)
        y_all = np.concatenate(all_y, axis=0).astype(np.int32)
        perm  = rng.permutation(len(X_all))
        X_all, y_all = X_all[perm], y_all[perm]

        grp_all = f_out.create_group("all")
        grp_all.create_dataset("waveforms",  data=X_all, dtype="float32",
                               compression="gzip", compression_opts=4)
        grp_all.create_dataset("polarities", data=y_all, dtype="int32")

        print(f"  /all: {len(X_all):7d} samples  "
              f"(pos={int(y_all.sum())}, neg={int((y_all==0).sum())})")

    print(f"\nDone. Output: {OUT_H5}")


if __name__ == "__main__":
    main()
