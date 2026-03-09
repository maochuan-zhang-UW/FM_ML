"""
02_build_eval_dataset_h5.py
============================
Build validation dataset (~10 000 samples) for AxialPolCap from A_wave_val.h5.

Uses the same noise augmentation + flip strategy as the training script so
the val set covers the same SNR/time-shift distribution.

Pipeline per station
--------------------
1. Load waveforms and polarities from A_wave_val.h5.
2. For each template produce two base entries:
     a. original  wf  + time shift
     b. flipped  -wf  + different time shift   (polarity inverted)
3. Fill to TARGET_PER_STATION with synthetic copies:
     original copies : wf  + lognormal-SNR noise + time shift
     flip copies     : -wf + different lognormal-SNR noise + time shift
4. Center-crop 200 → 64 samples (samples 68:132, P-arrival at sample 100).
5. Max-normalize each segment independently.
6. Convert polarity: -1 → 0,  +1 → 1.
7. Save per-station groups + merged /all group to one HDF5 file.

Output layout
-------------
TMSF_Val_002/val_dataset.h5
  /{STA}/waveforms   (N, 64, 1)  float32
  /{STA}/polarities  (N,)        int32    (0 or 1)
  /all/waveforms     (N_total, 64, 1)  float32   (shuffled)
  /all/polarities    (N_total,)        int32
  attrs: source_wf, source_noise, fs_hz, target_per_station, seed, stations

Run from any directory:
    python /path/to/01-scripts/data_preparation/02_build_eval_dataset_h5.py
"""

import os
import numpy as np
import h5py
from scipy.stats import lognorm
from scipy.interpolate import CubicSpline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VAL_H5    = os.path.join(ROOT, "02-data", "K_aug", "A_wave_val.h5")
NOISE_H5  = os.path.join(ROOT, "02-data", "A_wave_noise_10000.h5")
OUT_DIR   = os.path.join(ROOT, "02-data", "K_aug", "TMSF_Val_002")
OUT_H5    = os.path.join(OUT_DIR, "val_dataset.h5")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]

TARGET_TOTAL_ALL   = 10_000
TARGET_PER_STATION = TARGET_TOTAL_ALL // len(STATIONS)   # 1 428

RANDOM_SEED = 42

# Waveform geometry (100 Hz, 200 samples = 2 s, P-arrival at sample 100)
FS            = 100
N_SAMPLES_RAW = 200
CROP_CENTER   = 100
CROP_HALF     = 32
CROP_START    = CROP_CENTER - CROP_HALF   # 68
CROP_END      = CROP_CENTER + CROP_HALF   # 132

# Time-shift parameters (same as training)
SIGMA_SEC     = 0.02
SIGMA_SAMPLES = SIGMA_SEC * FS          # 2 samples
MAX_SHIFT_SEC = 0.08
MAX_SHIFT_SMP = MAX_SHIFT_SEC * FS      # 8 samples

# SNR windows and clip range
SNR_NOISE_WIN = (0,  80)
SNR_SIG_WIN   = (80, 160)
SNR_MIN_DB    = 0.5
SNR_MAX_DB    = 60.0


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
    rms_s = _rms(wf[SNR_SIG_WIN[0]:SNR_SIG_WIN[1]])
    rms_n = _rms(wf[SNR_NOISE_WIN[0]:SNR_NOISE_WIN[1]]) + 1e-12
    return 20.0 * np.log10(rms_s / rms_n)


def scale_noise_to_snr(signal_wf, noise_wf, target_snr_db):
    rms_sig          = _rms(signal_wf[SNR_SIG_WIN[0]:SNR_SIG_WIN[1]])
    rms_n            = _rms(noise_wf) + 1e-12
    rms_noise_target = rms_sig / (10.0 ** (target_snr_db / 20.0))
    return (rms_noise_target / rms_n) * noise_wf


def get_noise_200(noise_pool, rng):
    """Concatenate two random 100-sample snippets → 200-sample noise trace."""
    i1, i2 = rng.integers(0, len(noise_pool), size=2)
    return np.concatenate([noise_pool[i1], noise_pool[i2]])


def apply_time_shift(wf, rng):
    shift = float(np.clip(rng.normal(0.0, SIGMA_SAMPLES),
                          -MAX_SHIFT_SMP, MAX_SHIFT_SMP))
    if abs(shift) < 1e-6:
        return wf.copy()
    t      = np.arange(N_SAMPLES_RAW) / float(FS)
    dt_sec = shift / float(FS)
    cs     = CubicSpline(t, wf, extrapolate=True)
    return cs(t + dt_sec)


def crop_and_normalize(wf):
    segment = wf[CROP_START:CROP_END].copy()
    max_val = np.max(np.abs(segment))
    if max_val > 1e-12:
        segment = segment / max_val
    return segment.astype(np.float32).reshape(64, 1)


def pol_to_binary(pol_raw):
    return 1 if int(pol_raw) > 0 else 0


# ---------------------------------------------------------------------------
# Load noise pool
# ---------------------------------------------------------------------------

def load_noise_pool(h5_path):
    pool = []
    with h5py.File(h5_path, "r") as f:
        for sta in STATIONS:
            if sta not in f:
                continue
            wf = f[f"{sta}/waveforms"][:]
            pool.extend(wf.astype(np.float64))
    print(f"  Noise pool: {len(pool)} snippets (100 samples each)")
    return pool


# ---------------------------------------------------------------------------
# Per-station augmentation  (same logic as training script)
# ---------------------------------------------------------------------------

def fit_lognormal(waveforms):
    snr_vals = np.array([compute_snr_db(wf) for wf in waveforms])
    pos_snr  = snr_vals[snr_vals > 0]
    if len(pos_snr) < 2:
        pos_snr = np.array([5.0, 10.0, 15.0])
    return lognorm.fit(pos_snr, floc=0)


def draw_snr(shape, loc, scale, rng):
    return float(np.clip(
        lognorm.rvs(shape, loc=loc, scale=scale, random_state=rng),
        SNR_MIN_DB, SNR_MAX_DB
    ))


def augment_station(sta, templates, noise_pool, rng):
    n_tmpl = len(templates)
    if n_tmpl == 0:
        print(f"  [WARN] No templates for {sta}; skipping.")
        return np.empty((0, 64, 1), np.float32), np.empty((0,), np.int32)

    lg_shape, lg_loc, lg_scale = fit_lognormal([wf for wf, _ in templates])

    n_base       = 2 * n_tmpl
    n_synthetic  = max(0, TARGET_PER_STATION - n_base)
    n_synth_orig = n_synthetic // 2
    n_synth_flip = n_synthetic - n_synth_orig

    copies_per_orig = int(np.ceil(n_synth_orig / n_tmpl)) if n_tmpl else 0
    copies_per_flip = int(np.ceil(n_synth_flip / n_tmpl)) if n_tmpl else 0

    X_list, y_list = [], []
    synth_orig_count = 0
    synth_flip_count = 0

    # Base: original + flipped
    for wf_raw, pol_raw in _progress(templates, desc=f"{sta} base  ", total=n_tmpl):
        pol_bin = pol_to_binary(pol_raw)
        X_list.append(crop_and_normalize(apply_time_shift(wf_raw, rng)))
        y_list.append(pol_bin)
        X_list.append(crop_and_normalize(apply_time_shift(-wf_raw, rng)))
        y_list.append(1 - pol_bin)

    # Synthetic from originals
    for wf_raw, pol_raw in _progress(templates, desc=f"{sta} synth ", total=n_tmpl):
        pol_bin = pol_to_binary(pol_raw)
        for _ in range(copies_per_orig):
            if synth_orig_count >= n_synth_orig:
                break
            noise_200    = get_noise_200(noise_pool, rng)
            target_snr   = draw_snr(lg_shape, lg_loc, lg_scale, rng)
            scaled_noise = scale_noise_to_snr(wf_raw, noise_200, target_snr)
            X_list.append(crop_and_normalize(apply_time_shift(wf_raw + scaled_noise, rng)))
            y_list.append(pol_bin)
            synth_orig_count += 1
        if synth_orig_count >= n_synth_orig:
            break

    # Synthetic from flips
    for wf_raw, pol_raw in _progress(templates, desc=f"{sta} flip  ", total=n_tmpl):
        pol_bin_flip = 1 - pol_to_binary(pol_raw)
        wf_flip      = -wf_raw
        for _ in range(copies_per_flip):
            if synth_flip_count >= n_synth_flip:
                break
            noise_200    = get_noise_200(noise_pool, rng)
            target_snr   = draw_snr(lg_shape, lg_loc, lg_scale, rng)
            scaled_noise = scale_noise_to_snr(wf_flip, noise_200, target_snr)
            X_list.append(crop_and_normalize(apply_time_shift(wf_flip + scaled_noise, rng)))
            y_list.append(pol_bin_flip)
            synth_flip_count += 1
        if synth_flip_count >= n_synth_flip:
            break

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"  {sta}: {len(X):5d} samples  "
          f"(pos={int(y.sum())}, neg={int((y==0).sum())})")
    return X, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("02_build_eval_dataset_h5.py")
    print("=" * 60)
    print(f"Source waveforms : {VAL_H5}")
    print(f"Source noise     : {NOISE_H5}")
    print(f"Output           : {OUT_H5}")
    print(f"Target per station: {TARGET_PER_STATION}  (total ~{TARGET_TOTAL_ALL})")
    print()

    print("Loading noise pool ...")
    noise_pool = load_noise_pool(NOISE_H5)
    print()

    all_X, all_y = [], []

    with h5py.File(VAL_H5, "r") as f_in, h5py.File(OUT_H5, "w") as f_out:
        f_out.attrs["source_wf"]          = VAL_H5
        f_out.attrs["source_noise"]       = NOISE_H5
        f_out.attrs["fs_hz"]              = FS
        f_out.attrs["target_per_station"] = TARGET_PER_STATION
        f_out.attrs["seed"]               = RANDOM_SEED
        f_out.attrs["stations"]           = STATIONS

        for sta in STATIONS:
            print(f"\n{'=' * 40}")
            print(f"Station: {sta}")
            print(f"{'=' * 40}")

            if sta not in f_in:
                print(f"  [WARN] {sta} not found in source; skipping.")
                continue

            wf_all  = f_in[f"{sta}/waveforms"][:]
            pol_all = f_in[f"{sta}/polarities"][:]

            templates = [
                (wf_all[i].astype(np.float64), int(pol_all[i]))
                for i in range(len(wf_all))
                if int(pol_all[i]) != 0
            ]

            if not templates:
                print(f"  No valid waveforms — skipping.")
                continue

            X, y = augment_station(sta, templates, noise_pool, rng)
            if len(X) == 0:
                continue

            perm = rng.permutation(len(X))
            X, y = X[perm], y[perm]

            grp = f_out.create_group(sta)
            grp.create_dataset("waveforms",  data=X, dtype="float32",
                               compression="gzip", compression_opts=4)
            grp.create_dataset("polarities", data=y, dtype="int32")

            all_X.append(X)
            all_y.append(y)

        # Merged /all group
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

        print(f"  /all: {len(X_all):6d} samples  "
              f"(pos={int(y_all.sum())}, neg={int((y_all==0).sum())})")

    print(f"\nDone. Output: {OUT_H5}")


if __name__ == "__main__":
    main()
