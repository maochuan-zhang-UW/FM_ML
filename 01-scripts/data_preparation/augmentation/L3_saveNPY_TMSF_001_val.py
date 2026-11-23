# make the timeshift smaller
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# ======================================
#           PATHS & SETTINGS
# ======================================
data_dir = "/Users/mcZhang/Documents/GitHub/FM_ML/02-data/K_aug/Val"
data_dirsa = "/Users/mcZhang/Documents/GitHub/FM_ML/02-data/K_aug/TMSF_Val_001"

os.makedirs(data_dirsa, exist_ok=True)

station_files = [
    "AS1_V_add.mat",
    "AS2_V_add.mat",
    "CC1_V_add.mat",
    "EC1_V_add.mat",
    "EC2_V_add.mat",
    "EC3_V_add.mat",
    "ID1_V_add.mat",
]

# ---------- Time-shift settings ----------
DT = 0.01                 # 1 sample = 0.01 s
MAX_SHIFT_SEC = 0.04      # time shift range [-0.04, 0.04] s
MAX_SHIFT_SAMPLES = int(MAX_SHIFT_SEC / DT)   # = 4 samples

# std dev of the normal distribution in *samples*
SIGMA_SAMPLES = 1.0       # most shifts small, some up to ±4 after clipping

# ======================================
#   COLLECTORS FOR ALL-STATION ARRAYS
# ======================================
all_X = []   # list of (N_station, 64, 1)
all_y = []   # list of (N_station,)
all_shifts = []  # collect all time shifts (in samples)

# ======================================
#         MAIN LOOP PER STATION
# ======================================
for fname in station_files:
    fpath = os.path.join(data_dir, fname)
    print("\n==========================")
    print(f"Loading {fname}")

    # Load .mat file
    mat = loadmat(fpath, squeeze_me=True, struct_as_record=False)

    # Detect MATLAB struct variable (ignore __header__, __globals__, __version__)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"❌ Unexpected variable names in {fname}: {keys}")

    varname = keys[0]
    print(f"   ➤ Found MATLAB variable: {varname}")

    data_struct = mat[varname]

    # Station name from variable, e.g. "AS1_T_add" → "AS1"
    station = varname.split("_")[0]
    print(f"   ➤ Processing station: {station}")

    w_field = f"W_{station}"
    man_field = f"Man_{station}"

    # Make sure we always iterate over a 1D array of structs
    if not isinstance(data_struct, np.ndarray):
        items = np.array([data_struct])
    else:
        items = data_struct.ravel()

    waveforms = []
    labels = []

    # ======================================
    #      EXTRACT WAVEFORM + POLARITY
    # ======================================
    for item in items:
        # waveform (assumed length ~200)
        w = getattr(item, w_field)
        w = np.asarray(w).squeeze()

        # ----- draw random time shift (in samples) -----
        shift_samples = int(
            np.round(np.random.normal(loc=0.0, scale=SIGMA_SAMPLES))
        )
        shift_samples = int(
            np.clip(shift_samples, -MAX_SHIFT_SAMPLES, MAX_SHIFT_SAMPLES)
        )
        all_shifts.append(shift_samples)
        # -----------------------------------------------

        # center crop 64 samples with time shift
        center = len(w) // 2 + shift_samples
        start = center - 32
        end = center + 32

        # (for 200-sample traces, this stays in-bounds; add padding if needed)
        w_crop = w[start:end]

        # normalize individually
        max_val = np.max(np.abs(w_crop))
        if max_val == 0:
            max_val = 1.0
        w_norm = (w_crop / max_val).astype(np.float32)

        waveforms.append(w_norm.reshape(-1, 1))

        # polarity: -1 -> 0, +1 -> 1
        pol = getattr(item, man_field)
        pol = int(np.asarray(pol).squeeze())
        labels.append(0 if pol == -1 else 1)

    X = np.stack(waveforms, axis=0).astype(np.float32)  # (N, 64, 1)
    y = np.array(labels, dtype=np.int32)                # (N,)

    # ======================================
    #          SAVE NPY PER STATION
    # ======================================
    out_ts = os.path.join(data_dirsa, f"timeseries_{station}.npy")
    out_pol = os.path.join(data_dirsa, f"polarities_{station}.npy")

    np.save(out_ts, X)
    np.save(out_pol, y)

    print(f"   ✅ Saved {X.shape[0]} samples → {os.path.basename(out_ts)}, {os.path.basename(out_pol)}")

    # ======================================
    #   APPEND TO GLOBAL ALL-STATION LISTS
    # ======================================
    all_X.append(X)
    all_y.append(y)

# ======================================
#     SAVE COMBINED ALL-STATION FILES
# ======================================
X_all = np.concatenate(all_X, axis=0)   # (N_total, 64, 1)
y_all = np.concatenate(all_y, axis=0)   # (N_total,)

out_ts_all = os.path.join(data_dirsa, "timeseries_all.npy")
out_pol_all = os.path.join(data_dirsa, "polarities_all.npy")

np.save(out_ts_all, X_all)
np.save(out_pol_all, y_all)

print("\n🎉 All stations processed.")
print(f"   ✅ Combined file shapes: X_all={X_all.shape}, y_all={y_all.shape}")
print(f"   ✅ Saved → {os.path.basename(out_ts_all)}, {os.path.basename(out_pol_all)}")

# ======================================
#          PLOT TIME-SHIFT HISTOGRAM
# ======================================
shifts_sec = np.array(all_shifts) * DT   # convert samples → seconds

plt.figure()
plt.hist(shifts_sec, bins=np.arange(-0.045, 0.05, 0.01), edgecolor='black')
plt.xlabel("Time Shift (seconds)")
plt.ylabel("Count")
plt.title("Histogram of Time Shifts (-0.04 to 0.04 s)")
plt.grid(True)
plt.tight_layout()
plt.show()

print("Histogram plotted. Number of shifts:", len(all_shifts))
