import os
import numpy as np
from scipy.io import loadmat

# ======================================
#           PATHS & SETTINGS
# ======================================
# Folder where your .mat files live
# (change this if needed, e.g. "./02-data/Val")
data_dir = "./02-data/K_aug/Val"

station_files = [
    "AS1_V_add.mat",
    "AS2_V_add.mat",
    "CC1_V_add.mat",
    "EC1_V_add.mat",
    "EC2_V_add.mat",
    "EC3_V_add.mat",
    "ID1_V_add.mat",
]

# ======================================
#   COLLECTORS FOR ALL-STATION ARRAYS
# ======================================
all_X = []   # list of (N_station, 64, 1)
all_y = []   # list of (N_station,)

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

    # Station name from variable, e.g. "AS1_V_add" → "AS1"
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
        # waveform (assumed length 200)
        w = getattr(item, w_field)
        w = np.asarray(w).squeeze()

        # center crop 64 samples
        center = len(w) // 2
        w_crop = w[center - 32:center + 32]

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
    out_ts = os.path.join(data_dir, f"timeseries_{station}.npy")
    out_pol = os.path.join(data_dir, f"polarities_{station}.npy")

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

out_ts_all = os.path.join(data_dir, "timeseries_all.npy")
out_pol_all = os.path.join(data_dir, "polarities_all.npy")

np.save(out_ts_all, X_all)
np.save(out_pol_all, y_all)

print("\n🎉 All stations processed.")
print(f"   ✅ Combined file shapes: X_all={X_all.shape}, y_all={y_all.shape}")
print(f"   ✅ Saved → {os.path.basename(out_ts_all)}, {os.path.basename(out_pol_all)}")
