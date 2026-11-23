import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os

# ======================================
#            PATHS & SETTINGS
# ======================================
data_dir = "./02-data/K_aug/Org"
data_dirsa = "./02-data/K_aug/TMSF_Tra_001"

station_files = [
    "AS1_T_add.mat",
    "AS2_T_add.mat",
    "CC1_T_add.mat",
    "EC1_T_add.mat",
    "EC2_T_add.mat",
    "EC3_T_add.mat",
    "ID1_T_add.mat",
]

# ---------- Time-shift settings ----------
DT = 0.01                  # 1 sample = 0.01 s
MAX_SHIFT_SEC = 0.04       # +/- 0.04 seconds
MAX_SHIFT_SAMPLES = int(MAX_SHIFT_SEC / DT)   # = 4 samples
SIGMA_SAMPLES = 1.0        # normal distribution standard deviation

# ======================================
#     ACCUMULATORS (all-station merge)
# ======================================
X_train_all, y_train_all, shift_train_all = [], [], []
X_val_all,   y_val_all,   shift_val_all   = [], [], []
X_test_all,  y_test_all,  shift_test_all  = [], [], []

def summarize_balance(name, y):
    unique, counts = np.unique(y, return_counts=True)
    summary = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"   Balance {name}: {summary}")
    return summary


# ======================================
#     MAIN LOOP – PER STATION
# ======================================
for fname in station_files:
    fpath = os.path.join(data_dir, fname)
    print(f"\n==========================\nLoading {fname}")

    mat = loadmat(fpath)

    keys = [k for k in mat.keys() if not k.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"❌ Unexpected variable names in {fname}: {keys}")

    varname = keys[0]
    print(f"   ➤ Found MATLAB variable: {varname}")

    data_struct = mat[varname].squeeze()
    station = varname.split("_")[0]

    print(f"   ➤ Processing station: {station}")

    wField   = f"W_{station}"
    manField = f"Man_{station}"

    waveforms, labels, shifts = [], [], []

    # ======================================
    #   Extract waveform + polarity + shift
    # ======================================
    for item in data_struct:

        # waveform
        w = item[wField].squeeze()

        # ----- Random time shift -----
        shift_samples = int(np.round(np.random.normal(loc=0.0, scale=SIGMA_SAMPLES)))
        shift_samples = int(np.clip(shift_samples, -MAX_SHIFT_SAMPLES, MAX_SHIFT_SAMPLES))
        shifts.append(shift_samples)

        # ----- Crop with time shift -----
        center = len(w) // 2 + shift_samples
        start, end = center - 32, center + 32
        w_crop = w[start:end]

        # normalize
        max_val = np.max(np.abs(w_crop))
        if max_val == 0:
            max_val = 1.0
        w_norm = w_crop / max_val

        waveforms.append(w_norm.reshape(-1, 1))

        pol = int(item[manField].squeeze())
        labels.append(0 if pol == -1 else 1)

    X = np.array(waveforms, dtype=np.float32)
    y = np.array(labels,    dtype=np.int32)
    shifts = np.array(shifts, dtype=np.int32)

    # ======================================
    #     TRAIN / VAL / TEST SPLIT
    # ======================================
    X_train, X_temp, y_train, y_temp, shift_train, shift_temp = train_test_split(
        X, y, shifts, test_size=0.2, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test, shift_val, shift_test = train_test_split(
        X_temp, y_temp, shift_temp,
        test_size=0.5, random_state=42, stratify=y_temp
    )

    # ======================================
    #      SAVE PER-STATION .NPY FILES
    # ======================================
    np.save(os.path.join(data_dirsa, f"train_timeseries_{station}.npy"), X_train)
    np.save(os.path.join(data_dirsa, f"train_polarities_{station}.npy"), y_train)
    #np.save(os.path.join(data_dirsa, f"train_shifts_{station}.npy"), shift_train)

    np.save(os.path.join(data_dirsa, f"val_timeseries_{station}.npy"), X_val)
    np.save(os.path.join(data_dirsa, f"val_polarities_{station}.npy"), y_val)
    #np.save(os.path.join(data_dirsa, f"val_shifts_{station}.npy"), shift_val)

    np.save(os.path.join(data_dirsa, f"test_timeseries_{station}.npy"), X_test)
    np.save(os.path.join(data_dirsa, f"test_polarities_{station}.npy"), y_test)
    #np.save(os.path.join(data_dirsa, f"test_shifts_{station}.npy"), shift_test)

    # print summary
    print(f"✅ {station}: {X.shape[0]} samples → Train {X_train.shape[0]}, Val {X_val.shape[0]}, Test {X_test.shape[0]}")
    summarize_balance("Train", y_train)
    summarize_balance("Val",   y_val)
    summarize_balance("Test",  y_test)

    # Append to global merge
    X_train_all.append(X_train)
    y_train_all.append(y_train)
    shift_train_all.append(shift_train)

    X_val_all.append(X_val)
    y_val_all.append(y_val)
    shift_val_all.append(shift_val)

    X_test_all.append(X_test)
    y_test_all.append(y_test)
    shift_test_all.append(shift_test)


# ======================================
#     MERGE ALL STATIONS TOGETHER
# ======================================
X_train_all = np.vstack(X_train_all)
y_train_all = np.hstack(y_train_all)
shift_train_all = np.hstack(shift_train_all)

X_val_all = np.vstack(X_val_all)
y_val_all = np.hstack(y_val_all)
shift_val_all = np.hstack(shift_val_all)

X_test_all = np.vstack(X_test_all)
y_test_all = np.hstack(y_test_all)
shift_test_all = np.hstack(shift_test_all)

# Shuffle
X_train_all, y_train_all, shift_train_all = shuffle(
    X_train_all, y_train_all, shift_train_all, random_state=42
)
X_val_all, y_val_all, shift_val_all = shuffle(
    X_val_all, y_val_all, shift_val_all, random_state=42
)
X_test_all, y_test_all, shift_test_all = shuffle(
    X_test_all, y_test_all, shift_test_all, random_state=42
)

print("\n📊 Final merged dataset:")
print(f"Train: {X_train_all.shape}, shifts {shift_train_all.shape}")
print(f"Val:   {X_val_all.shape}, shifts {shift_val_all.shape}")
print(f"Test:  {X_test_all.shape}, shifts {shift_test_all.shape}")

summarize_balance("Train-All", y_train_all)
summarize_balance("Val-All",   y_val_all)
summarize_balance("Test-All",  y_test_all)

# Save merged files
np.save(os.path.join(data_dirsa, "train_timeseries_all.npy"), X_train_all)
np.save(os.path.join(data_dirsa, "train_polarities_all.npy"), y_train_all)
#np.save(os.path.join(data_dirsa, "train_shifts_all.npy"), shift_train_all)

np.save(os.path.join(data_dirsa, "val_timeseries_all.npy"), X_val_all)
np.save(os.path.join(data_dirsa, "val_polarities_all.npy"), y_val_all)
#np.save(os.path.join(data_dirsa, "val_shifts_all.npy"), shift_val_all)

np.save(os.path.join(data_dirsa, "test_timeseries_all.npy"), X_test_all)
np.save(os.path.join(data_dirsa, "test_polarities_all.npy"), y_test_all)
#np.save(os.path.join(data_dirsa, "test_shifts_all.npy"), shift_test_all)

print("\n✅ All datasets with time shifts saved successfully.")

# ======================================
#     HISTOGRAM OF ALL TIME SHIFTS
# ======================================
import matplotlib.pyplot as plt

# Combine all shifts (train + val + test)
all_shifts = np.concatenate([shift_train_all, shift_val_all, shift_test_all])

# Convert from samples → seconds
all_shifts_sec = all_shifts * DT

plt.figure(figsize=(8, 5))
plt.hist(all_shifts_sec,
         bins=np.arange(-0.045, 0.05, 0.01),
         edgecolor='black')

plt.xlabel("Time Shift (seconds)", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.title("Time-shift Distribution for All Stations", fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"📊 Time-shift histogram plotted. Total shifts: {len(all_shifts)}")
