import numpy as np
from scipy.io import loadmat
import os

# 1) Load the MATLAB struct
mat = loadmat(
    "/Users/mczhang/Documents/GitHub/FM3/02-data/E_Po/E_Po_test1000.mat",
    squeeze_me=True,
    struct_as_record=False
)
Felix = mat["Felix"]           # shape (n_events,)

# 2) Define which fields (channels) to extract
#channels  = ["W_AS1","W_AS2","W_CC1","W_EC1","W_EC2","W_EC3","W_ID1"]
channels  = ["W_AS1"]
n_events  = Felix.shape[0]
n_ch      = len(channels)
n_samples = 64                  # we want samples 18–81 inclusive
start_idx = 18
end_idx   = start_idx + n_samples  # 82

# 3) Allocate output array: (events × channels × samples)
X = np.zeros((n_events, n_ch, n_samples), dtype=np.float32)

# 4) Loop over events and channels, slicing samples 18:82
for i, rec in enumerate(Felix):
    for j, ch in enumerate(channels):
        # pull out and flatten
        w = np.array(getattr(rec, ch), dtype=np.float32).flatten()

        # pad if too short
        if w.size < end_idx:
            pad_width = end_idx - w.size
            w = np.pad(w, (0, pad_width), mode="constant", constant_values=0.0)

        # skip all-zero traces
        if np.max(np.abs(w)) == 0:
            print(f"Event {i+1} {ch}: all-zero waveform, skipping")
            continue

        # slice out exactly 64 samples
        X[i, j, :] = w[start_idx:end_idx]

# 5) Save to .npy
out_dir = "./data"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "Felix_waveforms_64.npy")
np.save(out_path, X)

print(f"Saved array of shape {X.shape} → {out_path}")
