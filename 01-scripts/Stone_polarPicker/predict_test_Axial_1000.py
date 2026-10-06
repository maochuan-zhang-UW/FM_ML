#!/usr/bin/env python3
import numpy as np
import os
from scipy.io import loadmat, savemat
from predict import polarPredict
import matplotlib.pyplot as plt   # ← import plotting

# ────────────────────────────────────────────────────────
# 0) Paths & parameters
# ────────────────────────────────────────────────────────
mat_file         = "/Users/mczhang/Documents/GitHub/FM5_ML/02-data/E_wave_forML_downsampled.mat"
output_mat_file  = "/Users/mczhang/Documents/GitHub/FM5_ML/02-data/E_wave_forML_downsampled_ML_Stone.mat"
model_path       = "./models/polarityModel_20240710.keras"

channels         = ['W_AS1','W_AS2','W_CC1','W_EC1','W_EC2','W_EC3','W_ID1']
start_idx        = 32
n_samples        = 64
end_idx          = start_idx + n_samples  # 82

os.makedirs(os.path.dirname(output_mat_file), exist_ok=True)

# ────────────────────────────────────────────────────────
# 1) Load MATLAB struct
# ────────────────────────────────────────────────────────
md    = loadmat(mat_file, squeeze_me=True, struct_as_record=False)
Felix = md['Felix']
N     = Felix.shape[0]
print(f"Loaded {N} events from {mat_file}")

# ────────────────────────────────────────────────────────
# 2) Prepare storage for outputs
# ────────────────────────────────────────────────────────
pred_labels = {ch: np.full((N,), 'x', dtype='<U10') for ch in channels}
pred_probs  = {ch: np.zeros((N,), dtype=np.float32)  for ch in channels}

# ────────────────────────────────────────────────────────
# 3) Init predictor
# ────────────────────────────────────────────────────────
predictor = polarPredict(mode="predict", model=model_path)

# ────────────────────────────────────────────────────────
# 4) Loop & process
# ────────────────────────────────────────────────────────
for i, rec in enumerate(Felix):
    for ch in channels:
        # extract & flatten
        w = np.array(getattr(rec, ch), dtype=np.float32).flatten()
        if w.size < end_idx:
            w = np.pad(w, (0, end_idx - w.size), 'constant')
        w64 = w[start_idx:end_idx]  # 64 samples
        
        if np.max(np.abs(w64)) == 0:
            continue
        # 4.1) Linear detrend
        t = np.arange(n_samples, dtype=np.float32)
        slope, intercept = np.polyfit(t, w64, 1)
        w64 = w64 - (slope*t + intercept)

        # 4.2) Normalize to [-1, 1]
        max_abs = np.max(np.abs(w64))
        if max_abs == 0:
            continue
        w64 = w64 / max_abs
        
        
        
        # --- PLOT this input so you can inspect it ---
        # plt.figure(figsize=(4,2))
        # plt.plot(w64, linewidth=1)
        # plt.title(f"Event {i+1} – {ch}")
        # plt.xlabel("Sample index")
        # plt.ylabel("Normalized amp.")
        # plt.tight_layout()
        # plt.show()
        
        # reshape for model
        x_i = w64.reshape(1, n_samples, 1)
        # --- predict ---
        label, prob = predictor.predict(x_i)[0]
        pred_labels[ch][i] = label
        pred_probs[ch][i]  = prob

# ────────────────────────────────────────────────────────
# 5) Attach fields & save
# ────────────────────────────────────────────────────────
for i, rec in enumerate(Felix):
    for ch in channels:
        suffix     = ch.split('_',1)[1]
        setattr(rec, f'Po_ML_Ian_{suffix}',    pred_labels[ch][i])
        setattr(rec, f'PoCon_ML_Ian_{suffix}', pred_probs[ch][i])

savemat(output_mat_file, {'Felix': Felix})
print("Saved augmented struct →", output_mat_file)
