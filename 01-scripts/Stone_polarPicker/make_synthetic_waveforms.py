#!/usr/bin/env python
import numpy as np
import os

# Parameters
N = 50       # number of waveforms
L = 64       # samples per trace
fs = 100     # Hz sampling rate
t = np.arange(L) / fs  # time vector in seconds

# Prepare output array
X = np.zeros((N, L), dtype=np.float32)

for i in range(N):
    # 1) Gaussian pulse centered at sample L/2
    center_time = (L/2) / fs
    sigma = 0.01  # pulse width in seconds
    pulse = np.exp(-((t - center_time)**2) / (2 * sigma**2))
    
    # 2) add white noise
    noise = np.random.normal(0, 0.2, size=L)
    w = pulse + noise

    # 3) linear detrend
    slope, intercept = np.polyfit(t, w, 1)
    trend = slope * t + intercept
    w = w - trend

    # 4) normalize to [-1, 1]
    w = w / np.max(np.abs(w))

    # store
    X[i] = w

# ensure output directory exists
out_dir = "./data"
os.makedirs(out_dir, exist_ok=True)

# save to .npy
out_path = os.path.join(out_dir, "Portland_data_polarityInput_newdownload.npy")
np.save(out_path, X)

print(f"Saved {N} synthetic waveforms of length {L} at {fs} Hz → {out_path}")
