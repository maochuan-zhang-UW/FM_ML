import numpy as np
import matplotlib.pyplot as plt

# Load the synthetic dataset you generated
X = np.load("./data/Portland_data_polarityInput_newdownload.npy")

# Sampling parameters
fs = 100                  # sampling rate in Hz
L = X.shape[1]            # number of samples per trace
t = np.arange(L) / fs     # time axis in seconds

# Pick one example trace (e.g., the first one)
w = X[0]

# Plot it
plt.figure(figsize=(8, 4))
plt.plot(t, w, linewidth=1.5)
plt.xlabel("Time (s)")
plt.ylabel("Normalized amplitude")
plt.title("Example Synthetic Waveform (Trace 1)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

