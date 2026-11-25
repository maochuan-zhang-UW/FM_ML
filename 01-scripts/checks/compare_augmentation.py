"""Check that the Python augmentation (02_augment.py) is statistically
equivalent to the archived MATLAB augmentation used for the paper.

Per station it compares the number of waveforms, the polarity balance and
the distribution of the waveform SNR (20 log10 of the rms after / before the
pick), with a two-sample Kolmogorov-Smirnov test.

Usage (from the repository root):
  python 01-scripts/checks/compare_augmentation.py \
      --python 02-data/augmented/train_org.h5 \
      --matlab <archive>/K_aug/Org --suffix _T_add.mat
"""
import argparse
import os

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.stats import ks_2samp

STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]


def snr_db(w):
    r = lambda x: np.sqrt(np.mean(np.square(x), axis=-1))
    return 20 * np.log10(r(w[:, 80:160]) / r(w[:, 0:80]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", required=True)
    ap.add_argument("--matlab", required=True)
    ap.add_argument("--suffix", default="_T_add.mat")
    a = ap.parse_args()
    print(f"{'sta':4s} {'N py':>7s} {'N mat':>7s} {'%pos py':>8s} {'%pos mat':>8s} "
          f"{'SNR med py':>10s} {'SNR med mat':>11s} {'SNR IQR py':>13s} {'SNR IQR mat':>13s} {'KS D':>6s}")
    with h5py.File(a.python, "r") as h5:
        for st in STATIONS:
            wp, yp = h5[f"{st}/waveform"][()], h5[f"{st}/manual_polarity"][()]
            mat = loadmat(os.path.join(a.matlab, st + a.suffix))
            d = mat[[k for k in mat if not k.startswith("__")][0]].squeeze()
            wm = np.stack([np.asarray(x[f"W_{st}"], float).ravel() for x in d])
            ym = np.array([int(np.asarray(x[f"Man_{st}"]).squeeze()) for x in d])
            sp, sm = snr_db(wp), snr_db(wm)
            q = lambda s: f"{np.percentile(s, 25):5.1f}-{np.percentile(s, 75):5.1f}"
            print(f"{st:4s} {len(yp):7d} {len(ym):7d} {100 * (yp == 1).mean():8.1f} {100 * (ym == 1).mean():8.1f} "
                  f"{np.median(sp):10.2f} {np.median(sm):11.2f} {q(sp):>13s} {q(sm):>13s} "
                  f"{ks_2samp(sp, sm).statistic:6.3f}")


if __name__ == "__main__":
    main()
