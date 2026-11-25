"""Step 2: augment the templates with real seafloor noise.

Python port of K_augument_mimic_org.m, K_augument_mimic_step.m,
K_augument_mimic_step_0_10.m and the augmentation part of L_saveNPY_Validate.m.

For every template of a station the output holds the original waveform
followed by floor(target / N_templates) noisy copies. Each copy is
  template + noise * (rms(template) / 10^(SNR/20)) / rms(noise),
with a noise window drawn at random from that station's noise library and a
target SNR (dB) drawn from
  lognormal : a lognormal fit to the station's observed SNR (> 0 dB),
              MLE mean and unbiased standard deviation of log(SNR), as
              MATLAB's fitdist(..., 'Lognormal')
  uniform   : uniform between --snr-range LOW HIGH.

Sets used in the paper
  train_org     : --split train --snr lognormal --target 15000   (original SNR)
  train_step    : --split train --snr uniform --snr-range 10 30 --target 15000 (high SNR)
  train_step010 : --split train --snr uniform --snr-range 0 10 --target 15000  (low SNR)
  test          : --split test  --snr lognormal --target 5000    (common test set)

Random numbers differ from MATLAB's, so the waveforms are statistically
equivalent to, not identical with, the archived datasets used for the paper.

Output HDF5: {STA}/waveform (N, 200), manual_polarity, catalog_polarity,
event_id, template_index, is_original, snr_target_db (NaN for originals) and
snr_db = 20 log10(rms(w[80:160]) / rms(w[0:80])).

Usage (from the repository root):
  python 01-scripts/02_augment.py --split train --snr lognormal --target 15000 \
      --out 02-data/augmented/train_org.h5
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]


def rms(x, axis=-1):
    return np.sqrt(np.mean(np.square(x), axis=axis))


def snr_db(w):
    return 20 * np.log10(rms(w[..., 80:160]) / rms(w[..., 0:80]))


def lognormal_params(observed):
    x = np.log(observed[observed > 0])
    return x.mean(), x.std(ddof=1)


def augment_station(templates, noise, n_copies, snr_targets, rng):
    n_t = len(templates)
    n_out = n_t * (1 + n_copies)
    out = np.empty((n_out, templates.shape[1]))
    tidx = np.repeat(np.arange(n_t), 1 + n_copies)
    is_orig = np.tile(np.r_[True, np.zeros(n_copies, bool)], n_t)
    target = np.full(n_out, np.nan)
    target[~is_orig] = snr_targets
    out[is_orig] = templates
    sig = templates[tidx[~is_orig]]
    nz = noise[rng.integers(0, len(noise), size=len(sig))]
    scale = (rms(sig) / 10 ** (snr_targets / 20)) / rms(nz)
    out[~is_orig] = sig + scale[:, None] * nz
    return out, tidx, is_orig, target


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", default=str(ROOT / "02-data" / "prepared" / "inputs.h5"))
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--snr", choices=["lognormal", "uniform"], required=True)
    ap.add_argument("--snr-range", type=float, nargs=2, metavar=("LOW", "HIGH"))
    ap.add_argument("--target", type=int, required=True, help="approximate samples per station")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.snr == "uniform" and not a.snr_range:
        ap.error("--snr uniform needs --snr-range LOW HIGH")
    rng = np.random.default_rng(a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.inputs, "r") as src, h5py.File(a.out, "w") as dst:
        dst.attrs.update(split=a.split, snr=a.snr, target=a.target, seed=a.seed,
                         snr_range=a.snr_range or [np.nan, np.nan])
        for st in STATIONS:
            t = src[f"templates/{a.split}/{st}"]
            temps = t["waveform"][()]
            n_copies = int(np.floor(a.target / len(temps)))   # MATLAB: for k = 1:target/N
            n_syn = len(temps) * n_copies
            if a.snr == "lognormal":
                mu, sigma = lognormal_params(src[f"snr/{st}/snr_db"][()])
                targets = rng.lognormal(mu, sigma, n_syn)
            else:
                targets = rng.uniform(*a.snr_range, n_syn)
            w, tidx, is_orig, target = augment_station(temps, src[f"noise/{st}/waveform"][()],
                                                       n_copies, targets, rng)
            g = dst.create_group(st)
            g.create_dataset("waveform", data=w, compression="gzip")
            for f in ("manual_polarity", "catalog_polarity", "event_id"):
                g[f] = t[f][()][tidx]
            g["template_index"] = tidx
            g["is_original"] = is_orig
            g["snr_target_db"] = target
            g["snr_db"] = snr_db(w)
            print(f"{st}: {len(temps)} templates x {1 + n_copies} = {len(w)} waveforms")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
