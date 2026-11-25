"""Benchmark of four published polarity classifiers on the Axial benchmark set
(Fig. 3, Fig. S1, Table S2).

Input: 02-data/raw/benchmark/A_wave_dB15_DT_CFM_EQP_PolCAP.mat, a struct array
Felix (6,801 events, 7 stations) with the reference polarity Po_{STA} and the
predictions of each pretrained model applied zero-shot (+1 up, -1 down, 0 no
prediction), written by the scripts in benchmark/inference/:

  DiTingMotion  PoML_W_{STA}   (Zhao et al. 2023)
  CFM           CFM_W_{STA}    (Messuti et al. 2023)
  EQPolarity    EQP_{STA}      (Chen et al. 2024)
  PolarCAP      PolCAP_{STA}   (Chakraborty et al. 2022)

A pick is scored when both the reference and the prediction are non-zero.
Output: 03-figs/benchmark_confusion_matrix.csv with, per model and station,
PT (up -> up), NT (down -> down), PF (down -> up), NF (up -> down), N and the
accuracy, plus the pooled row used as the Average in Fig. 3.

Usage (from the repository root): python 01-scripts/benchmark/score_benchmark.py
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]
MODELS = {"DiTingMotion": "PoML_W_{}", "CFM": "CFM_W_{}", "EQPolarity": "EQP_{}", "PolarCAP": "PolCAP_{}"}


def first(e, field):
    v = np.asarray(getattr(e, field, []))
    return float(v.ravel()[0]) if v.size else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(ROOT / "02-data" / "raw" / "benchmark" / "A_wave_dB15_DT_CFM_EQP_PolCAP.mat"))
    ap.add_argument("--out", default=str(ROOT / "03-figs" / "benchmark_confusion_matrix.csv"))
    a = ap.parse_args()
    ev = np.atleast_1d(loadmat(a.input, squeeze_me=True, struct_as_record=False)["Felix"])
    rows = []
    for model, pat in MODELS.items():
        total = np.zeros(4, int)
        for st in STATIONS:
            ref = np.sign([first(e, f"Po_{st}") for e in ev])
            pred = np.sign([first(e, pat.format(st)) for e in ev])
            k = (ref != 0) & (pred != 0)
            r, p = ref[k], pred[k]
            c = np.array([((r == 1) & (p == 1)).sum(), ((r == -1) & (p == -1)).sum(),
                          ((r == -1) & (p == 1)).sum(), ((r == 1) & (p == -1)).sum()])
            total += c
            rows.append([model, st, *c, c.sum(), round(100 * (c[0] + c[1]) / c.sum(), 2)])
        rows.append([model, "ALL (pooled)", *total, total.sum(), round(100 * (total[0] + total[1]) / total.sum(), 2)])
        print(f"{model:13s} " + " ".join(f"{r[1]} {r[-1]:.1f}" for r in rows[-8:-1]) + f"  pooled {rows[-1][-1]:.1f}")
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Station", "PT", "NT", "PF", "NF", "N", "Accuracy(%)"])
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
