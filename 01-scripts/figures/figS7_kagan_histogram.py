"""Fig. S7: distribution of the Kagan angle between the CC and AxialPolCap
composite mechanisms of the same cluster, 2015-2021.

Filled bars: clusters whose two mechanisms are both quality A or B (Table S3),
the comparison quoted in Section 4.1. Outline: all paired clusters. The median
for each quality grade is printed.

Input: 02-data/focal_mechanisms_2015_2021/mechanisms.csv (07_focal_mechanisms.py)
Usage (from the repository root): python 01-scripts/figures/figS7_kagan_histogram.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KMAX = 120


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "03-figs" / "GJI_FigureS7.png"))
    a = ap.parse_args()
    df = pd.read_csv(ROOT / "02-data" / "focal_mechanisms_2015_2021" / "mechanisms.csv").dropna(subset=["kagan"])
    ab = df[df.quality_cc.isin(list("AB")) & df.quality_ml.isin(list("AB"))].kagan.values
    allk = df.kagan.values
    print(f"all paired clusters: {len(allk)}, median {np.median(allk):.1f} deg")
    print(f"both A or B: {len(ab)}, median {np.median(ab):.1f} deg, within 15/30 deg: "
          f"{100 * (ab <= 15).mean():.0f}% / {100 * (ab <= 30).mean():.0f}%")
    for q in "ABCD":
        k = df[(df.quality_cc == q) & (df.quality_ml == q)].kagan
        print(f"  both {q}: {len(k)}, median {k.median():.1f} deg")
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    fig.subplots_adjust(left=0.165, bottom=0.175, right=0.965, top=0.95)
    bins = np.arange(0, KMAX + 1, 5)
    ax.hist(allk, bins=bins, histtype="step", color="0.45", lw=1.0,
            label=f"All paired clusters (n = {len(allk):,})")
    ax.hist(ab, bins=bins, color=(0.25, 0.45, 0.80), ec="k", lw=0.4, alpha=0.9,
            label=f"Both quality A–B (n = {len(ab):,})")
    md = np.median(ab)
    ax.axvline(md, color="r", ls="--", lw=1.2)
    ax.text(md + 3, ax.get_ylim()[1] * 0.60, f"median {md:.0f}°", color="r", fontsize=8)
    ax.set_xlim(0, KMAX)
    ax.set_xticks(range(0, KMAX + 1, 20))
    ax.set_xlabel("Kagan angle (°)", fontsize=8.5)
    ax.set_ylabel("Number of clusters", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.grid(alpha=0.15)
    ax.set_axisbelow(True)
    fig.savefig(a.out, dpi=300)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
