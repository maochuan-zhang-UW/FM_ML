"""Fig. S7: distribution of the Kagan angle between the CC and AxialPolCap
composite mechanisms of the same cluster, 2015-2021 (all paired clusters,
irrespective of quality grade).

Input: 02-data/focal_mechanisms_2015_2021/mechanisms.csv (07_focal_mechanisms.py)
Usage (from the repository root): python 01-scripts/figures/figS7_kagan_histogram.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KMAX = 120


def main():
    k = pd.read_csv(ROOT / "02-data" / "focal_mechanisms_2015_2021" / "mechanisms.csv").kagan.dropna().values
    md = np.median(k)
    print(f"paired clusters: {len(k)}, median {md:.1f} deg, mean {k.mean():.1f} deg")
    for th in (15, 30, 45):
        print(f"  within {th} deg: {100 * (k <= th).mean():.1f}%")
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    fig.subplots_adjust(left=0.165, bottom=0.175, right=0.965, top=0.95)
    ax.hist(k, bins=np.arange(0, KMAX + 1, 5), color=(0.25, 0.45, 0.80), ec="k", lw=0.4, alpha=0.9)
    ax.axvline(md, color="r", ls="--", lw=1.2)
    ax.text(md + 4, ax.get_ylim()[1] * 0.92, f"median {md:.0f}°", color="r", fontsize=8)
    ax.set_xlim(0, KMAX)
    ax.set_xticks(range(0, KMAX + 1, 20))
    ax.set_xlabel("Kagan angle (°)", fontsize=8.5)
    ax.set_ylabel("Number of clusters", fontsize=8.5)
    ax.grid(alpha=0.15)
    ax.set_axisbelow(True)
    out = ROOT / "03-figs" / "GJI_FigureS7.png"
    fig.savefig(out, dpi=300)
    print("wrote", out)


if __name__ == "__main__":
    main()
