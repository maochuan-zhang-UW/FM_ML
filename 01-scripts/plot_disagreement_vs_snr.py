#!/usr/bin/env python3
"""Supplementary figure: CC-AxialPolCap polarity disagreement as a function of SNR.

Supports the statement in the Discussion that disagreements between the two
methods occur predominantly on low signal-to-noise arrivals.

    python 01-scripts/plot_disagreement_vs_snr.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig_fm5_ml"

import matplotlib
matplotlib.use("Agg")                      # never open a window
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
MAT = REPO_ROOT / "02-data" / "A_wave_2015_2021_h5_conf80_predictions.mat"
OUT = REPO_ROOT / "03-figs" / "FigureS_disagreement_vs_snr_python.png"
STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]
STATIONS_AX = ["AX" + s for s in STATIONS]


def load_helpers():
    src = REPO_ROOT / "01-scripts" / "make_manuscript_figures.py"
    spec = importlib.util.spec_from_file_location("make_manuscript_figures", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def snr_db(w: np.ndarray) -> float:
    """SNR from the 64-sample window: post-arrival vs pre-arrival RMS."""
    w = np.asarray(w, dtype=float).ravel()
    if w.size < 64:
        return np.nan
    noise = np.sqrt(np.mean(w[:32] ** 2))
    sig = np.sqrt(np.mean(w[32:] ** 2))
    if noise <= 0:
        return np.nan
    return 20.0 * np.log10((sig + 1e-12) / (noise + 1e-12))


def collect():
    felix = np.atleast_1d(loadmat(str(MAT), struct_as_record=False, squeeze_me=True)["Felix"])
    sta_idx, snr, disagree = [], [], []
    for ev in felix:
        for k, s in enumerate(STATIONS):
            po = getattr(ev, f"Po_{s}", None)
            w = getattr(ev, f"W_{s}", None)
            if po is None or w is None:
                continue
            po = np.atleast_1d(po).ravel()
            if po.size < 2:
                continue
            cc, ml = float(po[0]), float(po[1])
            if cc == 0 or ml == 0:          # one method gave no polarity
                continue
            d = snr_db(w)
            if np.isfinite(d):
                sta_idx.append(k); snr.append(d); disagree.append(cc != ml)
    return np.array(sta_idx), np.array(snr), np.array(disagree, dtype=float)


def main() -> Path:
    fig_helpers = load_helpers()
    fig_helpers.apply_manuscript_style()
    sta, snr, dis = collect()
    print(f"picks with a polarity from both methods: {len(snr):,}")
    print(f"overall disagreement: {100*dis.mean():.2f}%")

    edges = np.arange(-10, 52.5, 2.5)
    ctr = 0.5 * (edges[:-1] + edges[1:])

    def rate(mask):
        r, n = np.full(ctr.size, np.nan), np.zeros(ctr.size)
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            m = mask & (snr >= a) & (snr < b)
            n[i] = m.sum()
            if n[i] >= 30:
                r[i] = 100.0 * dis[m].mean()
        return r, n

    fig, axes = plt.subplots(1, 2, figsize=(fig_helpers.FIG_WIDTH_IN, 3.0),
                             constrained_layout=True)

    # ---- (a) all stations, with the pick distribution behind ----------
    ax = axes[0]
    r_all, n_all = rate(np.ones_like(snr, dtype=bool))
    axb = ax.twinx()
    axb.bar(ctr, n_all / 1000.0, width=2.3, color="0.86", edgecolor="none", zorder=0)
    axb.set_ylabel("Picks per bin (thousands)", fontsize=8, color="0.45")
    axb.tick_params(axis="y", labelsize=7.5, colors="0.45")
    axb.set_zorder(0)
    ax.set_zorder(1); ax.patch.set_visible(False)
    ax.plot(ctr, r_all, "o-", color=(0.20, 0.35, 0.75), markersize=3.2, linewidth=1.5,
            zorder=3)
    # Catalog-average disagreement, for reference against the SNR trend.
    mean_all = 100.0 * dis.mean()
    ax.axhline(mean_all, color="crimson", linestyle="--", linewidth=1.2, zorder=4)
    ax.text(49, mean_all + 1.2, f"catalog average {mean_all:.1f}%", color="crimson",
            fontsize=7.2, ha="right", va="bottom", zorder=5)
    ax.set_xlabel("SNR (dB)", fontsize=8.5)
    ax.set_ylabel("Polarity disagreement (%)", fontsize=8.5)
    ax.set_xlim(-10, 50); ax.set_ylim(0, 56)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=7.5)
    ax.text(0.030, 0.965, "(a)", transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top", ha="left", zorder=6)

    # ---- (b) per station ----------------------------------------------
    ax = axes[1]
    cmap = plt.get_cmap("viridis")
    for k, name in enumerate(STATIONS_AX):
        r, _ = rate(sta == k)
        ax.plot(ctr, r, "-", color=cmap(k / 6.0), linewidth=1.3, label=name)
    ax.axhline(mean_all, color="crimson", linestyle="--", linewidth=1.2, zorder=4)
    ax.text(49, mean_all + 1.4, f"catalog average {mean_all:.1f}%", color="crimson",
            fontsize=7.2, ha="right", va="bottom", zorder=5)
    ax.set_xlabel("SNR (dB)", fontsize=8.5)
    ax.set_ylabel("Polarity disagreement (%)", fontsize=8.5)
    ax.set_xlim(-10, 50); ax.set_ylim(0, 72)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=7.5)
    ax.legend(ncol=4, fontsize=6.5, loc="upper right", handlelength=1.2,
              columnspacing=0.8, borderaxespad=0.3, framealpha=0.92)
    ax.text(0.030, 0.965, "(b)", transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top", ha="left", zorder=6)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT}")
    return OUT


if __name__ == "__main__":
    main()
