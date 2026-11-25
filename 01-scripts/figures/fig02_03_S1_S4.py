#!/usr/bin/env python3
"""Figs 2, 3, S1 and S4 and Table S2.

  2  : template, noise and augmented waveforms at AXAS2
  3  : benchmark of four published classifiers and CC on the common test data
  S1 : benchmark confusion matrices
  S4 : examples of 2015-2021 picks where AxialPolCap and CC disagree
  Table S2 (written with Fig. 3): per-station benchmark accuracies

Usage (from the repository root):
  python 01-scripts/figures/fig02_03_S1_S4.py [--figures 2,3,S1,S4] [--outdir 03-figs]
"""

from __future__ import annotations

import argparse
import gzip
import math
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Prevent repeated matplotlib cache warnings on read-only HOME config dirs.
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig_fm5_ml"

# Cartopy caches Natural Earth data under a user data dir by default. In this sandbox that
# location may be read-only, so redirect to a writable temp folder.
if "CARTOPY_DATA_DIR" not in os.environ:
    os.environ["CARTOPY_DATA_DIR"] = "/tmp/cartopy_fm5_ml"

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from scipy.io import loadmat, netcdf_file
from scipy.signal import resample
from scipy.stats import lognorm


STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]
STATIONS_AVG = STATIONS + ["Average"]
STATIONS_AX = ["AXAS1", "AXAS2", "AXCC1", "AXEC1", "AXEC2", "AXEC3", "AXID1"]
STATIONS_AX_AVG = STATIONS_AX + ["Average"]

# ---------------------------------------------------------------------------
# Manuscript figure style (revision 1, in response to review comments 19/25/32/47)
#
# Two rules drive the choices below.
#
# 1. Figures are drawn at their final printed width (FIG_WIDTH_IN) so Word does
#    not have to scale them down.  Previously several panels were 10-19 inches
#    wide and were shrunk to ~6.5 in on the page, which reduced 10 pt labels to
#    3-6 pt.  Drawing at print size keeps the point sizes below honest.
# 2. A single semantic palette is shared across Figures 4/6/7/8/9.  The three
#    reference models introduced in Figure 6 keep their Figure 6 colour wherever
#    they reappear as a baseline; every *new* category introduced in a later
#    figure gets a colour that is not used in Figure 6.
# ---------------------------------------------------------------------------

FIG_WIDTH_IN = 7.0          # SRL single-page text width
FIG_WIDTH_WIDE_IN = 7.2     # for 4-column panel grids

# ---------------------------------------------------------------------------
# Original manuscript palette (restored at the author's request).
#
# Figures 6, 7, 8 and 9 each use the same three pastels in positional order,
# exactly as in the original scripts.  Figure 4 adds two more for its five
# benchmark methods.
# ---------------------------------------------------------------------------

C_SCRATCH = (0.95, 0.80, 0.45)    # amber
C_LOSO = (0.95, 0.60, 0.60)       # coral pink
C_TRANSFER = (0.60, 0.90, 0.90)   # turquoise
C_VAR_1 = (0.60, 0.80, 0.60)      # green   (Figure 4 only)
C_VAR_2 = (0.75, 0.65, 0.95)      # purple  (Figure 4 only)

# Figures 5 and 6 each compare two variant conditions against a Figure 4
# baseline.  The baseline bar keeps its Figure 4 hue; the two variants share
# this grey ramp in both figures, so hue always means "which model of Figure 4"
# and never "which variant".  Both variables are ordered -- SNR band in
# Figure 5, time shift in Figure 6 -- so the ramp darkens along that order and
# the figures stay readable in greyscale.
C_ALT_1 = (0.72, 0.72, 0.75)      # milder variant  (high SNR / sigma = 0.01 s)
C_ALT_2 = (0.45, 0.45, 0.50)      # harsher variant (low SNR / sigma = 0.02 s)

SERIES3 = [C_SCRATCH, C_LOSO, C_TRANSFER]

C_MODEL_ALL, C_MODEL_LOSO, C_MODEL_TL = C_SCRATCH, C_LOSO, C_TRANSFER
C_CAT_A, C_CAT_B, C_CAT_C = C_SCRATCH, C_LOSO, C_TRANSFER
C_VAR_3 = C_VAR_1
C_REFERENCE = C_VAR_2

C_BENCH = [C_SCRATCH, C_LOSO, C_TRANSFER, C_VAR_1, C_VAR_2]

A_PICK = "red"
A_NOTE = "blue"

# First-motion polarity colours (Figure 2, review comment: colour-code the
# traces by polarity).  Up / compressional = red, down / dilatational = blue;
# traces without a polarity (ambient noise) are neutral grey.
C_UP = "#c62828"
C_DOWN = "#1f4fb4"
C_NOPOL = "0.45"
A_INK = "k"

SEQ_CMAP = "jet"

TINT_NEUTRAL = "#f6f6f6"
TINT_CONV = "#e8f4ff"
TINT_POOL = "#e8ffe8"
TINT_REG = "#fff7e8"
TINT_LATENT = "#f0ecff"
TINT_CLASS = "#ffeef2"


def apply_manuscript_style() -> None:
    """Set global matplotlib style for every manuscript figure.

    Arial replaces Times New Roman throughout (review comment 19: axis labels
    should be sans-serif and legible at printed size).
    """
    import matplotlib

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.borderpad": 0.4,
        "figure.titlesize": 11,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.6,
        "savefig.dpi": 400,
        "mathtext.fontset": "dejavusans",
    })


def panel_label(ax: plt.Axes, text: str, dx: float = 0.0, dy: float = 1.012) -> None:
    """Consistent (a)/(b) panel labels, placed above the axes so they cannot
    collide with an in-axes legend."""
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            va="bottom", ha="left")


# From FM4/01-scripts/Before22OBSs/x_plot_2F_ineachstation.m
STATION_COORDS = {
    "AS1": (-129.9992, 45.9336),
    "AS2": (-130.0141, 45.9338),
    "CC1": (-130.0089, 45.9547),
    "EC1": (-129.9797, 45.9496),
    "EC2": (-129.9738, 45.9397),
    "EC3": (-129.9785, 45.9361),
    "ID1": (-129.9780, 45.9257),
}

# From FM4/01-scripts/findcommonshallowEast/axial_calderaRim.m
CALDERA_RIM = np.array(
    [
        [-130.004785563058, 45.9207755734405],
        [-130.010476202888, 45.9238241104543],
        [-130.018881564079, 45.9351908809594],
        [-130.023946125193, 45.9412238501725],
        [-130.028718653506, 45.949881200114],
        [-130.03045121938, 45.9511765797916],
        [-130.03067948565, 45.9542732243167],
        [-130.031733279709, 45.9558130656063],
        [-130.0314446535, 45.9586760104296],
        [-130.036188782208, 45.9656647517656],
        [-130.036950110789, 45.9698291665232],
        [-130.039953347, 45.9750458167927],
        [-130.038595675479, 45.9847117727418],
        [-130.035927416999, 45.9883113986506],
        [-130.018067675296, 45.993358288674],
        [-130.013629193751, 45.993755284135],
        [-130.010365710979, 45.9929499241491],
        [-130.008647442296, 45.9924883829037],
        [-130.007262470669, 45.9915471582374],
        [-130.006042022411, 45.9902469280907],
        [-130.00517862949, 45.989777805361],
        [-130.001868199523, 45.9863506519894],
        [-130.001154359192, 45.9846883853932],
        [-130.000949059432, 45.9827833001814],
        [-129.99939353433, 45.9818434725493],
        [-129.997797388662, 45.9786395525337],
        [-129.995357566829, 45.9760388622191],
        [-129.993956176267, 45.9741441737512],
        [-129.993678708114, 45.9681875631427],
        [-129.993140494256, 45.9667620754035],
        [-129.992087550788, 45.9652218741086],
        [-129.991186410747, 45.9626077204113],
        [-129.989604036931, 45.960118640732],
        [-129.989238986151, 45.9588108137369],
        [-129.989728217453, 45.9574955894078],
        [-129.98548409867, 45.9494279735802],
        [-129.98478812249, 45.9487188881587],
    ],
    dtype=float,
)






def load_struct_array(mat_path: Path, var_name: str) -> np.ndarray:
    d = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if var_name not in d:
        raise KeyError(f"{var_name} not found in {mat_path}")
    arr = d[var_name]
    if isinstance(arr, np.ndarray):
        return arr.ravel()
    return np.array([arr], dtype=object)


def load_scalar_or_array(mat_path: Path, var_name: str):
    d = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if var_name not in d:
        raise KeyError(f"{var_name} not found in {mat_path}")
    return d[var_name]


def get_value(obj, field: str, default=None):
    if not hasattr(obj, field):
        return default
    v = getattr(obj, field)
    if isinstance(v, np.ndarray) and v.size == 0:
        return default
    return v


def as_1d(a) -> np.ndarray:
    if a is None:
        return np.array([])
    if isinstance(a, np.ndarray):
        return a.astype(float).ravel()
    if isinstance(a, (list, tuple)):
        return np.asarray(a, dtype=float).ravel()
    return np.array([a], dtype=float)


def norm_wave(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float).ravel()
    if w.size == 0:
        return w
    m = np.max(np.abs(w))
    if m == 0:
        return w
    return w / m


def choose_existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p is not None and p.exists():
            return p
    return None








def save_figure(fig: plt.Figure, out_path: Path, dpi: int = 300) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)








def _plot_stacked_waves(
    ax: plt.Axes,
    entries: np.ndarray,
    station: str,
    n_plot: int,
    offset: float,
    use_po_label: bool,
    half_rate: bool,
    color: str = "b",
) -> None:
    wave_field = f"W_{station}"
    po_field = f"Po_{station}"
    count = 0
    for e in entries:
        w = get_value(e, wave_field, None)
        if not isinstance(w, np.ndarray) or w.size <= 1:
            continue
        w = np.asarray(w, dtype=float).ravel()
        if half_rate and w.size >= 2:
            w = w[::2]
        w = norm_wave(w)
        x = np.linspace(-0.5, 0.49, w.size)
        yoff = count * offset
        ax.plot(x, w + yoff, "-", color=color, linewidth=1.0)
        if use_po_label:
            po = get_value(e, po_field, None)
            if po is not None:
                ax.text(-0.3, yoff + 0.35, f"Po: {int(np.sign(float(po)))}", fontsize=7, color="black")
        count += 1
        if count >= n_plot:
            break
    ax.set_title(f"Station {station}", fontsize=9)
    ax.set_ylim([-offset, max(offset, count * offset)])
    ax.set_yticks([])
    ax.grid(alpha=0.2)
    ax.set_box_aspect(2.4)




def figure_02_merged(paths: Paths, outdir: Path) -> Path:
    """Merged replacement for Figures 2 and 3, restricted to one station (AXAS2).

    (a) high-SNR template waveforms, five of each manual polarity
    (b) ambient-noise waveforms
    (c) the same template augmented with noise across a range of SNRs

    The original figure_02 / figure_03 builders are left in place; this writes
    to a separate file so the two-figure version stays available as a backup.
    """
    sta = "AS2"
    n_trace = 10
    offset = 2.0
    wave_field = f"W_{sta}"
    po_field = f"Po_{sta}"

    template = load_struct_array(paths.repo_root / "02-data" / "raw" / "A_wave_dB20_cleaned.mat", "Felix")
    noise = load_struct_array(paths.repo_root / "02-data" / "raw" / "A_wave_noise_10000.mat", "Felix")

    def valid_wave(e):
        w = as_1d(get_value(e, wave_field, np.array([])))
        return w if w.size >= 200 else None

    def first_motion(w):
        """Sign and strength of the first half-cycle after the P arrival.

        Panel (a) plots the arrival at sample 100.  Strength is the peak of that
        first half-cycle divided by the pre-arrival RMS, so it says how far the
        first swing rises above the background.
        """
        pre = np.sqrt(np.mean(w[:90] ** 2)) + 1e-15
        seg = w[100:130]
        sign = int(np.sign(seg[0]))
        k = 1
        while k < seg.size and np.sign(seg[k]) == sign:
            k += 1
        return sign, float(np.max(np.abs(seg[:k])) / pre)

    # Five templates of each polarity, interleaved, so the panel shows both
    # cases.  A template is only used if its first motion actually reads as the
    # polarity it is labelled with: the first swing must share the sign of Po
    # and rise well clear of the pre-arrival noise.  Taking the first five of
    # each polarity without this test picked up emergent onsets and traces whose
    # first swing opposes the label, which is the opposite of what the panel is
    # meant to show.
    min_clarity = 40.0
    pos, neg = [], []
    for e in template:
        w = valid_wave(e)
        if w is None:
            continue
        po = get_value(e, po_field, None)
        if po is None:
            continue
        po = int(np.sign(float(po)))
        if po == 0:
            continue
        sign, clarity = first_motion(w[:200])
        if sign != po or clarity < min_clarity:
            continue
        bucket = pos if po > 0 else neg
        if len(bucket) < n_trace // 2:
            bucket.append((w, po))
        if len(pos) >= n_trace // 2 and len(neg) >= n_trace // 2:
            break
    picked = [x for pair in zip(neg, pos) for x in pair]

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, 5.6), constrained_layout=True)

    # ---- (a) templates ---------------------------------------------------
    ax = axes[0]
    for i, (w, po) in enumerate(picked):
        wn = norm_wave(w[::2])
        t = np.linspace(-0.5, 0.49, wn.size)
        yoff = i * offset
        c = C_UP if po > 0 else C_DOWN
        ax.plot(t, wn + yoff, "-", color=c, linewidth=1.0)
        ax.text(-0.46, yoff + 0.55, f"Po: {po:+d}", fontsize=7, color=c,
                fontweight="bold")
    ax.set_ylabel("Normalized amplitude")

    # ---- (b) noise -------------------------------------------------------
    # The same noise entries feed panel (c), so collect them once using the
    # validity test _plot_stacked_waves applies, and hand that list to both.
    noise_used = []
    for e in noise:
        w = get_value(e, wave_field, None)
        if not isinstance(w, np.ndarray) or w.size <= 1:
            continue
        noise_used.append(e)
        if len(noise_used) >= n_trace:
            break

    _plot_stacked_waves(axes[1], noise_used, sta, n_plot=n_trace, offset=offset,
                        use_po_label=False, half_rate=False, color=C_NOPOL)
    axes[1].set_title("")
    axes[1].set_box_aspect(None)

    # ---- (c) one template from (a) + the noise of (b), across a range of SNR
    # Panel (c) is built from the topmost template of (a) and the noise traces
    # of (b): row i of (c) is that template plus row i of (b), scaled to a
    # different target SNR.  Using one real template throughout means every
    # synthetic trace carries the same first motion as its parent in (a), which
    # is what makes the augmented waveforms usable as labelled training data.
    base_wave, base_po = picked[-1]       # the topmost template drawn in (a)
    signal = base_wave[:200]

    # Target SNRs come from the lognormal fitted to this station's empirical SNR
    # values -- the same distribution the training set is augmented from -- so
    # (c) shows the noise levels the model is actually trained on rather than an
    # arbitrary ladder.  Evenly spaced quantiles of that fit are used instead of
    # random draws: the spread is representative and no two traces land on top
    # of each other.
    snr_file = choose_existing(
        paths.repo_root / "02-data" / "raw" / "H_noise_dB20_snrValue.mat",
        paths.repo_root / "02-data" / "H_noi" / "H_noise_dB20_snrValue.mat",
    )
    if snr_file is not None:
        snr_values = load_scalar_or_array(snr_file, "snrValues")
        snr_data = np.asarray(snr_values[STATIONS.index(sta)], dtype=float).ravel()
    else:                                  # fall back to the templates themselves
        snr_data = np.array([
            20 * np.log10((np.sqrt(np.mean(w[80:160] ** 2)) + 1e-12)
                          / (np.sqrt(np.mean(w[:80] ** 2)) + 1e-12))
            for w, _ in picked])
    snr_data = snr_data[np.isfinite(snr_data) & (snr_data > 0)]
    shape, loc, scale = lognorm.fit(snr_data, floc=0)
    targets = lognorm.ppf(np.linspace(0.05, 0.95, n_trace - 1), shape, loc, scale)

    traces, trace_db = [], []
    for i, target_db in enumerate(targets):
        w = as_1d(get_value(noise_used[i], wave_field, np.array([])))
        w = np.pad(w, (0, 200 - w.size)) if w.size < 200 else w[:200]
        rms_target = np.sqrt(np.mean(signal ** 2)) / (10 ** (target_db / 20))
        traces.append(signal + (rms_target / (np.sqrt(np.mean(w ** 2)) + 1e-12)) * w)
        trace_db.append(target_db)
    traces.append(signal.copy())          # unaugmented template on top
    # The template carries no target, so label it with its own measured SNR.
    trace_db.append(20 * np.log10((np.sqrt(np.mean(signal[80:160] ** 2)) + 1e-12)
                                  / (np.sqrt(np.mean(signal[:80] ** 2)) + 1e-12)))

    ax = axes[2]
    ticks, labels = [], []
    for i, w in enumerate(traces):
        # Label the SNR the trace was built to, not a value re-measured over a
        # window that straddles the arrival: the latter is not monotonic in the
        # target and collides at the low end.
        snr_db = trace_db[i]
        is_base = i == len(traces) - 1
        wn = norm_wave(w[::2])
        t = np.linspace(-0.5, 0.49, wn.size)
        yoff = i * offset
        # Every synthetic trace inherits the polarity of its parent template
        # in (a), so it takes the same polarity colour; the unaugmented
        # template itself is drawn heavier.
        c = C_UP if base_po > 0 else C_DOWN
        ax.plot(t, wn + yoff, "-", color=c,
                linewidth=1.6 if is_base else 0.9)
        ax.text(-0.46, yoff + 0.55, f"Po: {base_po:+d}", fontsize=7, color=c,
                fontweight="bold")
        ticks.append(yoff)
        labels.append(f"{snr_db:.1f} dB")
    ax.yaxis.tick_right()
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(axis="y", length=0)
    for lab, is_base in zip(ax.get_yticklabels(), [False] * (len(traces) - 1) + [True]):
        lab.set_color("k")
        if is_base:
            lab.set_fontweight("bold")

    for ax in axes:
        ax.set_xlim([-0.5, 0.5])
        # The extra 1.2 leaves a clear strip above the top trace for the
        # in-axes panel label.
        ax.set_ylim([-offset, n_trace * offset + 1.2])
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.2)
    for ax in axes[:2]:
        ax.set_yticks([])

    from matplotlib.lines import Line2D
    fig.legend(
        [Line2D([], [], color=C_UP, lw=1.6), Line2D([], [], color=C_DOWN, lw=1.6),
         Line2D([], [], color=C_NOPOL, lw=1.6)],
        ["Up (Po = +1)", "Down (Po = \u22121)", "Noise (no polarity)"],
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
        fontsize=8.5)

    # Panel labels sit inside the top-left corner of each axes.
    for ax, lab in zip(axes, ("(a)", "(b)", "(c)")):
        panel_label(ax, lab, dx=0.025, dy=0.955)

    out = outdir / "Figure02_merged_python.png"
    save_figure(fig, out)
    return out




def _grouped_bar(ax: plt.Axes, data: np.ndarray, labels: Sequence[str], series: Sequence[str], colors: Sequence, ylim):
    n_groups, n_series = data.shape
    x = np.arange(n_groups)
    width = 0.82 / n_series
    for i in range(n_series):
        ax.bar(
            x - 0.41 + width / 2 + i * width,
            data[:, i],
            width=width,
            color=colors[i],
            label=series[i],
            edgecolor="k",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(ylim)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    # Separate the "Average" group from the per-station groups.
    if labels and str(labels[-1]).lower().startswith("average"):
        ax.axvline(n_groups - 1.5, color="0.55", linestyle=":", linewidth=0.9, zorder=0)


# ---------------------------------------------------------------------------
# Benchmark confusion matrices (rows = actual Up/Down, cols = predicted
# Up/Down).  These are the counts behind both the Average bar of Figure 3 and
# the supplementary confusion-matrix figure, so the two cannot drift apart.
# DiTingMotion is evaluated on fewer waveforms than the other three because it
# returns no prediction for part of the set.
# ---------------------------------------------------------------------------
# Benchmark results from 01-scripts/benchmark/score_benchmark.py (zero-shot
# predictions of the four published models on the Axial benchmark set).
BENCH_CSV = Path(__file__).resolve().parents[2] / "03-figs" / "benchmark_confusion_matrix.csv"
BENCH_MODELS = ["DiTingMotion", "CFM", "EQPolarity", "PolarCAP"]
# Cross-correlation accuracy per station on the benchmark templates
# (H_compare_Man_CC.m in the benchmark bundle); its inputs are not in the data archive.
CC_ACCURACY = np.array([0.99972, 0.95991, 0.99134, 0.93391, 0.84833, 0.87012, 0.80463]) * 100


def _half_up(x: float) -> float:
    from decimal import Decimal, ROUND_HALF_UP
    return float(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _load_benchmark():
    import csv
    rows = list(csv.DictReader(open(BENCH_CSV)))
    acc, cm = {}, {}
    for m in BENCH_MODELS:
        acc[m] = np.array([_half_up(float(r["Accuracy(%)"])) for s in STATIONS for r in rows
                           if r["Model"] == m and r["Station"] == s])
        r = next(r for r in rows if r["Model"] == m and r["Station"] == "ALL (pooled)")
        # rows = actual Up/Down, columns = predicted Up/Down
        cm[m] = [[int(r["PT"]), int(r["NF"])], [int(r["PF"]), int(r["NT"])]]
    return acc, cm


BENCH_ACC, CM_COUNTS = _load_benchmark()


def figure_s_confusion(paths: Paths, outdir: Path) -> Path:
    """Supplementary confusion matrices for the four benchmarked DL models.

    Styled to match the other manuscript figures: same text width, same 300 dpi
    export, same panel-label convention.
    """
    from matplotlib import cm as _cm
    from matplotlib.colors import Normalize

    order = ["DiTingMotion", "CFM", "EQPolarity", "PolarCAP"]
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, 6.1), constrained_layout=True)
    norm = Normalize(vmin=0, vmax=100)
    cmap = _cm.get_cmap("Blues")

    for ax, name, lab in zip(axes.ravel(), order, "abcd"):
        counts = np.array(CM_COUNTS[name], dtype=float)
        row_pct = 100.0 * counts / counts.sum(axis=1, keepdims=True)
        n = int(counts.sum())
        acc = 100.0 * (counts[0, 0] + counts[1, 1]) / n

        ax.imshow(row_pct, cmap=cmap, norm=norm, aspect="equal")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(counts[i, j])}\n({row_pct[i, j]:.1f}%)",
                        ha="center", va="center", fontsize=8.5,
                        color="white" if row_pct[i, j] > 55 else "k")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred. Up", "Pred. Down"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Up", "Actual Down"], fontsize=8)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
        ax.set_title(f"({lab}) {name}\nn = {n:,}, accuracy = {acc:.1f}%", fontsize=9)

    cb = fig.colorbar(_cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                      fraction=0.040, pad=0.02)
    cb.set_label("Row-normalized (%)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    out = outdir / "FigureS_confusion_python.png"
    save_figure(fig, out)
    return out


def figure_04(paths: Paths, outdir: Path) -> Path:
    colors = C_BENCH
    # DiTingMotion, EQPolarity and PolarCAP re-run 2026-09-18; CFM and the
    # cross-correlation baseline are unchanged and keep their original values.
    diting, cfm, eqp, polcap = (BENCH_ACC[m] for m in BENCH_MODELS)
    cc = CC_ACCURACY
    # The Average bar is the pooled accuracy over all waveforms, not the
    # unweighted mean of the seven station accuracies: it is the value the
    # confusion matrices of Figure S1 report, and it weights each station by how
    # many waveforms it contributes.  Taken from CM_COUNTS below so the bar and
    # the confusion matrix can never disagree.
    pooled = {k: 100.0 * (v[0][0] + v[1][1]) / sum(v[0] + v[1])
              for k, v in CM_COUNTS.items()}
    data = np.vstack(
        [
            np.r_[diting, pooled["DiTingMotion"]],
            np.r_[cfm, pooled["CFM"]],
            np.r_[eqp, pooled["EQPolarity"]],
            np.r_[polcap, pooled["PolarCAP"]],
            np.r_[cc, cc.mean()],   # no confusion matrix for CC; station mean
        ]
    ).T

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.6), constrained_layout=True)
    _grouped_bar(
        ax, data, STATIONS_AX_AVG,
        ["DiTingMotion", "CFM", "EQPolarity", "PolarCAP", "Cross-correlation"],
        colors, (60, 104),
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Station")
    ax.set_title("")
    ax.legend(ncol=5, loc="upper center", columnspacing=1.0, handlelength=1.3,
              handletextpad=0.4, borderaxespad=0.2, fontsize=7.5)
    out = outdir / "Figure04_python.png"
    save_figure(fig, out)
    return out












def _build_conflict_data(paths: Paths):
    # 2015-2021 catalogue windows and AxialPolCap predictions (05_apply_to_catalog.py):
    # W_{STA} = 64-sample window at 100 Hz, P pick at sample 32;
    # Po_{STA} = [CC, ML, confidence, entropy]. A disagreement is a pick with
    # confidence >= 0.95 whose ML polarity differs from CC (Section 4.1).
    clu_path = paths.repo_root / "02-data" / "A_wave_2015_2021_AxialPolCap_final_conf80_predictions.mat"
    po_clu = load_struct_array(clu_path, "Felix")
    felix = po_clu
    id_to_idx = {int(get_value(e, "ID", -1)): i for i, e in enumerate(felix)}

    po_fields = [f"Po_{s}" for s in STATIONS]
    w_fields = [f"W_{s}" for s in STATIONS]

    conflict_wave: Dict[str, List[np.ndarray]] = {s: [] for s in STATIONS}
    conflict_labels: Dict[str, List[str]] = {s: [] for s in STATIONS}
    valid_loc: Dict[str, List[Tuple[float, float]]] = {s: [] for s in STATIONS}
    conflict_loc: Dict[str, List[Tuple[float, float]]] = {s: [] for s in STATIONS}
    conflict_id: Dict[str, List[int]] = {s: [] for s in STATIONS}

    for row in po_clu:
        rid = int(get_value(row, "ID", -1))
        j = id_to_idx.get(rid, None)
        if j is None:
            continue
        src = felix[j]
        lat = float(get_value(src, "lat", np.nan))
        lon = float(get_value(src, "lon", np.nan))
        for po_f, w_f, sta in zip(po_fields, w_fields, STATIONS):
            val = get_value(row, po_f, None)
            if val is None:
                continue
            val = as_1d(val)
            if val.size < 3:
                continue
            p1, p2 = float(val[0]), float(val[1])  # CC, ML
            if p1 == 0 or p2 == 0 or float(val[2]) < 0.95:
                continue
            valid_loc[sta].append((lon, lat))
            if p1 != p2:
                w = as_1d(get_value(src, w_f, np.array([])))
                if w.size > 0:
                    conflict_wave[sta].append(w)
                    conflict_labels[sta].append(f"{int(np.sign(p1))} -> {int(np.sign(p2))}")
                    conflict_loc[sta].append((lon, lat))
                    conflict_id[sta].append(rid)
    return conflict_wave, conflict_labels, valid_loc, conflict_loc, conflict_id


def figure_s4(paths: Paths, outdir: Path) -> Path:
    """Fig. S4: CC/ML polarity disagreements, styled like Fig. 2.

    Two panels, each holding ten randomly drawn conflicting waveforms, stacked
    and aligned on the catalog P pick.  Every station contributes at least one
    trace to a panel and the three remaining slots are filled at random from the
    stations that still have unused disagreements; the two panels draw disjoint
    sets of events.  Each trace is labelled on the right with its station and
    event ID.  Traces are the 64-sample, 100 Hz catalogue windows classified by
    AxialPolCap, so the 2015-2021 disagreements of Section 4.1 are shown.
    """
    conflict_wave, conflict_labels, _, _, conflict_id = _build_conflict_data(paths)

    pick, half = 32, 25          # 100 Hz, P pick at sample 32 -> +/- 0.25 s
    offset = 2.0
    rng = np.random.default_rng(7)

    sym = {"1": r"$\uparrow$", "-1": r"$\downarrow$"}
    def pretty(label: str) -> str:
        cc_s, ml_s = label.split(" -> ")
        return f"CC {sym.get(cc_s.strip(), '?')} / ML {sym.get(ml_s.strip(), '?')}"

    n_per_panel = 10

    # Usable conflicting traces per station, shuffled once up front so that the
    # two panels consume disjoint events.
    pools, cursor = {}, {}
    for sta in STATIONS:
        idx = [i for i, w in enumerate(conflict_wave[sta])
               if as_1d(w).size >= pick + half]
        rng.shuffle(idx)
        pools[sta], cursor[sta] = idx, 0

    def take(sta: str):
        """Next unused conflicting trace for this station, or None."""
        if cursor[sta] >= len(pools[sta]):
            return None
        j = pools[sta][cursor[sta]]
        cursor[sta] += 1
        return j

    # two independent draws: every station once, then random extras up to ten
    draws_meta = []
    for _ in range(2):
        picked = []
        for k, (sta, sta_ax) in enumerate(zip(STATIONS, STATIONS_AX)):
            j = take(sta)
            if j is not None:
                picked.append((k, sta, sta_ax, j))
        while len(picked) < n_per_panel:
            avail = [(k, sta, sta_ax)
                     for k, (sta, sta_ax) in enumerate(zip(STATIONS, STATIONS_AX))
                     if cursor[sta] < len(pools[sta])]
            if not avail:
                break
            k, sta, sta_ax = avail[int(rng.integers(len(avail)))]
            picked.append((k, sta, sta_ax, take(sta)))
        picked.sort(key=lambda r: r[0])          # keep the station order on the page
        draws_meta.append(picked)

    draws = [[(sta_ax, conflict_id[sta][j], as_1d(conflict_wave[sta][j]),
               conflict_labels[sta][j])
              for _, sta, sta_ax, j in picked]
             for picked in draws_meta]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 7.2), constrained_layout=True)
    t = np.arange(-half, half) / 100.0
    for ax, picked in zip(axes, draws):
        ticks, labels = [], []
        for i, (sta_ax, eid, w, lab) in enumerate(picked):
            seg = norm_wave(w[pick - half:pick + half])
            yoff = i * offset
            ax.plot(t, seg + yoff, "b-", linewidth=1.0)
            ax.text(-0.243, yoff + 0.62, pretty(lab), color=A_NOTE, fontsize=7,
                    va="center", ha="left")
            ticks.append(yoff)
            labels.append(f"{sta_ax}  {eid}")
        ax.axvline(0.0, color=A_PICK, linestyle="--", linewidth=0.9)
        ax.yaxis.tick_right()
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=7)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim([-0.25, 0.25])
        ax.set_ylim([-offset, len(picked) * offset])
        ax.set_xlabel("Time from P pick (s)")
        ax.grid(alpha=0.2)

    for ax, lab in zip(axes, ("(a)", "(b)")):
        panel_label(ax, lab)

    out = outdir / "FigureS4_python.png"
    save_figure(fig, out)
    return out




def write_benchmark_table(outdir: Path) -> Path:
    """Supplementary table of per-station benchmark accuracies (review comment 16:
    "the reader might wonder which performs best of the DL models").

    Numbers are the same ones plotted in Figure 4.
    """
    # DiTingMotion, EQPolarity and PolarCAP re-run 2026-09-18; CFM and the
    # cross-correlation baseline are unchanged and keep their original values.
    diting, cfm, eqp, polcap = (BENCH_ACC[m] for m in BENCH_MODELS)
    cc = CC_ACCURACY

    methods = [
        ("DiTingMotion", diting),
        ("CFM", cfm),
        ("EQPolarity", eqp),
        ("PolarCAP", polcap),
        ("Cross-correlation", cc),
    ]

    lines = [
        "Table S1. Station-wise P-wave first-motion polarity accuracy (%) on the "
        "synthetic benchmark data set, for four published deep-learning classifiers "
        "and the cross-correlation method. Accuracies are measured against the "
        "manually verified template polarities.",
        "",
        "| Method | " + " | ".join(STATIONS_AX) + " | Average |",
        "|" + "---|" * (len(STATIONS_AX) + 2),
    ]
    # Average column is the pooled accuracy, matching the Average bar of
    # Figure 3 and the confusion matrices of Figure S1.
    pooled = {k: 100.0 * (v[0][0] + v[1][1]) / sum(v[0] + v[1])
              for k, v in CM_COUNTS.items()}
    avg = {name: pooled.get(name, arr.mean()) for name, arr in methods}

    for name, arr in methods:
        cells = " | ".join(f"{v:.1f}" for v in arr)
        lines.append(f"| {name} | {cells} | **{avg[name]:.1f}** |")

    best = max(methods[:4], key=lambda m: avg[m[0]])
    worst = min(methods[:4], key=lambda m: avg[m[0]])
    lines += [
        "",
        f"Best deep-learning model: {best[0]} ({avg[best[0]]:.1f}% average).",
        f"Worst deep-learning model: {worst[0]} ({avg[worst[0]]:.1f}% average).",
        f"Deep-learning range: {min(avg[m[0]] for m in methods[:4]):.1f}-"
        f"{max(avg[m[0]] for m in methods[:4]):.1f}%; "
        f"cross-correlation: {cc.mean():.1f}%.",
    ]

    out = outdir / "TableS1_benchmark_accuracy.md"
    out.write_text("\n".join(lines) + "\n")
    return out


@dataclass
class Paths:
    repo_root: Path


FIGURES = {"2": (figure_02_merged, "GJI_Figure02.png"), "3": (figure_04, "GJI_Figure03.png"),
           "S1": (figure_s_confusion, "GJI_FigureS1.png"), "S4": (figure_s4, "GJI_FigureS4.png")}


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figures", default="2,3,S1,S4", help="comma list of 2, 3, S1, S4")
    parser.add_argument("--outdir", default=str(repo / "03-figs"))
    args = parser.parse_args()
    apply_manuscript_style()
    paths, outdir = Paths(repo_root=repo), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for fid in [f.strip() for f in args.figures.split(",")]:
        fn, name = FIGURES[fid]
        written = Path(fn(paths, outdir))
        written.replace(outdir / name)
        print(f"Fig. {fid} -> {outdir / name}")
        if fid == "3":
            print(f"Table S2 -> {write_benchmark_table(outdir)}")


if __name__ == "__main__":
    main()
