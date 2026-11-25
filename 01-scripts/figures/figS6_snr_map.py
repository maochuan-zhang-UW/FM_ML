#!/usr/bin/env python3
"""Fig. S6: spatial variation of P-wave signal-to-noise ratio across the Axial network.

Companion to Figure S4: the same 200 m grid, the same per-cell minimum and the
same panel layout, but coloured by the median SNR of the picks in each cell
rather than by the AxialPolCap-CC disagreement rate.  Plotting the two on the
same cells lets the reader judge directly whether the spatial pattern of
disagreement follows the spatial pattern of SNR.

SNR values are computed from the catalogue windows exactly as in Figure 1d
(fig01_background.py), so the two figures cannot disagree.  By default the cells are restricted to the picks actually compared in
Section 4.1: template station-event pairs are dropped, and a pick is kept only
when AxialPolCap returns a class probability of at least CONF_MIN and the
cross-correlation method also returns a polarity.

    python 01-scripts/figures/figS6_snr_map.py \
        --grid-m 200 --min-per-cell 150 \
        --output 03-figs/GJI_FigureS6.png
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig_fm5_ml"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_MAT = REPO_ROOT / "02-data" / "raw" / "A_wave_dB20_cleaned.mat"
NOISE_END = 20
PRED_MAT = REPO_ROOT / "02-data" / "A_wave_2015_2021_AxialPolCap_final_conf80_predictions.mat"
OUTPUT = REPO_ROOT / "03-figs" / "GJI_FigureS6.png"

STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]
GRID_SIZE_M = 200.0
MIN_PER_CELL = 150
CONF_MIN = 0.95
LON_LIM = [-130.031, -129.97]
LAT_LIM = [45.92, 45.972]


def fig_helpers():
    spec = importlib.util.spec_from_file_location(
        "mmf", REPO_ROOT / "01-scripts" / "figures" / "fig02_03_S1_S4.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_snr():
    """Per station-event SNR joined to the event location.

    SNR = 20 log10 of the peak absolute amplitude after / before sample
    NOISE_END of the 64-sample catalogue window, as in Fig. 1d
    (fig01_background.py). Template flags are per event, as in Fig. 1.
    """
    felix = np.atleast_1d(loadmat(str(PRED_MAT), struct_as_record=False, squeeze_me=True)["Felix"])
    tmpl = {int(e.ID) for e in np.atleast_1d(
        loadmat(str(TEMPLATE_MAT), struct_as_record=False, squeeze_me=True)["Felix"])}
    snr, ev, sta, ist, lon, lat = [], [], [], [], {}, {}
    for e in felix:
        eid = int(e.ID)
        lon[eid], lat[eid] = float(e.lon), float(e.lat)
        for i, s in enumerate(STATIONS):
            w = np.abs(np.asarray(getattr(e, f"W_{s}", []), float).ravel())
            if w.size <= NOISE_END:
                continue
            nz, sg = w[:NOISE_END].max(), w[NOISE_END:].max()
            if nz > 0 and np.isfinite(sg):
                snr.append(20 * np.log10(sg / nz)); ev.append(eid); sta.append(i + 1); ist.append(eid in tmpl)
    return (np.array(snr), np.array(ev, np.int64), np.array(sta), np.array(ist, int), lon, lat)


def load_compared_pairs():
    """(station, event id) pairs that enter the Section 4.1 comparison."""
    felix = np.atleast_1d(
        loadmat(str(PRED_MAT), struct_as_record=False, squeeze_me=True)["Felix"]
    )
    keep = {s: set() for s in STATIONS}
    for ev in felix:
        eid = int(np.atleast_1d(ev.ID).ravel()[0])
        for sta in STATIONS:
            po = getattr(ev, f"Po_{sta}", None)
            if po is None:
                continue
            po = np.atleast_1d(po).ravel()
            if po.size < 3:
                continue
            # po = [CC polarity, ML polarity, confidence, entropy]
            if po[0] == 0 or po[1] == 0 or float(po[2]) < CONF_MIN:
                continue
            keep[sta].add(eid)
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-m", type=float, default=GRID_SIZE_M)
    ap.add_argument("--min-per-cell", type=int, default=MIN_PER_CELL)
    ap.add_argument("--vmin", type=float, default=8.0)
    ap.add_argument("--vmax", type=float, default=28.0)
    ap.add_argument("--all-picks", action="store_true",
                    help="use every non-template pick instead of only those "
                         "compared in Section 4.1")
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()

    fh = fig_helpers()
    fh.apply_manuscript_style()

    snr, snr_ev, snr_sta, snr_tmpl, lon_of, lat_of = load_snr()
    compared = None if args.all_picks else load_compared_pairs()

    mean_lat = np.deg2rad(np.mean(LAT_LIM))
    aspect = 1.0 / np.cos(mean_lat)
    dlat = args.grid_m / 111190.0
    dlon = args.grid_m / (111190.0 * np.cos(mean_lat))
    lon_edges = np.arange(LON_LIM[0], LON_LIM[1] + dlon, dlon)
    lat_edges = np.arange(LAT_LIM[0], LAT_LIM[1] + dlat, dlat)

    norm = Normalize(args.vmin, args.vmax)
    cmap = plt.get_cmap("viridis")
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, axes = plt.subplots(2, 4, figsize=(fh.FIG_WIDTH_WIDE_IN, 5.4),
                             constrained_layout=True)
    kept_cells = 0
    print(f"grid {args.grid_m:.0f} m, minimum {args.min_per_cell} picks per cell, "
          f"{'all non-template picks' if args.all_picks else 'Section 4.1 compared picks'}")

    for i, sta in enumerate(STATIONS):
        ax = axes.ravel()[i]
        sel = (snr_sta == i + 1) & (snr_tmpl == 0) & np.isfinite(snr)
        ids = snr_ev[sel]
        vals = snr[sel]
        if compared is not None:
            ok = np.fromiter((e in compared[sta] for e in ids.tolist()),
                             dtype=bool, count=ids.size)
            ids, vals = ids[ok], vals[ok]
        xs = np.fromiter((lon_of.get(e, np.nan) for e in ids.tolist()),
                         dtype=float, count=ids.size)
        ys = np.fromiter((lat_of.get(e, np.nan) for e in ids.tolist()),
                         dtype=float, count=ids.size)
        good = np.isfinite(xs) & np.isfinite(ys)
        xs, ys, vals = xs[good], ys[good], vals[good]

        ix = np.digitize(xs, lon_edges) - 1
        iy = np.digitize(ys, lat_edges) - 1
        inside = (ix >= 0) & (ix < len(lon_edges) - 1) & (iy >= 0) & (iy < len(lat_edges) - 1)
        ix, iy, vals = ix[inside], iy[inside], vals[inside]

        flat = ix * (len(lat_edges) - 1) + iy
        order = np.argsort(flat, kind="stable")
        flat, vsorted = flat[order], vals[order]
        bounds = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1], True])
        n_cell = 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < args.min_per_cell:
                continue
            med = float(np.median(vsorted[a:b]))
            cx, cy = divmod(int(flat[a]), len(lat_edges) - 1)
            ax.fill([lon_edges[cx], lon_edges[cx + 1], lon_edges[cx + 1], lon_edges[cx]],
                    [lat_edges[cy], lat_edges[cy], lat_edges[cy + 1], lat_edges[cy + 1]],
                    facecolor=cmap(norm(med)), edgecolor="none", zorder=1)
            n_cell += 1
        kept_cells += n_cell

        ax.plot(fh.CALDERA_RIM[:, 0], fh.CALDERA_RIM[:, 1], "k-", linewidth=1.0, zorder=3)
        if sta in fh.STATION_COORDS:
            x, y = fh.STATION_COORDS[sta]
            ax.plot(x, y, "s", color="k", markersize=4.5, markeredgecolor="w",
                    markeredgewidth=0.6, zorder=4)
        median_all = float(np.median(vals)) if vals.size else float("nan")
        print(f"  {fh.STATIONS_AX[i]}: n={vals.size:6d}  median={median_all:5.1f} dB  cells={n_cell}")
        ax.set_title(f"{fh.STATIONS_AX[i]}  ({median_all:.1f} dB)", fontsize=9, pad=3)
        ax.set_xlim(LON_LIM)
        ax.set_ylim(LAT_LIM)
        ax.set_aspect(aspect)
        ax.set_xticks([-130.02, -130.00, -129.98])
        ax.set_yticks([45.93, 45.94, 45.95, 45.96, 45.97])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=7.5)
        if i % 4:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Latitude", fontsize=8.5)
        if i < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Longitude", fontsize=8.5)

    cax = axes.ravel()[7]
    cax.axis("off")
    cb = fig.colorbar(mappable, ax=cax, fraction=0.55, aspect=18, extend="both")
    cb.set_label("Median P-wave SNR (dB)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)

    print(f"  grid cells drawn: {kept_cells}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
