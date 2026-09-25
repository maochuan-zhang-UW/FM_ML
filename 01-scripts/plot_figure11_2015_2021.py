#!/usr/bin/env python3
"""Plot Figure 11-style polarity conflict maps for the 2015-2021 catalog.

The cell size, the per-cell pick minimum and the colour-scale cap are all
command-line options, because the main-text and supplementary versions of this
map use different settings.  The defaults reproduce the compact main-text
figure (50 m cells, 50 picks minimum, scale to 50%).  Supplementary Figure S4
uses 200 m cells, a 150-pick minimum and a 30% scale cap:

    python 01-scripts/plot_figure11_2015_2021.py \
        --grid-m 200 --min-per-cell 150 --ratio-max 30 --exclude-templates \
        --output 03-figs/SRL_FigureS4.png
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig_fm5_ml"

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter

# Style is inherited from make_manuscript_figures.apply_manuscript_style() so that
# this figure matches the rest of the manuscript (review comment 19: sans-serif,
# legible at printed size).


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "02-data" / "A_wave_2015_2021_h5_conf80_predictions.mat"
DEFAULT_OUTPUT = REPO_ROOT / "03-figs" / "Figure11_2015_2021_h5_conf80_compact_python.png"
# Defaults for the compact main-text figure.  Cell size and the per-cell
# minimum trade against each other: halving the cell quarters the picks it
# holds, so a finer grid shows more detail but drops more cells below the
# minimum.  Override all three on the command line (see the module docstring).
GRID_SIZE_M = 50.0
MIN_EVENTS_PER_GRID = 50
RATIO_MAX_DEFAULT = 50.0     # colour-scale cap, per cent disagreement
CONF_MIN = 0.95              # class-probability threshold used in Section 4.1
# Station-event pairs used as training templates.  Section 4.1 excludes them from
# the CC/ML comparison so the reported agreement is measured only on picks the
# model never saw; pass --exclude-templates to do the same here.
TEMPLATE_H5 = REPO_ROOT / "02-data" / "A_wave_dB20_cleaned.h5"


def load_figure_helpers():
    source = REPO_ROOT / "01-scripts" / "make_manuscript_figures.py"
    spec = importlib.util.spec_from_file_location("make_manuscript_figures", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helpers from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_template_ids(h5_path: Path, stations) -> Dict[str, set]:
    """Event ids used as training templates, per station."""
    import h5py
    out: Dict[str, set] = {}
    with h5py.File(h5_path, "r") as fh:
        for sta in stations:
            out[sta] = {int(i) for i in fh[sta]["event_id"][:]} if sta in fh else set()
    return out


def build_conflict_locations(fig, mat_path: Path, template_ids=None):
    felix = fig.load_struct_array(mat_path, "Felix")
    n_excluded = 0

    valid_loc: Dict[str, List[Tuple[float, float]]] = {s: [] for s in fig.STATIONS}
    conflict_loc: Dict[str, List[Tuple[float, float]]] = {s: [] for s in fig.STATIONS}
    counts = {s: {"agree": 0, "conflict": 0, "valid": 0} for s in fig.STATIONS}

    for event in felix:
        lon = float(fig.get_value(event, "lon", np.nan))
        lat = float(fig.get_value(event, "lat", np.nan))
        if not np.isfinite(lon) or not np.isfinite(lat):
            continue

        rid = int(fig.get_value(event, "ID", -1))
        for sta in fig.STATIONS:
            val = fig.as_1d(fig.get_value(event, f"Po_{sta}", None))
            if val.size < 2:
                continue
            if template_ids is not None and rid in template_ids[sta]:
                n_excluded += 1
                continue
            manual, predicted = float(val[0]), float(val[1])
            if manual == 0 or predicted == 0:
                continue
            # Same screening as Section 4.1: keep only confident predictions.
            if val.size >= 3 and float(val[2]) < CONF_MIN:
                continue

            valid_loc[sta].append((lon, lat))
            counts[sta]["valid"] += 1
            if manual != predicted:
                conflict_loc[sta].append((lon, lat))
                counts[sta]["conflict"] += 1
            else:
                counts[sta]["agree"] += 1

    if template_ids is not None:
        print(f"Excluded {n_excluded:,} template station-event pairs from the comparison")
    return valid_loc, conflict_loc, counts


def west_wall_diagnostic(fig_helpers, valid_loc, conflict_loc) -> None:
    """Quantify the West Wall disagreement WW flagged (review comment 48).

    Prints, per station, the ML-CC disagreement rate inside a box covering the
    western caldera wall versus the rest of the catalog, so the pattern can be
    described quantitatively in the text instead of left for the reader to spot.
    """
    # West wall box, read off the caldera rim: western wall between the SW and NW
    # bends of the rim.
    wl_lon = (-130.0265, -130.0090)
    wl_lat = (45.9290, 45.9520)

    def in_box(pts):
        if pts.size == 0:
            return np.zeros(0, dtype=bool)
        return (
            (pts[:, 0] >= wl_lon[0]) & (pts[:, 0] <= wl_lon[1])
            & (pts[:, 1] >= wl_lat[0]) & (pts[:, 1] <= wl_lat[1])
        )

    print("")
    print(f"West Wall box: lon {wl_lon[0]} to {wl_lon[1]}, lat {wl_lat[0]} to {wl_lat[1]}")
    print(f"{'station':>8}  {'westwall N':>10} {'westwall %':>10}  {'elsewhere N':>11} {'elsewhere %':>11}")
    for sta in fig_helpers.STATIONS:
        v = np.array(valid_loc[sta], dtype=float) if valid_loc[sta] else np.empty((0, 2))
        c = np.array(conflict_loc[sta], dtype=float) if conflict_loc[sta] else np.empty((0, 2))
        vw, cw = in_box(v), in_box(c)
        n_w, n_o = int(vw.sum()), int((~vw).sum())
        k_w, k_o = int(cw.sum()), int((~cw).sum())
        pw = 100.0 * k_w / n_w if n_w else float("nan")
        po = 100.0 * k_o / n_o if n_o else float("nan")
        print(f"{'AX' + sta:>8}  {n_w:>10d} {pw:>10.1f}  {n_o:>11d} {po:>11.1f}")


def plot_figure11_2015_2021(
    mat_path: Path,
    out_path: Path,
    grid_size_m: float = GRID_SIZE_M,
    min_per_cell: int = MIN_EVENTS_PER_GRID,
    ratio_max: float = RATIO_MAX_DEFAULT,
    exclude_templates: bool = False,
) -> Path:
    fig_helpers = load_figure_helpers()
    fig_helpers.apply_manuscript_style()
    template_ids = (load_template_ids(TEMPLATE_H5, fig_helpers.STATIONS)
                    if exclude_templates else None)
    valid_loc, conflict_loc, counts = build_conflict_locations(
        fig_helpers, mat_path, template_ids)

    lon_lim = [-130.031, -129.97]
    lat_lim = [45.92, 45.972]
    mean_lat_rad = np.deg2rad(np.mean(lat_lim))
    geo_aspect = 1.0 / np.cos(mean_lat_rad)
    lat_step = grid_size_m / 111_190.0
    lon_step = grid_size_m / (111_190.0 * np.cos(mean_lat_rad))
    lon_edges = np.arange(lon_lim[0], lon_lim[1] + lon_step, lon_step)
    lat_edges = np.arange(lat_lim[0], lat_lim[1] + lat_step, lat_step)
    # Cell rates run to >90%, so the scale is capped where it clips little and
    # the ``extend="max"`` arrow carries the rest.
    ratio_norm = Normalize(vmin=0.0, vmax=ratio_max)
    ratio_cmap = plt.get_cmap("YlOrRd")
    ratio_mappable = plt.cm.ScalarMappable(norm=ratio_norm, cmap=ratio_cmap)

    cells_drawn = 0
    cell_counts: List[float] = []

    n_col = 4
    figure, axes = plt.subplots(
        2, n_col, figsize=(fig_helpers.FIG_WIDTH_WIDE_IN, 5.4), constrained_layout=True
    )
    for i, sta in enumerate(fig_helpers.STATIONS):
        ax = axes.ravel()[i]
        sta_ax = fig_helpers.STATIONS_AX[i]
        valid = np.array(valid_loc[sta], dtype=float) if valid_loc[sta] else np.empty((0, 2))
        conf = np.array(conflict_loc[sta], dtype=float) if conflict_loc[sta] else np.empty((0, 2))

        if valid.size > 0:
            total_grid, _, _ = np.histogram2d(valid[:, 0], valid[:, 1], bins=[lon_edges, lat_edges])
            if conf.size > 0:
                conflict_grid, _, _ = np.histogram2d(conf[:, 0], conf[:, 1], bins=[lon_edges, lat_edges])
            else:
                conflict_grid = np.zeros_like(total_grid)

            for ix in range(total_grid.shape[0]):
                for iy in range(total_grid.shape[1]):
                    total = total_grid[ix, iy]
                    if total < min_per_cell:
                        continue
                    cells_drawn += 1
                    cell_counts.append(float(total))
                    ratio = 100.0 * conflict_grid[ix, iy] / total
                    x0, x1 = lon_edges[ix], lon_edges[ix + 1]
                    y0, y1 = lat_edges[iy], lat_edges[iy + 1]
                    ax.fill(
                        [x0, x1, x1, x0],
                        [y0, y0, y1, y1],
                        facecolor=ratio_cmap(ratio_norm(ratio)),
                        edgecolor="none",
                        zorder=1,
                    )

        ax.plot(fig_helpers.CALDERA_RIM[:, 0], fig_helpers.CALDERA_RIM[:, 1], "k-",
                linewidth=1.0, zorder=3)
        if sta in fig_helpers.STATION_COORDS:
            slon, slat = fig_helpers.STATION_COORDS[sta]
            ax.plot(slon, slat, "s", color="k", markersize=4.5, markeredgecolor="w",
                    markeredgewidth=0.6, zorder=4)

        # Station-average disagreement rate, so the reader gets the number from
        # the figure rather than having to infer it from the colours.
        c = counts[sta]
        pct = 100.0 * c["conflict"] / c["valid"] if c["valid"] else float("nan")
        ax.set_title(f"{sta_ax}  ({pct:.1f}%)", fontsize=9, pad=3)

        ax.set_xlim(lon_lim)
        ax.set_ylim(lat_lim)
        ax.set_aspect(geo_aspect)
        ax.set_xticks([-130.02, -130.00, -129.98])
        ax.set_yticks([45.93, 45.94, 45.95, 45.96, 45.97])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.15, linewidth=0.4)
        ax.set_axisbelow(False)
        if i % n_col == 0:
            ax.set_ylabel("Latitude", fontsize=9)
        else:
            ax.set_yticklabels([])
        if i >= n_col - 1:
            ax.set_xlabel("Longitude", fontsize=9)
        else:
            ax.set_xticklabels([])

    # Colourbar occupies the unused 8th cell.
    cax = axes.ravel()[7]
    cax.set_axis_off()
    cbar = figure.colorbar(ratio_mappable, ax=cax, fraction=0.55, aspect=18,
                           extend="max")
    cbar.set_label("ML\u2013CC polarity disagreement (%)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)
    figure.suptitle("")
    fig_helpers.save_figure(figure, out_path)

    print(f"Wrote {out_path}")
    print(f"Grid {grid_size_m:.0f} m, minimum {min_per_cell} picks per cell, "
          f"scale capped at {ratio_max:.0f}%")
    if cell_counts:
        arr = np.asarray(cell_counts)
        print(f"  cells drawn: {cells_drawn} (picks per cell: min {arr.min():.0f}, "
              f"median {np.median(arr):.0f}, max {arr.max():.0f})")
    else:
        print("  cells drawn: 0")
    print("Station valid/agree/conflict counts:")
    for sta in fig_helpers.STATIONS:
        c = counts[sta]
        pct = 100.0 * c["conflict"] / c["valid"] if c["valid"] else float("nan")
        print(f"  {sta}: valid={c['valid']}, agree={c['agree']}, conflict={c['conflict']} ({pct:.2f}%)")

    west_wall_diagnostic(fig_helpers, valid_loc, conflict_loc)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Prediction .mat file")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output PNG path")
    parser.add_argument("--grid-m", type=float, default=GRID_SIZE_M,
                        help="Grid cell size in metres")
    parser.add_argument("--min-per-cell", type=int, default=MIN_EVENTS_PER_GRID,
                        help="Minimum polarity picks a cell needs to be drawn")
    parser.add_argument("--ratio-max", type=float, default=RATIO_MAX_DEFAULT,
                        help="Upper limit of the disagreement colour scale (%%)")
    parser.add_argument("--exclude-templates", action="store_true",
                        help="Drop station-event pairs used as training templates")
    args = parser.parse_args()

    mat_path = Path(args.input)
    out_path = Path(args.output)
    if not mat_path.exists():
        raise FileNotFoundError(mat_path)
    plot_figure11_2015_2021(mat_path, out_path, grid_size_m=args.grid_m,
                            min_per_cell=args.min_per_cell, ratio_max=args.ratio_max,
                            exclude_templates=args.exclude_templates)


if __name__ == "__main__":
    main()
