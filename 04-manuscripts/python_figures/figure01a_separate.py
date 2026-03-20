#!/usr/bin/env python3
"""
Standalone Figure 1a remake in Python.

Updates requested:
- Separate script for Figure 1a only
- Top-right global-location inset
- Geographic x/y ratio adjustment for better map appearance
- Times New Roman for all text
"""

from __future__ import annotations

import argparse
import gzip
import math
import os
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Avoid matplotlib cache warnings on restricted HOME dirs.
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig_fm5_ml"

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.io import loadmat, netcdf_file


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


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def choose_existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p is not None and p.exists():
            return p
    return None


def load_scalar_or_array(mat_path: Path, var_name: str):
    d = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if var_name not in d:
        raise KeyError(f"{var_name} not found in {mat_path}")
    return d[var_name]


def parse_matlab_lava1998(lava_m_path: Path) -> Optional[np.ndarray]:
    if not lava_m_path.exists():
        return None
    txt = lava_m_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"lava\(1\)\.xy\s*=\s*\[(.*?)\];", txt, re.S)
    if not m:
        return None
    block = m.group(1)
    pairs: List[Tuple[float, float]] = []
    for line in block.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                pairs.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not pairs:
        return None
    return np.array(pairs, dtype=float)


def load_tabular_points(path: Path, gzipped: bool = False) -> np.ndarray:
    open_fn = gzip.open if gzipped else open
    with open_fn(path, "rt", encoding="utf-8", errors="ignore") as f:
        data = np.genfromtxt(f, delimiter="\t", names=True, dtype=None, encoding="utf-8")
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def set_geo_aspect(ax: plt.Axes, lon_lim: Sequence[float], lat_lim: Sequence[float]) -> None:
    span_x = (lon_lim[1] - lon_lim[0]) * math.cos(math.radians(np.mean(lat_lim)))
    span_y = lat_lim[1] - lat_lim[0]
    if span_y > 0:
        ax.set_aspect(span_x / span_y)


def plot_station_markers(ax: plt.Axes) -> None:
    for lon, lat in STATION_COORDS.values():
        ax.plot(lon, lat, "ks", markersize=5, markerfacecolor="k")


def plot_global_inset(fig: plt.Figure) -> None:
    # top-right global location inset
    ax_in = fig.add_axes([0.71, 0.69, 0.25, 0.25], facecolor="#f7fbff")
    ax_in.set_xlim([-180, 180])
    ax_in.set_ylim([-90, 90])
    ax_in.set_box_aspect(0.5)

    # Simple lat/lon grid for orientation
    for lon in [-120, -60, 0, 60, 120]:
        ax_in.plot([lon, lon], [-90, 90], color="0.85", linewidth=0.6, zorder=1)
    for lat in [-60, -30, 0, 30, 60]:
        ax_in.plot([-180, 180], [lat, lat], color="0.85", linewidth=0.6, zorder=1)

    # Axial Seamount location
    axial_lon, axial_lat = -130.0, 45.95
    ax_in.plot(axial_lon, axial_lat, marker="*", color="crimson", markersize=12, zorder=5)
    ax_in.text(
        axial_lon + 8,
        axial_lat + 6,
        "Axial Seamount",
        color="crimson",
        fontsize=9,
        ha="left",
        va="bottom",
        zorder=5,
    )

    ax_in.set_title("Global location", pad=4)
    ax_in.set_xticks([-180, 0, 180])
    ax_in.set_yticks([-90, 0, 90])
    ax_in.grid(False)


def make_figure_1a(repo_root: Path, fm_root: Path, outdir: Path) -> Path:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9.2, 7.2), constrained_layout=True)

    topo_file = choose_existing(
        repo_root / "04-manuscripts" / "Axial-em300-gmt-25m.grd",
        fm_root / "02-data" / "Alldata" / "Axial-em300-gmt-25m.grd",
    )
    if topo_file is not None:
        with netcdf_file(str(topo_file), "r") as nc:
            x = nc.variables["x"].data.copy()
            y = nc.variables["y"].data.copy()
            z = nc.variables["z"].data.copy()
        lon_min, lon_max = -130.1, -129.9
        lat_min, lat_max = 45.85, 46.1
        xi = (x >= lon_min) & (x <= lon_max)
        yi = (y >= lat_min) & (y <= lat_max)
        x2d, y2d = np.meshgrid(x[xi], y[yi])
        z2d = z[np.ix_(yi, xi)]

        cmap_file = choose_existing(fm_root / "02-data" / "Alldata" / "ColormapZMC.mat")
        if cmap_file is not None:
            cmap_raw = load_scalar_or_array(cmap_file, "ColormapZMC")
            cmap = ListedColormap(np.clip(cmap_raw * 0.7 + 0.3, 0, 1))
        else:
            cmap = "terrain"

        c = ax.contourf(x2d, y2d, z2d, levels=20, cmap=cmap)
        cb = fig.colorbar(c, ax=ax, shrink=0.72, pad=0.02)
        cb.set_label("Depth (m)")
    else:
        ax.text(0.02, 0.98, "Topography grid not found", transform=ax.transAxes, va="top")

    lava_2015 = choose_existing(
        fm_root / "02-data/Alldata/Fissures2015/JdF:Axial_Clague/Axial-2015-lava-points-geo-v2.txt"
    )
    lava_2011 = choose_existing(
        fm_root / "02-data/Alldata/Fissures2011/JdF:Axial_Clague/Axial-2011-lava-points-geo-v2.txt.gz"
    )
    fiss_2015 = choose_existing(
        fm_root / "02-data/Alldata/Fissures2015/JdF:Axial_Clague/Axial-2015-fissures-points-geo-v2.txt"
    )
    fiss_2011 = choose_existing(
        fm_root / "02-data/Alldata/Fissures2011/JdF:Axial_Clague/Axial-2011-fissures-points-geo-v2.txt"
    )
    fiss_1998 = choose_existing(fm_root / "02-data" / "Alldata" / "Axial-1998-Fissures.txt")

    if lava_2015 is not None:
        d = load_tabular_points(lava_2015, gzipped=False)
        for fid in np.unique(d["ORIG_FID"]):
            g = d[d["ORIG_FID"] == fid]
            ax.fill(g["LONGITUDE"], g["LATITUDE"], color=(0.0, 0.5, 0.0), alpha=0.35, linewidth=0)

    if lava_2011 is not None:
        d = load_tabular_points(lava_2011, gzipped=True)
        for fid in np.unique(d["ORIG_FID"]):
            g = d[d["ORIG_FID"] == fid]
            ax.fill(g["LONGITUDE"], g["LATITUDE"], color=(0.0, 0.0, 0.9), alpha=0.25, linewidth=0)

    lava_1998_mat = choose_existing(fm_root / "04-final-paper" / "axial_lava1998.m")
    lava_1998 = parse_matlab_lava1998(lava_1998_mat) if lava_1998_mat else None
    if lava_1998 is not None:
        ax.fill(lava_1998[:, 0], lava_1998[:, 1], color=(0.5, 0.0, 0.0), alpha=0.2, linewidth=0)

    for fiss_path in (fiss_2015, fiss_2011):
        if fiss_path is None:
            continue
        d = load_tabular_points(fiss_path, gzipped=False)
        for fid in np.unique(d["ORIG_FID"]):
            g = d[d["ORIG_FID"] == fid]
            ax.plot(g["LONGITUDE"], g["LATITUDE"], "k-", linewidth=0.8)

    if fiss_1998 is not None:
        rows: List[Tuple[float, float, float]] = []
        with open(fiss_1998, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    fid = float(parts[0])
                    lon_s, lat_s = parts[1].split(",")
                    rows.append((fid, float(lon_s), float(lat_s)))
                except Exception:
                    continue
        if rows:
            arr = np.array(rows, dtype=float)
            for fid in np.unique(arr[:, 0]):
                g = arr[arr[:, 0] == fid]
                ax.plot(g[:, 1], g[:, 2], "k-", linewidth=0.8)

    ax.plot(CALDERA_RIM[:, 0], CALDERA_RIM[:, 1], "k-", linewidth=2.0)
    plot_station_markers(ax)

    # Manuscript box shown in panel a.
    box_lon = np.array([-130.0438, -130.0438, -129.9649, -129.9649, -130.0438])
    box_lat = np.array([45.9142, 45.9952, 45.9952, 45.9142, 45.9142])
    ax.plot(box_lon, box_lat, "b-", linewidth=1.6)

    lon_lim = [-130.1, -129.9]
    lat_lim = [45.85, 46.1]
    ax.set_xlim(lon_lim)
    ax.set_ylim(lat_lim)
    set_geo_aspect(ax, lon_lim, lat_lim)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Figure 1a: Bathymetry, fissures, and lava flows")
    ax.grid(alpha=0.2)

    plot_global_inset(fig)

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "Figure01a_python_updated.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    default_repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Standalone Figure 1a builder")
    parser.add_argument("--repo-root", default=str(default_repo))
    parser.add_argument("--fm-root", default="/Users/mczhang/Documents/GitHub/FM")
    parser.add_argument("--outdir", default=str(default_repo / "04-manuscripts" / "python_figures" / "output"))
    args = parser.parse_args()

    out = make_figure_1a(Path(args.repo_root), Path(args.fm_root), Path(args.outdir))
    print(f"[OK] Updated Figure 1a saved to: {out}")


if __name__ == "__main__":
    main()
