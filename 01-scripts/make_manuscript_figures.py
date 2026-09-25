#!/usr/bin/env python3
"""
Rebuild manuscript figures (Figure 1-14) in Python.

Default paths are configured for this local environment:
  - FM5_ML repo (current repo)
  - sibling FM / FM3 / FM4 repos used by the original MATLAB scripts
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


@dataclass
class Paths:
    repo_root: Path
    fm_root: Path
    fm3_root: Path
    fm4_root: Path
    docx_path: Path


def matlab_datenum(year: int, month: int, day: int, hour: int = 0, minute: int = 0, sec: int = 0) -> float:
    dt = datetime(year, month, day, hour, minute, sec)
    frac = (dt - datetime(year, month, day)).total_seconds() / 86400.0
    return dt.toordinal() + 366 + frac


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


def parse_matlab_lava1998(lava_m_path: Path) -> Optional[List[np.ndarray]]:
    if not lava_m_path.exists():
        return None
    txt = lava_m_path.read_text(encoding="utf-8", errors="ignore")
    polys: List[np.ndarray] = []
    for m in re.finditer(r"lava\(\s*\d+\s*\)\.xy\s*=\s*\[(.*?)\];", txt, re.S):
        block = m.group(1)
        pairs: List[Tuple[float, float]] = []
        for line in block.splitlines():
            line = line.strip().replace("...", "").rstrip(",")
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    pairs.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        if pairs:
            polys.append(np.array(pairs, dtype=float))
    return polys or None


def load_tabular_points(path: Path, gzipped: bool = False) -> np.ndarray:
    open_fn = gzip.open if gzipped else open
    with open_fn(path, "rt", encoding="utf-8", errors="ignore") as f:
        data = np.genfromtxt(f, delimiter="\t", names=True, dtype=None, encoding="utf-8")
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def extract_docx_image(paths: Paths, fig_num: int, out_path: Path) -> bool:
    if not paths.docx_path.exists():
        return False
    media_name = f"word/media/image{fig_num}.png"
    with zipfile.ZipFile(paths.docx_path, "r") as zf:
        if media_name not in zf.namelist():
            return False
        raw = zf.read(media_name)
    out_path.write_bytes(raw)
    return True


def save_figure(fig: plt.Figure, out_path: Path, dpi: int = 300) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _set_geo_aspect(ax: plt.Axes, lon_lim: Sequence[float], lat_lim: Sequence[float]) -> None:
    span_x = (lon_lim[1] - lon_lim[0]) * math.cos(math.radians(np.mean(lat_lim)))
    span_y = lat_lim[1] - lat_lim[0]
    if span_y > 0:
        ax.set_aspect(span_x / span_y)


def _plot_station_markers(
    ax: plt.Axes,
    label: bool = True,
    markersize: float = 5,
    fontsize: float = 7,
    text_dx: float = 0.0002,
    text_dy: float = 0.0002,
) -> None:
    for code, (lon, lat) in STATION_COORDS.items():
        ax.plot(lon, lat, "ks", markersize=markersize, markerfacecolor="k")
        if label:
            ax.text(lon + text_dx, lat + text_dy, code, fontsize=fontsize)


def figure_01(paths: Paths, outdir: Path) -> Path:
    # Port of /Users/mczhang/Documents/GitHub/FM3/04-final-paper/Figure01_Background_combine_Final.m
    style = {
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }

    def latlon2xy(dlat: np.ndarray, dlon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Hardwired for Axial Seamount: origin at AXCC1, rotation = -20 degrees (see FM scripts).
        dlato = 45.9547
        dlono = -130.0089
        rota = -20.0
        xltkm = 111.19
        xlnkm = xltkm * math.cos(math.radians(dlato))
        dlat_km = (np.asarray(dlat, dtype=float) - dlato) * xltkm
        dlon_km = (np.asarray(dlon, dtype=float) - dlono) * xlnkm
        snr = math.sin(math.radians(rota))
        csr = math.cos(math.radians(rota))
        y = csr * dlat_km + snr * dlon_km
        x = csr * dlon_km - snr * dlat_km
        return x, y

    with plt.rc_context(style):
        from matplotlib.lines import Line2D

        fig = plt.figure(figsize=(10.0, 6.48))
        fig.patch.set_facecolor("white")

        ax1 = fig.add_axes([0.05, 0.11, 0.35, 0.8])
        ax2 = fig.add_axes([0.5, 0.11, 0.35, 0.8])

        # --- Panel (a): bathymetry + lava/fissures + rotated box + globe inset ---
        topo_file = choose_existing(
            paths.repo_root / "04-manuscripts" / "Axial-em300-gmt-25m.grd",
            paths.fm_root / "02-data" / "Alldata" / "Axial-em300-gmt-25m.grd",
        )
        cset = None
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

            cmap_file = choose_existing(paths.fm_root / "02-data" / "Alldata" / "ColormapZMC.mat")
            if cmap_file is not None:
                cmap_raw = load_scalar_or_array(cmap_file, "ColormapZMC")
                cmap = ListedColormap(np.clip(cmap_raw, 0, 1))
            else:
                cmap = "terrain"
            cset = ax1.contourf(x2d, y2d, z2d, levels=20, cmap=cmap, antialiased=False)
        else:
            ax1.text(0.02, 0.98, "Topography grid not found", transform=ax1.transAxes, va="top")

        # Rotated box corners (lat/lon) matching the MATLAB figure.
        box_lat = np.array([45.9074, 45.9835, 46.0020, 45.9259, 45.9074])
        box_lon = np.array([-130.0255, -130.0653, -129.9923, -129.9525, -130.0255])
        ax1.plot(box_lon, box_lat, "b-", linewidth=2.0)

        alpha = 0.5
        lava_2015 = choose_existing(
            paths.fm_root
            / "02-data/Alldata/Fissures2015/JdF:Axial_Clague/Axial-2015-lava-points-geo-v2.txt"
        )
        lava_2011 = choose_existing(
            paths.fm_root
            / "02-data/Alldata/Fissures2011/JdF:Axial_Clague/Axial-2011-lava-points-geo-v2.txt.gz"
        )
        fiss_2015 = choose_existing(
            paths.fm_root
            / "02-data/Alldata/Fissures2015/JdF:Axial_Clague/Axial-2015-fissures-points-geo-v2.txt"
        )
        fiss_2011 = choose_existing(
            paths.fm_root
            / "02-data/Alldata/Fissures2011/JdF:Axial_Clague/Axial-2011-fissures-points-geo-v2.txt"
        )
        fiss_1998 = choose_existing(paths.fm_root / "02-data" / "Alldata" / "Axial-1998-Fissures.txt")

        fissure_handle = None
        for fiss_path in (fiss_2015, fiss_2011):
            if fiss_path is None:
                continue
            d = load_tabular_points(fiss_path, gzipped=False)
            for fid in np.unique(d["ORIG_FID"]):
                g = d[d["ORIG_FID"] == fid]
                line = ax1.plot(g["LONGITUDE"], g["LATITUDE"], "k-", linewidth=1.0)
                if fissure_handle is None and line:
                    fissure_handle = line[0]

        if fiss_1998 is not None:
            rows = []
            with open(fiss_1998, "rt", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.replace(",", " ").split()
                    if len(parts) < 3:
                        continue
                    try:
                        fid = float(parts[0])
                        lon = float(parts[1])
                        lat = float(parts[2])
                        rows.append((fid, lon, lat))
                    except Exception:
                        continue
            if rows:
                arr = np.array(rows, dtype=float)
                for fid in np.unique(arr[:, 0]):
                    g = arr[arr[:, 0] == fid]
                    line = ax1.plot(g[:, 1], g[:, 2], "k-", linewidth=1.0)
                    if fissure_handle is None and line:
                        fissure_handle = line[0]

        ax1.plot(CALDERA_RIM[:, 0], CALDERA_RIM[:, 1], "k-", linewidth=2.0)

        lava_flow_2011_handle = None
        if lava_2011 is not None:
            d = load_tabular_points(lava_2011, gzipped=True)
            for fid in np.unique(d["ORIG_FID"]):
                g = d[d["ORIG_FID"] == fid]
                patches = ax1.fill(
                    g["LONGITUDE"],
                    g["LATITUDE"],
                    facecolor=(0.0, 0.0, 0.9),
                    edgecolor="none",
                    alpha=alpha,
                    linewidth=0,
                )
                if lava_flow_2011_handle is None and patches:
                    lava_flow_2011_handle = patches[0]

        lava_flow_2015_handle = None
        if lava_2015 is not None:
            d = load_tabular_points(lava_2015, gzipped=False)
            for fid in np.unique(d["ORIG_FID"]):
                g = d[d["ORIG_FID"] == fid]
                patches = ax1.fill(
                    g["LONGITUDE"],
                    g["LATITUDE"],
                    facecolor=(0.0, 0.5, 0.0),
                    edgecolor="none",
                    alpha=alpha,
                    linewidth=0,
                )
                if lava_flow_2015_handle is None and patches:
                    lava_flow_2015_handle = patches[0]

        lava_flow_1998_handle = None
        lava_1998_mat = choose_existing(paths.fm_root / "04-final-paper" / "axial_lava1998.m")
        lava_1998_polys = parse_matlab_lava1998(lava_1998_mat) if lava_1998_mat else None
        if lava_1998_polys is not None:
            for poly in lava_1998_polys:
                patches = ax1.fill(
                    poly[:, 0],
                    poly[:, 1],
                    facecolor=(0.5, 0.0, 0.0),
                    edgecolor="none",
                    alpha=alpha,
                    linewidth=0,
                )
                if lava_flow_1998_handle is None and patches:
                    lava_flow_1998_handle = patches[0]

        lons = [v[0] for v in STATION_COORDS.values()]
        lats = [v[1] for v in STATION_COORDS.values()]
        station_handle = ax1.plot(lons, lats, "sk", markerfacecolor="k", markersize=8)[0]

        handles = []
        labels = []
        if station_handle is not None:
            handles.append(station_handle)
            labels.append("OOI OBS Stations")
        if fissure_handle is not None:
            handles.append(fissure_handle)
            labels.append("Fissures")
        if lava_flow_1998_handle is not None:
            handles.append(lava_flow_1998_handle)
            labels.append("Lava Flows 1998")
        if lava_flow_2011_handle is not None:
            handles.append(lava_flow_2011_handle)
            labels.append("Lava Flows 2011")
        if lava_flow_2015_handle is not None:
            handles.append(lava_flow_2015_handle)
            labels.append("Lava Flows 2015")
        if handles:
            ax1.legend(handles, labels, loc="lower left", fontsize=8, frameon=True, framealpha=1.0)

        ax1.text(-130.04, 45.96, "AXIAL CALDERA", fontsize=10, color="w", fontweight="bold")
        ax1.text(-130.03, 46.01, "NORTH RIFT ZONE", fontsize=10, color="k", rotation=75)
        ax1.text(-130.00, 45.852, "SOUTH RIFT ZONE", fontsize=10, color="k", rotation=75)
        ax1.text(-130.035, 45.932, "ASHES", fontsize=8, color="w", fontweight="bold")
        ax1.text(-129.985, 45.917, "INTERNATIONAL", fontsize=8, color="w", fontweight="bold")
        ax1.text(-130.09, 46.09, "(a)", fontsize=12, color="w", fontweight="bold")

        lon_lim = [-130.1, -129.9]
        lat_lim = [45.85, 46.1]
        ax1.set_xlim(lon_lim)
        ax1.set_ylim(lat_lim)
        _set_geo_aspect(ax1, lon_lim, lat_lim)
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        ax1.grid(True)

        if cset is not None:
            cax1 = fig.add_axes([0.31, 0.14, 0.02, 0.2])
            cb1 = fig.colorbar(cset, cax=cax1)
            cb1.set_label("Depth (m)")

        # Globe inset: try Cartopy (adds coastlines) when data/network is available;
        # otherwise fall back to a self-contained 3D globe (graticule + star only).
        globe_pos = [0.14, 0.60, 0.40, 0.40]
        axg = None
        try:
            import cartopy
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            Path(os.environ["CARTOPY_DATA_DIR"]).mkdir(parents=True, exist_ok=True)
            cartopy.config["data_dir"] = os.environ["CARTOPY_DATA_DIR"]

            axg = fig.add_axes(globe_pos, projection=ccrs.Orthographic(-135, 45))
            axg.set_global()
            axg.set_facecolor("white")
            axg.add_feature(cfeature.LAND.with_scale("110m"), facecolor=(0.1, 0.5, 0.5), edgecolor="none")
            axg.coastlines(resolution="110m", linewidth=0.6, color="k")
            axg.gridlines(draw_labels=False, linewidth=0.4, color="0.5", alpha=0.8)
            axg.plot(-128.0, 46.0, marker="*", markersize=10, color="r", transform=ccrs.PlateCarree())

            # Force a draw so any NaturalEarth download failures happen here (so we can fall back cleanly).
            fig.canvas.draw()
        except Exception:
            if axg is not None:
                try:
                    axg.remove()
                except Exception:
                    pass

            axg = fig.add_axes(globe_pos, projection="3d")
            axg.set_axis_off()
            axg.set_box_aspect((1, 1, 1))
            axg.set_xlim(-1.05, 1.05)
            axg.set_ylim(-1.05, 1.05)
            axg.set_zlim(-1.05, 1.05)
            try:
                axg.set_proj_type("ortho")
            except Exception:
                pass
            # MATLAB: view(-37.5, 46.8). The 3D fallback globe uses a different axis convention,
            # so we flip azimuth by 180 deg to keep the Axial star on the visible hemisphere.
            axg.view_init(elev=46.8, azim=142.5)

            u = np.linspace(0, 2 * np.pi, 120)
            v = np.linspace(-np.pi / 2, np.pi / 2, 60)
            xs = np.outer(np.cos(v), np.cos(u))
            ys = np.outer(np.cos(v), np.sin(u))
            zs = np.outer(np.sin(v), np.ones_like(u))
            axg.plot_surface(xs, ys, zs, color="#f7f7f7", linewidth=0, shade=True, antialiased=False)

            for lon_deg in range(-180, 181, 30):
                lon = np.deg2rad(lon_deg)
                lat = np.deg2rad(np.linspace(-90, 90, 181))
                xg = np.cos(lat) * np.cos(lon) * 1.01
                yg = np.cos(lat) * np.sin(lon) * 1.01
                zg = np.sin(lat) * 1.01
                axg.plot(xg, yg, zg, color="0.25", linewidth=1.0, alpha=0.95)
            for lat_deg in range(-60, 61, 30):
                lat = np.deg2rad(lat_deg)
                lon = np.deg2rad(np.linspace(-180, 180, 361))
                xg = np.cos(lat) * np.cos(lon) * 1.01
                yg = np.cos(lat) * np.sin(lon) * 1.01
                zg = np.sin(lat) * np.ones_like(lon) * 1.01
                axg.plot(xg, yg, zg, color="0.25", linewidth=1.0, alpha=0.95)

            star_lon = np.deg2rad(-128.0)
            star_lat = np.deg2rad(46.0)
            sx = np.cos(star_lat) * np.cos(star_lon) * 1.12
            sy = np.cos(star_lat) * np.sin(star_lon) * 1.12
            sz = np.sin(star_lat) * 1.12
            axg.scatter([sx], [sy], [sz], color="red", s=140, marker="*", depthshade=False)

        # --- Panel (b): depth scatter in x/y (km) + k-means boundaries + connector lines ---
        sc = None
        w_catalog = choose_existing(paths.fm3_root / "02-data" / "A_All" / "Felix_kmean_morethan5.mat")
        felix = None
        if w_catalog is not None:
            felix = load_struct_array(w_catalog, "Felix")
            lon = np.array([get_value(x, "lon", np.nan) for x in felix], dtype=float)
            lat = np.array([get_value(x, "lat", np.nan) for x in felix], dtype=float)
            dep = np.array([get_value(x, "depth", np.nan) for x in felix], dtype=float)
            m = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(dep) & (dep <= 2)
            xkm, ykm = latlon2xy(lat[m], lon[m])
            sc = ax2.scatter(xkm, ykm, c=dep[m], s=2, cmap="summer_r", linewidths=0)

        ax2.set_aspect("equal", adjustable="box")
        ax2.set_xlim([-3, 3])
        ax2.set_ylim([-4.5, 4.5])

        kb = choose_existing(paths.fm_root / "04-final-paper" / "kameanBoundary.mat")
        if kb is not None:
            d = loadmat(str(kb), squeeze_me=True)
            C = np.asarray(d.get("C"), dtype=float)
            num_points = np.asarray(d.get("numPoints"), dtype=int).ravel()
            vx = np.asarray(d.get("vx"), dtype=float)
            vy = np.asarray(d.get("vy"), dtype=float)
            names_cluster = ["R7", "R6", "R5", "R4", "R3", "R2", "R1"]
            if C.size and num_points.size >= 7:
                for i in range(7):
                    ax2.text(
                        C[i, 0] - 0.1,
                        C[i, 1] + 0.1,
                        f"{int(num_points[i])}: {names_cluster[i]}",
                        va="bottom",
                        ha="right",
                    )
                ax2.plot(C[:, 0], C[:, 1], linestyle="None", marker="x", color="c", markersize=15, markeredgewidth=3)
            if vx.ndim == 2 and vy.ndim == 2 and vx.shape == vy.shape:
                for j in range(vx.shape[1]):
                    ax2.plot(vx[:, j], vy[:, j], "k-.", linewidth=1.0)

        # Caldera rim (convert lon/lat -> x/y km)
        xr, yr = latlon2xy(CALDERA_RIM[:, 1], CALDERA_RIM[:, 0])
        ax2.plot(xr, yr, "k-", linewidth=3.0)

        # Stations (lon/lat -> x/y km)
        for code, (slon, slat) in STATION_COORDS.items():
            xs, ys = latlon2xy(np.array([slat]), np.array([slon]))
            ax2.plot(xs[0], ys[0], "s", markeredgecolor="k", markerfacecolor="k", markersize=10)
            ax2.text(xs[0] + 0.1, ys[0], code)

        # Highlight a small set of events (as in MATLAB).
        if felix is not None:
            ids = np.array([get_value(x, "ID", np.nan) for x in felix], dtype=float)
            highlight_ids = np.array(
                [
                    1225535,
                    1341701,
                    1334022,
                    1225960,
                    1316811,
                    1321478,
                    1315292,
                    1315619,
                    1346145,
                    1340523,
                    1335762,
                    1346359,
                    1501394,
                    1336032,
                    1225536,
                    1343586,
                    1347474,
                    1325518,
                    1340556,
                    1315224,
                    1330369,
                    1225831,
                    1327219,
                    1328527,
                    1298463,
                    1312638,
                ],
                dtype=float,
            )
            highlight_idx = np.flatnonzero(np.isin(ids, highlight_ids))
            if highlight_idx.size > 1:
                rest = highlight_idx[1:]
                lon_h = np.array([get_value(felix[i], "lon", np.nan) for i in rest], dtype=float)
                lat_h = np.array([get_value(felix[i], "lat", np.nan) for i in rest], dtype=float)
                m_h = np.isfinite(lon_h) & np.isfinite(lat_h)
                if np.any(m_h):
                    xh, yh = latlon2xy(lat_h[m_h], lon_h[m_h])
                    ax2.scatter(xh, yh, s=5, c="b", linewidths=0)
            if highlight_idx.size >= 1:
                i0 = int(highlight_idx[0])
                lon0 = float(get_value(felix[i0], "lon", np.nan))
                lat0 = float(get_value(felix[i0], "lat", np.nan))
                if np.isfinite(lon0) and np.isfinite(lat0):
                    x0, y0 = latlon2xy(np.array([lat0]), np.array([lon0]))
                    ax2.scatter(x0, y0, s=15, c="r", linewidths=0)

        ax2.text(-2.8, 1, "North", color="blue", fontsize=12, rotation=90, va="center")
        ax2.text(-2.8, -2, "South", color="blue", fontsize=12, rotation=90, va="center")
        ax2.text(-1.5, -4.3, "West Wall", color="blue", fontsize=12, ha="center")
        ax2.text(1.5, -4.3, "East Wall", color="blue", fontsize=12, ha="center")
        ax2.set_xlabel("x-Distance (km)")
        ax2.set_ylabel("y-Distance (km)")
        ax2.grid(True)
        ax2.text(-2.8, 4.1, "(b)", fontsize=12, color="k", fontweight="bold")

        if sc is not None:
            cax2 = fig.add_axes([0.80, 0.65, 0.02, 0.2])
            cb2 = fig.colorbar(sc, cax=cax2)
            cb2.set_label("Depth (km)")

        # Connector dashed lines (figure-normalized coords)
        xlim1 = ax1.get_xlim()
        ylim1 = ax1.get_ylim()
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        xlim2 = ax2.get_xlim()
        ylim2 = ax2.get_ylim()

        top_right_121 = (-129.9923, 46.0020)
        bottom_right_121 = (-129.9525, 45.9259)
        top_right_norm_x1 = pos1.x0 + pos1.width * (top_right_121[0] - xlim1[0]) / (xlim1[1] - xlim1[0])
        top_right_norm_y1 = pos1.y0 + pos1.height * (top_right_121[1] - ylim1[0]) / (ylim1[1] - ylim1[0])
        bottom_right_norm_y1 = pos1.y0 + pos1.height * (bottom_right_121[1] - ylim1[0]) / (ylim1[1] - ylim1[0])

        top_left_122 = (-3.0, 4.5)
        bottom_left_122 = (-3.0, -4.5)
        top_left_norm_x2 = pos2.x0 + pos2.width * (top_left_122[0] - xlim2[0]) / (xlim2[1] - xlim2[0])
        top_left_norm_y2 = pos2.y0 + pos2.height * (top_left_122[1] - ylim2[0]) / (ylim2[1] - ylim2[0])
        bottom_left_norm_x2 = pos2.x0 + pos2.width * (bottom_left_122[0] - xlim2[0]) / (xlim2[1] - xlim2[0])
        bottom_left_norm_y2 = pos2.y0 + pos2.height * (bottom_left_122[1] - ylim2[0]) / (ylim2[1] - ylim2[0])

        fig.add_artist(
            Line2D(
                [top_right_norm_x1, top_left_norm_x2],
                [top_right_norm_y1, top_left_norm_y2],
                transform=fig.transFigure,
                color="blue",
                linewidth=1,
                linestyle="--",
            )
        )
        fig.add_artist(
            Line2D(
                [0.295, bottom_left_norm_x2],
                [bottom_right_norm_y1, bottom_left_norm_y2],
                transform=fig.transFigure,
                color="blue",
                linewidth=1,
                linestyle="--",
            )
        )

        out = outdir / "Figure01_python.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return out


def _plot_stacked_waves(
    ax: plt.Axes,
    entries: np.ndarray,
    station: str,
    n_plot: int,
    offset: float,
    use_po_label: bool,
    half_rate: bool,
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
        ax.plot(x, w + yoff, "b-", linewidth=1.0)
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


def figure_02(paths: Paths, outdir: Path) -> Path:
    template_mat = paths.repo_root / "02-data" / "A_wave_dB20_cleaned.mat"
    noise_mat = paths.repo_root / "02-data" / "A_wave_noise_10000.mat"
    template = load_struct_array(template_mat, "Felix")
    noise = load_struct_array(noise_mat, "Felix")

    fig, axes = plt.subplots(2, 7, figsize=(18, 10), constrained_layout=True)
    for i, sta in enumerate(STATIONS):
        _plot_stacked_waves(
            axes[0, i], template, sta, n_plot=20, offset=2.0, use_po_label=True, half_rate=True
        )
        _plot_stacked_waves(
            axes[1, i], noise, sta, n_plot=20, offset=2.0, use_po_label=False, half_rate=False
        )
        if i == 0:
            axes[0, i].set_ylabel("Normalized amplitude")
            axes[1, i].set_ylabel("Normalized amplitude")
        axes[1, i].set_xlabel("Time (s)")
    out = outdir / "Figure02_python.png"
    save_figure(fig, out)
    return out


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

    template = load_struct_array(paths.repo_root / "02-data" / "A_wave_dB20_cleaned.mat", "Felix")
    noise = load_struct_array(paths.repo_root / "02-data" / "A_wave_noise_10000.mat", "Felix")

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
        ax.plot(t, wn + yoff, "b-", linewidth=1.0)
        ax.text(-0.46, yoff + 0.55, f"Po: {po:+d}", fontsize=7, color="k")
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
                        use_po_label=False, half_rate=False)
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
        paths.repo_root / "02-data" / "H_Noi" / "H_noise_dB20_snrValue.mat",
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
        ax.plot(t, wn + yoff, color="red" if is_base else "black",
                linewidth=1.4 if is_base else 0.9)
        # Every synthetic trace inherits the polarity of its parent template
        # in (a); label it the same way (a) does so that stays visible.
        ax.text(-0.46, yoff + 0.55, f"Po: {base_po:+d}", fontsize=7, color="k")
        ticks.append(yoff)
        labels.append(f"{snr_db:.1f} dB")
    ax.yaxis.tick_right()
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(axis="y", length=0)
    for lab, is_base in zip(ax.get_yticklabels(), [False] * (len(traces) - 1) + [True]):
        lab.set_color(A_PICK if is_base else A_NOTE)

    for ax in axes:
        ax.set_xlim([-0.5, 0.5])
        # The extra 1.2 leaves a clear strip above the top trace for the
        # in-axes panel label.
        ax.set_ylim([-offset, n_trace * offset + 1.2])
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.2)
    for ax in axes[:2]:
        ax.set_yticks([])

    # Panel labels sit inside the top-left corner of each axes.
    for ax, lab in zip(axes, ("(a)", "(b)", "(c)")):
        panel_label(ax, lab, dx=0.025, dy=0.955)

    out = outdir / "Figure02_merged_python.png"
    save_figure(fig, out)
    return out


def figure_03(paths: Paths, outdir: Path) -> Path:
    # Template waveforms: prefer station-specific file, fall back to cleaned catalog
    as1_path = choose_existing(
        paths.repo_root / "02-data" / "K_aug" / "AS1.mat",
        paths.repo_root / "02-data" / "AS1.mat",
        paths.repo_root / "02-data" / "A_wave_dB20_cleaned.mat",
    )
    as1_struct = "AS1" if as1_path is not None and as1_path.stem == "AS1" else "Felix"
    as1 = load_struct_array(as1_path, as1_struct)

    # Noise waveforms: prefer original noise file, fall back to large noise catalog
    noise_path = choose_existing(
        paths.repo_root / "02-data" / "H_Noi" / "H_Noise_200.mat",
        paths.repo_root / "02-data" / "H_noi" / "H_Noise_200.mat",
        paths.repo_root / "02-data" / "A_wave_noise_10000.mat",
    )
    noise = load_struct_array(noise_path, "Felix")

    # SNR distribution: prefer pre-computed file, fall back to computing from templates
    snr_file = choose_existing(
        paths.repo_root / "02-data" / "H_Noi" / "H_noise_dB20_snrValue.mat",
        paths.repo_root / "02-data" / "H_noi" / "H_noise_dB20_snrValue.mat",
    )
    if snr_file is not None:
        snr_values = load_scalar_or_array(snr_file, "snrValues")
        snr_data = np.asarray(snr_values[0], dtype=float).ravel()
        snr_data = snr_data[np.isfinite(snr_data) & (snr_data > 0)]
    else:
        snrs = []
        for ev in as1:
            w = as_1d(get_value(ev, "W_AS1", np.array([])))
            if w.size >= 160:
                rms_n = np.sqrt(np.mean(w[:80] ** 2))
                rms_s = np.sqrt(np.mean(w[80:160] ** 2))
                if rms_n > 0:
                    db = 20 * np.log10((rms_s + 1e-12) / (rms_n + 1e-12))
                    if np.isfinite(db) and db > 0:
                        snrs.append(db)
        snr_data = np.array(snrs)
    shape, loc, scale = lognorm.fit(snr_data, floc=0)
    rng = np.random.default_rng(3)

    # Find first event with a valid W_AS1 waveform (not all events cover every station)
    base = next((ev for ev in as1 if as_1d(get_value(ev, "W_AS1", np.array([]))).size >= 200), None)
    if base is None:
        raise RuntimeError("AS1 waveform not found for Figure 3")
    signal = as_1d(get_value(base, "W_AS1", np.array([])))
    signal = signal[:200]

    traces: List[np.ndarray] = []
    snr_labels: List[float] = []
    for k in range(11):
        if k == 0:
            syn = signal.copy()
        else:
            target_db = float(rng.uniform(5, 45))
            valid = False
            while not valid:
                n = noise[rng.integers(0, len(noise))]
                w = as_1d(get_value(n, "W_AS1", np.array([])))
                valid = w.size >= 50 and np.max(np.abs(w)) > 0
            # Pad or trim noise to match signal length (200 samples)
            if w.size < 200:
                w = np.pad(w, (0, 200 - w.size))
            else:
                w = w[:200]
            rms_signal = np.sqrt(np.mean(signal**2))
            rms_noise_target = rms_signal / (10 ** (target_db / 20))
            rms_noise = np.sqrt(np.mean(w**2))
            syn = signal + (rms_noise_target / (rms_noise + 1e-12)) * w
        nseg = syn[:80]
        sseg = syn[80:160]
        snr_db = 20 * np.log10((np.sqrt(np.mean(sseg**2)) + 1e-12) / (np.sqrt(np.mean(nseg**2)) + 1e-12))
        snr_labels.append(float(snr_db))
        traces.append(syn)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 5.2), constrained_layout=True)
    for i, w in enumerate(traces[:11]):
        w = norm_wave(w)
        yoff = i * 2.0
        color = "red" if i == 0 else "black"
        ax.plot(np.arange(w.size), w + yoff, color=color, linewidth=1.6 if i == 0 else 1.0)
        txt_color = A_PICK if i == 0 else A_NOTE
        ax.text(205, yoff, f"{snr_labels[i]:.1f} dB", fontsize=9, color=txt_color, va="center")
    ax.set_xlim([0, 240])
    ax.set_ylim([-1, 22])
    ax.set_yticks([])
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Normalized waveform (stacked)")
    ax.set_title("")
    ax.grid(alpha=0.15)
    out = outdir / "Figure03_python.png"
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
CM_COUNTS = {
    "DiTingMotion": [[5685, 723], [742, 1179]],
    "CFM": [[8609, 1032], [939, 6548]],
    "EQPolarity": [[8648, 993], [1091, 6396]],
    "PolarCAP": [[8633, 1008], [1066, 6421]],
}


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
    diting = np.array([94.1, 70.9, 85.8, 76.2, 81.6, 66.2, 84.8])
    cfm = np.array([97.0, 81.0, 97.4, 88.3, 87.1, 82.9, 74.5])
    eqp = np.array([96.5, 79.4, 96.8, 87.1, 86.0, 83.3, 75.4])
    polcap = np.array([96.1, 79.8, 97.3, 87.8, 86.4, 81.9, 75.4])
    cc = np.array([0.99972, 0.95991, 0.99134, 0.93391, 0.84833, 0.87012, 0.80463]) * 100
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


def figure_05(paths: Paths, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.5), constrained_layout=True)
    ax.axis("off")

    def box(x, y, w, h, text, fc=TINT_NEUTRAL, ec="black", fs=10):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor=fc, edgecolor=ec, linewidth=1.2)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5))

    # Encoder
    box(0.03, 0.44, 0.12, 0.12, "Input\n1 x 200")
    box(0.20, 0.44, 0.12, 0.12, "Conv1D\n32, k=32", fc=TINT_CONV)
    box(0.36, 0.44, 0.10, 0.12, "MaxPool", fc=TINT_POOL)
    box(0.50, 0.44, 0.12, 0.12, "Conv1D\n8, k=16", fc=TINT_CONV)
    box(0.66, 0.44, 0.10, 0.12, "Dropout\n0.3", fc=TINT_REG)
    ell = Ellipse((0.84, 0.50), 0.13, 0.14, facecolor=TINT_LATENT, edgecolor="black")
    ax.add_patch(ell)
    ax.text(0.84, 0.50, "Latent\ncode", ha="center", va="center", fontsize=10)

    # Decoder
    box(0.50, 0.16, 0.12, 0.12, "Conv1D\n8, k=16", fc=TINT_CONV)
    box(0.34, 0.16, 0.12, 0.12, "Upsample", fc=TINT_POOL)
    box(0.18, 0.16, 0.12, 0.12, "Conv1D\n32, k=32", fc=TINT_CONV)
    box(0.03, 0.16, 0.12, 0.12, "Output\n1 x 200")

    # Classifier head
    box(0.80, 0.76, 0.17, 0.10, "Softmax classifier\nP(Up), P(Down)", fc=TINT_CLASS)

    arrow(0.15, 0.50, 0.20, 0.50)
    arrow(0.32, 0.50, 0.36, 0.50)
    arrow(0.46, 0.50, 0.50, 0.50)
    arrow(0.62, 0.50, 0.66, 0.50)
    arrow(0.76, 0.50, 0.775, 0.50)
    arrow(0.84, 0.43, 0.56, 0.28)
    arrow(0.50, 0.22, 0.46, 0.22)
    arrow(0.34, 0.22, 0.30, 0.22)
    arrow(0.18, 0.22, 0.15, 0.22)
    arrow(0.86, 0.57, 0.88, 0.76)

    ax.text(0.39, 0.62, "Encoder", fontsize=12, fontweight="bold")
    ax.text(0.26, 0.34, "Decoder", fontsize=12, fontweight="bold")
    ax.text(0.02, 0.95, "", fontsize=14, fontweight="bold")

    out = outdir / "Figure05_python.png"
    save_figure(fig, out)
    return out


def figure_06(paths: Paths, outdir: Path) -> Path:
    # Reference palette for the whole manuscript: these three colours are defined
    # here and reused as baselines in Figures 7, 8 and 9.
    colors = SERIES3
    acc_all = np.array([0.9935, 0.9858, 0.9876, 0.9867, 0.9868, 0.9871, 0.9644])
    acc_loso = np.array([0.9964, 0.9855, 0.9902, 0.9869, 0.9854, 0.9877, 0.9653])
    acc_fine = np.array([0.9829, 0.9484, 0.9653, 0.9585, 0.9534, 0.9386, 0.9013])
    data = np.vstack(
        [np.r_[acc_all, acc_all.mean()], np.r_[acc_loso, acc_loso.mean()], np.r_[acc_fine, acc_fine.mean()]]
    ).T * 100

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.4), constrained_layout=True)
    _grouped_bar(
        ax,
        data,
        STATIONS_AX_AVG,
        ["Trained on all stations", "Leave-one-station-out", "Transfer learning"],
        colors,
        (88, 101.6),
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Station")
    ax.set_title("")
    ax.legend(ncol=3, loc="upper center", columnspacing=1.2, handlelength=1.4,
              handletextpad=0.4, borderaxespad=0.2)
    out = outdir / "Figure06_python.png"
    save_figure(fig, out)
    return out


def figure_07(paths: Paths, outdir: Path) -> Path:
    # Colour rule: the original-SNR bars ARE the Figure 4 models, so each panel
    # keeps that model's Figure 4 colour -- amber for the model trained from
    # scratch on all stations, turquoise for the transfer-learning model.  The
    # two SNR variants take the same grey ramp Figure 6 uses for its time-shift
    # variants, darkening from the high-SNR to the low-SNR training set, so the
    # baseline is the only coloured bar in either figure.
    # Series order is [high, original, low].
    colors_train = [C_ALT_1, C_SCRATCH, C_ALT_2]
    colors_transfer = [C_ALT_1, C_TRANSFER, C_ALT_2]
    # New model (training) and transfer model under SNR conditions
    A5 = np.array([0.9935, 0.9858, 0.9876, 0.9867, 0.9868, 0.9871, 0.9644, 0.9850]) * 100  # Orig
    B5 = np.array([0.9859, 0.9505, 0.9739, 0.9584, 0.9623, 0.9569, 0.9341, 0.9604]) * 100  # High
    C5 = np.array([0.9940, 0.9731, 0.9868, 0.9760, 0.9690, 0.9850, 0.9770, 0.9801]) * 100  # Low
    A6 = np.array([0.9869, 0.9555, 0.9706, 0.9621, 0.9583, 0.9507, 0.9334, 0.9596]) * 100  # Orig
    B6 = np.array([0.9827, 0.9425, 0.9682, 0.9617, 0.9558, 0.9437, 0.9184, 0.9534]) * 100  # High
    C6 = np.array([0.9879, 0.9537, 0.9667, 0.9693, 0.9567, 0.9555, 0.9385, 0.9612]) * 100  # Low
    data_train = np.vstack([B5, A5, C5]).T
    data_transfer = np.vstack([B6, A6, C6]).T

    series = ["Trained on high SNR", "Trained on original SNR", "Trained on low SNR"]

    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH_IN, 6.0), constrained_layout=True)
    _grouped_bar(axes[0], data_train, STATIONS_AX_AVG, series, colors_train, (90, 102.4))
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("")
    axes[0].legend(ncol=3, loc="upper center", columnspacing=1.2, handlelength=1.4,
                   handletextpad=0.4, borderaxespad=0.2, fontsize=8)
    panel_label(axes[0], "(a) Model trained from scratch")

    _grouped_bar(axes[1], data_transfer, STATIONS_AX_AVG, series, colors_transfer, (90, 102.0))
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xlabel("Station")
    axes[1].set_title("")
    axes[1].legend(ncol=3, loc="upper center", columnspacing=1.2, handlelength=1.4,
                   handletextpad=0.4, borderaxespad=0.2, fontsize=8)
    panel_label(axes[1], "(b) Transfer learning")

    out = outdir / "Figure07_python.png"
    save_figure(fig, out)
    return out


def figure_08(paths: Paths, outdir: Path) -> Path:
    # Colour rule, as in Figure 5: the sigma = 0 bars ARE the Figure 4 models, so
    # each panel keeps that model's Figure 4 colour -- amber for the model
    # trained from scratch on all stations, turquoise for the transfer-learning
    # model.  The two shifted conditions take the same grey ramp Figure 5 uses
    # for its SNR variants, so the baseline is the only coloured bar.
    # Series order is [sigma = 0.00, 0.01, 0.02].
    colors_train = [C_SCRATCH, C_ALT_1, C_ALT_2]
    colors_transfer = [C_TRANSFER, C_ALT_1, C_ALT_2]
    ft02 = np.array([0.8278, 0.7496, 0.8066, 0.7712, 0.7533, 0.7358, 0.7455, 0.7700]) * 100
    ft01 = np.array([0.9524, 0.8793, 0.9167, 0.9123, 0.8909, 0.8689, 0.8678, 0.8984]) * 100
    ft00 = np.array([0.9869, 0.9555, 0.9706, 0.9621, 0.9583, 0.9507, 0.9334, 0.9596]) * 100
    tr02 = np.array([0.8335, 0.7520, 0.8133, 0.7512, 0.7288, 0.7271, 0.7393, 0.7636]) * 100
    tr01 = np.array([0.9704, 0.8992, 0.9415, 0.9310, 0.9022, 0.9007, 0.8991, 0.9207]) * 100
    tr00 = np.array([0.9935, 0.9858, 0.9876, 0.9867, 0.9868, 0.9871, 0.9644, 0.9850]) * 100
    data_transfer = np.vstack([ft00, ft01, ft02]).T
    data_train = np.vstack([tr00, tr01, tr02]).T

    series = [r"$\sigma$ = 0.00 s", r"$\sigma$ = 0.01 s", r"$\sigma$ = 0.02 s"]

    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH_IN, 6.0), constrained_layout=True)
    _grouped_bar(axes[0], data_train, STATIONS_AX_AVG, series, colors_train, (70, 109))
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("")
    axes[0].legend(title="Time shift applied to test waveforms", ncol=3, loc="upper center",
                   columnspacing=1.2, handlelength=1.4, handletextpad=0.4,
                   borderaxespad=0.2, fontsize=8, title_fontsize=8)
    panel_label(axes[0], "(a) Model trained from scratch")

    _grouped_bar(axes[1], data_transfer, STATIONS_AX_AVG, series, colors_transfer, (70, 109))
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xlabel("Station")
    axes[1].set_title("")
    axes[1].legend(title="Time shift applied to test waveforms", ncol=3, loc="upper center",
                   columnspacing=1.2, handlelength=1.4, handletextpad=0.4,
                   borderaxespad=0.2, fontsize=8, title_fontsize=8)
    panel_label(axes[1], "(b) Transfer learning")

    out = outdir / "Figure08_python.png"
    save_figure(fig, out)
    return out


def figure_09(paths: Paths, outdir: Path) -> Path:
    # Training-shift colours follow Figure 8: 0.01 s -> C_CAT_A, 0.02 s -> C_CAT_B.
    # The off-diagonal condition (train 0.01 / test 0.02) is shown in neutral grey.
    colors = SERIES3
    a_train = np.array([0.9942, 0.9483, 0.9789, 0.9668, 0.9598, 0.9631, 0.9466, 0.9655]) * 100
    b_train = np.array([0.9722, 0.9052, 0.9389, 0.9244, 0.9192, 0.9254, 0.8892, 0.9250]) * 100
    c_train = np.array([0.9948, 0.9407, 0.9840, 0.9674, 0.9640, 0.9472, 0.9311, 0.9615]) * 100
    a_ft = np.array([0.9776, 0.8992, 0.9438, 0.9351, 0.9294, 0.8945, 0.8884, 0.9241]) * 100
    b_ft = np.array([0.8976, 0.8115, 0.8711, 0.8353, 0.8251, 0.7875, 0.7963, 0.8322]) * 100
    c_ft = np.array([0.9768, 0.8583, 0.9498, 0.9128, 0.9167, 0.8613, 0.8354, 0.9020]) * 100

    data_train = np.vstack([a_train, b_train, c_train]).T
    data_transfer = np.vstack([a_ft, b_ft, c_ft]).T

    series = [
        r"train $\sigma$ = 0.01 s, test $\sigma$ = 0.01 s",
        r"train $\sigma$ = 0.01 s, test $\sigma$ = 0.02 s",
        r"train $\sigma$ = 0.02 s, test $\sigma$ = 0.01 s",
    ]

    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH_IN, 6.0), constrained_layout=True)
    _grouped_bar(axes[0], data_train, STATIONS_AX_AVG, series, colors, (75, 107))
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("")
    axes[0].legend(ncol=2, loc="upper center", columnspacing=1.0, handlelength=1.4,
                   handletextpad=0.4, borderaxespad=0.2, fontsize=7.5)
    panel_label(axes[0], "(a) Model trained from scratch")

    _grouped_bar(axes[1], data_transfer, STATIONS_AX_AVG, series, colors, (75, 107))
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xlabel("Station")
    axes[1].set_title("")
    axes[1].legend(ncol=2, loc="upper center", columnspacing=1.0, handlelength=1.4,
                   handletextpad=0.4, borderaxespad=0.2, fontsize=7.5)
    panel_label(axes[1], "(b) Transfer learning")
    out = outdir / "Figure09_python.png"
    save_figure(fig, out)
    return out


def _build_conflict_data(paths: Paths):
    wave_path = choose_existing(
        paths.fm4_root / "02-data" / "Before22OBSs" / "A_All" / "A_wavelarge5.mat",
        paths.repo_root / "02-data" / "A_wavelarge5.mat",
    )
    clu_path = choose_existing(
        paths.fm4_root / "02-data" / "Before22OBSs" / "F_Cl" / "F_Cl_All_MLreplace_samecluster_conf.mat",
        paths.repo_root / "02-data" / "F_Cl_All_MLreplace_samecluster_conf.mat",
    )
    felix = load_struct_array(wave_path, "Felix")
    po_clu = load_struct_array(clu_path, "Po_Clu")
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
            if val.size < 2:
                continue
            p1, p2 = float(val[0]), float(val[1])  # CC, ML
            if p1 == 0 or p2 == 0:
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


def figure_10(paths: Paths, outdir: Path) -> Path:
    """Waveforms where the CC and ML polarity picks disagree.

    Redesigned in response to review comment 47 ("font sizes are ridiculously
    small; what am I meant to be getting out of this figure?"):

      * 2 x 4 panel grid at 7 in printed width instead of a 1 x 7 strip at 19 in,
        so nothing is scaled down on the page;
      * the window is trimmed to +/- 0.25 s about the pick, because the polarity
        decision is made on the first half cycle and the later coda only crowds
        the panel;
      * each trace is annotated as "CC / ML" with an explicit up/down symbol
        rather than the bare "1 -> -1" string.
    """
    conflict_wave, conflict_labels, _, _, _ = _build_conflict_data(paths)

    # Samples: the waveforms are 200 Hz, pick at sample 50 in the 100-sample view
    # used previously.  Keep 0.25 s (50 samples) either side of the pick.
    pick = 50
    half = 50
    n_trace = 6

    def pretty(label: str) -> str:
        cc_s, ml_s = label.split(" -> ")
        sym = {"1": r"$\uparrow$", "-1": r"$\downarrow$"}
        return f"CC {sym.get(cc_s.strip(), '?')} / ML {sym.get(ml_s.strip(), '?')}"

    fig, axes = plt.subplots(2, 4, figsize=(FIG_WIDTH_WIDE_IN, 5.0), constrained_layout=True)
    rng = np.random.default_rng(42)
    for k, (sta, sta_ax) in enumerate(zip(STATIONS, STATIONS_AX)):
        ax = axes.ravel()[k]
        waves = conflict_wave[sta]
        labels = conflict_labels[sta]
        if not waves:
            ax.set_title(f"{sta_ax} (no disagreements)", fontsize=9)
            ax.axis("off")
            continue
        n_plot = min(len(waves), n_trace)
        picks = rng.choice(len(waves), size=n_plot, replace=False)
        t = (np.arange(-half, half)) / 200.0
        for i, idx in enumerate(picks, start=1):
            w = as_1d(waves[idx])
            lo, hi = pick - half, pick + half
            if w.size < hi:
                continue
            seg = norm_wave(w[lo:hi])
            ax.plot(t, 0.42 * seg + i, "k-", linewidth=0.8)
            ax.text(-0.245, i + 0.30, pretty(labels[idx]), color=A_NOTE, fontsize=6.5,
                    va="center", ha="left")
        ax.axvline(0.0, color=A_PICK, linestyle="--", linewidth=0.9)
        ax.set_xlim([-0.25, 0.25])
        ax.set_ylim([0.3, n_plot + 0.9])
        ax.set_title(sta_ax, fontsize=9, pad=3)
        ax.set_yticks([])
        ax.set_xticks([-0.2, 0.0, 0.2])
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", alpha=0.15, linewidth=0.5)
        if k >= 4:
            ax.set_xlabel("Time from P pick (s)", fontsize=8.5)

    # Legend panel in the unused 8th cell.
    lax = axes.ravel()[7]
    lax.axis("off")
    lax.plot([], [], "k-", linewidth=0.8, label="Vertical-component waveform")
    lax.axvline(np.nan, color=A_PICK, linestyle="--", linewidth=0.9)
    lax.plot([], [], color=A_PICK, linestyle="--", linewidth=0.9, label="Catalog P pick")
    lax.legend(loc="center", fontsize=7.5, frameon=False)
    lax.text(0.5, 0.30, "Labels give the polarity assigned\nby cross-correlation (CC) and by\nAxialPolCap (ML) for the same trace.",
             transform=lax.transAxes, ha="center", va="top", fontsize=7, color=A_NOTE)

    fig.suptitle("")
    out = outdir / "Figure10_python.png"
    save_figure(fig, out)
    return out


def figure_s3(paths: Paths, outdir: Path) -> Path:
    """Figure S3: CC/ML polarity disagreements, styled like the merged Figure 2.

    Two panels, each holding ten randomly drawn conflicting waveforms, stacked
    and aligned on the catalog P pick.  Every station contributes at least one
    trace to a panel and the three remaining slots are filled at random from the
    stations that still have unused disagreements; the two panels draw disjoint
    sets of events.  The 1st, 3rd and 4th rows of panel (a), counted from the
    top, are then swapped for fresh traces of the same stations (see
    ``resample_top_rows_a`` below), which leaves panel (b) untouched.  Each trace
    is labelled on the right with its station and event ID.
    """
    conflict_wave, conflict_labels, _, _, conflict_id = _build_conflict_data(paths)

    pick, half = 50, 50          # 200 Hz, P pick at sample 50 -> +/- 0.25 s
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

    # Rows of panel (a), counted from the top, that are swapped for fresh traces
    # of the same station.  The replacements come from pool entries that neither
    # panel has consumed, so panel (b) is left exactly as drawn.
    resample_top_rows_a = (1, 3, 4)
    picked_a = draws_meta[0]
    for row in resample_top_rows_a:
        i = len(picked_a) - row                  # rows are stored bottom-up
        if not 0 <= i < len(picked_a):
            continue
        k, sta, sta_ax, _ = picked_a[i]
        j_new = take(sta)
        if j_new is not None:
            picked_a[i] = (k, sta, sta_ax, j_new)

    draws = [[(sta_ax, conflict_id[sta][j], as_1d(conflict_wave[sta][j]),
               conflict_labels[sta][j])
              for _, sta, sta_ax, j in picked]
             for picked in draws_meta]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 7.2), constrained_layout=True)
    t = np.arange(-half, half) / 200.0
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

    out = outdir / "FigureS03_python.png"
    save_figure(fig, out)
    return out


def figure_11(paths: Paths, outdir: Path) -> Path:
    _, _, valid_loc, conflict_loc, _ = _build_conflict_data(paths)
    lon_lim = [-130.031, -129.97]
    lat_lim = [45.92, 45.972]

    fig, axes = plt.subplots(3, 3, figsize=(FIG_WIDTH_IN, 6.6), constrained_layout=True)
    for i, sta in enumerate(STATIONS):
        ax = axes.ravel()[i]
        valid = np.array(valid_loc[sta], dtype=float) if valid_loc[sta] else np.empty((0, 2))
        conf = np.array(conflict_loc[sta], dtype=float) if conflict_loc[sta] else np.empty((0, 2))
        if valid.size > 0:
            ax.scatter(valid[:, 0], valid[:, 1], s=4, c="0.75", label="Agree")
        if conf.size > 0:
            ax.scatter(conf[:, 0], conf[:, 1], s=7, c="dodgerblue", label="Conflict")
        ax.plot(CALDERA_RIM[:, 0], CALDERA_RIM[:, 1], "k-", linewidth=1.5)
        if sta in STATION_COORDS:
            slon, slat = STATION_COORDS[sta]
            ax.plot(slon, slat, "ks", markersize=5, markerfacecolor="k")
            ax.text(slon, slat + 0.00035, f"AX{sta}", ha="center", fontsize=7)
        ax.set_xlim(lon_lim)
        ax.set_ylim(lat_lim)
        _set_geo_aspect(ax, lon_lim, lat_lim)
        ax.set_title(sta, fontsize=10)
        ax.grid(alpha=0.2)
        if i % 3 == 0:
            ax.set_ylabel("Latitude")
        if i >= 6:
            ax.set_xlabel("Longitude")
    axes.ravel()[7].axis("off")
    axes.ravel()[8].axis("off")
    fig.suptitle("")
    out = outdir / "Figure11_python.png"
    save_figure(fig, out)
    return out


def _color3_from_event(e) -> np.ndarray:
    c2 = np.asarray(get_value(e, "color2", np.array([0.0, 0.0, 0.0])), dtype=float).ravel()
    if c2.size < 3:
        c2 = np.array([0.0, 0.0, 0.0], dtype=float)
    q = str(get_value(e, "mechqual", "U"))
    if q in ("A", "B"):
        return np.clip(c2[:3], 0, 1)
    return np.clip(c2[:3] + 0.5 * (1 - c2[:3]), 0, 1)


def _focal_plane_curve(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=float).ravel()
    if abs(u[0]) < 1e-12:
        v = np.array([1.0, 0.0, 0.0])
    elif abs(u[1]) < 1e-12:
        v = np.array([0.0, 1.0, 0.0])
    else:
        v = np.array([1.0, -u[0] / u[1], 0.0], dtype=float)
        v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    a = np.linspace(1e-7, 2 * np.pi * 0.9999999, 360)
    vec = np.outer(np.cos(a), v) + np.outer(np.sin(a), w)
    vec = vec[vec[:, 2] <= 0]
    az = np.arctan2(vec[:, 0], vec[:, 1])
    theta = np.arccos(-vec[:, 2])
    p = np.sin(theta / 2.0)
    return p * np.sin(az), p * np.cos(az)


def _arc_append(xx: np.ndarray, yy: np.ndarray, a1: float, a2: float) -> Tuple[np.ndarray, np.ndarray]:
    # Port of MATLAB direction logic.
    if (a1 >= a2) and ((a1 - a2) <= np.pi):
        direc = 1
    elif (a2 >= a1) and ((a2 - a1) <= np.pi):
        direc = 2
    else:
        aa1 = a1 + 2 * np.pi if a1 < 0 else a1
        aa2 = a2 + 2 * np.pi if a2 < 0 else a2
        if (aa1 >= aa2) and ((aa1 - aa2) < np.pi):
            direc = 1
            a1, a2 = aa1, aa2
        elif (aa2 >= aa1) and ((aa2 - aa1) < np.pi):
            direc = 2
            a1, a2 = aa1, aa2
        else:
            direc = 1
    step = np.pi / 180.0
    if direc == 1:
        aa = np.arange(a2, a1 + step, step)
    else:
        aa = np.arange(a2, a1 - step, -step)
    xx2 = np.concatenate([xx, np.sin(aa) / np.sqrt(2.0)])
    yy2 = np.concatenate([yy, np.cos(aa) / np.sqrt(2.0)])
    return xx2, yy2


def plot_balloon(
    ax: plt.Axes, u1: Sequence[float], u2: Sequence[float], xx: float, yy: float, rr: float, scale: float, col
) -> None:
    try:
        u1 = np.asarray(u1, dtype=float).ravel()
        u2 = np.asarray(u2, dtype=float).ravel()
        if u1.size < 3 or u2.size < 3:
            return
        x1, y1 = _focal_plane_curve(u1)
        x2, y2 = _focal_plane_curve(u2)
        if x1.size < 3 or x2.size < 3:
            return

        d2 = (x1[:, None] - x2[None, :]) ** 2 + (y1[:, None] - y2[None, :]) ** 2
        i1, i2 = np.unravel_index(np.argmin(d2), d2.shape)

        if i1 == x1.size - 1:
            i1 = i1 - 1
        elif i1 != 0:
            p = np.min((x2 - x1[i1 + 1]) ** 2 + (y2 - y1[i1 + 1]) ** 2)
            m = np.min((x2 - x1[i1 - 1]) ** 2 + (y2 - y1[i1 - 1]) ** 2)
            if p > m:
                i1 = i1 - 1

        if i2 == x2.size - 1:
            i2 = i2 - 1
        elif i2 != 0:
            p = np.min((x1 - x2[i2 + 1]) ** 2 + (y1 - y2[i2 + 1]) ** 2)
            m = np.min((x1 - x2[i2 - 1]) ** 2 + (y1 - y2[i2 - 1]) ** 2)
            if p > m:
                i2 = i2 - 1

        A = np.array(
            [[x1[i1] - x1[i1 + 1], x2[i2 + 1] - x2[i2]], [y1[i1] - y1[i1 + 1], y2[i2 + 1] - y2[i2]]]
        )
        b = np.array([x2[i2 + 1] - x1[i1 + 1], y2[i2 + 1] - y1[i1 + 1]])
        w = np.linalg.solve(A, b)
        xi = w[0] * x1[i1] + (1 - w[0]) * x1[i1 + 1]
        yi = w[0] * y1[i1] + (1 - w[0]) * y1[i1 + 1]

        a = np.array([math.atan2(x1[0], y1[0]), math.atan2(x1[-1], y1[-1]), math.atan2(x2[0], y2[0]), math.atan2(x2[-1], y2[-1])])

        xx1 = np.concatenate([x1[: i1], np.array([xi]), x2[i2:]])
        yy1 = np.concatenate([y1[: i1], np.array([yi]), y2[i2:]])
        xx1, yy1 = _arc_append(xx1, yy1, a[0], a[3])

        xx2 = np.concatenate([x2[: i2 - 1 : -1], np.array([xi]), x1[i1:]])
        yy2 = np.concatenate([y2[: i2 - 1 : -1], np.array([yi]), y1[i1:]])
        xx2, yy2 = _arc_append(xx2, yy2, a[3], a[1])

        xx3 = np.concatenate([x2[: i2], np.array([xi]), x1[i1:]])
        yy3 = np.concatenate([y2[: i2], np.array([yi]), y1[i1:]])
        xx3, yy3 = _arc_append(xx3, yy3, a[2], a[1])

        xx4 = np.concatenate([x1[: i1], np.array([xi]), x2[i2 - 1 : 0 : -1]])
        yy4 = np.concatenate([y1[: i1], np.array([yi]), y2[i2 - 1 : 0 : -1]])
        xx4, yy4 = _arc_append(xx4, yy4, a[0], a[2])

        curves = [(xx1, yy1), (xx2, yy2), (xx3, yy3), (xx4, yy4)]
        maxn = max(c[0].size for c in curves)
        xmat = np.zeros((maxn, 4))
        ymat = np.zeros((maxn, 4))
        for i, (cx, cy) in enumerate(curves):
            xmat[: cx.size, i] = cx
            ymat[: cy.size, i] = cy
            if cx.size < maxn:
                xmat[cx.size :, i] = cx[-1]
                ymat[cy.size :, i] = cy[-1]

        ut = u1 + u2
        if ut[2] > 0:
            ut = -ut
        ut = ut / (np.linalg.norm(ut) + 1e-12)
        xt = ut[0] / np.sqrt(2.0)
        yt = ut[1] / np.sqrt(2.0)

        dx = xmat - xt
        dy = ymat - yt
        az = np.arctan2(dx, dy)
        nseg = np.zeros(4, dtype=int)
        for i in range(4):
            for minaz in np.arange(-np.pi, np.pi - 0.01, np.pi / 12):
                if np.any((az[:, i] >= minaz) & (az[:, i] <= minaz + np.pi / 12)):
                    nseg[i] += 1
        it = int(np.argmax(nseg))
        if it in (0, 2):
            use = [0, 2]
        else:
            use = [1, 3]

        # Circle and boundary.
        ax.add_patch(Circle((xx, yy), rr, facecolor="white", edgecolor="none", zorder=2))
        aa = np.linspace(0, 2 * np.pi, 180)
        ax.plot(np.sin(aa) * rr * scale + xx, np.cos(aa) * rr + yy, color="k", linewidth=0.3, zorder=3)

        for idx in use:
            cx = xmat[:, idx] * np.sqrt(2.0) * rr * scale + xx
            cy = ymat[:, idx] * np.sqrt(2.0) * rr + yy
            ax.fill(cx, cy, color=col, linewidth=0, zorder=3)
    except Exception:
        ax.plot(xx, yy, "o", markersize=1.5, color=col, zorder=3)


def figure_12(paths: Paths, outdir: Path) -> Path:
    cc_path = choose_existing(
        paths.fm3_root / "02-data" / "G_FM" / "G_2015Erp_polished.mat",
    )
    ml_path = choose_existing(
        paths.fm3_root / "02-data" / "G_FM" / "G_HASH_All_ML_sameClusterasbeforev_confidence.mat",
        paths.repo_root / "02-data" / "G_HASH_All_ML_sameClusterasbeforev_confidence.mat",
    )
    event_a = load_struct_array(cc_path, "event1")
    event_b = load_struct_array(ml_path, "event1")

    id_a = np.array([int(get_value(e, "id", -1)) for e in event_a], dtype=int)
    id_b = np.array([int(get_value(e, "id", -1)) for e in event_b], dtype=int)
    common = np.intersect1d(id_a, id_b)
    event_a = event_a[np.isin(id_a, common)]
    event_b = event_b[np.isin(id_b, common)]

    b_by_id = {int(get_value(e, "id", -1)): e for e in event_b}
    for i, e in enumerate(event_a):
        rid = int(get_value(e, "id", -1))
        b = b_by_id.get(rid)
        if b is None:
            continue
        e.lat = float(get_value(b, "lat", np.nan))
        e.lon = float(get_value(b, "lon", np.nan))
        e.depth = float(get_value(b, "depth", np.nan))
        e.time = float(get_value(b, "time", np.nan))

    def filt(arr: np.ndarray) -> np.ndarray:
        out = []
        for e in arr:
            q = str(get_value(e, "mechqual", "U"))
            if (get_value(e, "lat", 999) <= 45.969) and (get_value(e, "lon", -999) >= -130.03) and (q not in ("C", "D")):
                out.append(e)
        return np.array(out, dtype=object)

    event_a = filt(event_a)
    event_b = filt(event_b)

    # Re-match after filtering.
    id_a = np.array([int(get_value(e, "id", -1)) for e in event_a], dtype=int)
    id_b = np.array([int(get_value(e, "id", -1)) for e in event_b], dtype=int)
    common2 = np.intersect1d(id_a, id_b)
    event_a = event_a[np.isin(id_a, common2)]
    event_b = event_b[np.isin(id_b, common2)]

    date_bf = matlab_datenum(2015, 4, 24, 8, 0, 0)
    date_dr = matlab_datenum(2015, 5, 19, 0, 0, 0)
    lon_lim = [-130.031, -129.97]
    lat_lim = [45.92, 45.970]

    def in_period(ev, period):
        t = float(get_value(ev, "time", np.nan))
        if period == 0:
            return t < date_bf
        if period == 1:
            return date_bf <= t < date_dr
        return t >= date_dr

    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH_IN, 4.1), constrained_layout=True)
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    col_titles = ["Before", "During", "After"]
    catalogs = [event_a, event_b]
    max_events_per_panel = 900

    for r in range(2):
        for c in range(3):
            ax = axes[r, c]
            evp = [e for e in catalogs[r] if in_period(e, c)]
            if len(evp) > max_events_per_panel:
                idx = np.linspace(0, len(evp) - 1, max_events_per_panel, dtype=int)
                evp = [evp[i] for i in idx]
            for e in evp:
                u1 = as_1d(get_value(e, "avfnorm", np.array([])))
                u2 = as_1d(get_value(e, "avslip", np.array([])))
                if u1.size >= 3 and u2.size >= 3:
                    plot_balloon(
                        ax,
                        u1,
                        u2,
                        float(get_value(e, "lon", np.nan)),
                        float(get_value(e, "lat", np.nan)),
                        rr=0.0005,
                        scale=1.3,
                        col=_color3_from_event(e),
                    )
            ax.plot(CALDERA_RIM[:, 0], CALDERA_RIM[:, 1], "k-", linewidth=1.4)
            ax.set_xlim(lon_lim)
            ax.set_ylim(lat_lim)
            _set_geo_aspect(ax, lon_lim, lat_lim)
            ax.set_xticks([-130.02, -130.00, -129.98])
            ax.set_yticks([45.93, 45.94, 45.95, 45.96, 45.97])
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.tick_params(labelsize=7.5)
            if c > 0:
                ax.set_yticklabels([])
            if r == 0:
                ax.set_xticklabels([])
            ax.grid(alpha=0.2)
            ax.text(lon_lim[0] + 0.002, lat_lim[1] - 0.002, panel_labels[r * 3 + c], fontsize=12, fontweight="bold")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11)
            if c == 0:
                ax.set_ylabel("Latitude")
            else:
                ax.set_yticklabels([])
            if r == 1:
                ax.set_xlabel("Longitude")
            else:
                ax.set_xticklabels([])

    fig.suptitle("")
    out = outdir / "Figure12_python.png"
    save_figure(fig, out)
    return out


def _quatp(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q3 = np.zeros(4, dtype=float)
    q3[0] = q1[3] * q2[0] + q1[2] * q2[1] - q1[1] * q2[2] + q1[0] * q2[3]
    q3[1] = -q1[2] * q2[0] + q1[3] * q2[1] + q1[0] * q2[2] + q1[1] * q2[3]
    q3[2] = q1[1] * q2[0] - q1[0] * q2[1] + q1[3] * q2[2] + q1[2] * q2[3]
    q3[3] = -q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] + q1[3] * q2[3]
    return q3


def _quatd(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    qc1 = np.array([-q1[0], -q1[1], -q1[2], q1[3]], dtype=float)
    return _quatp(qc1, q2)


def _sphcoor(quat: np.ndarray) -> Tuple[float, float, float]:
    q = quat.copy()
    if q[3] < 0:
        q = -q
    q4n = np.sqrt(max(0.0, 1.0 - q[3] ** 2))
    costh = 1.0
    if abs(q4n) > 1e-10:
        costh = q[2] / q4n
    costh = np.clip(costh, -1.0, 1.0)
    theta = np.degrees(np.arccos(costh))
    angl = 2.0 * np.degrees(np.arccos(np.clip(q[3], -1.0, 1.0)))
    phi = 0.0
    if abs(q[0]) > 1e-10 or abs(q[1]) > 1e-10:
        phi = np.degrees(np.arctan2(q[1], q[0]))
    if phi < 0:
        phi += 360.0
    return angl, theta, phi


def _boxtest(q1: np.ndarray, icode: int) -> Tuple[np.ndarray, float]:
    quat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    if icode == 0:
        icode = 1
        qm = abs(q1[0])
        for ixc in range(2, 5):
            if abs(q1[ixc - 1]) > qm:
                qm = abs(q1[ixc - 1])
                icode = ixc
    if icode == 4:
        q2 = q1.copy()
    else:
        quatt = quat[:, icode - 1]
        q2 = _quatp(quatt, q1)
    if q2[3] < 0:
        q2 = -q2
    qm = q2[3]
    return q2, qm


def _f4r1(q1: np.ndarray, q2: np.ndarray, icode: int) -> np.ndarray:
    qr1, _ = _boxtest(q1, icode)
    return _quatd(qr1, q2)


def _quat_fps(dd: float, da: float, sa: float) -> np.ndarray:
    cdd = np.cos(np.radians(dd))
    sdd = np.sin(np.radians(dd))
    cda = np.cos(np.radians(da))
    sda = np.sin(np.radians(da))
    csa = np.cos(np.radians(sa))
    ssa = np.sin(np.radians(sa))
    s1 = csa * sdd - ssa * cda * cdd
    s2 = -csa * cdd - ssa * cda * sdd
    s3 = -ssa * sda
    v1 = sda * cdd
    v2 = sda * sdd
    v3 = -cda
    an1 = s2 * v3 - v2 * s3
    an2 = v1 * s3 - s1 * v3
    an3 = s1 * v2 - v1 * s2
    d2 = 1.0 / np.sqrt(2.0)
    t1 = (v1 + s1) * d2
    t2 = (v2 + s2) * d2
    t3 = (v3 + s3) * d2
    p1 = (v1 - s1) * d2
    p2 = (v2 - s2) * d2
    p3 = (v3 - s3) * d2
    u0 = (t1 + p2 + an3 + 1.0) / 4.0
    u1 = (t1 - p2 - an3 + 1.0) / 4.0
    u2 = (-t1 + p2 - an3 + 1.0) / 4.0
    u3 = (-t1 - p2 + an3 + 1.0) / 4.0
    um = max(u0, u1, u2, u3)
    if um == u0:
        u0 = np.sqrt(u0)
        u3 = (t2 - p1) / (4.0 * u0)
        u2 = (an1 - t3) / (4.0 * u0)
        u1 = (p3 - an2) / (4.0 * u0)
    elif um == u1:
        u1 = np.sqrt(u1)
        u2 = (t2 + p1) / (4.0 * u1)
        u3 = (an1 + t3) / (4.0 * u1)
        u0 = (p3 - an2) / (4.0 * u1)
    elif um == u2:
        u2 = np.sqrt(u2)
        u1 = (t2 + p1) / (4.0 * u2)
        u0 = (an1 - t3) / (4.0 * u2)
        u3 = (p3 + an2) / (4.0 * u2)
    else:
        u3 = np.sqrt(u3)
        u0 = (t2 - p1) / (4.0 * u3)
        u1 = (an1 + t3) / (4.0 * u3)
        u2 = (p3 + an2) / (4.0 * u3)
    quat = np.array([u1, u2, u3, u0], dtype=float)
    return quat


def kagan_angle(mech_old: Sequence[float], mech_new: Sequence[float]) -> float:
    q1 = _quat_fps(float(mech_old[0]), float(mech_old[1]), float(mech_old[2]))
    q2 = _quat_fps(float(mech_new[0]), float(mech_new[1]), float(mech_new[2]))
    best = 180.0
    for i in range(1, 5):
        qdum = _f4r1(q1, q2, i)
        rot, _, _ = _sphcoor(qdum)
        if rot < best:
            best = rot
    return float(best)


def figure_13(paths: Paths, outdir: Path) -> Path:
    ml_path = choose_existing(
        paths.fm3_root / "02-data" / "G_FM" / "G_HASH_All_ML_sameClusterasbeforev_confidence.mat",
        paths.repo_root / "02-data" / "G_HASH_All_ML_sameClusterasbeforev_confidence.mat",
    )
    cc_path = choose_existing(
        paths.fm3_root / "02-data" / "G_FM" / "G_2015Erp_polished.mat",
    )
    ml = load_struct_array(ml_path, "event1")
    cc = load_struct_array(cc_path, "event1")
    cc_by_id = {int(get_value(e, "id", -1)): e for e in cc}

    lon, lat, kg = [], [], []
    for e in ml:
        rid = int(get_value(e, "id", -1))
        c = cc_by_id.get(rid)
        if c is None:
            continue
        m1 = as_1d(get_value(e, "avmech", np.array([])))
        m2 = as_1d(get_value(c, "avmech", np.array([])))
        if m1.size < 3 or m2.size < 3:
            continue
        try:
            k = kagan_angle(m1[:3], m2[:3])
        except Exception:
            continue
        if np.isfinite(k):
            kg.append(k)
            lon.append(float(get_value(e, "lon", np.nan)))
            lat.append(float(get_value(e, "lat", np.nan)))

    kg = np.asarray(kg, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    m = np.isfinite(kg) & np.isfinite(lon) & np.isfinite(lat)
    kg, lon, lat = kg[m], lon[m], lat[m]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 3.0), constrained_layout=True)
    axes[0].hist(kg, bins=np.linspace(0, 120, 25), color=(0.2, 0.4, 0.9), edgecolor="k")
    axes[0].set_xlim([0, 120])
    axes[0].grid(alpha=0.2)
    axes[0].set_xlabel("Kagan angle (deg)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("")
    panel_label(axes[0], "(a)")

    sc = axes[1].scatter(lon, lat, c=kg, s=6, cmap=SEQ_CMAP, vmin=0, vmax=120,
                         linewidths=0)
    axes[1].plot(CALDERA_RIM[:, 0], CALDERA_RIM[:, 1], "k-", linewidth=1.0)
    axes[1].set_xlim([-130.03, -129.97])
    axes[1].set_ylim([45.92, 45.97])
    _set_geo_aspect(axes[1], [-130.03, -129.97], [45.92, 45.97])
    axes[1].grid(alpha=0.15, linewidth=0.5)
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].set_title("")
    # Absolute longitudes, matching Figure 11; the default offset notation
    # rendered these as "-1.3e2" which is unreadable.
    axes[1].set_xticks([-130.02, -130.00, -129.98])
    axes[1].set_yticks([45.93, 45.94, 45.95, 45.96, 45.97])
    axes[1].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[1].tick_params(labelsize=8)
    panel_label(axes[1], "(b)")
    cb = fig.colorbar(sc, ax=axes[1], shrink=0.85, aspect=18)
    cb.set_label("Kagan angle (deg)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    out = outdir / "Figure13_python.png"
    save_figure(fig, out)
    return out


def figure_14(paths: Paths, outdir: Path) -> Path:
    """Real-time focal-mechanism pipeline schematic.

    Re-laid out for the printed page width.  The previous version was drawn on a
    14 x 6 in canvas with 10-14 pt type; at the size Word actually placed it the
    labels overflowed their boxes and the right-hand column ran off the figure.
    Coordinates below are chosen for a 7.2 x 4.4 in canvas, and the in-figure
    title was removed because the caption carries it.
    """
    fig_w, fig_h = FIG_WIDTH_WIDE_IN, 4.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    FS = 7.0        # box text
    FS_HEAD = 7.5   # group headers

    def rect(x, y, w, h, txt, fc="white", bold=False):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor=fc, edgecolor="black", linewidth=0.8,
        ))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=FS_HEAD if bold else FS,
                fontweight="bold" if bold else "normal")

    def ellipse(cx, cy, w, h, txt, fc):
        ax.add_patch(Ellipse((cx, cy), w, h, facecolor=fc, edgecolor="black", linewidth=0.8))
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=FS)

    def hexagon(cx, cy, r, txt, fc):
        # Compensate for the non-square axes so the hexagon reads as regular.
        th = np.linspace(0, 2 * np.pi, 7)
        pts = np.c_[cx + r * np.cos(th), cy + r * (fig_w / fig_h) * np.sin(th)]
        ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor="black", linewidth=0.9))
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=FS, fontweight="bold")

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.25",
                                    shrinkA=1.5, shrinkB=1.5))

    # --- top flow: waveform -> model -> polarity -------------------------------
    rect(0.01, 0.865, 0.21, 0.062, "Input", TINT_POOL, bold=True)
    rect(0.01, 0.800, 0.21, 0.062, "3-component waveform")
    ellipse(0.42, 0.862, 0.21, 0.115, "AxialPolCap", TINT_REG)
    rect(0.62, 0.831, 0.22, 0.062, "First-motion polarity")
    arr(0.225, 0.862, 0.312, 0.862)
    arr(0.528, 0.862, 0.617, 0.862)

    # --- secondary inputs ------------------------------------------------------
    rect(0.26, 0.600, 0.22, 0.062, "S/P amplitude ratios")
    rect(0.01, 0.300, 0.21, 0.058, "Earthquake location")
    rect(0.01, 0.238, 0.21, 0.058, "Station location")
    rect(0.01, 0.176, 0.21, 0.058, "Velocity model")

    # --- inversion -------------------------------------------------------------
    hexagon(0.535, 0.400, 0.075, "SKHASH", TINT_CONV)
    arr(0.730, 0.828, 0.590, 0.505)
    arr(0.370, 0.597, 0.487, 0.487)
    arr(0.225, 0.245, 0.462, 0.372)

    # --- outputs ---------------------------------------------------------------
    rect(0.76, 0.520, 0.23, 0.062, "Focal mechanism", TINT_POOL, bold=True)
    rect(0.76, 0.458, 0.23, 0.058, "Strike")
    rect(0.76, 0.400, 0.23, 0.058, "Dip")
    rect(0.76, 0.342, 0.23, 0.058, "Rake")
    rect(0.76, 0.222, 0.23, 0.062, "Stress inversion", TINT_POOL, bold=True)
    rect(0.76, 0.160, 0.23, 0.058, "P axis")
    rect(0.76, 0.102, 0.23, 0.058, "T axis")
    rect(0.76, 0.044, 0.23, 0.058, "Shape ratio")
    arr(0.612, 0.400, 0.755, 0.400)
    arr(0.875, 0.340, 0.875, 0.288)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    out = outdir / "Figure14_python.png"
    save_figure(fig, out)
    return out


def write_benchmark_table(outdir: Path) -> Path:
    """Supplementary table of per-station benchmark accuracies (review comment 16:
    "the reader might wonder which performs best of the DL models").

    Numbers are the same ones plotted in Figure 4.
    """
    # DiTingMotion, EQPolarity and PolarCAP re-run 2026-09-18; CFM and the
    # cross-correlation baseline are unchanged and keep their original values.
    diting = np.array([94.1, 70.9, 85.8, 76.2, 81.6, 66.2, 84.8])
    cfm = np.array([97.0, 81.0, 97.4, 88.3, 87.1, 82.9, 74.5])
    eqp = np.array([96.5, 79.4, 96.8, 87.1, 86.0, 83.3, 75.4])
    polcap = np.array([96.1, 79.8, 97.3, 87.8, 86.4, 81.9, 75.4])
    cc = np.array([0.99972, 0.95991, 0.99134, 0.93391, 0.84833, 0.87012, 0.80463]) * 100

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


FIGURE_FUNCS = {
    1: figure_01,
    2: figure_02,
    3: figure_03,
    4: figure_04,
    5: figure_05,
    6: figure_06,
    7: figure_07,
    8: figure_08,
    9: figure_09,
    10: figure_10,
    11: figure_11,
    12: figure_12,
    13: figure_13,
    14: figure_14,
    15: figure_02_merged,   # merged replacement for Figures 2 and 3 (AXAS2 only)
    17: figure_s_confusion,  # supplementary: benchmark confusion matrices
    16: figure_s3,           # supplementary Figure S3 (replaces old Figure 10)
}


def parse_figure_ids(figures: str) -> List[int]:
    if figures.strip().lower() == "all":
        return list(range(1, 15))
    out = []
    for part in figures.split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v < 1 or v > 17:
            raise ValueError("Figure ids must be in [1,17] (15 = merged Figure 2+3, 16 = Figure S3, 17 = supplementary confusion matrices)")
        out.append(v)
    return sorted(set(out))


def main() -> None:
    default_repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate manuscript figures in Python")
    parser.add_argument("--figures", default="all", help="Comma list (e.g., 1,2,10) or 'all'")
    parser.add_argument(
        "--outdir",
        default=str(default_repo / "04-manuscripts" / "python_figures" / "output"),
        help="Output directory for generated figures",
    )
    parser.add_argument("--fm-root", default="/Users/mczhang/Documents/GitHub/FM")
    parser.add_argument("--fm3-root", default="/Users/mczhang/Documents/GitHub/FM3")
    parser.add_argument("--fm4-root", default="/Users/mczhang/Documents/GitHub/FM4")
    parser.add_argument(
        "--docx",
        default=str(default_repo / "04-manuscripts" / "MZhang_week_10 an edit.docx"),
        help="Manuscript .docx used for image fallback",
    )
    parser.add_argument(
        "--fallback-docx-image",
        action="store_true",
        help="If a figure build fails, extract imageN.png from docx as fallback.",
    )
    args = parser.parse_args()

    apply_manuscript_style()

    paths = Paths(
        repo_root=default_repo,
        fm_root=Path(args.fm_root),
        fm3_root=Path(args.fm3_root),
        fm4_root=Path(args.fm4_root),
        docx_path=Path(args.docx),
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig_ids = parse_figure_ids(args.figures)
    print(f"Generating figures: {fig_ids}")
    print(f"Output dir: {outdir}")

    for fid in fig_ids:
        fn = FIGURE_FUNCS[fid]
        out = outdir / f"Figure{fid:02d}_python.png"
        try:
            written = fn(paths, outdir)
            print(f"[OK] Figure {fid:02d} -> {written}")
        except Exception as exc:
            print(f"[FAIL] Figure {fid:02d}: {exc}")
            if args.fallback_docx_image:
                ok = extract_docx_image(paths, fid, out)
                if ok:
                    print(f"[FALLBACK] Extracted Figure {fid:02d} image from docx -> {out}")
                else:
                    print(f"[FALLBACK-FAIL] Could not extract image{fid}.png from docx")

    if 4 in fig_ids:
        tbl = write_benchmark_table(outdir)
        print(f"[OK] Table S1 -> {tbl}")


if __name__ == "__main__":
    main()
