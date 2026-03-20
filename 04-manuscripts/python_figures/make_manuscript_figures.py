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
from matplotlib.ticker import ScalarFormatter
from scipy.io import loadmat, netcdf_file
from scipy.signal import resample
from scipy.stats import lognorm


STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]
STATIONS_AVG = STATIONS + ["Average"]

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
        "font.family": "Helvetica",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
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
    template_mat = paths.repo_root / "02-data" / "A_wave" / "A_wave_dB20_cleaned.mat"
    noise_mat = paths.repo_root / "02-data" / "A_wave" / "A_wave_noise_10000.mat"
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
    fig.suptitle("Figure 2: Template and noise waveforms by station", fontsize=14)
    out = outdir / "Figure02_python.png"
    save_figure(fig, out)
    return out


def figure_03(paths: Paths, outdir: Path) -> Path:
    as1 = load_struct_array(paths.repo_root / "02-data" / "K_aug" / "AS1.mat", "AS1")
    snr_values = load_scalar_or_array(paths.repo_root / "02-data" / "H_Noi" / "H_noise_dB20_snrValue.mat", "snrValues")
    noise = load_struct_array(paths.repo_root / "02-data" / "H_Noi" / "H_Noise_200.mat", "Felix")

    snr_data = np.asarray(snr_values[0], dtype=float).ravel()
    snr_data = snr_data[np.isfinite(snr_data) & (snr_data > 0)]
    shape, loc, scale = lognorm.fit(snr_data, floc=0)
    rng = np.random.default_rng(3)

    base = as1[3]
    signal = as_1d(get_value(base, "W_AS1", np.array([])))
    signal = signal[:200]
    if signal.size == 0:
        raise RuntimeError("AS1 waveform not found for Figure 3")

    traces: List[np.ndarray] = [signal.copy()]
    snr_labels: List[float] = []
    for k in range(11):
        if k == 0:
            syn = signal.copy()
        else:
            target_db = lognorm.rvs(shape, loc=loc, scale=scale, random_state=rng)
            valid = False
            while not valid:
                n = noise[rng.integers(0, len(noise))]
                w = as_1d(get_value(n, "W_AS1", np.array([])))
                valid = w.size == 200 and np.max(np.abs(w)) > 0
            rms_signal = np.sqrt(np.mean(signal**2))
            rms_noise_target = rms_signal / (10 ** (target_db / 20))
            rms_noise = np.sqrt(np.mean(w**2))
            syn = signal + (rms_noise_target / (rms_noise + 1e-12)) * w
        nseg = syn[:80]
        sseg = syn[80:160]
        snr_db = 20 * np.log10((np.sqrt(np.mean(sseg**2)) + 1e-12) / (np.sqrt(np.mean(nseg**2)) + 1e-12))
        snr_labels.append(float(snr_db))
        traces.append(syn)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, w in enumerate(traces[:11]):
        w = norm_wave(w)
        yoff = i * 2.0
        color = "red" if i == 0 else "black"
        ax.plot(np.arange(w.size), w + yoff, color=color, linewidth=1.6 if i == 0 else 1.0)
        txt_color = "red" if i == 0 else "blue"
        ax.text(205, yoff, f"{snr_labels[i]:.1f} dB", fontsize=9, color=txt_color, va="center")
    ax.set_xlim([0, 240])
    ax.set_ylim([-1, 22])
    ax.set_yticks([])
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Normalized waveform (stacked)")
    ax.set_title("Figure 3: AS1 augmentation examples with target SNR")
    ax.grid(alpha=0.15)
    out = outdir / "Figure03_python.png"
    save_figure(fig, out)
    return out


def _grouped_bar(ax: plt.Axes, data: np.ndarray, labels: Sequence[str], series: Sequence[str], colors: Sequence, ylim):
    n_groups, n_series = data.shape
    x = np.arange(n_groups)
    width = 0.82 / n_series
    for i in range(n_series):
        ax.bar(x - 0.41 + width / 2 + i * width, data[:, i], width=width, color=colors[i], label=series[i], edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(ylim)
    ax.grid(axis="y", alpha=0.25)


def figure_04(paths: Paths, outdir: Path) -> Path:
    colors = [(0.95, 0.80, 0.45), (0.95, 0.60, 0.60), (0.60, 0.90, 0.90), (0.60, 0.80, 0.60), (0.75, 0.65, 0.95)]
    diting = np.array([97.03, 78.23, 97.86, 89.10, 88.04, 77.03, 73.23])
    cfm = np.array([97.03, 80.98, 97.35, 88.27, 87.12, 82.91, 74.46])
    eqp = np.array([92.85, 76.57, 90.85, 85.16, 83.09, 80.64, 72.49])
    polcap = np.array([0.7942, 0.7270, 0.8186, 0.7699, 0.8307, 0.7617, 0.6857]) * 100
    cc = np.array([0.99972, 0.95991, 0.99134, 0.93391, 0.84833, 0.87012, 0.80463]) * 100
    data = np.vstack(
        [
            np.r_[diting, diting.mean()],
            np.r_[cfm, cfm.mean()],
            np.r_[eqp, eqp.mean()],
            np.r_[polcap, polcap.mean()],
            np.r_[cc, cc.mean()],
        ]
    ).T

    fig, ax = plt.subplots(figsize=(11, 6))
    _grouped_bar(ax, data, STATIONS_AVG, ["DiTingMotion", "CFM", "EQPolarity", "AxialPolCap", "CC"], colors, (60, 100))
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Station")
    ax.set_title("Figure 4: Benchmark accuracy by station")
    ax.legend(ncol=3, fontsize=9)
    out = outdir / "Figure04_python.png"
    save_figure(fig, out)
    return out


def figure_05(paths: Paths, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f6f6f6", ec="black", fs=10):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor=fc, edgecolor=ec, linewidth=1.2)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5))

    # Encoder
    box(0.03, 0.44, 0.12, 0.12, "Input\n1 x 200")
    box(0.20, 0.44, 0.12, 0.12, "Conv1D\n32, k=32", fc="#e8f4ff")
    box(0.36, 0.44, 0.10, 0.12, "MaxPool", fc="#e8ffe8")
    box(0.50, 0.44, 0.12, 0.12, "Conv1D\n8, k=16", fc="#e8f4ff")
    box(0.66, 0.44, 0.10, 0.12, "Dropout\n0.3", fc="#fff7e8")
    ell = Ellipse((0.84, 0.50), 0.13, 0.14, facecolor="#f0ecff", edgecolor="black")
    ax.add_patch(ell)
    ax.text(0.84, 0.50, "Latent\ncode", ha="center", va="center", fontsize=10)

    # Decoder
    box(0.50, 0.16, 0.12, 0.12, "Conv1D\n8, k=16", fc="#e8f4ff")
    box(0.34, 0.16, 0.12, 0.12, "Upsample", fc="#e8ffe8")
    box(0.18, 0.16, 0.12, 0.12, "Conv1D\n32, k=32", fc="#e8f4ff")
    box(0.03, 0.16, 0.12, 0.12, "Output\n1 x 200")

    # Classifier head
    box(0.80, 0.76, 0.17, 0.10, "Softmax classifier\nP(Up), P(Down)", fc="#ffeef2")

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
    ax.text(0.02, 0.95, "Figure 5: Autoencoder architecture", fontsize=14, fontweight="bold")

    out = outdir / "Figure05_python.png"
    save_figure(fig, out)
    return out


def figure_06(paths: Paths, outdir: Path) -> Path:
    colors = [(0.95, 0.80, 0.45), (0.95, 0.60, 0.60), (0.60, 0.90, 0.90)]
    acc_all = np.array([0.9935, 0.9858, 0.9876, 0.9867, 0.9868, 0.9871, 0.9644])
    acc_loso = np.array([0.9964, 0.9855, 0.9902, 0.9869, 0.9854, 0.9877, 0.9653])
    acc_fine = np.array([0.9829, 0.9484, 0.9653, 0.9585, 0.9534, 0.9386, 0.9013])
    data = np.vstack(
        [np.r_[acc_all, acc_all.mean()], np.r_[acc_loso, acc_loso.mean()], np.r_[acc_fine, acc_fine.mean()]]
    ).T * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    _grouped_bar(
        ax,
        data,
        STATIONS_AVG,
        ["All-station train", "LOSO", "Transfer learning"],
        colors,
        (88, 100.5),
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Station")
    ax.set_title("Figure 6: Accuracy by training strategy")
    ax.legend(ncol=3, fontsize=9)
    out = outdir / "Figure06_python.png"
    save_figure(fig, out)
    return out


def figure_07(paths: Paths, outdir: Path) -> Path:
    colors = [(0.95, 0.80, 0.45), (0.95, 0.60, 0.60), (0.60, 0.90, 0.90)]
    # New model (training) and transfer model under SNR conditions
    A5 = np.array([0.9935, 0.9858, 0.9876, 0.9867, 0.9868, 0.9871, 0.9644, 0.9850]) * 100  # Orig
    B5 = np.array([0.9859, 0.9505, 0.9739, 0.9584, 0.9623, 0.9569, 0.9341, 0.9604]) * 100  # High
    C5 = np.array([0.9940, 0.9731, 0.9868, 0.9760, 0.9690, 0.9850, 0.9770, 0.9801]) * 100  # Low
    A6 = np.array([0.9869, 0.9555, 0.9706, 0.9621, 0.9583, 0.9507, 0.9334, 0.9596]) * 100  # Orig
    B6 = np.array([0.9827, 0.9425, 0.9682, 0.9617, 0.9558, 0.9437, 0.9184, 0.9534]) * 100  # High
    C6 = np.array([0.9879, 0.9537, 0.9667, 0.9693, 0.9567, 0.9555, 0.9385, 0.9612]) * 100  # Low
    data_train = np.vstack([B5, A5, C5]).T
    data_transfer = np.vstack([B6, A6, C6]).T

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    _grouped_bar(axes[0], data_train, STATIONS_AVG, ["High-SNR", "Orig-SNR", "Low-SNR"], colors, (90, 101))
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Figure 7a: New-model accuracy vs training SNR")
    axes[0].legend(ncol=3, fontsize=9)

    _grouped_bar(
        axes[1], data_transfer, STATIONS_AVG, ["High-SNR", "Orig-SNR", "Low-SNR"], colors, (90, 100.5)
    )
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xlabel("Station")
    axes[1].set_title("Figure 7b: Transfer-learning accuracy vs training SNR")
    axes[1].legend(ncol=3, fontsize=9)

    out = outdir / "Figure07_python.png"
    save_figure(fig, out)
    return out


def figure_08(paths: Paths, outdir: Path) -> Path:
    colors = [(0.95, 0.80, 0.45), (0.95, 0.60, 0.60), (0.60, 0.90, 0.90)]
    ft02 = np.array([0.8278, 0.7496, 0.8066, 0.7712, 0.7533, 0.7358, 0.7455, 0.7700]) * 100
    ft01 = np.array([0.9524, 0.8793, 0.9167, 0.9123, 0.8909, 0.8689, 0.8678, 0.8984]) * 100
    ft00 = np.array([0.9869, 0.9555, 0.9706, 0.9621, 0.9583, 0.9507, 0.9334, 0.9596]) * 100
    tr02 = np.array([0.8335, 0.7520, 0.8133, 0.7512, 0.7288, 0.7271, 0.7393, 0.7636]) * 100
    tr01 = np.array([0.9704, 0.8992, 0.9415, 0.9310, 0.9022, 0.9007, 0.8991, 0.9207]) * 100
    tr00 = np.array([0.9935, 0.9858, 0.9876, 0.9867, 0.9868, 0.9871, 0.9644, 0.9850]) * 100
    data_transfer = np.vstack([ft00, ft01, ft02]).T
    data_train = np.vstack([tr00, tr01, tr02]).T

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    _grouped_bar(axes[0], data_train, STATIONS_AVG, ["0.00 s", "0.01 s", "0.02 s"], colors, (70, 105))
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Figure 8a: New model accuracy vs imposed pick-time shift")
    axes[0].legend(title="Test-time shift", ncol=3, fontsize=9)

    _grouped_bar(axes[1], data_transfer, STATIONS_AVG, ["0.00 s", "0.01 s", "0.02 s"], colors, (70, 103))
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xlabel("Station")
    axes[1].set_title("Figure 8b: Transfer model accuracy vs imposed pick-time shift")
    axes[1].legend(title="Test-time shift", ncol=3, fontsize=9)

    out = outdir / "Figure08_python.png"
    save_figure(fig, out)
    return out


def figure_09(paths: Paths, outdir: Path) -> Path:
    colors = [(0.95, 0.80, 0.45), (0.95, 0.60, 0.60), (0.60, 0.90, 0.90)]
    a_train = np.array([0.9942, 0.9483, 0.9789, 0.9668, 0.9598, 0.9631, 0.9466, 0.9655]) * 100
    b_train = np.array([0.9722, 0.9052, 0.9389, 0.9244, 0.9192, 0.9254, 0.8892, 0.9250]) * 100
    c_train = np.array([0.9948, 0.9407, 0.9840, 0.9674, 0.9640, 0.9472, 0.9311, 0.9615]) * 100
    a_ft = np.array([0.9776, 0.8992, 0.9438, 0.9351, 0.9294, 0.8945, 0.8884, 0.9241]) * 100
    b_ft = np.array([0.8976, 0.8115, 0.8711, 0.8353, 0.8251, 0.7875, 0.7963, 0.8322]) * 100
    c_ft = np.array([0.9768, 0.8583, 0.9498, 0.9128, 0.9167, 0.8613, 0.8354, 0.9020]) * 100

    data_train = np.vstack([a_train, b_train, c_train]).T
    data_transfer = np.vstack([a_ft, b_ft, c_ft]).T

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    _grouped_bar(
        axes[0],
        data_train,
        STATIONS_AVG,
        ["Train 0.01 / Test 0.01", "Train 0.01 / Test 0.02", "Train 0.02 / Test 0.01"],
        colors,
        (75, 102),
    )
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Figure 9a: New model under train/test shift combinations")
    axes[0].legend(ncol=2, fontsize=8)

    _grouped_bar(
        axes[1],
        data_transfer,
        STATIONS_AVG,
        ["FT 0.01 / Test 0.01", "FT 0.01 / Test 0.02", "FT 0.02 / Test 0.01"],
        colors,
        (75, 102),
    )
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xlabel("Station")
    axes[1].set_title("Figure 9b: Transfer model under train/test shift combinations")
    axes[1].legend(ncol=2, fontsize=8)
    out = outdir / "Figure09_python.png"
    save_figure(fig, out)
    return out


def _build_conflict_data(paths: Paths):
    wave_path = paths.fm4_root / "02-data" / "Before22OBSs" / "A_All" / "A_wavelarge5.mat"
    clu_path = paths.fm4_root / "02-data" / "Before22OBSs" / "F_Cl" / "F_Cl_All_MLreplace_samecluster_conf.mat"
    felix = load_struct_array(wave_path, "Felix")
    po_clu = load_struct_array(clu_path, "Po_Clu")
    id_to_idx = {int(get_value(e, "ID", -1)): i for i, e in enumerate(felix)}

    po_fields = [f"Po_{s}" for s in STATIONS]
    w_fields = [f"W_{s}" for s in STATIONS]

    conflict_wave: Dict[str, List[np.ndarray]] = {s: [] for s in STATIONS}
    conflict_labels: Dict[str, List[str]] = {s: [] for s in STATIONS}
    valid_loc: Dict[str, List[Tuple[float, float]]] = {s: [] for s in STATIONS}
    conflict_loc: Dict[str, List[Tuple[float, float]]] = {s: [] for s in STATIONS}

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
    return conflict_wave, conflict_labels, valid_loc, conflict_loc


def figure_10(paths: Paths, outdir: Path) -> Path:
    conflict_wave, conflict_labels, _, _ = _build_conflict_data(paths)

    fig, axes = plt.subplots(1, 7, figsize=(19, 4.8), constrained_layout=True)
    rng = np.random.default_rng(42)
    for ax, sta in zip(axes, STATIONS):
        waves = conflict_wave[sta]
        labels = conflict_labels[sta]
        if not waves:
            ax.set_title(f"{sta}\n(no conflict)")
            ax.axis("off")
            continue
        n_plot = min(len(waves), 10)
        picks = rng.choice(len(waves), size=n_plot, replace=False)
        for i, idx in enumerate(picks, start=1):
            w = norm_wave(waves[idx][:100])  # match MATLAB xlim [0, 100]
            ax.plot(np.arange(w.size), w + i, "k-", linewidth=1.2)
            ax.text(3, i + 0.22, labels[idx], color="b", fontsize=8)
        ax.axvline(50, color="r", linestyle="--", linewidth=1.0)
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 11])
        ax.set_title(sta, fontsize=10)
        ax.set_yticks([])
        ax.grid(alpha=0.15)
    fig.suptitle("Figure 10: Conflicting CC vs ML polarity waveforms by station", fontsize=13)
    out = outdir / "Figure10_python.png"
    save_figure(fig, out)
    return out


def figure_11(paths: Paths, outdir: Path) -> Path:
    _, _, valid_loc, conflict_loc = _build_conflict_data(paths)
    lon_lim = [-130.031, -129.97]
    lat_lim = [45.92, 45.972]

    fig, axes = plt.subplots(3, 3, figsize=(11, 9), constrained_layout=True)
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
    fig.suptitle("Figure 11: Spatial agreement (gray) vs conflict (blue)", fontsize=13)
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
    cc_path = paths.fm3_root / "02-data" / "G_FM" / "G_2015Erp_polished.mat"
    ml_path = paths.fm3_root / "02-data" / "G_FM" / "G_HASH_All_ML_sameClusterasbeforev_confidence.mat"
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

    fig, axes = plt.subplots(2, 3, figsize=(12, 9), constrained_layout=True)
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

    fig.suptitle("Figure 12: Focal-mechanism comparison (CC vs AxialPolCap catalogs)", fontsize=13)
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
    ml_path = paths.fm3_root / "02-data" / "G_FM" / "G_HASH_All_ML_sameClusterasbeforev_confidence.mat"
    cc_path = paths.fm3_root / "02-data" / "G_FM" / "G_2015Erp_polished.mat"
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].hist(kg, bins=np.linspace(0, 120, 25), color=(0.2, 0.4, 0.9), edgecolor="k")
    axes[0].set_xlim([0, 120])
    axes[0].grid(alpha=0.2)
    axes[0].set_xlabel("Kagan angle (deg)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Figure 13a: Histogram (mean={np.mean(kg):.2f}, median={np.median(kg):.2f})")

    sc = axes[1].scatter(lon, lat, c=kg, s=7, cmap="jet")
    axes[1].plot(CALDERA_RIM[:, 0], CALDERA_RIM[:, 1], "k-", linewidth=1.5)
    axes[1].set_xlim([-130.03, -129.97])
    axes[1].set_ylim([45.92, 45.97])
    _set_geo_aspect(axes[1], [-130.03, -129.97], [45.92, 45.97])
    axes[1].grid(alpha=0.2)
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].set_title("Figure 13b: Spatial Kagan distribution")
    cb = fig.colorbar(sc, ax=axes[1], shrink=0.75)
    cb.set_label("Kagan (deg)")

    out = outdir / "Figure13_python.png"
    save_figure(fig, out)
    return out


def figure_14(paths: Paths, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    def rect(x, y, w, h, txt, fc, bold=False):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor=fc, edgecolor="black", linewidth=1.2)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=10, fontweight="bold" if bold else "normal")

    def ellipse(x, y, w, h, txt, fc):
        e = Ellipse((x + w / 2, y + h / 2), w, h, facecolor=fc, edgecolor="black", linewidth=1.2)
        ax.add_patch(e)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=10)

    def hexagon(cx, cy, r, txt, fc):
        th = np.linspace(0, 2 * np.pi, 7)
        pts = np.c_[cx + r * np.cos(th), cy + r * np.sin(th)]
        ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor="black", linewidth=1.3))
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=10)

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.4))

    rect(0.05, 0.70, 0.22, 0.07, "Input", (0.83, 0.93, 0.83), True)
    rect(0.05, 0.64, 0.22, 0.05, "3-channel waveform", (1, 1, 1))
    rect(0.33, 0.48, 0.18, 0.07, "S/P amplitude ratios", (1, 1, 1))
    ellipse(0.58, 0.57, 0.18, 0.10, "Trained + Fine-tuned", (1, 1, 0.85))
    rect(0.82, 0.58, 0.15, 0.08, "First-motion polarity", (1, 1, 1))
    rect(0.05, 0.27, 0.22, 0.05, "Earthquake location", (1, 1, 1))
    rect(0.05, 0.22, 0.22, 0.05, "Station location", (1, 1, 1))
    rect(0.05, 0.17, 0.22, 0.05, "Velocity model", (1, 1, 1))
    hexagon(0.62, 0.32, 0.07, "SKHASH", (0.80, 0.87, 1.0))
    rect(0.82, 0.40, 0.15, 0.07, "Focal mechanism", (0.83, 0.93, 0.83), True)
    rect(0.82, 0.35, 0.15, 0.05, "Strike", (1, 1, 1))
    rect(0.82, 0.30, 0.15, 0.05, "Dip", (1, 1, 1))
    rect(0.82, 0.25, 0.15, 0.05, "Rake", (1, 1, 1))
    rect(0.82, 0.15, 0.15, 0.07, "Stress inversion", (0.83, 0.93, 0.83), True)
    rect(0.82, 0.10, 0.15, 0.05, "P axis", (1, 1, 1))
    rect(0.82, 0.05, 0.15, 0.05, "T axis", (1, 1, 1))
    rect(0.82, 0.00, 0.15, 0.05, "Shape ratio", (1, 1, 1))

    arr(0.27, 0.615, 0.33, 0.615)
    arr(0.51, 0.615, 0.58, 0.615)
    arr(0.76, 0.615, 0.82, 0.615)
    arr(0.42, 0.48, 0.58, 0.36)
    arr(0.27, 0.215, 0.58, 0.31)
    arr(0.88, 0.57, 0.66, 0.35)
    arr(0.69, 0.32, 0.82, 0.32)

    ax.text(0.02, 0.98, "Figure 14: Real-time focal-mechanism pipeline", fontsize=14, fontweight="bold", va="top")

    out = outdir / "Figure14_python.png"
    save_figure(fig, out)
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
        if v < 1 or v > 14:
            raise ValueError("Figure ids must be in [1,14]")
        out.append(v)
    return sorted(set(out))


def main() -> None:
    default_repo = Path(__file__).resolve().parents[2]
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


if __name__ == "__main__":
    main()
