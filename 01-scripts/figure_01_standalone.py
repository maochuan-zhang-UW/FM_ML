#!/usr/bin/env python3
"""
Standalone script to regenerate Figure 1 of the AxialPolCap manuscript.

Figure 1: Two-panel map figure
  (a) Bathymetry + lava flows + fissures + caldera rim + globe inset
  (b) Earthquake depth scatter in rotated x/y (km) + k-means boundaries

Usage:
    conda run -n tf_macos python 01-scripts/figure_01_standalone.py
    conda run -n tf_macos python 01-scripts/figure_01_standalone.py --outdir 03-figs
"""
from __future__ import annotations

import argparse
import gzip
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig_fm5_ml"
if "CARTOPY_DATA_DIR" not in os.environ:
    os.environ["CARTOPY_DATA_DIR"] = "/tmp/cartopy_fm5_ml"

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy.io import loadmat, netcdf_file
from scipy.stats import lognorm  # noqa: F401  (imported for parity with main script)


# ---------------------------------------------------------------------------
# Station coordinates and caldera rim (from FM4 MATLAB scripts)
# ---------------------------------------------------------------------------

STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]

STATION_COORDS = {
    "AS1": (-129.9992, 45.9336),
    "AS2": (-130.0141, 45.9338),
    "CC1": (-130.0089, 45.9547),
    "EC1": (-129.9797, 45.9496),
    "EC2": (-129.9738, 45.9397),
    "EC3": (-129.9785, 45.9361),
    "ID1": (-129.9780, 45.9257),
}

CALDERA_RIM = np.array(
    [
        [-130.004785563058, 45.9207755734405],
        [-130.010476202888, 45.9238241104543],
        [-130.018881564079, 45.9351908809594],
        [-130.023946125193, 45.9412238501725],
        [-130.028718653506, 45.949881200114],
        [-130.03045121938,  45.9511765797916],
        [-130.03067948565,  45.9542732243167],
        [-130.031733279709, 45.9558130656063],
        [-130.0314446535,   45.9586760104296],
        [-130.036188782208, 45.9656647517656],
        [-130.036950110789, 45.9698291665232],
        [-130.039953347,    45.9750458167927],
        [-130.038595675479, 45.9847117727418],
        [-130.035927416999, 45.9883113986506],
        [-130.018067675296, 45.993358288674],
        [-130.013629193751, 45.993755284135],
        [-130.010365710979, 45.9929499241491],
        [-130.008647442296, 45.9924883829037],
        [-130.007262470669, 45.9915471582374],
        [-130.006042022411, 45.9902469280907],
        [-130.00517862949,  45.989777805361],
        [-130.001868199523, 45.9863506519894],
        [-130.001154359192, 45.9846883853932],
        [-130.000949059432, 45.9827833001814],
        [-129.99939353433,  45.9818434725493],
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
        [-129.98548409867,  45.9494279735802],
        [-129.98478812249,  45.9487188881587],
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

@dataclass
class Paths:
    repo_root: Path
    fm_root: Path
    fm3_root: Path
    fm4_root: Path


def choose_existing(*paths: Optional[Path]) -> Optional[Path]:
    for p in paths:
        if p is not None and p.exists():
            return p
    return None


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


def load_tabular_points(path: Path, gzipped: bool = False) -> np.ndarray:
    open_fn = gzip.open if gzipped else open
    with open_fn(path, "rt", encoding="utf-8", errors="ignore") as f:
        data = np.genfromtxt(f, delimiter="\t", names=True, dtype=None, encoding="utf-8")
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


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


def _set_geo_aspect(ax: plt.Axes, lon_lim, lat_lim) -> None:
    span_x = (lon_lim[1] - lon_lim[0]) * math.cos(math.radians(np.mean(lat_lim)))
    span_y = lat_lim[1] - lat_lim[0]
    if span_y > 0:
        ax.set_aspect(span_x / span_y)


def latlon2xy(dlat: np.ndarray, dlon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert lat/lon to rotated x/y (km). Origin = AXCC1, rotation = -20 deg."""
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


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------

def figure_01(paths: Paths, outdir: Path) -> Path:
    style = {
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }

    with plt.rc_context(style):
        fig = plt.figure(figsize=(10.0, 6.48))
        fig.patch.set_facecolor("white")

        ax1 = fig.add_axes([0.05, 0.11, 0.35, 0.8])
        ax2 = fig.add_axes([0.5, 0.11, 0.35, 0.8])

        # --- Panel (a): bathymetry + lava/fissures + caldera rim + globe inset ---
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

        # Rotated box corners matching the MATLAB figure
        box_lat = np.array([45.9074, 45.9835, 46.0020, 45.9259, 45.9074])
        box_lon = np.array([-130.0255, -130.0653, -129.9923, -129.9525, -130.0255])
        ax1.plot(box_lon, box_lat, "b-", linewidth=2.0)

        alpha = 0.5
        lava_2015 = choose_existing(
            paths.fm_root / "02-data/Alldata/Fissures2015/JdF:Axial_Clague/Axial-2015-lava-points-geo-v2.txt"
        )
        lava_2011 = choose_existing(
            paths.fm_root / "02-data/Alldata/Fissures2011/JdF:Axial_Clague/Axial-2011-lava-points-geo-v2.txt.gz"
        )
        fiss_2015 = choose_existing(
            paths.fm_root / "02-data/Alldata/Fissures2015/JdF:Axial_Clague/Axial-2015-fissures-points-geo-v2.txt"
        )
        fiss_2011 = choose_existing(
            paths.fm_root / "02-data/Alldata/Fissures2011/JdF:Axial_Clague/Axial-2011-fissures-points-geo-v2.txt"
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
                        rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
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
                    g["LONGITUDE"], g["LATITUDE"],
                    facecolor=(0.0, 0.0, 0.9), edgecolor="none", alpha=alpha, linewidth=0,
                )
                if lava_flow_2011_handle is None and patches:
                    lava_flow_2011_handle = patches[0]

        lava_flow_2015_handle = None
        if lava_2015 is not None:
            d = load_tabular_points(lava_2015, gzipped=False)
            for fid in np.unique(d["ORIG_FID"]):
                g = d[d["ORIG_FID"] == fid]
                patches = ax1.fill(
                    g["LONGITUDE"], g["LATITUDE"],
                    facecolor=(0.0, 0.5, 0.0), edgecolor="none", alpha=alpha, linewidth=0,
                )
                if lava_flow_2015_handle is None and patches:
                    lava_flow_2015_handle = patches[0]

        lava_flow_1998_handle = None
        lava_1998_mat = choose_existing(paths.fm_root / "04-final-paper" / "axial_lava1998.m")
        lava_1998_polys = parse_matlab_lava1998(lava_1998_mat) if lava_1998_mat else None
        if lava_1998_polys is not None:
            for poly in lava_1998_polys:
                patches = ax1.fill(
                    poly[:, 0], poly[:, 1],
                    facecolor=(0.5, 0.0, 0.0), edgecolor="none", alpha=alpha, linewidth=0,
                )
                if lava_flow_1998_handle is None and patches:
                    lava_flow_1998_handle = patches[0]

        lons = [v[0] for v in STATION_COORDS.values()]
        lats = [v[1] for v in STATION_COORDS.values()]
        station_handle = ax1.plot(lons, lats, "sk", markerfacecolor="k", markersize=8)[0]

        handles, labels = [], []
        if station_handle is not None:
            handles.append(station_handle); labels.append("OOI OBS Stations")
        if fissure_handle is not None:
            handles.append(fissure_handle); labels.append("Fissures")
        if lava_flow_1998_handle is not None:
            handles.append(lava_flow_1998_handle); labels.append("Lava Flows 1998")
        if lava_flow_2011_handle is not None:
            handles.append(lava_flow_2011_handle); labels.append("Lava Flows 2011")
        if lava_flow_2015_handle is not None:
            handles.append(lava_flow_2015_handle); labels.append("Lava Flows 2015")
        if handles:
            ax1.legend(handles, labels, loc="lower left", fontsize=8, frameon=True, framealpha=1.0)

        ax1.text(-130.04, 45.96,  "AXIAL CALDERA",  fontsize=10, color="w", fontweight="bold")
        ax1.text(-130.03, 46.01,  "NORTH RIFT ZONE", fontsize=10, color="k", rotation=75)
        ax1.text(-130.00, 45.852, "SOUTH RIFT ZONE", fontsize=10, color="k", rotation=75)
        ax1.text(-130.035, 45.932, "ASHES",           fontsize=8,  color="w", fontweight="bold")
        ax1.text(-129.985, 45.917, "INTERNATIONAL",   fontsize=8,  color="w", fontweight="bold")
        ax1.text(-130.09,  46.09,  "(a)",              fontsize=12, color="w", fontweight="bold")

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

        # Globe inset: Cartopy if available, otherwise 3-D fallback
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

        # --- Panel (b): depth scatter in rotated x/y km + k-means boundaries ---
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
                        C[i, 0] - 0.1, C[i, 1] + 0.1,
                        f"{int(num_points[i])}: {names_cluster[i]}",
                        va="bottom", ha="right",
                    )
                ax2.plot(C[:, 0], C[:, 1], linestyle="None", marker="x",
                         color="c", markersize=15, markeredgewidth=3)
            if vx.ndim == 2 and vy.ndim == 2 and vx.shape == vy.shape:
                for j in range(vx.shape[1]):
                    ax2.plot(vx[:, j], vy[:, j], "k-.", linewidth=1.0)

        xr, yr = latlon2xy(CALDERA_RIM[:, 1], CALDERA_RIM[:, 0])
        ax2.plot(xr, yr, "k-", linewidth=3.0)

        for code, (slon, slat) in STATION_COORDS.items():
            xs, ys = latlon2xy(np.array([slat]), np.array([slon]))
            ax2.plot(xs[0], ys[0], "s", markeredgecolor="k", markerfacecolor="k", markersize=10)
            ax2.text(xs[0] + 0.1, ys[0], code)

        if felix is not None:
            ids = np.array([get_value(x, "ID", np.nan) for x in felix], dtype=float)
            highlight_ids = np.array(
                [1225535, 1341701, 1334022, 1225960, 1316811, 1321478, 1315292,
                 1315619, 1346145, 1340523, 1335762, 1346359, 1501394, 1336032,
                 1225536, 1343586, 1347474, 1325518, 1340556, 1315224, 1330369,
                 1225831, 1327219, 1328527, 1298463, 1312638],
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

        ax2.text(-2.8,  1,    "North",      color="blue", fontsize=12, rotation=90, va="center")
        ax2.text(-2.8, -2,    "South",      color="blue", fontsize=12, rotation=90, va="center")
        ax2.text(-1.5, -4.3,  "West Wall",  color="blue", fontsize=12, ha="center")
        ax2.text( 1.5, -4.3,  "East Wall",  color="blue", fontsize=12, ha="center")
        ax2.set_xlabel("x-Distance (km)")
        ax2.set_ylabel("y-Distance (km)")
        ax2.grid(True)
        ax2.text(-2.8, 4.1, "(b)", fontsize=12, color="k", fontweight="bold")

        if sc is not None:
            cax2 = fig.add_axes([0.80, 0.65, 0.02, 0.2])
            cb2 = fig.colorbar(sc, cax=cax2)
            cb2.set_label("Depth (km)")

        # Connector dashed lines between the two panels
        xlim1 = ax1.get_xlim()
        ylim1 = ax1.get_ylim()
        pos1  = ax1.get_position()
        pos2  = ax2.get_position()
        xlim2 = ax2.get_xlim()
        ylim2 = ax2.get_ylim()

        top_right_121    = (-129.9923, 46.0020)
        bottom_right_121 = (-129.9525, 45.9259)
        top_right_norm_x1    = pos1.x0 + pos1.width  * (top_right_121[0]    - xlim1[0]) / (xlim1[1] - xlim1[0])
        top_right_norm_y1    = pos1.y0 + pos1.height * (top_right_121[1]    - ylim1[0]) / (ylim1[1] - ylim1[0])
        bottom_right_norm_y1 = pos1.y0 + pos1.height * (bottom_right_121[1] - ylim1[0]) / (ylim1[1] - ylim1[0])

        top_left_122    = (-3.0,  4.5)
        bottom_left_122 = (-3.0, -4.5)
        top_left_norm_x2    = pos2.x0 + pos2.width  * (top_left_122[0]    - xlim2[0]) / (xlim2[1] - xlim2[0])
        top_left_norm_y2    = pos2.y0 + pos2.height * (top_left_122[1]    - ylim2[0]) / (ylim2[1] - ylim2[0])
        bottom_left_norm_x2 = pos2.x0 + pos2.width  * (bottom_left_122[0] - xlim2[0]) / (xlim2[1] - xlim2[0])
        bottom_left_norm_y2 = pos2.y0 + pos2.height * (bottom_left_122[1] - ylim2[0]) / (ylim2[1] - ylim2[0])

        fig.add_artist(Line2D(
            [top_right_norm_x1, top_left_norm_x2],
            [top_right_norm_y1, top_left_norm_y2],
            transform=fig.transFigure, color="blue", linewidth=1, linestyle="--",
        ))
        fig.add_artist(Line2D(
            [0.295, bottom_left_norm_x2],
            [bottom_right_norm_y1, bottom_left_norm_y2],
            transform=fig.transFigure, color="blue", linewidth=1, linestyle="--",
        ))

        out = outdir / "Figure01_python.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Saved: {out}")
        return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    default_repo = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Generate Figure 1 of the AxialPolCap manuscript")
    parser.add_argument(
        "--outdir",
        default=str(default_repo / "03-figs"),
        help="Output directory (default: 03-figs/)",
    )
    parser.add_argument("--fm-root",  default="/Users/mczhang/Documents/GitHub/FM")
    parser.add_argument("--fm3-root", default="/Users/mczhang/Documents/GitHub/FM3")
    parser.add_argument("--fm4-root", default="/Users/mczhang/Documents/GitHub/FM4")
    args = parser.parse_args()

    paths = Paths(
        repo_root=default_repo,
        fm_root=Path(args.fm_root),
        fm3_root=Path(args.fm3_root),
        fm4_root=Path(args.fm4_root),
    )

    figure_01(paths, Path(args.outdir))


if __name__ == "__main__":
    main()
