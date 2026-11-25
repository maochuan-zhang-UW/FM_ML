"""Fig. 1: (a) bathymetry, eruptive features, caldera rim and OBS stations;
(b) catalogue earthquakes in local x/y (km) with the template events;
(c) magnitude and (d) P-wave SNR distributions, templates vs the rest.

Python port of Figure01_background.m and Figure01_panelCD_prep.m.
SNR in (d) is 20 log10 of the peak absolute amplitude after / before sample 20
of each 64-sample catalogue window (P pick at sample 32).

Inputs (defaults are the local data layout; see --help)
  07-files/  bathymetry grid, fissures, lava flows, caldera rim, coastlines, evMw.txt
  02-data/raw/Felix_kmean_morethan5.mat  catalogue earthquakes with >= 5 P and S picks
  02-data/raw/A_wave_dB20_cleaned.mat    templates
  02-data/raw/ColormapZMC.mat            bathymetry colour map
  catalogue windows (02-data/A_wave_2015_2021_AxialPolCap_final_conf80_predictions.mat)

Usage (from the repository root): python 01-scripts/figures/fig01_background.py
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.io import loadmat, netcdf_file

ROOT = Path(__file__).resolve().parents[2]
FILES = ROOT / "07-files"
STATIONS = {"AS1": (-129.9992, 45.9336), "AS2": (-130.0141, 45.9338), "CC1": (-130.0089, 45.9547),
            "EC1": (-129.9797, 45.9496), "EC2": (-129.9738, 45.9397), "EC3": (-129.9785, 45.9361),
            "ID1": (-129.9780, 45.9257)}
TEMPLATE_COL, NORMAL_COL = (0.00, 0.20, 0.90), (0.55, 0.55, 0.55)
NOISE_END = 20


def matlab_blocks(path):
    """Numeric [lon, lat] arrays from the '= [ ... ]' blocks of a MATLAB data file."""
    out = []
    for blk in re.findall(r"=\s*\[(.*?)\]", Path(path).read_text(errors="ignore"), re.S):
        rows = []
        for ln in blk.splitlines():
            ln = ln.split("%")[0].replace("...", "").strip().rstrip(";")
            vals = [v for v in re.split(r"[,\s]+", ln) if v]
            if len(vals) >= 2:
                rows.append([float(vals[0]), float(vals[1])])
        if rows:
            out.append(np.array(rows))
    return out


def grouped_lines(path):
    """Polylines from a tab-separated ORIG_FID/LONGITUDE/LATITUDE file (any line endings)."""
    txt = Path(path).read_text(errors="ignore").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    a = np.array([[float(v) for v in ln.split("\t")[:3]] for ln in txt[1:] if ln.strip()])
    return [a[a[:, 0] == i, 1:] for i in np.unique(a[:, 0])]


def fill_gaps(Z):
    """Replace NaNs by the nearest valid value along columns, then rows
    (as MATLAB fillmissing(..., 'nearest') in Figure01_background.m)."""
    from scipy.ndimage import distance_transform_edt
    bad = ~np.isfinite(Z)
    if not bad.any():
        return Z
    idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
    return Z[tuple(idx)]


def latlon2xy(lat, lon):
    """Local x/y (km), origin at AXCC1, no rotation."""
    lat0, lon0, km = 45.9547, -130.0089, 111.19
    return (np.asarray(lon) - lon0) * km * np.cos(np.radians(lat0)), (np.asarray(lat) - lat0) * km


def ortho(lat, lon, lat0=35.0, lon0=-145.0):
    lat, lon, lat0, lon0 = map(np.radians, (np.asarray(lat, float), np.asarray(lon, float), lat0, lon0))
    cosc = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0)
    x = np.cos(lat) * np.sin(lon - lon0)
    y = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0)
    x[cosc < 0] = np.nan
    y[cosc < 0] = np.nan
    return x, y


def inset_globe(fig):
    ax = fig.add_axes([0.250, 0.800, 0.145, 0.145])
    th = np.linspace(0, 2 * np.pi, 400)
    ax.fill(np.cos(th), np.sin(th), color="w", ec=(0.25, 0.25, 0.25), lw=1.0)
    for lon in range(-180, 181, 30):
        ax.plot(*ortho(np.arange(-90, 91, 2), np.full(91, lon)), color=(0.75,) * 3, lw=0.4)
    for lat in range(-60, 61, 30):
        ax.plot(*ortho(np.full(181, lat), np.arange(-180, 181, 2)), color=(0.75,) * 3, lw=0.4)
    c = np.loadtxt(FILES / "coastlines_ne110m.txt")
    ax.plot(*ortho(c[:, 0], c[:, 1]), color="k", lw=0.8)
    ax.plot(*ortho([46.0], [-130.0]), "r*", ms=11, mec="k", mew=0.5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect(1)
    ax.axis("off")


def label(ax, text, colour="k"):
    ax.text(0.02, 0.96, text, transform=ax.transAxes, fontsize=10, fontweight="bold", color=colour, va="top")


def panel_a(fig, rim, colormap):
    ax = fig.add_axes([0.085, 0.40, 0.32, 0.55])
    lon_lim, lat_lim = (-130.10, -129.90), (45.85, 46.10)
    with netcdf_file(FILES / "Axial-em300-gmt-25m.grd", "r", mmap=False) as nc:
        x, y, z = nc.variables["x"][:].copy(), nc.variables["y"][:].copy(), nc.variables["z"][:].copy()
    xm, ym = (x >= lon_lim[0]) & (x <= lon_lim[1]), (y >= lat_lim[0]) & (y <= lat_lim[1])
    Z = z[np.ix_(ym, xm)].astype(float) / 1000.0
    Z = fill_gaps(Z)
    cm = ListedColormap(loadmat(colormap)["ColormapZMC"] * 0.7 + 0.3) if Path(colormap).exists() else "viridis"
    cf = ax.contourf(x[xm], y[ym], Z, 20, cmap=cm)
    ax.plot([-130.0438, -130.0438, -129.9649, -129.9649, -130.0438],
            [45.9142, 45.9952, 45.9952, 45.9142, 45.9142], "b-", lw=2)
    fiss = []
    for f in ("Fissures2015/JdF:Axial_Clague/Axial-2015-fissures-points-geo-v2.txt",
              "Fissures2011/JdF:Axial_Clague/Axial-2011-fissures-points-geo-v2.txt"):
        fiss += grouped_lines(FILES / f)
    a98 = np.array([[float(v) for v in re.split(r"[,\s]+", ln.strip())[:3]]
                    for ln in open(FILES / "Axial-1998-Fissures.txt") if ln.strip()])
    fiss += [a98[a98[:, 0] == i, 1:] for i in np.unique(a98[:, 0])]
    h_f = [ax.plot(f[:, 0], f[:, 1], "k-", lw=1)[0] for f in fiss][0]
    h_f.set_label("Fissures")
    ax.plot(rim[:, 0], rim[:, 1], "k-", lw=2)
    handles = []
    for name, fname, col in (("Lava Flows 1998", "axial_lava1998.m", (0.5, 0, 0)),
                             ("Lava Flows 2011", "axial_lava2011.m", (0, 0, 0.9)),
                             ("Lava Flows 2015", "axial_lava2015.m", (0, 0.5, 0))):
        polys = [ax.fill(p[:, 0], p[:, 1], color=col, lw=0, alpha=0.5)[0] for p in matlab_blocks(FILES / fname)]
        polys[0].set_label(name)
        handles.append(polys[0])
    h_s, = ax.plot(*np.array(list(STATIONS.values())).T, "ks", mfc="k", ms=8, ls="", label="OBS")
    ax.legend(handles=[h_s, h_f] + handles, loc="lower left", fontsize=6, handlelength=1.2, framealpha=1)
    for t, xy, c, fs, rot in (("AXIAL CALDERA", (-130.037, 45.965), "w", 8, 0), ("NORTH RIFT", (-130.038, 46.012), "k", 8, 75),
                              ("SOUTH RIFT", (-129.985, 45.858), "k", 8, 75), ("ASHES", (-130.035, 45.932), "w", 7, 0),
                              ("CASM", (-130.033, 45.990), "w", 7, 0), ("INTERNATIONAL", (-130.012, 45.922), "w", 7, 0),
                              ("DISTRICT", (-130.006, 45.917), "w", 7, 0)):
        ax.text(*xy, t, fontsize=fs, color=c, fontweight="bold", rotation=rot)
    ax.text(-130.09, 46.09, "(a)", fontsize=10, color="w", fontweight="bold", va="top")
    ax.set_xlim(lon_lim)
    ax.set_ylim(lat_lim)
    ax.set_aspect(1 / np.cos(np.radians(np.mean(lat_lim))))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.3)
    cax = fig.add_axes([0.300, 0.435, 0.013, 0.085])
    fig.colorbar(cf, cax=cax, ticks=[-2.4, -2.2, -2.0, -1.8, -1.6]).set_label("Depth (km)")


def panel_b(fig, rim, cat_lat, cat_lon, cat_dep, is_t):
    ax = fig.add_axes([0.50, 0.40, 0.35, 0.55])
    bx, by = latlon2xy(cat_lat, cat_lon)
    sc = ax.scatter(bx, by, s=1, c=cat_dep, cmap=plt.get_cmap("summer").reversed(), lw=0)
    ax.scatter(bx[is_t], by[is_t], s=2, color=TEMPLATE_COL, lw=0)
    ax.plot(*latlon2xy(rim[:, 1], rim[:, 0]), "k-", lw=3)
    for name, (lo, la) in STATIONS.items():
        sx, sy = latlon2xy(la, lo)
        ax.plot(sx, sy, "ks", mfc="k", ms=9)
        ax.text(sx + 0.15, sy, "AX" + name, fontsize=7, va="center")
    ax.set_xlim(-2.7, 3.4)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect(1)
    ax.text(-2.5, 4.1, "(b)", fontsize=10, fontweight="bold")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.grid(alpha=0.3)
    cax = fig.add_axes([0.762, 0.80, 0.018, 0.13])
    fig.colorbar(sc, cax=cax, ticks=[0.5, 1, 1.5, 2]).set_label("Depth (km)")


def panel_cd(fig, mw_t, mw_n, snr_t, snr_n):
    ax = fig.add_axes([0.08, 0.07, 0.33, 0.24])
    allm = np.r_[mw_n, mw_t]
    edges = np.arange(np.floor(allm.min() * 10) / 10, np.ceil(allm.max() * 10) / 10 + 0.05, 0.1)
    ax.hist(mw_n, edges, color=NORMAL_COL, alpha=0.75, label=f"Non-template (n = {len(mw_n)})")
    ax.hist(mw_t, edges, color=TEMPLATE_COL, alpha=0.75, label=f"Templates (n = {len(mw_t)})")
    ax.set_xlim(np.percentile(allm, 0.2), np.percentile(allm, 99.9))
    ax.set_xlabel("Magnitude ($M_w$)")
    ax.set_ylabel("Earthquake count")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=7.5)
    label(ax, "(c)")
    ax = fig.add_axes([0.54, 0.07, 0.33, 0.24])
    edges = np.arange(-10, 81, 2)
    n1, *_ = ax.hist(snr_n, edges, color=NORMAL_COL, alpha=0.75, label=f"Non-template (n = {len(snr_n)})")
    n2, *_ = ax.hist(snr_t, edges, color=TEMPLATE_COL, alpha=0.75, label=f"Templates (n = {len(snr_t)})")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, max(n1.max(), n2.max()) * 1.42)
    ax.axvline(10, color=(0.15,) * 3, ls="--", lw=1.1)
    ax.text(9, ax.get_ylim()[1] * 0.97, "SNR = 10 dB", rotation=90, fontsize=7.5, ha="right", va="top")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Waveform count")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(4, 4))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=7.5)
    label(ax, "(d)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=str(ROOT / "02-data" / "raw" / "Felix_kmean_morethan5.mat"))
    ap.add_argument("--templates", default=str(ROOT / "02-data" / "raw" / "A_wave_dB20_cleaned.mat"))
    ap.add_argument("--windows", default=str(ROOT / "02-data" / "A_wave_2015_2021_AxialPolCap_final_conf80_predictions.mat"))
    ap.add_argument("--colormap", default=str(ROOT / "02-data" / "raw" / "ColormapZMC.mat"))
    ap.add_argument("--out", default=str(ROOT / "03-figs" / "GJI_Figure01.png"))
    a = ap.parse_args()
    rim = matlab_blocks(FILES / "axial_calderaRim.m")[0]

    cat = np.atleast_1d(loadmat(a.catalog, squeeze_me=True, struct_as_record=False)["Felix"])
    cat = [e for e in cat if float(e.depth) <= 2]
    cid = np.array([int(e.ID) for e in cat])
    tid = {int(e.ID) for e in np.atleast_1d(loadmat(a.templates, squeeze_me=True, struct_as_record=False)["Felix"])}
    is_t = np.isin(cid, list(tid))
    mw = dict(np.loadtxt(FILES / "evMw.txt"))
    mags = np.array([mw.get(i, np.nan) for i in cid])

    win = np.atleast_1d(loadmat(a.windows, squeeze_me=True, struct_as_record=False)["Felix"])
    keep = set(cid.tolist())
    snr, snr_t = [], []
    for e in win:
        i = int(e.ID)
        if i not in keep:
            continue
        for s in STATIONS:
            w = np.abs(np.asarray(getattr(e, f"W_{s}", []), float).ravel())
            if w.size <= NOISE_END:
                continue
            nz, sg = w[:NOISE_END].max(), w[NOISE_END:].max()
            if nz > 0 and np.isfinite(sg):
                snr.append(20 * np.log10(sg / nz))
                snr_t.append(i in tid)
    snr, snr_t = np.array(snr), np.array(snr_t)

    plt.rcParams.update({"font.family": "Arial", "font.size": 9})
    fig = plt.figure(figsize=(7.2, 7.2))
    panel_a(fig, rim, a.colormap)
    panel_b(fig, rim, np.array([float(e.lat) for e in cat]), np.array([float(e.lon) for e in cat]),
            np.array([float(e.depth) for e in cat]), is_t)
    panel_cd(fig, mags[is_t & np.isfinite(mags)], mags[~is_t & np.isfinite(mags)], snr[snr_t], snr[~snr_t])
    inset_globe(fig)
    fig.savefig(a.out, dpi=300)
    print(f"(b) {len(cid)} earthquakes, {is_t.sum()} templates; (d) {len(snr)} waveforms "
          f"(template median {np.median(snr[snr_t]):.1f} dB, rest {np.median(snr[~snr_t]):.1f} dB)")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
