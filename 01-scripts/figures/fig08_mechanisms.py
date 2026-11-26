"""Fig. 8: composite focal mechanisms from CC (a-c) and AxialPolCap (d-f)
polarities before, during and after the 2015 eruption, and the Kagan angle
between the two mechanisms of each cluster (g-i).

Only clusters whose CC and AxialPolCap mechanisms are both well constrained
(quality A or B, Table S3; --quality) and classified as normal, reverse or
strike-slip by both (Table S4) are drawn, so all panels of a column show the
same clusters.

Input: 02-data/focal_mechanisms_2015_2021/mechanisms.csv (07_focal_mechanisms.py)
Usage (from the repository root): python 01-scripts/figures/fig08_mechanisms.py
"""
import argparse
import re
import sys
from pathlib import Path

import contourpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01-scripts"))
from mechanism_tools import sdr_to_frame  # noqa: E402

ERUPT, ENDCO = pd.Timestamp("2015-04-24"), pd.Timestamp("2015-05-25")
LON_LIM, LAT_LIM = (-130.030, -129.970), (45.920, 45.972)
TYPE_COL = {"N": (0.00, 0.20, 0.90), "R": (0.85, 0.10, 0.10), "S": (0.00, 0.60, 0.20)}
TYPE_NAME = {"N": "Normal", "R": "Reverse", "S": "Strike-slip"}
RADIUS = 0.00075            # beachball radius, degrees of latitude
KMAX = 120

# Lower-hemisphere equal-area grid on the unit disc, shared by every beachball
_G = np.linspace(-1, 1, 61)
_X, _Y = np.meshgrid(_G, _G)
_R = np.hypot(_X, _Y)
_INC = 2 * np.arcsin(np.clip(_R / np.sqrt(2), 0, 1))         # angle from vertical-down
_AZ = np.arctan2(_X, _Y)                                       # x = east, y = north
_V = np.stack([np.sin(_INC) * np.cos(_AZ), np.sin(_INC) * np.sin(_AZ), np.cos(_INC)], -1)


def caldera_rim():
    txt = (ROOT / "07-files" / "axial_calderaRim.m").read_text()
    body = txt[txt.index("[") + 1:txt.index("]")]
    return np.array([[float(v) for v in re.split(r"[,\s]+", ln.strip()) if v]
                     for ln in body.strip().splitlines() if ln.strip()])


def compressional_paths(strike, dip, rake):
    """Unit-disc polygons of the compressional quadrants (P first motion up)."""
    U = sdr_to_frame(strike, dip, rake)
    t, p = U[:, 0], U[:, 2]
    amp = (_V @ t) ** 2 - (_V @ p) ** 2                 # sign of P radiation
    amp[_R > 1] = -1
    filled = contourpy.contour_generator(_X, _Y, amp, fill_type="OuterCode").filled(0, np.inf)
    return [MPath(v, c) for v, c in zip(*filled)]


def beachball(ax, lon, lat, sdr, colour, aspect):
    tf = lambda verts: np.column_stack([lon + verts[:, 0] * RADIUS * aspect, lat + verts[:, 1] * RADIUS])
    th = np.linspace(0, 2 * np.pi, 60)
    ring = np.column_stack([np.cos(th), np.sin(th)])
    ax.fill(*tf(ring).T, color="w", lw=0, zorder=2)
    for pth in compressional_paths(*sdr):
        ax.add_patch(PathPatch(MPath(tf(pth.vertices), pth.codes), fc=colour, ec="none", zorder=3))
    ax.plot(*tf(ring).T, color="k", lw=0.25, zorder=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quality", default="AB", help="mechanism quality grades kept for both methods")
    ap.add_argument("--out", default=str(ROOT / "03-figs" / "GJI_Figure08.png"))
    a = ap.parse_args()
    df = pd.read_csv(ROOT / "02-data" / "focal_mechanisms_2015_2021" / "mechanisms.csv", parse_dates=["time"])
    df = df[df.kagan.notna() & df.quality_cc.isin(list(a.quality)) & df.quality_ml.isin(list(a.quality))
            & df.fault_type_cc.isin(list("NRS")) & df.fault_type_ml.isin(list("NRS"))]
    periods = {"Before": df.time < ERUPT, "During": (df.time >= ERUPT) & (df.time < ENDCO), "After": df.time >= ENDCO}
    rim = caldera_rim()
    aspect = 1 / np.cos(np.radians(np.mean(LAT_LIM)))
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 7.5})
    fig, axs = plt.subplots(3, 3, figsize=(7.0, 8.8), sharex=True, sharey=True,
                            gridspec_kw=dict(left=0.10, right=0.98, top=0.96, bottom=0.12, hspace=0.20, wspace=0.06))
    cmap = plt.get_cmap("viridis", 24)
    rows = ("CC polarities", "AxialPolCap polarities", "Kagan angle")
    for c, (per, sel) in enumerate(periods.items()):
        d = df[sel]
        for r in range(3):
            ax = axs[r, c]
            if r < 2:
                s = ("cc", "ml")[r]
                for _, m in d.iterrows():
                    beachball(ax, m.lon, m.lat, (m[f"strike_{s}"], m[f"dip_{s}"], m[f"rake_{s}"]),
                              TYPE_COL[m[f"fault_type_{s}"]], aspect)
            else:
                o = d.sort_values("kagan")                      # largest disagreements on top
                sc = ax.scatter(o.lon, o.lat, s=5, c=o.kagan, cmap=cmap, vmin=0, vmax=KMAX, lw=0, zorder=3)
            ax.plot(rim[:, 0], rim[:, 1], "k-", lw=1.0, zorder=5)
            ax.set_xlim(LON_LIM)
            ax.set_ylim(LAT_LIM)
            ax.set_aspect(aspect)
            ax.set_xticks([-130.02, -130.00, -129.98])
            ax.set_yticks([45.92, 45.94, 45.96])
            ax.xaxis.set_major_formatter("{x:.2f}")
            ax.yaxis.set_major_formatter("{x:.2f}")
            ax.text(0.035, 0.955, f"({'abcdefghi'[3 * r + c]})", transform=ax.transAxes, fontsize=9,
                    fontweight="bold", va="top")
            ax.text(0.965, 0.955, f"n = {len(d)}", transform=ax.transAxes, fontsize=7, ha="right", va="top")
            if r == 0:
                ax.set_title(per, fontsize=9, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{rows[r]}\nLatitude", fontsize=8)
            if r == 2:
                ax.set_xlabel("Longitude", fontsize=8)
        print(f"{per}: n = {len(d)}")
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=TYPE_COL[k], mec="k", mew=0.5, ms=6, label=TYPE_NAME[k])
               for k in "NRS"]
    y_gap = (axs[1, 0].get_position().y0 + axs[2, 0].get_position().y1) / 2
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.54, y_gap), ncol=3, frameon=False, fontsize=8)
    cax = fig.add_axes([0.32, 0.05, 0.44, 0.014])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=range(0, KMAX + 1, 20))
    cb.set_label("Kagan angle (°)", fontsize=8)
    out = Path(a.out)
    fig.savefig(out, dpi=300)
    print("wrote", out, f"(clusters plotted: {len(df)})")


if __name__ == "__main__":
    main()
