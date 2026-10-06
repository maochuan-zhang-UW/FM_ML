#!/usr/bin/env python3
"""
Supplementary figure: pooled (all-station) PT/NT/PF/NF confusion matrix for
the CC (cross-correlation) baseline shown in Figure 4.

Same convention as FigureS1 (Positive=Up, Negative=Down, True=matches manual
Po_<station> pick). Unlike the 4 DL models in FigureS1 -- which are all
scored on the same shared 6801-event/7-station Axial benchmark set
(02-data/F_ML/A_wave_dB15_DT_CFM_EQP_PolCAP.mat) -- CC is scored on its own,
separate ~1000-event/station set (02-data/D_man/D_manual_<station>_CCPo.mat),
produced by a semi-manual pick pipeline (01-scripts/B_CC_combined.m ->
C_align_manualpick*.m -> D_manual_pick_station*.m). The two sets are largely
non-overlapping (~145/1000 CC1 events in common), so this pooled 87.79%
should NOT be read as directly comparable to the DL models' pooled numbers
in FigureS1, and it does not match the CC value currently hardcoded in
Figure 4 (91.6% avg) -- that number's exact source dataset has not been
traced yet.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig_fm5_ml"

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from make_manuscript_figures import STATIONS, finalize_srl_size, save_figure
from figureS1_confusion_matrices import plot_one_matrix
import make_manuscript_figures as mmf

CC_FIELD = {s: (f"{s}_Po" if s == "CC1" else f"{s}_CCPo") for s in STATIONS}


def pooled_cc_confusion(repo_root: Path) -> dict:
    tot = {"PT": 0, "NT": 0, "PF": 0, "NF": 0}
    for s in STATIONS:
        path = repo_root / "02-data" / "D_man" / f"D_manual_{s}_CCPo.mat"
        d = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        felix = d["Felix"]
        po = np.array([np.sign(float(np.asarray(getattr(r, f"Po_{s}")).flat[0]))
                       if np.asarray(getattr(r, f"Po_{s}")).size else 0.0 for r in felix])
        cc = np.array([np.sign(float(np.asarray(getattr(r, CC_FIELD[s])).flat[0]))
                       if np.asarray(getattr(r, CC_FIELD[s])).size else 0.0 for r in felix])
        mask = (po != 0) & (cc != 0)
        po_m, cc_m = po[mask], cc[mask]
        tot["PT"] += int(np.sum((cc_m == 1) & (po_m == 1)))
        tot["NT"] += int(np.sum((cc_m == -1) & (po_m == -1)))
        tot["PF"] += int(np.sum((cc_m == 1) & (po_m == -1)))
        tot["NF"] += int(np.sum((cc_m == -1) & (po_m == 1)))
    return tot


def figure_s2(repo_root: Path, outdir: Path) -> Path:
    counts = pooled_cc_confusion(repo_root)

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    fig.subplots_adjust(left=0.20, right=0.85, top=0.82, bottom=0.15)
    im = plot_one_matrix(ax, counts, "", "CC (cross-correlation)")

    cax = fig.add_axes([0.88, 0.15, 0.04, 0.6])
    fig.colorbar(im, cax=cax, label="Row-normalized (%)")

    out = outdir / "FigureS2_confusion_matrix_CC.png"
    finalize_srl_size(fig, width_in=mmf.SRL_SINGLE_COL_IN)
    save_figure(fig, out)
    return out


def main() -> None:
    default_repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build the CC pooled confusion-matrix supplementary figure")
    parser.add_argument("--outdir", default=str(default_repo / "03-output"))
    parser.add_argument("--srl", action="store_true")
    args = parser.parse_args()
    mmf.SRL_MODE = args.srl

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = figure_s2(default_repo, outdir)
    print(f"[OK] Figure S2 -> {out}")


if __name__ == "__main__":
    main()
