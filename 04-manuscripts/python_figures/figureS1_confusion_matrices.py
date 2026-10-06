#!/usr/bin/env python3
"""
Supplementary figure: pooled (all-station) PT/NT/PF/NF confusion matrix for
each of the 4 DL models compared in Figure 4 (DiTingMotion, CFM, EQPolarity,
PolarCAP), as one figure with 4 subplots.

Convention: Positive (P) = Up (+1), Negative (N) = Down (-1);
True = matches manual Po_<station> pick, False = does not.
  PT = predicted Up,   true  (= TP)      NF = predicted Down, false (= FN)
  NT = predicted Down, true  (= TN)      PF = predicted Up,   false (= FP)

Source data: 02-data/F_ML/A_wave_dB15_DT_CFM_EQP_PolCAP.mat (see
01-scripts/{DiTing-FOCALFLOW,CFM,eqpolarity,PolarCAP} for how each model's
predictions were generated).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig_fm5_ml"

import matplotlib.pyplot as plt
import numpy as np

from make_manuscript_figures import (
    STATIONS,
    finalize_srl_size,
    get_value,
    load_struct_array,
    save_figure,
)
import make_manuscript_figures as mmf

MODELS = {
    "DiTingMotion": "PoML_W_",
    "CFM": "CFM_W_",
    "EQPolarity": "EQP_",
    "PolarCAP": "PolCAP_",
}


def pooled_confusion(felix: np.ndarray, prefix: str) -> dict:
    tot = {"PT": 0, "NT": 0, "PF": 0, "NF": 0}
    for s in STATIONS:
        po = np.array([get_value(r, f"Po_{s}", 0) for r in felix], dtype=float)
        pred = np.array([get_value(r, f"{prefix}{s}", 0) for r in felix], dtype=float)
        mask = (po != 0) & (pred != 0)
        po_m, pred_m = po[mask], pred[mask]
        tot["PT"] += int(np.sum((pred_m == 1) & (po_m == 1)))
        tot["NT"] += int(np.sum((pred_m == -1) & (po_m == -1)))
        tot["PF"] += int(np.sum((pred_m == 1) & (po_m == -1)))
        tot["NF"] += int(np.sum((pred_m == -1) & (po_m == 1)))
    return tot


def plot_one_matrix(ax: plt.Axes, counts: dict, panel_label: str, model_name: str) -> None:
    # rows = actual (Up, Down); cols = predicted (Up, Down)
    # row 0 (actual Up):   [PT, NF]
    # row 1 (actual Down): [PF, NT]
    mat = np.array([[counts["PT"], counts["NF"]], [counts["PF"], counts["NT"]]], dtype=float)
    row_sums = mat.sum(axis=1, keepdims=True)
    pct = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums != 0) * 100

    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="equal")
    for i in range(2):
        for j in range(2):
            color = "white" if pct[i, j] > 60 else "black"
            ax.text(
                j, i, f"{int(mat[i, j])}\n({pct[i, j]:.1f}%)",
                ha="center", va="center", color=color, fontsize=10,
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred. Up", "Pred. Down"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual Up", "Actual Down"])
    n = int(mat.sum())
    acc = 100 * (counts["PT"] + counts["NT"]) / n if n else float("nan")
    # Always shown (in both SRL and non-SRL mode): this identifies which
    # model/panel is which, not a redundant "Figure N:" caption.
    ax.set_title(f"{panel_label} {model_name}\nn={n}, acc={acc:.1f}%", loc="left", fontsize=10)
    return im


def figure_s1(paths, outdir: Path) -> Path:
    master_mat = paths.repo_root / "02-data" / "F_ML" / "A_wave_dB15_DT_CFM_EQP_PolCAP.mat"
    felix = load_struct_array(master_mat, "Felix")

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.0))
    fig.subplots_adjust(left=0.10, right=0.86, top=0.90, bottom=0.08, wspace=0.35, hspace=0.55)
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    im = None
    for ax, label, (model_name, prefix) in zip(axes.flat, panel_labels, MODELS.items()):
        counts = pooled_confusion(felix, prefix)
        im = plot_one_matrix(ax, counts, label, model_name)

    cax = fig.add_axes([0.89, 0.15, 0.025, 0.7])
    fig.colorbar(im, cax=cax, label="Row-normalized (%)")

    out = outdir / "FigureS1_confusion_matrices.png"
    finalize_srl_size(fig)
    save_figure(fig, out)
    return out


def main() -> None:
    default_repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build the pooled confusion-matrix supplementary figure")
    parser.add_argument("--outdir", default=str(default_repo / "03-output"))
    parser.add_argument("--srl", action="store_true", help="Strip descriptive title text (panel labels are kept)")
    args = parser.parse_args()

    mmf.SRL_MODE = args.srl

    paths = mmf.Paths(
        repo_root=default_repo,
        fm_root=Path("/Users/mczhang/Documents/GitHub/FM"),
        fm3_root=Path("/Users/mczhang/Documents/GitHub/FM3"),
        fm4_root=Path("/Users/mczhang/Documents/GitHub/FM4_7OBS"),
        docx_path=default_repo / "04-manuscripts" / "MZhang_FM_ML_V1_WW.docx",
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = figure_s1(paths, outdir)
    print(f"[OK] Figure S1 -> {out}")


if __name__ == "__main__":
    main()
