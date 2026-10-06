#!/usr/bin/env python3
"""
Supplementary figure: pooled confusion matrix for the CC stacked-template
matched-filter test (01-scripts/X_CC_matched_filter_7station.py) run on the
SAME shared 6801-event/7-station set used for the 4 DL models in FigureS1 --
unlike FigureS2 (CC scored on its own separate ~1000-event/station set),
this one is directly comparable to FigureS1's numbers.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig_fm5_ml"

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "01-scripts"))
from X_CC_matched_filter_7station import run as run_cc_matched_filter

from make_manuscript_figures import finalize_srl_size, save_figure
from figureS1_confusion_matrices import plot_one_matrix
import make_manuscript_figures as mmf


def figure_s3(outdir: Path) -> Path:
    rows = run_cc_matched_filter()
    pooled = next(r for r in rows if r[0] == "ALL (pooled)")
    counts = {"PT": pooled[1], "NT": pooled[2], "PF": pooled[3], "NF": pooled[4]}

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    fig.subplots_adjust(left=0.20, right=0.85, top=0.82, bottom=0.15)
    im = plot_one_matrix(ax, counts, "", "CC (matched filter, shared set)")

    cax = fig.add_axes([0.88, 0.15, 0.04, 0.6])
    fig.colorbar(im, cax=cax, label="Row-normalized (%)")

    out = outdir / "FigureS3_confusion_matrix_CC_sharedset.png"
    finalize_srl_size(fig, width_in=mmf.SRL_SINGLE_COL_IN)
    save_figure(fig, out)
    return out


def main() -> None:
    default_repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build the shared-set CC matched-filter confusion-matrix figure")
    parser.add_argument("--outdir", default=str(default_repo / "03-output"))
    parser.add_argument("--srl", action="store_true")
    args = parser.parse_args()
    mmf.SRL_MODE = args.srl

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = figure_s3(outdir)
    print(f"[OK] Figure S3 -> {out}")


if __name__ == "__main__":
    main()
