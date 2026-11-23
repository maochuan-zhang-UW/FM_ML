"""GJI Figs 4-7 and S3: training strategies, SNR, and time-shift experiments.

Reads the per-run JSON results written by
01-scripts/training/retrain_fig4_strategies.py into 06-models/retrain_20261122/
(see 01-scripts/training/run_training_strategies.sh for the full set of runs).

Transfer-learning model in Figs 4-7: PolarCAP with its pretrained softmax
classifier frozen and the encoder retrained ("freezecls").
Every bar is the mean of three seeds (LOSO: one run per held-out station).
Average bar = unweighted mean of the seven station accuracies.

Outputs -> 03-figs/GJI_Figure04.png ... GJI_Figure07.png, GJI_FigureS3.png,
and the plotted values in 03-figs/training_strategy_numbers.json.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "06-models" / "retrain_20261122"
OUT = ROOT / "03-figs"
ST = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]

spec = importlib.util.spec_from_file_location("mmf", ROOT / "01-scripts" / "make_manuscript_figures.py")
h = importlib.util.module_from_spec(spec)
sys.modules["mmf"] = h
spec.loader.exec_module(h)
h.apply_manuscript_style()


def runs(data, strat):
    out = [json.load(open(p)) for p in sorted(RUNS.glob(f"{data}_{strat}_s[0-9].json"))]
    assert out, (data, strat)
    return out


def bars(data, strat, test="Val"):
    """Per-station mean over seeds (%) + station average as 8th element."""
    A = np.array([[r["eval"][test][s] for s in ST] for r in runs(data, strat)]).mean(0) * 100
    return np.r_[A, A.mean()]


def loso():
    A = np.array([json.load(open(RUNS / f"TMSF_Tra_000_scratch_s0_loso{s}.json"))["eval"]["Val"][s]
                  for s in ST]) * 100
    return np.r_[A, A.mean()]


TL = "freezecls"
NUMBERS = {}


def fig4():
    series = {"Trained on all stations": bars("TMSF_Tra_000", "scratch"),
              "Leave-one-station-out": loso(),
              "Transfer learning": bars("TMSF_Tra_000", TL)}
    NUMBERS["fig4"] = series
    data = np.vstack(list(series.values())).T
    fig, ax = plt.subplots(figsize=(h.FIG_WIDTH_IN, 3.4), constrained_layout=True)
    h._grouped_bar(ax, data, h.STATIONS_AX_AVG, list(series), h.SERIES3, (88, 101.6))
    ax.set_ylabel("Accuracy (%)"); ax.set_xlabel("Station")
    ax.legend(ncol=3, loc="upper center", columnspacing=1.2, handlelength=1.4,
              handletextpad=0.4, borderaxespad=0.2)
    p = OUT / "GJI_Figure04.png"; h.save_figure(fig, p); return p


def two_panel(name, rows_a, rows_b, series, cols_a, cols_b, ylim, legend_title=None, ncol=3, fs=8):
    data_a = np.vstack(rows_a).T; data_b = np.vstack(rows_b).T
    fig, axes = plt.subplots(2, 1, figsize=(h.FIG_WIDTH_IN, 6.0), constrained_layout=True)
    for ax, d, c, lab in [(axes[0], data_a, cols_a, "(a) Model trained from scratch"),
                          (axes[1], data_b, cols_b, "(b) Transfer learning")]:
        h._grouped_bar(ax, d, h.STATIONS_AX_AVG, series, c, ylim)
        ax.set_ylabel("Accuracy (%)")
        kw = dict(title=legend_title, title_fontsize=8) if legend_title else {}
        ax.legend(ncol=ncol, loc="upper center", columnspacing=1.2, handlelength=1.4,
                  handletextpad=0.4, borderaxespad=0.2, fontsize=fs, **kw)
        h.panel_label(ax, lab)
    axes[1].set_xlabel("Station")
    p = OUT / name; h.save_figure(fig, p); return p


def fig5():
    a = [bars("STEP", "scratch"), bars("TMSF_Tra_000", "scratch"), bars("STEP010", "scratch")]
    b = [bars("STEP", TL), bars("TMSF_Tra_000", TL), bars("STEP010", TL)]
    NUMBERS["fig5"] = {"scratch": dict(zip(["high", "orig", "low"], a)),
                       "transfer": dict(zip(["high", "orig", "low"], b))}
    return two_panel("GJI_Figure05.png", a, b,
                     ["Trained on high SNR", "Trained on original SNR", "Trained on low SNR"],
                     [h.C_ALT_1, h.C_SCRATCH, h.C_ALT_2], [h.C_ALT_1, h.C_TRANSFER, h.C_ALT_2],
                     (88, 102.4))


def fig6():
    tests = ["Val", "TMSF_Val_001", "TMSF_Val"]
    a = [bars("TMSF_Tra_000", "scratch", t) for t in tests]
    b = [bars("TMSF_Tra_000", TL, t) for t in tests]
    NUMBERS["fig6"] = {"scratch": dict(zip(tests, a)), "transfer": dict(zip(tests, b))}
    return two_panel("GJI_Figure06.png", a, b,
                     [r"$\sigma$ = 0.00 s", r"$\sigma$ = 0.01 s", r"$\sigma$ = 0.02 s"],
                     [h.C_SCRATCH, h.C_ALT_1, h.C_ALT_2], [h.C_TRANSFER, h.C_ALT_1, h.C_ALT_2],
                     (65, 109), legend_title="Time shift applied to test waveforms")


def fig7():
    combos = [("TMSF_Tra_001", "TMSF_Val_001"), ("TMSF_Tra_001", "TMSF_Val"), ("TMSF_Tra_002s", "TMSF_Val_001")]
    a = [bars(d, "scratch", t) for d, t in combos]
    b = [bars(d, TL, t) for d, t in combos]
    NUMBERS["fig7"] = {"scratch": a, "transfer": b,
                       "scratch_tr02_te02": bars("TMSF_Tra_002s", "scratch", "TMSF_Val"),
                       "transfer_tr02_te02": bars("TMSF_Tra_002s", TL, "TMSF_Val")}
    return two_panel("GJI_Figure07.png", a, b,
                     [r"train $\sigma$ = 0.01 s, test $\sigma$ = 0.01 s",
                      r"train $\sigma$ = 0.01 s, test $\sigma$ = 0.02 s",
                      r"train $\sigma$ = 0.02 s, test $\sigma$ = 0.01 s"],
                     h.SERIES3, h.SERIES3, (75, 107), ncol=2, fs=7.5)


def figS3():
    strategies = [
        ("scratch",   "AxialPolCap, trained from scratch",               h.C_SCRATCH),
        ("freezecls", "PolarCAP, classifier frozen, encoder retrained",   h.C_VAR_1),
        ("finetune",  "PolarCAP, all layers retrained (full fine-tuning)", h.C_VAR_2),
        ("probe",     "PolarCAP, encoder frozen, classifier retrained",   h.C_TRANSFER),
    ]
    fig, ax = plt.subplots(figsize=(h.FIG_WIDTH_IN, 3.4), constrained_layout=True)
    NUMBERS["figS3"] = {}
    for s, label, c in strategies:
        rs = runs("TMSF_Tra_000", s)
        n = max(len(r["val_acc_hist"]) for r in rs)
        for r in rs:
            y = np.array(r["val_acc_hist"]) * 100
            ax.plot(np.arange(1, len(y) + 1), y, "-", color=c, lw=0.8, alpha=0.6)
        L = min(len(r["val_acc_hist"]) for r in rs)
        mu = np.array([r["val_acc_hist"][:L] for r in rs]).mean(0) * 100
        test = np.mean([r["eval"]["Val"]["station_mean"] for r in rs]) * 100
        NUMBERS["figS3"][s] = dict(test=test, epochs=[r["epochs_run"] for r in rs],
                                   n_trainable=rs[0]["n_trainable"], ep1=mu[0], best_val=max(max(r["val_acc_hist"]) for r in rs) * 100)
        ax.plot(np.arange(1, L + 1), mu, "-", color=c, lw=2.0, marker="o", markevery=5, ms=4,
                mec="k", mew=0.6,
                label=f"{label} ({rs[0]['n_trainable']:,} trainable parameters); test {test:.1f}%")
    ax.set_xlim(0, 41); ax.set_ylim(85, 100)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation polarity accuracy (%)")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=6.8, handlelength=2.0)
    p = OUT / "GJI_FigureS3.png"; h.save_figure(fig, p); return p


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig4, fig5, fig6, fig7, figS3):
        print("saved", f())
    js = {k: (v if k == "figS3" else json.loads(json.dumps(v, default=lambda a: [round(float(x), 2) for x in a])))
          for k, v in NUMBERS.items()}
    json.dump(js, open(OUT / "training_strategy_numbers.json", "w"), indent=1, default=float)
    print(json.dumps(js, indent=0, default=float)[:6000])
