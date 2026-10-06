#!/usr/bin/env python3
"""
CC (cross-correlation) polarity test on the same shared 6801-event/7-station
Axial Seamount benchmark set used for the 4 DL models in Figure 4
(02-data/F_ML/A_wave_dB15_DT_CFM_EQP_PolCAP.mat), for a direct, apples-to-
apples comparison.

The original CC method (01-scripts/B_CC_forplot_pickCC1.m) does full O(N^2)
pairwise event correlation + clustering, which doesn't scale to 6801 events
and whose clustering step's code hasn't been located. This is a simplified
but same-principle "matched filter" approach instead:

  1. Per station, stack a reference "Up" template by z-scoring and averaging
     the 0.2 s window around the P arrival (sample index 100 of the 200
     sample / 200 Hz window -- empirically verified, see Figure 4 rebuild
     notes) for every event with an existing manual Po_<station> == 1 label.
  2. For every event in the shared set, z-score the same window (searching
     +/- 10 samples of lag for best alignment) and compute its normalized
     cross-correlation against the template. Leave-one-out: an event that
     itself contributed to the "Up" template is scored against a template
     with its own contribution removed (via a running sum, not recomputed
     from scratch per event), so no event is ever scored against a template
     built partly from itself. Down-labeled events never contribute to the
     template, so this only matters for Up-labeled events.
  3. Threshold |correlation| >= 0.7 (matches P.c.ccthreshold in
     B_CC_forplot_pickCC1.m) to decide Up (positive correlation) / Down
     (negative correlation); below threshold -> undecided (excluded from
     scoring, same convention as the DL models' 0/skip).

Note: this still does not reproduce the original graph-clustering CC method
-- it is a different (simpler), same-principle algorithm, not a faithful
reimplementation.
"""
import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat

MASTER = Path('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/F_ML/A_wave_dB15_DT_CFM_EQP_PolCAP.mat')
STATIONS = ['AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1']

ARRIVAL = 100
HALF_WIN = 20
MAX_LAG = 10
CC_THRESHOLD = 0.7


def zscore(x: np.ndarray):
    x = x - x.mean()
    sd = x.std()
    return None if sd == 0 else x / sd


def run(master_mat: Path = MASTER):
    d = loadmat(str(master_mat))
    fx = d['Felix'][0]
    n_events = len(fx)

    rows = []
    tot = {'PT': 0, 'NT': 0, 'PF': 0, 'NF': 0}
    for s in STATIONS:
        po_all = fx[f'Po_{s}'].astype(float).flatten()
        W = fx[f'W_{s}']

        # up_windows: event_index -> z-scored template window, for every
        # event that itself contributes to the "Up" template.
        up_windows = {}
        for i in range(n_events):
            w = np.asarray(W[i]).flatten()
            if w.size != 200 or po_all[i] != 1:
                continue
            seg = zscore(w[ARRIVAL - HALF_WIN:ARRIVAL + HALF_WIN])
            if seg is not None:
                up_windows[i] = seg
        n_up = len(up_windows)
        sum_vec = np.sum(list(up_windows.values()), axis=0)
        full_template = sum_vec / n_up
        full_template = full_template / np.std(full_template)

        counts = {'PT': 0, 'NT': 0, 'PF': 0, 'NF': 0}
        for i in range(n_events):
            po = po_all[i]
            if po == 0:
                continue
            w = np.asarray(W[i]).flatten()
            if w.size != 200:
                continue

            # Leave-one-out: if this event helped build the template, score
            # it against a template with its own contribution removed, so no
            # event is ever scored against a template it was part of.
            if i in up_windows:
                loo_mean = (sum_vec - up_windows[i]) / (n_up - 1)
                template = loo_mean / np.std(loo_mean)
            else:
                template = full_template

            best_abs_corr, best_corr = 0.0, 0.0
            for lag in range(-MAX_LAG, MAX_LAG + 1):
                c0 = ARRIVAL + lag
                sub = w[c0 - HALF_WIN:c0 + HALF_WIN]
                if sub.size != 2 * HALF_WIN:
                    continue
                z = zscore(sub)
                if z is None:
                    continue
                corr = float(np.dot(z, template) / (2 * HALF_WIN))
                if abs(corr) > best_abs_corr:
                    best_abs_corr, best_corr = abs(corr), corr

            if best_abs_corr < CC_THRESHOLD:
                continue
            pred = 1 if best_corr > 0 else -1
            if pred == 1 and po == 1:
                counts['PT'] += 1
            elif pred == -1 and po == -1:
                counts['NT'] += 1
            elif pred == 1 and po == -1:
                counts['PF'] += 1
            elif pred == -1 and po == 1:
                counts['NF'] += 1

        n = sum(counts.values())
        acc = 100 * (counts['PT'] + counts['NT']) / n if n else float('nan')
        rows.append([s, counts['PT'], counts['NT'], counts['PF'], counts['NF'], n, round(acc, 2)])
        for k in tot:
            tot[k] += counts[k]

    n = sum(tot.values())
    acc = 100 * (tot['PT'] + tot['NT']) / n
    rows.append(['ALL (pooled)', tot['PT'], tot['NT'], tot['PF'], tot['NF'], n, round(acc, 2)])
    return rows


if __name__ == '__main__':
    rows = run()
    header = ['Station', 'PT', 'NT', 'PF', 'NF', 'N', 'Accuracy(%)']
    print('\t'.join(header))
    for r in rows:
        print('\t'.join(str(x) for x in r))

    out_csv = Path('/Users/mczhang/Documents/GitHub/FM5_ML/03-output/CC_matched_filter_confusion_matrix.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f'\nSaved to {out_csv}')
