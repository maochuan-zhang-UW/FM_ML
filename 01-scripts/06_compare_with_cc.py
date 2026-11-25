"""Step 6: agreement between AxialPolCap and CC polarities on the 2015-2021
catalogue (Section 4.1).

Picks are retained when the AxialPolCap confidence is >= --conf (0.95).
Agreement is computed over retained picks that also have a CC polarity,
excluding the station-event pairs used as training templates. It is also
reported per station, for events with >= 5 retained picks (the composite
mechanism criterion) and for events retained at all seven stations.

Input: output of 05_apply_to_catalog.py; templates (event_id per station) in
02-data/A_wave_dB20_cleaned.h5.
Usage (from the repository root): python 01-scripts/06_compare_with_cc.py
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default=str(ROOT / "02-data" / "A_wave_2015_2021_AxialPolCap_final_conf80_predictions.mat"))
    ap.add_argument("--templates", default=str(ROOT / "02-data" / "A_wave_dB20_cleaned.h5"))
    ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--out", default=str(ROOT / "02-data" / "section41_agreement.json"))
    a = ap.parse_args()
    with h5py.File(a.templates, "r") as h:
        tid = {s: {int(i) for i in h[s]["event_id"][:]} for s in STATIONS if s in h}
    ev = np.atleast_1d(loadmat(a.predictions, squeeze_me=True, struct_as_record=False)["Felix"])
    n = dict(waveforms=0, retained=0, both=0, templates=0, events_ge5=0)
    agree = {k: [0, 0] for k in ("all_with_templates", "independent", "events_ge5", "all_seven")}
    per_sta = {s: [0, 0] for s in STATIONS}
    for e in ev:
        eid, picks = int(e.ID), []
        for s in STATIONS:
            v = np.atleast_1d(getattr(e, f"Po_{s}", [])).astype(float).ravel()
            if v.size < 4:
                continue
            n["waveforms"] += 1
            if v[1] == 0 or v[2] < a.conf:
                continue
            n["retained"] += 1
            picks.append((s, v[0], v[1]))
        n["events_ge5"] += len(picks) >= 5
        for s, cc, ml in picks:
            if cc == 0:
                continue
            ok = int(np.sign(cc) == np.sign(ml))
            n["both"] += 1
            agree["all_with_templates"][0] += ok
            agree["all_with_templates"][1] += 1
            if eid in tid.get(s, ()):
                n["templates"] += 1
                continue
            for key, use in (("independent", True), ("events_ge5", len(picks) >= 5), ("all_seven", len(picks) == 7)):
                if use:
                    agree[key][0] += ok
                    agree[key][1] += 1
            per_sta[s][0] += ok
            per_sta[s][1] += 1
    pct = lambda x: 100.0 * x[0] / x[1] if x[1] else float("nan")
    res = {"counts": n, "agreement_pct": {k: pct(v) for k, v in agree.items()},
           "n_picks": {k: v[1] for k, v in agree.items()},
           "per_station_pct": {s: pct(v) for s, v in per_sta.items()}}
    print(f"waveforms {n['waveforms']:,}; retained (conf >= {a.conf}) {n['retained']:,} "
          f"({100 * n['retained'] / n['waveforms']:.1f}%); with CC {n['both']:,}; templates {n['templates']:,}; "
          f"independent {agree['independent'][1]:,}")
    print(f"agreement {pct(agree['independent']):.1f}% (with templates {pct(agree['all_with_templates']):.1f}%); "
          f"events with >= 5 retained picks {n['events_ge5']:,}: {pct(agree['events_ge5']):.1f}% over "
          f"{agree['events_ge5'][1]:,} picks; all seven stations {pct(agree['all_seven']):.1f}%")
    print("per station: " + "  ".join(f"{s} {pct(v):.1f}" for s, v in per_sta.items()))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
