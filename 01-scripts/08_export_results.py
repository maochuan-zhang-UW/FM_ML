"""Step 8: export the AxialPolCap polarities of the 2015-2021 catalogue as a
flat CSV, one row per event and station with a waveform window.

Columns
  event_id          catalogue event ID (Wang et al. 2024)
  origin_time       UTC, ISO 8601
  lat, lon          degrees
  depth_km          km below sea level
  station           AS1, AS2, CC1, EC1, EC2, EC3, ID1
  cc_polarity       cross-correlation polarity: 1 up, -1 down, 0 none
  ml_polarity       AxialPolCap polarity: 1 up, -1 down, 0 where the confidence
                    is below 0.80 (the threshold stored by step 5)
  confidence        maximum class probability
  entropy           -sum(p ln p) of the class probabilities
  retained          1 if confidence >= --conf (0.95), the picks used in Section 4
  template          1 if this event-station pair is a training template

Input: output of 05_apply_to_catalog.py; templates in
02-data/raw/A_wave_dB20_cleaned.h5.
Usage (from the repository root): python 01-scripts/08_export_results.py
"""
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]
COLUMNS = ["event_id", "origin_time", "lat", "lon", "depth_km", "station", "cc_polarity",
           "ml_polarity", "confidence", "entropy", "retained", "template"]


def datenum_to_iso(d):
    t = datetime.fromordinal(int(d)) + timedelta(days=float(d) % 1) - timedelta(days=366)
    return t.isoformat(timespec="milliseconds")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default=str(ROOT / "02-data" / "A_wave_2015_2021_AxialPolCap_final_conf80_predictions.mat"))
    ap.add_argument("--templates", default=str(ROOT / "02-data" / "raw" / "A_wave_dB20_cleaned.h5"))
    ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--out", default=str(ROOT / "02-data" / "results" / "AxialPolCap_polarities_2015_2021.csv"))
    a = ap.parse_args()
    with h5py.File(a.templates, "r") as h:
        tid = {s: {int(i) for i in h[s]["event_id"][:]} for s in STATIONS if s in h}
    ev = np.atleast_1d(loadmat(a.predictions, squeeze_me=True, struct_as_record=False)["Felix"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = dict(rows=0, retained=0, events=set())
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for e in ev:
            eid = int(e.ID)
            head = [eid, datenum_to_iso(e.on), f"{float(e.lat):.6f}", f"{float(e.lon):.6f}", f"{float(e.depth):.3f}"]
            for s in STATIONS:
                v = np.atleast_1d(getattr(e, f"Po_{s}", [])).astype(float).ravel()
                if v.size < 4:
                    continue
                kept = int(v[1] != 0 and v[2] >= a.conf)
                w.writerow(head + [s, int(np.sign(v[0])), int(np.sign(v[1])), f"{v[2]:.5f}", f"{v[3]:.5f}",
                                   kept, int(eid in tid.get(s, ()))])
                n["rows"] += 1
                n["retained"] += kept
                n["events"].add(eid)
    print(f"{n['rows']:,} picks of {len(n['events']):,} events; retained (conf >= {a.conf}) {n['retained']:,}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
