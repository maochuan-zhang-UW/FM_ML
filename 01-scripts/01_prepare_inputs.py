"""Step 1: convert the MATLAB inputs to one HDF5 file.

Inputs (02-data/raw/, MATLAB .mat as delivered by the catalogue workflow)
  Template_divide.mat        templates per station; {STA}_T = training templates
                             (80%), {STA}_V = held-out test templates (20%).
                             Each has a 200-sample waveform W_{STA} (100 Hz,
                             P pick at sample 100), the manual polarity Man_{STA}
                             (-1/+1) and the catalogue (CC) polarity Po_{STA}.
  H_Noise_200.mat            200-sample pre-event noise windows per station
                             (struct array Felix, fields W_{STA}).
  H_noise_dB20_snrValue.mat  observed SNR (dB) of the catalogue picks per station.

Output: 02-data/prepared/inputs.h5
  templates/{train,test}/{STA}/waveform        (N, 200) float64
                               manual_polarity (N,) int8, -1/+1
                               catalog_polarity(N,) int8
                               event_id, lon, lat, depth (N,)
  noise/{STA}/waveform                         (M, 200) float64, valid windows only
  snr/{STA}/snr_db                             observed SNR values (dB)

Usage (from the repository root):
  python 01-scripts/01_prepare_inputs.py [--raw-dir 02-data/raw] [--out 02-data/prepared/inputs.h5]
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
STATIONS = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]


def write_templates(h5, mat):
    for split, suffix in (("train", "_T"), ("test", "_V")):
        for st in STATIONS:
            items = np.atleast_1d(mat[st + suffix])
            g = h5.create_group(f"templates/{split}/{st}")
            g["waveform"] = np.stack([np.asarray(getattr(e, f"W_{st}"), float).ravel() for e in items])
            g["manual_polarity"] = np.array([int(getattr(e, f"Man_{st}")) for e in items], np.int8)
            g["catalog_polarity"] = np.array([int(getattr(e, f"Po_{st}")) for e in items], np.int8)
            for f in ("ID", "lon", "lat", "depth"):
                g["event_id" if f == "ID" else f] = np.array([getattr(e, f) for e in items])
            print(f"templates/{split}/{st}: {len(items)}")


def write_noise(h5, felix):
    for st in STATIONS:
        w = []
        for e in felix:
            x = np.asarray(getattr(e, f"W_{st}", []), float).ravel()
            if x.size == 200 and np.any(x):          # same validity test as the MATLAB code
                w.append(x)
        h5.create_dataset(f"noise/{st}/waveform", data=np.stack(w), compression="gzip")
        print(f"noise/{st}: {len(w)} windows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=str(ROOT / "02-data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "02-data" / "prepared" / "inputs.h5"))
    a = ap.parse_args()
    raw = Path(a.raw_dir)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.out, "w") as h5:
        write_templates(h5, loadmat(raw / "Template_divide.mat", squeeze_me=True, struct_as_record=False))
        snr = loadmat(raw / "H_noise_dB20_snrValue.mat", squeeze_me=True)["snrValues"]
        for st, v in zip(STATIONS, snr):
            h5[f"snr/{st}/snr_db"] = np.asarray(v, float).ravel()
        write_noise(h5, np.atleast_1d(loadmat(raw / "H_Noise_200.mat", squeeze_me=True,
                                              struct_as_record=False)["Felix"]))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
