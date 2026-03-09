"""
convert_mat_to_h5.py
====================
Convert two source .mat files into HDF5 format.

Input files
-----------
  02-data/A_wave_dB20_cleaned.mat   -- 2009 seismic events, 200-sample waveforms
                                       at 100 Hz (-1 s to +1 s around P-arrival)
  02-data/A_wave_noise_10000.mat    -- 10 000 noise snippets, 100-sample at 100 Hz

Output files
------------
  02-data/A_wave_dB20_cleaned.h5
  02-data/A_wave_noise_10000.h5

HDF5 layout
-----------
A_wave_dB20_cleaned.h5
  /AS1/
    waveforms   (N, 200)  float32   -- raw 100-Hz waveform, -1 s to +1 s
    polarities  (N,)      int8      -- -1 = negative, +1 = positive, 0 = no pick
    event_id    (N,)      int32     -- Felix.ID
    sp          (N,)      float32   -- signal power (Felix.SP_{STA})
    ddt         (N,)      float32   -- P-arrival travel-time offset in seconds
  /AS2/ ... /ID1/                   -- same structure for every station
  attrs: source, fs_hz, n_events_total, stations

A_wave_noise_10000.h5
  /AS1/
    waveforms   (10000, 100)  float32   -- 100-Hz noise snippet, 1 s long
  /AS2/ ... /ID1/
  attrs: source, fs_hz, n_events_total, stations

Note on polarities
------------------
  Po = 0 at a given station means no manual pick was made there; the waveform
  field is also empty in that case. Only events with a valid 200-sample
  waveform are included in each station group.

Run from FM_ML/ root:
    conda activate tf-2.14.0
    python 01-scripts/convert_mat_to_h5.py
"""

import os
import numpy as np
import scipy.io
import h5py

# ---------------------------------------------------------------------------
# Paths  (resolved relative to FM_ML/ root, regardless of working directory)
# ---------------------------------------------------------------------------
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAVEFORM_MAT = os.path.join(ROOT, "02-data", "A_wave_dB20_cleaned.mat")
NOISE_MAT    = os.path.join(ROOT, "02-data", "A_wave_noise_10000.mat")
WAVEFORM_H5  = os.path.join(ROOT, "02-data", "A_wave_dB20_cleaned.h5")
NOISE_H5     = os.path.join(ROOT, "02-data", "A_wave_noise_10000.h5")

STATIONS     = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]
WF_LEN       = 200   # expected waveform length (samples)
NOISE_LEN    = 100   # expected noise length (samples)


# ---------------------------------------------------------------------------
# Convert waveform file
# ---------------------------------------------------------------------------

def convert_waveforms(mat_path: str, h5_path: str) -> None:
    print(f"\nLoading {mat_path} ...")
    mat   = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    felix = mat["Felix"]
    n_events = len(felix)
    print(f"  {n_events} events in struct array")

    with h5py.File(h5_path, "w") as f:
        f.attrs["source"]         = mat_path
        f.attrs["fs_hz"]          = 100
        f.attrs["n_events_total"] = n_events
        f.attrs["stations"]       = STATIONS

        for sta in STATIONS:
            wf_field  = f"W_{sta}"
            pol_field = f"Po_{sta}"
            sp_field  = f"SP_{sta}"
            ddt_field = f"DDt_{sta}"

            waveforms, polarities, event_ids, sps, ddts = [], [], [], [], []

            for ev in felix:
                wf = np.asarray(getattr(ev, wf_field, []), dtype=np.float64).ravel()
                if len(wf) != WF_LEN:          # skip empty / wrong-length
                    continue

                pol = int(getattr(ev, pol_field, 0))
                sp  = float(getattr(ev, sp_field,  0.0))
                ddt = float(getattr(ev, ddt_field, 0.0))
                eid = int(getattr(ev, "ID", 0))

                waveforms.append(wf.astype(np.float32))
                polarities.append(pol)
                event_ids.append(eid)
                sps.append(sp)
                ddts.append(ddt)

            n = len(waveforms)
            grp = f.create_group(sta)
            if n == 0:
                print(f"  {sta}: no valid waveforms — empty group created")
                continue

            grp.create_dataset("waveforms",  data=np.stack(waveforms), dtype="float32",
                               compression="gzip", compression_opts=4)
            grp.create_dataset("polarities", data=np.array(polarities, dtype=np.int8))
            grp.create_dataset("event_id",   data=np.array(event_ids,  dtype=np.int32))
            grp.create_dataset("sp",         data=np.array(sps,        dtype=np.float32))
            grp.create_dataset("ddt",        data=np.array(ddts,       dtype=np.float32))

            n_pos = sum(p == 1  for p in polarities)
            n_neg = sum(p == -1 for p in polarities)
            n_unk = sum(p == 0  for p in polarities)
            print(f"  {sta}: {n:4d} waveforms  "
                  f"(pos={n_pos}, neg={n_neg}, no-pick={n_unk})")

    print(f"  Saved -> {h5_path}")


# ---------------------------------------------------------------------------
# Convert noise file
# ---------------------------------------------------------------------------

def convert_noise(mat_path: str, h5_path: str) -> None:
    print(f"\nLoading {mat_path} ...")
    mat   = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    felix = mat["Felix"]
    n_events = len(felix)
    print(f"  {n_events} noise snippets in struct array")

    with h5py.File(h5_path, "w") as f:
        f.attrs["source"]         = mat_path
        f.attrs["fs_hz"]          = 100
        f.attrs["n_events_total"] = n_events
        f.attrs["stations"]       = STATIONS

        for sta in STATIONS:
            wf_field = f"W_{sta}"
            waveforms = []

            for ev in felix:
                wf = np.asarray(getattr(ev, wf_field, []), dtype=np.float64).ravel()
                if len(wf) != NOISE_LEN:
                    continue
                waveforms.append(wf.astype(np.float32))

            n = len(waveforms)
            grp = f.create_group(sta)
            if n == 0:
                print(f"  {sta}: no valid noise snippets — empty group created")
                continue

            grp.create_dataset("waveforms", data=np.stack(waveforms), dtype="float32",
                               compression="gzip", compression_opts=4)
            print(f"  {sta}: {n:5d} noise snippets  shape={np.stack(waveforms).shape}")

    print(f"  Saved -> {h5_path}")


# ---------------------------------------------------------------------------
# Verification: print HDF5 tree and a quick sanity check
# ---------------------------------------------------------------------------

def verify_h5(h5_path: str) -> None:
    print(f"\n--- Verifying {h5_path} ---")
    with h5py.File(h5_path, "r") as f:
        print(f"  attrs: { {k: f.attrs[k] for k in f.attrs} }")
        for sta in f.keys():
            grp = f[sta]
            for ds_name in grp.keys():
                ds = grp[ds_name]
                print(f"  /{sta}/{ds_name}: shape={ds.shape}, dtype={ds.dtype}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(os.path.join(ROOT, "02-data"), exist_ok=True)

    print("=" * 60)
    print("convert_mat_to_h5.py")
    print("=" * 60)

    convert_waveforms(WAVEFORM_MAT, WAVEFORM_H5)
    convert_noise(NOISE_MAT, NOISE_H5)

    verify_h5(WAVEFORM_H5)
    verify_h5(NOISE_H5)

    print("\nDone.")


if __name__ == "__main__":
    main()
