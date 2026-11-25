"""Step 7: composite focal mechanisms from CC and AxialPolCap polarities,
2015-2021 (Section 4.1, Figs 8 and S7, Table S3).

For every cross-correlation cluster (Zhang et al. in preparation) two
composite mechanisms are computed with identical SKHASH settings, velocity
model, station corrections and S/P amplitude ratios:

  cc : the catalogue (CC) polarities of every event in the cluster
  ml : AxialPolCap polarities with class probability >= --conf (0.95), pooled
       over the events of the cluster that have at least --min-confident (5)
       such polarities; picks below the threshold are dropped

Each composite is written in SKHASH's hash3 format with one pseudo-station
per event and station (AS1A, AS1B, ...), so repeated observations at a station
are kept. The SKHASH parameters are those of the near-real-time pipeline.

Inputs
  --clusters    MATLAB struct array Po_Clu: ID, on (datenum), lat, lon, depth,
                Cluster, Po_{STA} (CC polarity), NSP_{STA} (noise and P/S amplitudes)
  --predictions 05_apply_to_catalog.py output: Felix with
                Po_{STA} = [CC polarity, prediction, confidence, entropy]
  --geometry    station coordinates (staLat, staLon) and 1-D Vp profile (avgProfile)
  --skhash      path to SKHASH.py (https://code.usgs.gov/esc/SKHASH)

Stages (--stage): inputs (SKHASH input files and clusters.csv), invert (run
SKHASH for cc and ml), merge (mechanisms.csv); the default runs all three.

Output (--out): {cc,ml}/ SKHASH inputs and out.txt, and mechanisms.csv with
one row per cluster: location, time, number of events, both mechanisms with
their quality, fault type (Table S4) and the Kagan angle between them.

Usage (from the repository root; inputs default to 02-data/raw/):
  python 01-scripts/07_focal_mechanisms.py --skhash <SKHASH>/SKHASH/SKHASH.py
"""
import argparse
import collections
import os
import datetime
import string
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mechanism_tools import fault_type, kagan_angle, read_skhash  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STA = ["AS1", "AS2", "CC1", "EC1", "EC2", "EC3", "ID1"]
CHAN = {"AS1": "EHZ", "AS2": "EHZ", "CC1": "HHZ", "EC1": "EHZ", "EC2": "HHZ", "EC3": "EHZ", "ID1": "EHZ"}
NET = "OO"
SP_COR = {"AS1": 0.1761, "AS2": -0.3010, "CC1": -0.6990, "EC1": -0.3010,
          "EC2": -0.6990, "EC3": -0.1249, "ID1": -0.6990}     # log10 S/P station corrections
LETTERS = string.ascii_uppercase
MAX_CLUSTER = 100            # clusters with >= 100 events are not inverted
SEH, SEZ, MAG = 0.30, 0.20, 1.00
SKHASH_PARAMS = {            # near-real-time pipeline settings
    "npolmin": 8, "dang": 5, "nmc": 30, "maxout": 500, "ratmin": 3, "badfrac": 0.1,
    "qbadfrac": 0.3, "delmax": 24, "cangle": 45, "prob_max": 0.2, "max_agap": 240,
    "max_pgap": 90, "look_dep": "0 13 1", "look_del": "0 100 1", "num_cpus": 4,
}


def datenum_to_datetime(d):
    return datetime.datetime.fromordinal(int(d) - 366) + datetime.timedelta(days=float(d) % 1)


def load_events(clusters, predictions):
    """One dict per event with CC polarities, AxialPolCap (pred, conf) and S/P data."""
    pc = np.atleast_1d(loadmat(clusters, squeeze_me=True, struct_as_record=False)["Po_Clu"])
    pr = np.atleast_1d(loadmat(predictions, squeeze_me=True, struct_as_record=False)["Felix"])
    pred = {}
    for e in pr:
        d = {}
        for s in STA:
            v = np.atleast_1d(getattr(e, f"Po_{s}", [])).astype(float).ravel()
            if v.size >= 4 and v[1] != 0:
                d[s] = (int(np.sign(v[1])), float(v[2]))
        pred[int(e.ID)] = d
    ev = []
    for e in pc:
        first = lambda f: float(np.atleast_1d(getattr(e, f)).ravel()[0])
        ev.append({
            "id": int(first("ID")), "on": first("on"), "lat": first("lat"), "lon": first("lon"),
            "depth": first("depth"), "cluster": int(first("Cluster")),
            "cc": {s: int(first(f"Po_{s}")) for s in STA},
            "ml": pred.get(int(first("ID")), {}),
            "nsp": {s: np.atleast_1d(getattr(e, f"NSP_{s}")).astype(float).ravel() for s in STA},
        })
    return ev


def select_polarities(ev, source, conf, min_confident):
    """{event id: {station: +1/-1}} for one polarity source."""
    out = {}
    for e in ev:
        if source == "cc":
            p = {s: v for s, v in e["cc"].items() if v != 0}
        else:
            p = {s: pol for s, (pol, c) in e["ml"].items() if c >= conf}
            if len(p) < min_confident:
                continue
        if p:
            out[e["id"]] = p
    return out


def sp_row(name, s, v):
    """hash3 amplitude line, or None (NSP = [noise, S amp, P amp] or [S noise, P noise, S amp, P amp])."""
    if v.size > 3:
        qn_p, qn_s, ap, asx = v[1], v[0], v[3], v[2]
    elif v.size == 3:
        qn_p, qn_s, ap, asx = v[0], v[0], v[2], v[1]
    else:
        return None
    if not (np.isfinite(ap) and np.isfinite(asx)) or ap == 0:
        return None
    return "%-4s %-3s %-2s %8.3f %8.3f %10.3f %10.3f %10.3f %10.3f" % (name, CHAN[s], NET, 0, 0, qn_p, qn_s, ap, asx)


def write_inputs(outd, ev, pols, geometry):
    """SKHASH hash3 inputs: one composite per cluster."""
    outd.mkdir(parents=True, exist_ok=True)
    byid = {e["id"]: e for e in ev}
    size = collections.Counter(e["cluster"] for e in ev)
    grp = collections.defaultdict(list)
    for eid in pols:
        c = byid[eid]["cluster"]
        if size[c] < MAX_CLUSTER:
            grp[c].append(eid)
    ph, amp = [], []
    for c in sorted(grp):
        ids = sorted(grp[c], key=lambda i: byid[i]["on"])[:len(LETTERS)]
        es = [byid[i] for i in ids]
        t = datenum_to_datetime(np.mean([e["on"] for e in es]))
        lat, lon = np.mean([e["lat"] for e in es]), np.mean([e["lon"] for e in es])
        dep = np.mean([e["depth"] for e in es])
        h = [" "] * 165

        def put(text, col):
            h[col:col + len(text)] = list(text)
        put("%4d%02d%02d%02d%02d%5.2f" % (t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6), 0)
        put("%2d%1s%5.2f" % (int(abs(lat)), "N" if lat >= 0 else "S", (abs(lat) % 1) * 60), 17)
        put("%3d%1s%5.2f" % (int(abs(lon)), "W" if lon < 0 else "E", (abs(lon) % 1) * 60), 25)
        put("%5.2f" % dep, 34)
        put("%5.2f" % SEH, 88)
        put("%5.2f" % SEZ, 94)
        put("%4.2f" % MAG, 139)
        put("%16d" % c, 149)
        ph.append("".join(h).rstrip())
        rows = []
        for L, e in zip(LETTERS, es):
            for s, p in sorted(pols[e["id"]].items()):
                ph.append("%-4s %-2s  %-3s %1s %1s" % (s + L, NET, CHAN[s], "I", "U" if p > 0 else "D"))
                r = sp_row(s + L, s, e["nsp"][s])
                if r:
                    rows.append(r)
        ph.append("    ")
        amp.append("%-16s%6d" % (c, len(rows)))
        amp.extend(rows)
    (outd / "phase.dat").write_text("\n".join(ph) + "\n")
    (outd / "amp.dat").write_text("\n".join(amp) + "\n")

    G = loadmat(geometry, squeeze_me=True)
    sla, slo = np.asarray(G["staLat"]).ravel(), np.asarray(G["staLon"]).ravel()
    st, cor = [], []
    for k, s in enumerate(STA):
        for L in LETTERS:
            row = [" "] * 92
            for text, col in (("%-4s" % (s + L), 0), ("%-3s" % CHAN[s], 5), ("AXIAL SEAMOUNT OBS", 10),
                              ("%9.5f" % sla[k], 41), ("%10.5f" % slo[k], 51), ("%5d" % 0, 62),
                              ("1997/09/19", 68), ("3000/01/01", 79), ("%-2s" % NET, 90)):
                row[col:col + len(text)] = list(text)
            st.append("".join(row).rstrip())
            cor.append("%-4s %-3s %-2s %7.4f" % (s + L, CHAN[s], NET, SP_COR[s]))
    (outd / "stations.txt").write_text("\n".join(sorted(st)) + "\n")
    (outd / "statcor.txt").write_text("\n".join(sorted(cor)) + "\n")
    (outd / "reverse.txt").write_text("")
    (outd / "vmodel.txt").write_text("# Depth (km), Vp (km/s)\n" + "\n".join(
        f"{z:.3f}, {vp:.4f}" for z, vp in np.asarray(G["avgProfile"])) + "\n")
    ctl = ["$input_format", "hash3"]
    for k, f in (("stfile", "stations.txt"), ("plfile", "reverse.txt"), ("corfile", "statcor.txt"),
                 ("ampfile", "amp.dat"), ("fpfile", "phase.dat"), ("outfile1", "out.txt"),
                 ("vmodel_paths", "vmodel.txt")):
        ctl += [f"${k}", str((outd / f).resolve())]
    for k, v in SKHASH_PARAMS.items():
        ctl += [f"${k}", str(v)]
    (outd / "control_file.txt").write_text("\n\n".join(f"{a}\n{b}" for a, b in zip(ctl[::2], ctl[1::2])) + "\n")
    return {c: len(v) for c, v in grp.items()}


def build_inputs(a, out):
    """Stage 1 (own process, as the cluster file is large): SKHASH inputs + clusters.csv."""
    ev = load_events(a.clusters, a.predictions)
    print(f"{len(ev):,} clustered events")
    nev = {}
    for src in ("cc", "ml"):
        pols = select_polarities(ev, src, a.conf, a.min_confident)
        nev[src] = write_inputs(out / src, ev, pols, a.geometry)
        print(f"{src}: {len(pols):,} events, {sum(len(p) for p in pols.values()):,} polarities, "
              f"{len(nev[src]):,} composites")
    byc = collections.defaultdict(list)
    for e in ev:
        byc[e["cluster"]].append(e)
    pd.DataFrame([{"cluster": c, "time": datenum_to_datetime(np.mean([e["on"] for e in es])),
                   "lat": np.mean([e["lat"] for e in es]), "lon": np.mean([e["lon"] for e in es]),
                   "depth": np.mean([e["depth"] for e in es]), "n_events": len(es),
                   "n_events_cc": nev["cc"].get(c, 0), "n_events_ml": nev["ml"].get(c, 0)}
                  for c, es in sorted(byc.items())]).to_csv(out / "clusters.csv", index=False)


def merge(out):
    """Stage 3: pair the CC and ML mechanisms of each cluster."""
    mech = {s: read_skhash(out / s / "out.txt") for s in ("cc", "ml")}
    rows = []
    for _, r in pd.read_csv(out / "clusters.csv").iterrows():
        c = int(r.cluster)
        if c not in mech["cc"].index and c not in mech["ml"].index:
            continue
        r = r.to_dict()
        for s in ("cc", "ml"):
            if c in mech[s].index:
                m = mech[s].loc[c]
                r.update({f"{k}_{s}": m[k] for k in ("strike", "dip", "rake", "quality", "num_p_pol",
                                                    "num_sp_ratios", "polarity_misfit", "prob_mech",
                                                    "fault_plane_uncertainty", "sta_distribution_ratio")})
                r[f"fault_type_{s}"] = fault_type(m.strike, m.dip, m.rake)
        if c in mech["cc"].index and c in mech["ml"].index:
            r["kagan"] = kagan_angle(*[(mech[s].loc[c, "strike"], mech[s].loc[c, "dip"], mech[s].loc[c, "rake"])
                                       for s in ("cc", "ml")])
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "mechanisms.csv", index=False)
    k = df["kagan"].dropna()
    print(f"mechanisms: CC {df.strike_cc.notna().sum():,}, ML {df.strike_ml.notna().sum():,}, paired {len(k):,}")
    print(f"Kagan angle: median {k.median():.1f} deg, 25-75% {k.quantile(.25):.1f}-{k.quantile(.75):.1f} deg")
    print("ML quality:", df.quality_ml.value_counts().sort_index().to_dict())
    print(f"ML polarities per mechanism: {int(df.num_p_pol_ml.min())}-{int(df.num_p_pol_ml.max())}, "
          f"median {df.num_p_pol_ml.median():.0f}")
    print("wrote", out / "mechanisms.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters", default=str(ROOT / "02-data" / "raw" / "F_2015Erp_V1V2V3_Po_polished.mat"))
    ap.add_argument("--predictions", default=str(ROOT / "02-data" / "A_wave_2015_2021_AxialPolCap_final_conf80_predictions.mat"))
    ap.add_argument("--geometry", default=str(ROOT / "02-data" / "raw" / "TakeOffAzimuth0203_ext2026.mat"))
    ap.add_argument("--skhash", default=os.environ.get("SKHASH"), required="SKHASH" not in os.environ,
                    help="path to SKHASH.py (or set the SKHASH environment variable)")
    ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--min-confident", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "02-data" / "focal_mechanisms_2015_2021"))
    ap.add_argument("--stage", choices=["all", "inputs", "invert", "merge"], default="all")
    a = ap.parse_args()
    out = Path(a.out)
    if a.stage == "inputs":
        return build_inputs(a, out)
    if a.stage == "all":
        subprocess.run([sys.executable, __file__] + sys.argv[1:] + ["--stage", "inputs"], check=True)
    if a.stage in ("all", "invert"):
        for src in ("cc", "ml"):
            print(f"SKHASH: {src}", flush=True)
            subprocess.run([sys.executable, Path(a.skhash).name, str((out / src / "control_file.txt").resolve())],
                           cwd=Path(a.skhash).parent, check=True, stdout=subprocess.DEVNULL)
    merge(out)


if __name__ == "__main__":
    main()
