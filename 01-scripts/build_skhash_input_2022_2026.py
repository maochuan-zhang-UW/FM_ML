#!/usr/bin/env python3
"""Build SKHASH (hash3 input_format) composite-mechanism input for 2022-01 to
2026-03, from either the cross-correlation or the AxialPolCap (ML) polarities.

This generalises build_skhash_input_ML_2022_2026.py so the two catalogs can be
compared fairly.  The CC and ML runs are built from the SAME event set, the
SAME composite grouping, the SAME S/P amplitudes and the SAME station geometry;
the ONLY thing that differs between them is the polarity value.  Any difference
in the resulting mechanisms is therefore attributable to the polarity method,
which is the comparison the paper makes.

Earlier numbers compared a fresh 1-D run against a pre-existing CC mechanism
catalog computed with different settings, so the Kagan angles mixed the polarity
difference together with the difference in inversion setup.

  python 01-scripts/build_skhash_input_2022_2026.py --polarity cc
  python 01-scripts/build_skhash_input_2022_2026.py --polarity ml
  python 01-scripts/build_skhash_input_2022_2026.py --polarity ml --no-sp

--no-sp omits the amplitude file, giving a polarity-only inversion.  HASH's
predicted S/P ratio carries a factor 4.9 calibrated for land stations with a
free surface, which is not appropriate for seafloor OBS, so the polarity-only
run is the cleaner comparison as well as a diagnostic.
"""
from __future__ import annotations
import argparse, collections, datetime, string
from pathlib import Path
import h5py, numpy as np, scipy.io as sio

GH   = Path.home()/"Documents"/"GitHub"
ML   = GH/"FM4_7OBS"/"02-data"/"F_Cl"/"F_Cl_ML_RT2026.mat"
CC   = GH/"FM4"/"02-data"/"G_final_Po_Clu_COMB_RT2026AGUV2.mat"
GEOM = GH/"FM4"/"02-data"/"TakeOffAzimuth0203_ext2026.mat"
STA  = ["AS1","AS2","CC1","EC1","EC2","EC3","ID1"]
CHAN = {"AS1":"EHZ","AS2":"EHZ","CC1":"HHZ","EC1":"EHZ","EC2":"HHZ","EC3":"EHZ","ID1":"EHZ"}
NET  = "OO"
SP_COR = {"AS1":0.1761,"AS2":-0.3010,"CC1":-0.6990,"EC1":-0.3010,
          "EC2":-0.6990,"EC3":-0.1249,"ID1":-0.6990}
START = datetime.datetime(2022,1,1).toordinal()+366
MIN_EVENTS = 3
SEH, SEZ, MAG = 0.30, 0.20, 1.00
LETTERS = string.ascii_uppercase


def h5cols(path, names):
    h = h5py.File(path, "r"); g = h["Po_Clu"]; out = {}
    for n in names:
        flat = g[n][()].ravel()
        if n.startswith("NSP_"):
            out[n] = [np.atleast_1d(h[r][()] if isinstance(r, h5py.Reference) else r).ravel()
                      for r in flat]
        else:
            a = np.empty(len(flat))
            for i, r in enumerate(flat):
                v = h[r][()] if isinstance(r, h5py.Reference) else r
                a[i] = np.atleast_1d(v).ravel()[0]
            out[n] = a
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--polarity", choices=["cc", "ml"], required=True)
    ap.add_argument("--no-sp", action="store_true", help="omit S/P ratios")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    tag = args.polarity + ("_pol" if args.no_sp else "_sp")
    OUTD = Path(args.outdir) if args.outdir else GH/"FM_ML"/"02-data"/f"skhash_{tag}_2022_2026"
    OUTD.mkdir(parents=True, exist_ok=True)

    # ---- CC file: cluster membership and CC polarities (scipy-readable v7) ----
    pc = np.atleast_1d(sio.loadmat(str(CC), struct_as_record=False, squeeze_me=True)["Po_Clu"])
    g1 = lambda e, f: float(np.atleast_1d(getattr(e, f)).ravel()[0])
    cc_cluster = {}
    cc_pol = {}
    for e in pc:
        eid = int(g1(e, "ID"))
        cc_cluster[eid] = int(g1(e, "Cluster"))
        cc_pol[eid] = {s: g1(e, "Po_" + s) for s in STA}

    # ---- ML file: ML polarities, locations, and the S/P amplitudes ----
    d = h5cols(ML, ["ID", "on", "lat", "lon", "depth"]
               + [f"Po_{s}" for s in STA] + [f"NSP_{s}" for s in STA])
    n = len(d["ID"]); keep = d["on"] >= START

    # Same event set for both runs: 2022+, present in both files, cluster known.
    grp = collections.defaultdict(list)
    for i in range(n):
        if not keep[i]:
            continue
        eid = int(d["ID"][i])
        if eid in cc_cluster and eid in cc_pol:
            grp[cc_cluster[eid]].append(i)
    grp = {c: v for c, v in grp.items() if len(v) >= MIN_EVENTS}
    print(f"[{tag}] events 2022+ in both files: "
          f"{sum(len(v) for v in grp.values()):,};  composites: {len(grp):,}")

    def polarity(i, s):
        return cc_pol[int(d["ID"][i])][s] if args.polarity == "cc" else d[f"Po_{s}"][i]

    ph = []; amp = []; mapping = []; npol = namp = neq = 0; maxlet = 0
    for i_c, c in enumerate(sorted(grp), start=1):
        idx = grp[c]; hid = i_c + 99
        t = (datetime.datetime.fromordinal(int(np.mean(d["on"][idx])) - 366)
             + datetime.timedelta(days=float(np.mean(d["on"][idx])) % 1))
        lat = float(np.mean(d["lat"][idx])); lon = float(np.mean(d["lon"][idx]))
        dep = float(np.mean(d["depth"][idx]))
        latd = int(abs(lat)); latm = (abs(lat) - latd) * 60
        lond = int(abs(lon)); lonm = (abs(lon) - lond) * 60
        h = [" "] * 165

        def put(s, a):
            for k, ch in enumerate(s):
                h[a + k] = ch
        put("%4d%02d%02d%02d%02d%5.2f" % (t.year, t.month, t.day, t.hour, t.minute,
                                          t.second + t.microsecond / 1e6), 0)
        put("%2d%1s%5.2f" % (latd, "N" if lat >= 0 else "S", latm), 17)
        put("%3d%1s%5.2f" % (lond, "W" if lon < 0 else "E", lonm), 25)
        put("%5.2f" % dep, 34)
        put("%5.2f" % SEH, 88)
        put("%5.2f" % SEZ, 94)
        put("%4.2f" % MAG, 139)
        put("%16d" % hid, 149)
        ph.append("".join(h).rstrip())

        arows = []; cpol = 0
        for j, e in enumerate(idx, start=1):
            if j > len(LETTERS):
                break
            maxlet = max(maxlet, j); neq += 1
            for s in STA:
                nm = s + LETTERS[j - 1]
                p = polarity(e, s)
                if p != 0:
                    ph.append("%-4s %-2s  %-3s %1s %1s" % (nm, NET, CHAN[s], "I",
                                                           "U" if p > 0 else "D"))
                    cpol += 1; npol += 1
                if args.no_sp:
                    continue
                v = d[f"NSP_{s}"][e]
                # NSP_STATION = [noise_amp, S_amp, P_amp]  (A_build_RT2026.m:15)
                if v.size > 3:
                    qn_p, qn_s, a_p, a_s = v[1], v[0], v[3], v[2]
                elif v.size == 3:
                    qn_p, qn_s, a_p, a_s = v[0], v[0], v[2], v[1]
                else:
                    continue
                if not (np.isfinite(a_p) and np.isfinite(a_s)) or a_p == 0:
                    continue
                arows.append("%-4s %-3s %-2s %8.3f %8.3f %10.3f %10.3f %10.3f %10.3f"
                             % (nm, CHAN[s], NET, 0.0, 0.0, qn_p, qn_s, a_p, a_s))
                namp += 1
        ph.append("    ")
        if not args.no_sp:
            amp.append("%-16s%6d" % (hid, len(arows))); amp.extend(arows)
        mapping.append((hid, c, len(idx), cpol))

    (OUTD/"phase.dat").write_text("\n".join(ph) + "\n")
    if not args.no_sp:
        (OUTD/"amp.dat").write_text("\n".join(amp) + "\n")
    with open(OUTD/"cluster_id_map.csv", "w") as f:
        f.write("skhash_event_id,cc_cluster,n_events,n_polarities\n")
        for r in mapping:
            f.write(",".join(str(x) for x in r) + "\n")

    G = sio.loadmat(str(GEOM), struct_as_record=False, squeeze_me=True)
    sla = np.asarray(G["staLat"]).ravel(); slo = np.asarray(G["staLon"]).ravel()
    st = []; cor = []
    for k, s in enumerate(STA):
        for L in LETTERS:
            row = [" "] * 92

            def p2(x, a):
                for q, ch in enumerate(x):
                    row[a + q] = ch
            p2("%-4s" % (s + L), 0); p2("%-3s" % CHAN[s], 5)
            p2("AXIAL SEAMOUNT OBS", 10)
            p2("%9.5f" % sla[k], 41); p2("%10.5f" % slo[k], 51); p2("%5d" % 0, 62)
            p2("1997/09/19", 68); p2("3000/01/01", 79); p2("%-2s" % NET, 90)
            st.append("".join(row).rstrip())
            cor.append("%-4s %-3s %-2s %7.4f" % (s + L, CHAN[s], NET, SP_COR[s]))
    (OUTD/"stations.txt").write_text("\n".join(sorted(st)) + "\n")
    (OUTD/"statcor.txt").write_text("\n".join(sorted(cor)) + "\n")
    (OUTD/"reverse.txt").write_text("")
    prof = np.asarray(G["avgProfile"])
    (OUTD/"vmodel.txt").write_text("# Depth (km), Vp (km/s)\n"
                                   + "\n".join(f"{z:.3f}, {vp:.4f}" for z, vp in prof) + "\n")

    ctl = ["## SKHASH driver3 control file - %s composites, 2022-01 to 2026-03" % tag, "",
           "$input_format", "hash3", "",
           "$stfile", f"{OUTD}/stations.txt", "",
           "$plfile", f"{OUTD}/reverse.txt", "",
           "$corfile", f"{OUTD}/statcor.txt", ""]
    if not args.no_sp:
        ctl += ["$ampfile", f"{OUTD}/amp.dat", ""]
    ctl += ["$fpfile", f"{OUTD}/phase.dat", "",
            "$outfile1", f"{OUTD}/out.txt", "",
            "$outfile2", f"{OUTD}/out2.txt", "",
            "$vmodel_paths", f"{OUTD}/vmodel.txt", "",
            # look_dep/look_del must give exactly nd0=14 and nx0=101 table bins
            "$look_dep", "0 13 1", "", "$look_del", "0 100 1", "",
            "$npolmin", "8", "", "$dang", "5", "", "$nmc", "30", "",
            "$maxout", "500", "", "$ratmin", "3", "", "$badfrac", "0.1", "",
            "$qbadfrac", "0.3", "", "$delmax", "24", "", "$cangle", "45", "",
            "$prob_max", "0.2", "", "$max_agap", "240", "", "$max_pgap", "90", ""]
    (OUTD/"control_file.txt").write_text("\n".join(ctl) + "\n")

    print(f"[{tag}] phase.dat : {len(grp):,} composites, {neq:,} earthquakes, {npol:,} polarities")
    if not args.no_sp:
        print(f"[{tag}] amp.dat   : {namp:,} S/P observations")
    print(f"[{tag}] saved to {OUTD}")


if __name__ == "__main__":
    main()
