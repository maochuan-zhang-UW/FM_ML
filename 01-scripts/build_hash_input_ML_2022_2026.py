#!/usr/bin/env python3
"""Build HASH driver-3 (1-D) composite-mechanism input from AxialPolCap (ML)
polarities, 2022-01 to 2026-03.

Uses both first-motion polarities and S/P amplitude ratios, as the CC catalog
does, so hash_driver3 (four outputs) is the driver rather than the polarity-only
build.

Follows the convention already used for the CC catalog: one event header per
CLUSTER, then every earthquake's polarities as separate observations, with the
station renamed by a per-earthquake letter (AS1A, AS1B, AS1C ...) so HASH does
not collapse repeated observations at one physical station. Polarities are NOT
majority-voted. HASH traces rays through the 1-D model, so no azimuth or
take-off angle is supplied.

Composite groups are the CROSS-CORRELATION clusters, joined by EVENT ID. The
three source files number their clusters independently and the numberings do
not correspond, so cluster id is never used as a key. Identical groups mean the
ML and CC mechanisms differ only in polarity.

Output in 02-data/hash_ML_2022_2026/:
    phase.dat      composite polarity input (fpfile)
    station.dat    AS1A..ID1Z at the true station sites (stfile)
    reverse.dat    empty (plfile)
    Acor.dat       per-station S/P ratio corrections (corfile)
    amp.dat        S and P amplitudes + noise (ampfile)
    velmod.dat     1-D P velocity model
    hash.input     control file
    cluster_id_map.csv   HASH event id <-> CC cluster id

    python 01-scripts/build_hash_input_ML_2022_2026.py
"""
from __future__ import annotations
import collections, datetime, string
from pathlib import Path
import h5py, numpy as np, scipy.io as sio

GH   = Path.home()/"Documents"/"GitHub"
ML   = GH/"FM4_7OBS"/"02-data"/"F_Cl"/"F_Cl_ML_RT2026.mat"
CC   = GH/"FM4"/"02-data"/"G_final_Po_Clu_COMB_RT2026AGUV2.mat"
GEOM = GH/"FM4"/"02-data"/"TakeOffAzimuth0203_ext2026.mat"   # stations + velocity model only
OUTD = GH/"FM_ML"/"02-data"/"hash_ML_2022_2026"
STA  = ["AS1","AS2","CC1","EC1","EC2","EC3","ID1"]
CHAN = {"AS1":"EHZ","AS2":"EHZ","CC1":"HHZ","EC1":"EHZ","EC2":"HHZ","EC3":"EHZ","ID1":"EHZ"}
NET  = "OO"
START = datetime.datetime(2022,1,1).toordinal()+366
MIN_EVENTS = 3
SP_COR = {"AS1":0.1761,"AS2":-0.3010,"CC1":-0.6990,"EC1":-0.3010,
          "EC2":-0.6990,"EC3":-0.1249,"ID1":-0.6990}   # from Modified_Acor.dat
SEH, SEZ, MAG = 0.30, 0.20, 1.00      # horiz/vert location uncertainty (km), nominal magnitude
LETTERS = string.ascii_uppercase

def h5cols(path, names):
    h=h5py.File(path,"r"); g=h["Po_Clu"]; out={}
    for n in names:
        flat=g[n][()].ravel(); a=np.empty(len(flat))
        for i,r in enumerate(flat):
            v=h[r][()] if isinstance(r,h5py.Reference) else r
            a[i]=np.atleast_1d(v).ravel()[0]
        out[n]=a
    return out

def h5nsp(path, names):
    """NSP_* are per-station [noise, S amplitude, P amplitude] vectors."""
    h=h5py.File(path,"r"); g=h["Po_Clu"]; out={}
    for n in names:
        out[n]=[np.atleast_1d(h[r][()] if isinstance(r,h5py.Reference) else r).ravel()
                for r in g[n][()].ravel()]
    return out

def dnum_to_dt(x):
    return datetime.datetime.fromordinal(int(x)-366)+datetime.timedelta(days=float(x)%1)

def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    pc=np.atleast_1d(sio.loadmat(str(CC),struct_as_record=False,squeeze_me=True)["Po_Clu"])
    g1=lambda e,f: float(np.atleast_1d(getattr(e,f)).ravel()[0])
    cc_cluster={int(g1(e,"ID")):int(g1(e,"Cluster")) for e in pc}

    d=h5cols(ML,["ID","on","lat","lon","depth"]+[f"Po_{s}" for s in STA])
    nsp=h5nsp(ML,[f"NSP_{s}" for s in STA])
    n=len(d["ID"]); keep=d["on"]>=START
    print(f"ML events {n:,}; 2022-01-01 onward {int(keep.sum()):,}")

    grp=collections.defaultdict(list)
    for i in range(n):
        if not keep[i]: continue
        c=cc_cluster.get(int(d["ID"][i]))
        if c is not None: grp[c].append(i)
    print(f"CC clusters populated by ML events: {len(grp):,}")
    grp={c:v for c,v in grp.items() if len(v)>=MIN_EVENTS}
    print(f"   with >= {MIN_EVENTS} events: {len(grp):,}")

    lines=[]; amp_blocks=[]; nobs=neq=namp=0; maxlet=0; mapping=[]
    for i,c in enumerate(sorted(grp), start=1):
        idx=grp[c]; hid=i+99
        t=dnum_to_dt(float(np.mean(d["on"][idx])))
        lat=float(np.mean(d["lat"][idx])); lon=float(np.mean(d["lon"][idx]))
        dep=float(np.mean(d["depth"][idx]))
        latd=int(abs(lat)); latm=(abs(lat)-latd)*60.0
        lond=int(abs(lon)); lonm=(abs(lon)-lond)*60.0
        # format 125: i4,4i2,f5.2,i2,a1,f5.2,i3,a1,f5.2,f5.2,1x,f5.2,1x,f5.2,1x,f4.2,i16
        lines.append("%4d%2d%2d%2d%2d%5.2f%2d%1s%5.2f%3d%1s%5.2f%5.2f %5.2f %5.2f %4.2f%16d" %
                     (t.year,t.month,t.day,t.hour,t.minute,t.second+t.microsecond/1e6,
                      latd,"N" if lat>=0 else "S",latm,
                      lond,"W" if lon<0 else "E",lonm,
                      dep,SEH,SEZ,MAG,hid))
        cobs=0; arows=[]
        for j,e in enumerate(idx, start=1):
            if j>len(LETTERS): break
            maxlet=max(maxlet,j); neq+=1
            for s in STA:
                p=d[f"Po_{s}"][e]
                if p!=0:
                    # format 135: a4,1x,a2,2x,a3,1x,a1,1x,a1
                    lines.append("%-4s %-2s  %-3s %1s %1s" %
                                 (s+LETTERS[j-1], NET, CHAN[s], "I", "U" if p>0 else "D"))
                    cobs+=1; nobs+=1
                # S/P amplitude ratio: NSP = [noise, S amp, P amp]
                v=nsp[f"NSP_{s}"][e]
                if v.size>3:   qns1,qns2,qsamp,qpamp = v[1],v[0],v[2],v[3]
                elif v.size==3: qns1,qns2,qsamp,qpamp = v[0],v[0],v[1],v[2]
                else: continue
                if not (np.isfinite(qpamp) and np.isfinite(qsamp)) or qpamp==0: continue
                # format 35: a4,1x,a3,1x,a2,17x,f10.3,1x,f10.3,1x,f10.3,1x,f10.3
                arows.append("%-4s %-3s %-2s%17s%10.3f %10.3f %10.3f %10.3f" %
                             (s+LETTERS[j-1], CHAN[s], NET, "", qns1, qns2, qpamp, qsamp))
                namp+=1
        lines.append("")                      # blank line ends the event
        amp_blocks.append((hid,arows))
        mapping.append((hid,c,len(idx),cobs))

    (OUTD/"phase.dat").write_text("\n".join(lines)+"\n")
    with open(OUTD/"cluster_id_map.csv","w") as f:
        f.write("hash_event_id,cc_cluster,n_events,n_polarities\n")
        for r in mapping: f.write(",".join(str(x) for x in r)+"\n")

    G=sio.loadmat(str(GEOM),struct_as_record=False,squeeze_me=True)
    sla=np.asarray(G["staLat"]).ravel(); slo=np.asarray(G["staLon"]).ravel()
    st=[]
    for k,s in enumerate(STA):
        for L in LETTERS:
            st.append("%-4s %-3s %9.5f %10.5f %5d %s" %
                      (s+L, CHAN[s], sla[k], slo[k], 0, NET))
    (OUTD/"station.dat").write_text("\n".join(sorted(st))+"\n")
    prof=np.asarray(G["avgProfile"])
    (OUTD/"velmod.dat").write_text("\n".join(f"{z}\t{vp:.5f}" for z,vp in prof)+"\n")
    (OUTD/"reverse.dat").write_text("")
    # amp.dat: "<evid> <nin>" then one line per observation
    al=[]
    for hid,arows in amp_blocks:
        al.append("%-8d%d" % (hid, len(arows)))
        al.extend(arows)
    (OUTD/"amp.dat").write_text("\n".join(al)+"\n")
    # Acor.dat: per-station S/P corrections, repeated for every letter.
    # GET_COR returns -999 for a station it cannot find and the observation is
    # then discarded, so every lettered station must be present.
    ac=[]
    for s in STA:
        for L in LETTERS:
            ac.append("%-4s %-3s %-2s %7.4f" % (s+L, CHAN[s], NET, SP_COR[s]))
    (OUTD/"Acor.dat").write_text("\n".join(sorted(ac))+"\n")
    # control file, in hash_driver3 (four-output) read order
    (OUTD/"hash.input").write_text("\n".join([
        "station.dat","reverse.dat","Acor.dat","amp.dat","phase.dat",
        "hashout1.dat","hashout2.dat","hashout3.dat","hashout4.dat",
        "240","120","5","30","300","2","0.0","2","0","0.3000","25","45","0.750000",
        "1","velmod.dat"])+"\n")

    print(f"\nphase.dat : {len(grp):,} composite events, {neq:,} earthquakes, {nobs:,} polarities")
    print(f"amp.dat   : {namp:,} S/P amplitude observations")
    print(f"   station letters used: A..{LETTERS[maxlet-1]}")
    print(f"station.dat: {len(st)} entries (7 x A-Z)")
    print(f"velmod.dat : {len(prof)} layers, {prof[0,0]:.1f}-{prof[-1,0]:.1f} km")
    v=np.array([m[3] for m in mapping])
    print(f"polarities per composite: median {np.median(v):.0f}, min {v.min()}, max {v.max()}")
    print(f"\nSaved to {OUTD}")

if __name__=="__main__":
    main()
