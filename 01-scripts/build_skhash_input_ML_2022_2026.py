#!/usr/bin/env python3
"""Build SKHASH (hash3 input_format) composite-mechanism input from AxialPolCap
(ML) polarities and S/P amplitude ratios, 2022-01 to 2026-03.

Same composite convention as the CC catalog: one header per CLUSTER, then every
earthquake's observations with a per-earthquake station letter (AS1A, AS1B ...)
so repeated observations at one physical station are not collapsed. Polarities
are NOT majority-voted. SKHASH traces rays through the 1-D model, so no azimuth
or take-off is supplied.

Composite groups are the CROSS-CORRELATION clusters, joined by EVENT ID; the
three source files number their clusters independently, so cluster id is never
used as a key.

NOTE the SKHASH hash3 readers differ from the modified HASH driver-3 this group
uses, and the files are written to SKHASH's spec:
  phase header : horiz uncert at cols 89-93, vert at 95-99, mag at 140-143,
                 event id at 150-165  (HASH put them at 41-45/47-51/53-56/57-72)
  event end    : a line of four or more spaces, not an empty line
  amp lines    : whitespace-split, noise_p/noise_s/amp_p/amp_s must be tokens
                 5-8, so two filler columns precede them
  stations     : long SCEDC layout, lat 42-50, lon 52-61, elev 63-67, net 91-92

    python 01-scripts/build_skhash_input_ML_2022_2026.py
"""
from __future__ import annotations
import collections, datetime, string
from pathlib import Path
import h5py, numpy as np, scipy.io as sio

GH   = Path.home()/"Documents"/"GitHub"
ML   = GH/"FM4_7OBS"/"02-data"/"F_Cl"/"F_Cl_ML_RT2026.mat"
CC   = GH/"FM4"/"02-data"/"G_final_Po_Clu_COMB_RT2026AGUV2.mat"
GEOM = GH/"FM4"/"02-data"/"TakeOffAzimuth0203_ext2026.mat"
OUTD = GH/"FM_ML"/"02-data"/"skhash_ML_2022_2026"
STA  = ["AS1","AS2","CC1","EC1","EC2","EC3","ID1"]
CHAN = {"AS1":"EHZ","AS2":"EHZ","CC1":"HHZ","EC1":"EHZ","EC2":"HHZ","EC3":"EHZ","ID1":"EHZ"}
NET  = "OO"
SP_COR = {"AS1":0.1761,"AS2":-0.3010,"CC1":-0.6990,"EC1":-0.3010,
          "EC2":-0.6990,"EC3":-0.1249,"ID1":-0.6990}
START = datetime.datetime(2022,1,1).toordinal()+366
MIN_EVENTS = 3
SEH, SEZ, MAG = 0.30, 0.20, 1.00
LETTERS = string.ascii_uppercase

def h5cols(path,names):
    h=h5py.File(path,"r"); g=h["Po_Clu"]; out={}
    for n in names:
        flat=g[n][()].ravel()
        if n.startswith("NSP_"):
            out[n]=[np.atleast_1d(h[r][()] if isinstance(r,h5py.Reference) else r).ravel() for r in flat]
        else:
            a=np.empty(len(flat))
            for i,r in enumerate(flat):
                v=h[r][()] if isinstance(r,h5py.Reference) else r
                a[i]=np.atleast_1d(v).ravel()[0]
            out[n]=a
    return out

def main():
    OUTD.mkdir(parents=True,exist_ok=True)
    pc=np.atleast_1d(sio.loadmat(str(CC),struct_as_record=False,squeeze_me=True)["Po_Clu"])
    g1=lambda e,f: float(np.atleast_1d(getattr(e,f)).ravel()[0])
    cc_cluster={int(g1(e,"ID")):int(g1(e,"Cluster")) for e in pc}

    d=h5cols(ML,["ID","on","lat","lon","depth"]+[f"Po_{s}" for s in STA]+[f"NSP_{s}" for s in STA])
    n=len(d["ID"]); keep=d["on"]>=START
    grp=collections.defaultdict(list)
    for i in range(n):
        if not keep[i]: continue
        c=cc_cluster.get(int(d["ID"][i]))
        if c is not None: grp[c].append(i)
    grp={c:v for c,v in grp.items() if len(v)>=MIN_EVENTS}
    print(f"ML events 2022+: {int(keep.sum()):,};  composites kept: {len(grp):,}")

    ph=[]; amp=[]; mapping=[]; npol=namp=neq=0; maxlet=0
    for i,c in enumerate(sorted(grp),start=1):
        idx=grp[c]; hid=i+99
        t=(datetime.datetime.fromordinal(int(np.mean(d["on"][idx]))-366)
           +datetime.timedelta(days=float(np.mean(d["on"][idx]))%1))
        lat=float(np.mean(d["lat"][idx])); lon=float(np.mean(d["lon"][idx]))
        dep=float(np.mean(d["depth"][idx]))
        latd=int(abs(lat)); latm=(abs(lat)-latd)*60
        lond=int(abs(lon)); lonm=(abs(lon)-lond)*60
        h=[" "]*165
        def put(s,a):
            for k,ch in enumerate(s): h[a+k]=ch
        put("%4d%02d%02d%02d%02d%5.2f"%(t.year,t.month,t.day,t.hour,t.minute,
                                        t.second+t.microsecond/1e6),0)
        put("%2d%1s%5.2f"%(latd,"N" if lat>=0 else "S",latm),17)
        put("%3d%1s%5.2f"%(lond,"W" if lon<0 else "E",lonm),25)
        put("%5.2f"%dep,34)
        put("%5.2f"%SEH,88)          # horz_uncert_km  -> cols 89-93
        put("%5.2f"%SEZ,94)          # vert_uncert_km  -> cols 95-99
        put("%4.2f"%MAG,139)         # magnitude       -> cols 140-143
        put("%16d"%hid,149)          # event id        -> cols 150-165
        ph.append("".join(h).rstrip())
        arows=[]; cpol=0
        for j,e in enumerate(idx,start=1):
            if j>len(LETTERS): break
            maxlet=max(maxlet,j); neq+=1
            for s in STA:
                nm=s+LETTERS[j-1]
                p=d[f"Po_{s}"][e]
                if p!=0:
                    # cols: [0:4] sta, [5:7] net, [9:12] chan, [13] onset, [15] polarity
                    ph.append("%-4s %-2s  %-3s %1s %1s"%(nm,NET,CHAN[s],"I","U" if p>0 else "D"))
                    cpol+=1; npol+=1
                v=d[f"NSP_{s}"][e]
                if v.size>3:    qn_p,qn_s,ap,asx = v[1],v[0],v[3],v[2]
                elif v.size==3: qn_p,qn_s,ap,asx = v[0],v[0],v[2],v[1]
                else: continue
                if not (np.isfinite(ap) and np.isfinite(asx)) or ap==0: continue
                # tokens 0,1,2 then two fillers, then noise_p,noise_s,amp_p,amp_s at 5-8
                arows.append("%-4s %-3s %-2s %8.3f %8.3f %10.3f %10.3f %10.3f %10.3f"%
                             (nm,CHAN[s],NET,0.0,0.0,qn_p,qn_s,ap,asx))
                namp+=1
        ph.append("    ")            # four spaces terminates the event
        amp.append("%-16s%6d"%(hid,len(arows))); amp.extend(arows)
        mapping.append((hid,c,len(idx),cpol))

    (OUTD/"phase.dat").write_text("\n".join(ph)+"\n")
    (OUTD/"amp.dat").write_text("\n".join(amp)+"\n")
    with open(OUTD/"cluster_id_map.csv","w") as f:
        f.write("skhash_event_id,cc_cluster,n_events,n_polarities\n")
        for r in mapping: f.write(",".join(str(x) for x in r)+"\n")

    G=sio.loadmat(str(GEOM),struct_as_record=False,squeeze_me=True)
    sla=np.asarray(G["staLat"]).ravel(); slo=np.asarray(G["staLon"]).ravel()
    st=[]; cor=[]
    for k,s in enumerate(STA):
        for L in LETTERS:
            row=[" "]*92
            def p2(x,a):
                for q,ch in enumerate(x): row[a+q]=ch
            p2("%-4s"%(s+L),0); p2("%-3s"%CHAN[s],5)
            p2("AXIAL SEAMOUNT OBS",10)
            p2("%9.5f"%sla[k],41); p2("%10.5f"%slo[k],51); p2("%5d"%0,62)
            p2("1997/09/19",68); p2("3000/01/01",79); p2("%-2s"%NET,90)
            st.append("".join(row).rstrip())
            cor.append("%-4s %-3s %-2s %7.4f"%(s+L,CHAN[s],NET,SP_COR[s]))
    (OUTD/"stations.txt").write_text("\n".join(sorted(st))+"\n")
    (OUTD/"statcor.txt").write_text("\n".join(sorted(cor))+"\n")
    (OUTD/"reverse.txt").write_text("")
    prof=np.asarray(G["avgProfile"])
    (OUTD/"vmodel.txt").write_text("# Depth (km), Vp (km/s)\n"
                                   +"\n".join(f"{z:.3f}, {vp:.4f}" for z,vp in prof)+"\n")
    (OUTD/"control_file.txt").write_text("\n".join([
        "## SKHASH driver3 control file - AxialPolCap composites, 2022-01 to 2026-03","",
        "$input_format","hash3","",
        "$stfile",f"{OUTD}/stations.txt","",
        "$plfile",f"{OUTD}/reverse.txt","",
        "$corfile",f"{OUTD}/statcor.txt","",
        "$ampfile",f"{OUTD}/amp.dat","",
        "$fpfile",f"{OUTD}/phase.dat","",
        "$outfile1",f"{OUTD}/out.txt","",
        "$outfile2",f"{OUTD}/out2.txt","",
        "$vmodel_paths",f"{OUTD}/vmodel.txt","",
        "$npolmin","8","","$dang","5","","$nmc","30","","$maxout","500","",
        "$ratmin","3","","$badfrac","0.1","","$qbadfrac","0.3","",
        "$delmax","25","","$cangle","45","","$prob_max","0.2",""])+"\n")

    print(f"phase.dat  : {len(grp):,} composites, {neq:,} earthquakes, {npol:,} polarities")
    print(f"amp.dat    : {namp:,} S/P observations")
    print(f"stations   : {len(st)} (7 x A-Z), letters used A..{LETTERS[maxlet-1]}")
    print(f"Saved to {OUTD}")

if __name__=="__main__":
    main()
