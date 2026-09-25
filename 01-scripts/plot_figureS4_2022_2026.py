#!/usr/bin/env python3
"""Figure S4 equivalent for the 2022 onward catalog: spatial pattern of
AxialPolCap-CC polarity disagreement.

Unlike the 2015-2021 version, none of these events was used to build the
training set, so this is a fully out-of-sample comparison.
"""
from __future__ import annotations
import datetime, importlib.util, os, sys
from pathlib import Path
if "MPLCONFIGDIR" not in os.environ: os.environ["MPLCONFIGDIR"]="/tmp/mplconfig_fm5_ml"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from scipy.io import loadmat

REPO = Path(__file__).resolve().parents[1]
MAT  = REPO/"01-scripts"/"FM_ML_bk"/"02-data"/"A_wave_2022_2025CC_with_predictions_Final_confidiencev2.mat"
OUT  = REPO/"03-figs"/"FigureS4_2022_2026_python.png"
STATIONS=["AS1","AS2","CC1","EC1","EC2","EC3","ID1"]
GRID_M=200.0; MIN_PER_CELL=100; CONF_MIN=0.95; RATIO_MAX=50.0
START=datetime.datetime(2022,1,1).toordinal()+366

def helpers():
    spec=importlib.util.spec_from_file_location("mmf", REPO/"01-scripts"/"make_manuscript_figures.py")
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
    return mod

def main():
    fh=helpers(); fh.apply_manuscript_style()
    F=np.atleast_1d(loadmat(str(MAT),struct_as_record=False,squeeze_me=True)["Felix"])
    valid={s:[] for s in STATIONS}; conf={s:[] for s in STATIONS}; cnt={s:[0,0] for s in STATIONS}
    for e in F:
        if float(np.atleast_1d(e.on).ravel()[0])<START: continue
        lon=float(np.atleast_1d(e.lon).ravel()[0]); lat=float(np.atleast_1d(e.lat).ravel()[0])
        if not (np.isfinite(lon) and np.isfinite(lat)): continue
        for s in STATIONS:
            po=getattr(e,f"Po_{s}",None)
            if po is None: continue
            po=np.atleast_1d(po).ravel()
            if po.size<3 or po[0]==0 or po[1]==0 or po[2]<CONF_MIN: continue
            valid[s].append((lon,lat)); cnt[s][0]+=1
            if po[0]!=po[1]: conf[s].append((lon,lat)); cnt[s][1]+=1
    print("2022-01-01 onward, confidence >= %.2f" % CONF_MIN)
    for s in STATIONS:
        n,c=cnt[s]; print(f"  AX{s}: n={n:,}  disagree={c:,} ({100*c/n:.2f}%)")
    tn=sum(v[0] for v in cnt.values()); tc=sum(v[1] for v in cnt.values())
    print(f"  TOTAL: n={tn:,}  disagree {100*tc/tn:.2f}%  -> agreement {100*(1-tc/tn):.2f}%")

    lon_lim=[-130.031,-129.97]; lat_lim=[45.92,45.972]
    mlat=np.deg2rad(np.mean(lat_lim)); aspect=1.0/np.cos(mlat)
    dlat=GRID_M/111190.0; dlon=GRID_M/(111190.0*np.cos(mlat))
    le=np.arange(lon_lim[0],lon_lim[1]+dlon,dlon); la=np.arange(lat_lim[0],lat_lim[1]+dlat,dlat)
    norm=Normalize(0,RATIO_MAX); cmap=plt.get_cmap("YlOrRd")
    mappable=plt.cm.ScalarMappable(norm=norm,cmap=cmap)

    fig,axes=plt.subplots(2,4,figsize=(fh.FIG_WIDTH_WIDE_IN,5.4),constrained_layout=True)
    kept=0
    for i,s in enumerate(STATIONS):
        ax=axes.ravel()[i]
        V=np.array(valid[s],float); C=np.array(conf[s],float) if conf[s] else np.empty((0,2))
        tg,_,_=np.histogram2d(V[:,0],V[:,1],bins=[le,la])
        cg,_,_=np.histogram2d(C[:,0],C[:,1],bins=[le,la]) if C.size else (np.zeros_like(tg),0,0)
        for ix in range(tg.shape[0]):
            for iy in range(tg.shape[1]):
                if tg[ix,iy]<MIN_PER_CELL: continue
                kept+=1
                r=100.0*cg[ix,iy]/tg[ix,iy]
                ax.fill([le[ix],le[ix+1],le[ix+1],le[ix]],[la[iy],la[iy],la[iy+1],la[iy+1]],
                        facecolor=cmap(norm(r)),edgecolor="none",zorder=1)
        ax.plot(fh.CALDERA_RIM[:,0],fh.CALDERA_RIM[:,1],"k-",linewidth=1.0,zorder=3)
        if s in fh.STATION_COORDS:
            x,y=fh.STATION_COORDS[s]
            ax.plot(x,y,"s",color="k",markersize=4.5,markeredgecolor="w",markeredgewidth=0.6,zorder=4)
        n,c=cnt[s]
        ax.set_title(f"{fh.STATIONS_AX[i]}  ({100*c/n:.1f}%)",fontsize=9,pad=3)
        ax.set_xlim(lon_lim); ax.set_ylim(lat_lim); ax.set_aspect(aspect)
        ax.set_xticks([-130.02,-130.00,-129.98]); ax.set_yticks([45.93,45.94,45.95,45.96,45.97])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=7.5)
        if i%4: ax.set_yticklabels([])
        else: ax.set_ylabel("Latitude",fontsize=8.5)
        if i<3: ax.set_xticklabels([])
        else: ax.set_xlabel("Longitude",fontsize=8.5)
    cax=axes.ravel()[7]; cax.axis("off")
    cb=fig.colorbar(mappable,ax=cax,fraction=0.55,aspect=18,extend="max")
    cb.set_label("ML–CC polarity disagreement (%)",fontsize=8.5); cb.ax.tick_params(labelsize=7.5)
    print(f"  grid cells drawn: {kept}")
    OUT.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT,dpi=300,bbox_inches="tight"); plt.close(fig)
    print("Saved:",OUT)

if __name__=="__main__": main()
