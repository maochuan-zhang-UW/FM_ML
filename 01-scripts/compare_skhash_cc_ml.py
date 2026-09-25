#!/usr/bin/env python3
"""Compare CC-polarity and ML-polarity SKHASH composite mechanisms, 2022-2026.

Reads the out.txt written by two SKHASH runs that share an event set, composite
grouping and S/P data, and differ only in the polarity used, then reports the
Kagan angle between the paired mechanisms.

  python 01-scripts/compare_skhash_cc_ml.py --tag sp
  python 01-scripts/compare_skhash_cc_ml.py --tag pol
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path.home()/"Documents"/"GitHub"/"FM_ML"/"02-data"

# Rotations by 180 deg about the T, B and P axes leave a double couple
# unchanged, so the Kagan angle is the smallest of the four equivalent rotations.
_SYM = [np.diag([1., 1., 1.]), np.diag([1., -1., -1.]),
        np.diag([-1., 1., -1.]), np.diag([-1., -1., 1.])]


def sdr_to_frame(strike, dip, rake):
    """Orthonormal frame [T, B, P] as columns, in a North-East-Down system."""
    s, d, r = np.radians(strike), np.radians(dip), np.radians(rake)
    n = np.array([-np.sin(d) * np.sin(s), np.sin(d) * np.cos(s), -np.cos(d)])
    u = np.array([np.cos(r) * np.cos(s) + np.cos(d) * np.sin(r) * np.sin(s),
                  np.cos(r) * np.sin(s) - np.cos(d) * np.sin(r) * np.cos(s),
                  -np.sin(r) * np.sin(d)])
    t = (n + u); t /= np.linalg.norm(t)
    p = (n - u); p /= np.linalg.norm(p)
    b = np.cross(t, p); b /= np.linalg.norm(b)
    U = np.column_stack([t, b, p])
    if np.linalg.det(U) < 0:
        U[:, 1] *= -1.0
    return U


def kagan_angle(sdr1, sdr2):
    U1, U2 = sdr_to_frame(*sdr1), sdr_to_frame(*sdr2)
    best = 180.0
    for S in _SYM:
        R = U2 @ S @ U1.T
        c = (np.trace(R) - 1.0) / 2.0
        best = min(best, float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
    return best


def _selftest():
    assert kagan_angle((30, 60, 90), (30, 60, 90)) < 1e-6
    # A pure double couple and its auxiliary plane are the same mechanism.
    assert kagan_angle((0, 90, 0), (90, 90, 180)) < 1e-6
    # 90 deg rotation about the null axis of a vertical strike-slip.
    assert abs(kagan_angle((0, 90, 0), (45, 90, 0)) - 45.0) < 1e-6


def plunge(vec):
    """Plunge in degrees of a unit vector in a North-East-Down frame."""
    return np.degrees(np.arcsin(np.clip(abs(vec[2]), 0, 1)))


def fault_type(strike, dip, rake):
    """Classify from P/T/B plunges, following the usual Zoback-style scheme."""
    U = sdr_to_frame(strike, dip, rake)
    pt, pb, pp = plunge(U[:, 0]), plunge(U[:, 1]), plunge(U[:, 2])
    if pp >= 52:
        return "normal"
    if pt >= 52:
        return "reverse"
    if pb >= 52:
        return "strike-slip"
    return "normal" if pp > pt else "reverse"


def load(tag, pol):
    """Preferred mechanism per composite.

    SKHASH writes one row per accepted solution, so an event whose polarities
    admit more than one mechanism family appears several times.  The rows are
    not ordered by preference within an event, so the solution with the highest
    prob_mech is taken rather than the first one listed.
    """
    f = DATA/f"skhash_{pol}_{tag}_2022_2026"/"out.txt"
    df = pd.read_csv(f)
    n_rows, n_ev = len(df), df.event_id.nunique()
    df = df.loc[df.groupby("event_id")["prob_mech"].idxmax()]
    print(f"  {pol}: {n_rows:,} solution rows -> {n_ev:,} composites "
          f"({n_rows - n_ev:,} alternate solutions dropped)")
    return df.set_index("event_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=["sp", "pol"], default="sp")
    ap.add_argument("--quality", default=None, help="keep only these qualities, e.g. AB")
    args = ap.parse_args()
    _selftest()

    cc, ml = load(args.tag, "cc"), load(args.tag, "ml")
    print(f"CC mechanisms: {len(cc):,}   ML mechanisms: {len(ml):,}")
    ids = cc.index.intersection(ml.index)
    print(f"paired composites: {len(ids):,}")

    # The CSV always holds the complete pairing, so downstream figures can filter
    # for themselves; --quality narrows only what is printed here.
    all_ids = ids
    if args.quality:
        keep = [i for i in ids
                if cc.loc[i, "quality"] in args.quality and ml.loc[i, "quality"] in args.quality]
        print(f"after quality filter [{args.quality}]: {len(keep):,}")
        ids = pd.Index(keep)

    k = np.array([kagan_angle(
        (cc.loc[i, "strike"], cc.loc[i, "dip"], cc.loc[i, "rake"]),
        (ml.loc[i, "strike"], ml.loc[i, "dip"], ml.loc[i, "rake"])) for i in ids])

    print(f"\nKagan angle  median {np.median(k):6.1f} deg   mean {k.mean():6.1f} deg")
    for q in (25, 50, 75, 90):
        print(f"   {q}th percentile: {np.percentile(k, q):5.1f}")
    for thr in (10, 20, 30, 45):
        print(f"   <= {thr:2d} deg: {100.0 * (k <= thr).mean():5.1f}%")

    print("\nquality distribution:")
    for src, df in (("CC", cc.loc[ids]), ("ML", ml.loc[ids])):
        vc = df["quality"].value_counts().sort_index()
        print("   " + src + ": " + "  ".join(f"{q}={n}" for q, n in vc.items()))

    print("\nfault-type agreement:")
    ft_cc = [fault_type(cc.loc[i, "strike"], cc.loc[i, "dip"], cc.loc[i, "rake"]) for i in ids]
    ft_ml = [fault_type(ml.loc[i, "strike"], ml.loc[i, "dip"], ml.loc[i, "rake"]) for i in ids]
    same = sum(a == b for a, b in zip(ft_cc, ft_ml))
    print(f"   {same:,}/{len(ids):,} = {100.0 * same / max(len(ids), 1):.1f}%")
    for t in ("normal", "reverse", "strike-slip"):
        print(f"   {t:12s} CC {ft_cc.count(t):5d}   ML {ft_ml.count(t):5d}")

    # Recompute over the full pairing for the CSV.
    ids = all_ids
    k = np.array([kagan_angle(
        (cc.loc[i, "strike"], cc.loc[i, "dip"], cc.loc[i, "rake"]),
        (ml.loc[i, "strike"], ml.loc[i, "dip"], ml.loc[i, "rake"])) for i in ids])
    ft_cc_all = [fault_type(cc.loc[i, "strike"], cc.loc[i, "dip"], cc.loc[i, "rake"]) for i in ids]
    ft_ml_all = [fault_type(ml.loc[i, "strike"], ml.loc[i, "dip"], ml.loc[i, "rake"]) for i in ids]
    out = DATA/f"skhash_kagan_{args.tag}_2022_2026.csv"
    pd.DataFrame({"event_id": ids, "kagan_deg": k,
                  "cc_strike": cc.loc[ids, "strike"].values,
                  "cc_dip": cc.loc[ids, "dip"].values,
                  "cc_rake": cc.loc[ids, "rake"].values,
                  "cc_quality": cc.loc[ids, "quality"].values,
                  "cc_plane_uncert": cc.loc[ids, "fault_plane_uncertainty"].values,
                  "ml_plane_uncert": ml.loc[ids, "fault_plane_uncertainty"].values,
                  "cc_num_p_pol": cc.loc[ids, "num_p_pol"].values,
                  "ml_num_p_pol": ml.loc[ids, "num_p_pol"].values,
                  "ml_strike": ml.loc[ids, "strike"].values,
                  "ml_dip": ml.loc[ids, "dip"].values,
                  "ml_rake": ml.loc[ids, "rake"].values,
                  "ml_quality": ml.loc[ids, "quality"].values,
                  "lat": cc.loc[ids, "origin_lat"].values,
                  "lon": cc.loc[ids, "origin_lon"].values,
                  "depth_km": cc.loc[ids, "origin_depth_km"].values,
                  "time": cc.loc[ids, "time"].values,
                  "cc_fault_type": ft_cc_all,
                  "ml_fault_type": ft_ml_all}).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
