"""Shared focal-mechanism utilities: P/T/B axes, fault type (Table S4),
Kagan angle, and reading SKHASH output."""
import numpy as np
import pandas as pd

# Rotations by 180 deg about the T, B and P axes leave a double couple
# unchanged, so the Kagan angle is the smallest of the four equivalent rotations.
_SYM = [np.diag([1., 1., 1.]), np.diag([1., -1., -1.]),
        np.diag([-1., 1., -1.]), np.diag([-1., -1., 1.])]


def sdr_to_frame(strike, dip, rake):
    """Orthonormal frame [T, B, P] (columns) in a North-East-Down system."""
    s, d, r = np.radians(strike), np.radians(dip), np.radians(rake)
    n = np.array([-np.sin(d) * np.sin(s), np.sin(d) * np.cos(s), -np.cos(d)])
    u = np.array([np.cos(r) * np.cos(s) + np.cos(d) * np.sin(r) * np.sin(s),
                  np.cos(r) * np.sin(s) - np.cos(d) * np.sin(r) * np.cos(s),
                  -np.sin(r) * np.sin(d)])
    t = (n + u) / np.linalg.norm(n + u)
    p = (n - u) / np.linalg.norm(n - u)
    b = np.cross(t, p)
    U = np.column_stack([t, b / np.linalg.norm(b), p])
    if np.linalg.det(U) < 0:
        U[:, 1] *= -1.0
    return U


def kagan_angle(sdr1, sdr2):
    """Minimum 3-D rotation (degrees) between two double couples (Kagan 1991)."""
    U1, U2 = sdr_to_frame(*sdr1), sdr_to_frame(*sdr2)
    best = 180.0
    for S in _SYM:
        c = (np.trace(U2 @ S @ U1.T) - 1.0) / 2.0
        best = min(best, float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
    return best


def axis_plunges(strike, dip, rake):
    """Plunges (degrees) of the P, T and B axes."""
    U = sdr_to_frame(strike, dip, rake)
    pl = lambda v: float(np.degrees(np.arcsin(np.clip(abs(v[2]), 0, 1))))
    return pl(U[:, 2]), pl(U[:, 0]), pl(U[:, 1])


def fault_type(strike, dip, rake):
    """Table S4 (modified from Zoback 1992); tests applied in order.
    Returns 'N' normal, 'R' reverse, 'S' strike-slip or 'U' unclassified."""
    p, t, b = axis_plunges(strike, dip, rake)
    if p >= 52 and t <= 35:
        return "N"
    if p <= 35 and t >= 52:
        return "R"
    if p <= 40 and b >= 45 and t <= 40:
        return "S"
    if p <= 20 and 40 <= t <= 52:        # oblique reverse
        return "R"
    if 40 <= p <= 52 and t <= 20:        # oblique normal
        return "N"
    return "U"


def read_skhash(path):
    """Preferred mechanism per event from a SKHASH out file.

    SKHASH writes one row per accepted solution; for events with multiple
    solutions the one with the highest mechanism probability is kept.
    """
    df = pd.read_csv(path)
    return df.loc[df.groupby("event_id")["prob_mech"].idxmax()].set_index("event_id")


def _selftest():
    assert kagan_angle((30, 60, 90), (30, 60, 90)) < 1e-6
    assert kagan_angle((0, 90, 0), (90, 90, 180)) < 1e-6        # auxiliary plane
    assert abs(kagan_angle((0, 90, 0), (45, 90, 0)) - 45.0) < 1e-6
    assert fault_type(0, 45, -90) == "N" and fault_type(0, 45, 90) == "R" and fault_type(0, 90, 0) == "S"


_selftest()
