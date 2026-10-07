"""Heun slip integrator. Same published motor contract as Euler; not imported by solve.py."""
from __future__ import annotations

import math


def te(v: float, s: float, r1: float, x1: float, r2: float, x2: float) -> float:
    s = max(1e-4, min(0.999, s))
    r2os = r2 / s
    return 3.0 * v * v * r2os / ((r1 + r2os) ** 2 + (x1 + x2) ** 2)


def iamp(v: float, s: float, r1: float, x1: float, r2: float, x2: float) -> float:
    s = max(1e-4, min(0.999, s))
    r2os = r2 / s
    return v / math.hypot(r1 + r2os, x1 + x2)


def heun_current(vth: float, tm, h: float, r1, x1, r2, x2, dt: float, s0: float, s_lo: float, s_hi: float):
    s = s0
    out = []
    for tload in tm:
        te1 = te(vth, s, r1, x1, r2, x2)
        sp = min(s_hi, max(s_lo, s + dt / h * (tload - te1)))
        te2 = te(vth, sp, r1, x1, r2, x2)
        s = min(s_hi, max(s_lo, s + dt / h * (tload - 0.5 * (te1 + te2))))
        out.append(iamp(vth, s, r1, x1, r2, x2))
    return out
