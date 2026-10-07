"""Gauss–Seidel ZIP load flow. Verifier-only; does not import solution/."""
from __future__ import annotations

import math

import numpy as np


def zip_scale(v: float, z: float, i: float, p: float) -> float:
    return z * v * v + i * v + p


def with_scales(branches: list[dict], scales: dict[str, float] | None) -> list[dict]:
    if not scales:
        return [dict(br) for br in branches]
    out = []
    for br in branches:
        row = dict(br)
        fac = float(scales.get(row["id"], 1.0))
        row["r_pu"] = float(row["r_pu"]) * fac
        row["x_pu"] = float(row["x_pu"]) * fac
        out.append(row)
    return out


def build_ybus(n: int, branches: list[dict], open_ids: set[str]) -> np.ndarray:
    y = np.zeros((n, n), dtype=complex)
    for br in branches:
        if br["id"] in open_ids:
            continue
        a, c = br["from"] - 1, br["to"] - 1
        yser = 1.0 / complex(br["r_pu"], br["x_pu"])
        ych = 1j * float(br.get("b_pu", 0.0)) / 2.0
        y[a, a] += yser + ych
        y[c, c] += yser + ych
        y[a, c] -= yser
        y[c, a] -= yser
    return y


def solve_gs(
    n: int,
    slack: int,
    v_slack: float,
    branches: list[dict],
    open_ids: set[str],
    p0: np.ndarray,
    q0: np.ndarray,
    zip_z: np.ndarray,
    zip_i: np.ndarray,
    zip_p: np.ndarray,
    max_iter: int = 400,
    residual: float = 5e-8,
) -> tuple[np.ndarray, np.ndarray, bool]:
    y = build_ybus(n, branches, open_ids)
    v = np.ones(n, dtype=complex)
    v[slack - 1] = v_slack + 0j
    acc = 1.6
    for _ in range(max_iter):
        vmax = 0.0
        for i in range(n):
            if i == slack - 1:
                continue
            vi = v[i]
            mag = abs(vi)
            s_zip = zip_scale(mag, zip_z[i], zip_i[i], zip_p[i])
            s_spec = complex(-p0[i] * s_zip, -q0[i] * s_zip)
            acc_sum = s_spec.conjugate() / vi.conjugate()
            for k in range(n):
                if k == i:
                    continue
                acc_sum -= y[i, k] * v[k]
            vnew = acc_sum / y[i, i]
            vnew = vi + acc * (vnew - vi)
            mag = min(1.2, max(0.7, abs(vnew)))
            ang = math.atan2(vnew.imag, vnew.real)
            vnew = mag * complex(math.cos(ang), math.sin(ang))
            vmax = max(vmax, abs(vnew - vi))
            v[i] = vnew
        if vmax < residual:
            mag = np.abs(v)
            ang = np.angle(v)
            return mag, ang, True
    return np.abs(v), np.angle(v), False
