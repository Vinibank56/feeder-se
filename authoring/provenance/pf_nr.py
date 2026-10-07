"""Polar Newton–Raphson load flow with ZIP injections. Oracle/generator path."""
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
        i, j = br["from"] - 1, br["to"] - 1
        z = complex(br["r_pu"], br["x_pu"])
        ych = 1j * float(br.get("b_pu", 0.0)) / 2.0
        yser = 1.0 / z
        y[i, i] += yser + ych
        y[j, j] += yser + ych
        y[i, j] -= yser
        y[j, i] -= yser
    return y


def connected(n: int, branches: list[dict], open_ids: set[str], slack: int = 1) -> bool:
    adj = {i: [] for i in range(1, n + 1)}
    for br in branches:
        if br["id"] in open_ids:
            continue
        adj[br["from"]].append(br["to"])
        adj[br["to"]].append(br["from"])
    seen = {slack}
    stack = [slack]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n


def solve_nr(
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
    vmax_iter: int = 40,
    residual: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, bool]:
    y = build_ybus(n, branches, open_ids)
    g, b = y.real, y.imag
    v = np.ones(n)
    th = np.zeros(n)
    v[slack - 1] = v_slack
    pq = [i for i in range(n) if i != slack - 1]
    m = len(pq)
    for _ in range(vmax_iter):
        p_calc = np.zeros(n)
        q_calc = np.zeros(n)
        for i in range(n):
            vi = v[i]
            for k in range(n):
                tik = th[i] - th[k]
                p_calc[i] += vi * v[k] * (g[i, k] * math.cos(tik) + b[i, k] * math.sin(tik))
                q_calc[i] += vi * v[k] * (g[i, k] * math.sin(tik) - b[i, k] * math.cos(tik))
        p_spec = np.array([-p0[i] * zip_scale(v[i], zip_z[i], zip_i[i], zip_p[i]) for i in range(n)])
        q_spec = np.array([-q0[i] * zip_scale(v[i], zip_z[i], zip_i[i], zip_p[i]) for i in range(n)])
        p_spec[slack - 1] = 0.0
        q_spec[slack - 1] = 0.0
        dp = p_spec - p_calc
        dq = q_spec - q_calc
        mis = np.concatenate([dp[pq], dq[pq]])
        if float(np.max(np.abs(mis))) < residual:
            return v, th, True
        jac = np.zeros((2 * m, 2 * m))
        for a, i in enumerate(pq):
            for c, k in enumerate(pq):
                tik = th[i] - th[k]
                if i == k:
                    jac[a, c] = -q_calc[i] - b[i, i] * v[i] * v[i]
                    jac[a, m + c] = p_calc[i] / v[i] + g[i, i] * v[i]
                    jac[m + a, c] = p_calc[i] - g[i, i] * v[i] * v[i]
                    jac[m + a, m + c] = q_calc[i] / v[i] - b[i, i] * v[i]
                else:
                    jac[a, c] = v[i] * v[k] * (g[i, k] * math.sin(tik) - b[i, k] * math.cos(tik))
                    jac[a, m + c] = v[i] * (g[i, k] * math.cos(tik) + b[i, k] * math.sin(tik))
                    jac[m + a, c] = -v[i] * v[k] * (g[i, k] * math.cos(tik) + b[i, k] * math.sin(tik))
                    jac[m + a, m + c] = v[i] * (g[i, k] * math.sin(tik) - b[i, k] * math.cos(tik))
            dz = 2.0 * zip_z[i] * v[i] + zip_i[i]
            jac[a, m + a] += p0[i] * dz
            jac[m + a, m + a] += q0[i] * dz
        try:
            dx = np.linalg.solve(jac, mis)
        except np.linalg.LinAlgError:
            return v, th, False
        for a, i in enumerate(pq):
            th[i] += dx[a]
            v[i] += dx[m + a]
            v[i] = min(1.2, max(0.7, v[i]))
    return v, th, False


def solve_gs_fast(
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
    max_iter: int = 80,
    residual: float = 2e-5,
) -> tuple[np.ndarray, np.ndarray, bool]:
    y = build_ybus(n, branches, open_ids)
    v = np.ones(n, dtype=complex)
    v[slack - 1] = v_slack + 0j
    acc = 1.5
    for _ in range(max_iter):
        vmax = 0.0
        for i in range(n):
            if i == slack - 1:
                continue
            mag = abs(v[i])
            s_zip = zip_z[i] * mag * mag + zip_i[i] * mag + zip_p[i]
            s_spec = complex(-p0[i] * s_zip, -q0[i] * s_zip)
            acc_sum = s_spec.conjugate() / np.conjugate(v[i]) - (y[i] @ v - y[i, i] * v[i])
            if abs(y[i, i]) < 1e-14:
                return np.abs(v), np.angle(v), False
            vnew = acc_sum / y[i, i]
            vnew = v[i] + acc * (vnew - v[i])
            mag = min(1.2, max(0.7, abs(vnew)))
            ang = math.atan2(vnew.imag, vnew.real)
            vnew = mag * complex(math.cos(ang), math.sin(ang))
            vmax = max(vmax, abs(vnew - v[i]))
            v[i] = vnew
        if vmax < residual:
            return np.abs(v), np.angle(v), True
    return np.abs(v), np.angle(v), False
