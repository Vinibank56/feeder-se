#!/usr/bin/env python3
"""Recover open switches, PMU bias, ZIP, withheld voltages, motor start."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from pf_nr import connected, solve_nr


def data_root() -> Path:
    p = Path("/app/data")
    return p if (p / "case.json").exists() else Path(__file__).resolve().parents[1] / "environment" / "data"


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def zip_from_classes(n: int, class_of: dict[int, str], zip_map: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zz = np.zeros(n)
    zi = np.zeros(n)
    zp = np.ones(n)
    for b, cl in class_of.items():
        zz[b - 1], zi[b - 1], zp[b - 1] = zip_map[cl]
    return zz, zi, zp


def fit_zip_pmu(hours, pmu_buses, slack) -> dict[str, tuple[float, float, float]]:
    by = {"res": [], "com": [], "ind": []}
    for rec in hours:
        for b in pmu_buses:
            if b == slack:
                continue
            if b not in rec["inj"]:
                continue
            cl = rec["inj"][b]["class"]
            p0 = float(rec["inj"][b]["p0_mw"])
            pmeas = float(rec["pmu"][b]["p_inj_mw"])
            v = float(rec["pmu"][b]["vmag_pu"])
            if abs(p0) < 1e-6:
                continue
            by[cl].append((v, -pmeas / p0))
    out = {}
    for cl, pts in by.items():
        if len(pts) < 4:
            out[cl] = (0.2, 0.2, 0.6)
            continue
        a = np.array([[v * v - 1.0, v - 1.0] for v, _o in pts], dtype=float)
        y = np.array([o - 1.0 for _v, o in pts], dtype=float)
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        z, i = float(coef[0]), float(coef[1])
        p = 1.0 - z - i
        out[cl] = (z, i, p)
    return out


def motor_te(v: float, s: float, r1, x1, r2, x2) -> float:
    s = max(1e-4, min(0.999, s))
    r2os = r2 / s
    den = (r1 + r2os) ** 2 + (x1 + x2) ** 2
    return 3.0 * v * v * r2os / den


def motor_i(v: float, s: float, r1, x1, r2, x2) -> float:
    s = max(1e-4, min(0.999, s))
    r2os = r2 / s
    return v / math.hypot(r1 + r2os, x1 + x2)


def sim_motor(vth, tm, h, r1, x1, r2, x2, dt, s0, s_lo, s_hi):
    s = s0
    amps = []
    for tload in tm:
        te = motor_te(vth, s, r1, x1, r2, x2)
        s = min(s_hi, max(s_lo, s + dt / h * (tload - te)))
        amps.append(motor_i(vth, s, r1, x1, r2, x2))
    return np.array(amps)


def pf_vmag_mse(n, slack, vslack, branches, opens, zz, zi, zp, hours, pmu_buses, sbase):
    acc = 0.0
    nobs = 0
    for rec in hours:
        p0 = np.zeros(n)
        q0 = np.zeros(n)
        for b, row in rec["inj"].items():
            p0[b - 1] = float(row["p0_mw"]) / sbase
            q0[b - 1] = float(row["q0_mvar"]) / sbase
        v, _th, ok = solve_nr(n, slack, vslack, branches, opens, p0, q0, zz, zi, zp)
        if not ok:
            return 50.0
        for b in pmu_buses:
            acc += (v[b - 1] - float(rec["pmu"][b]["vmag_pu"])) ** 2
            nobs += 1
    return acc / max(nobs, 1)


def greedy_fill(n_open, forced, n, slack, vslack, base_br, switchable, zz, zi, zp, hours, pmu_buses, sbase):
    cur = set(forced)
    while len(cur) < n_open:
        best = None
        for sid in switchable:
            if sid in cur:
                continue
            nxt = cur | {sid}
            if not connected(n, base_br, nxt, slack):
                continue
            err = pf_vmag_mse(n, slack, vslack, base_br, nxt, zz, zi, zp, hours, pmu_buses, sbase)
            if best is None or err < best[0]:
                best = (err, sid)
        if best is None:
            break
        cur.add(best[1])
    return cur


def one_exchange(cur, n, slack, vslack, base_br, switchable, zz, zi, zp, hours, pmu_buses, sbase):
    cur = set(cur)
    cur_e = pf_vmag_mse(n, slack, vslack, base_br, cur, zz, zi, zp, hours, pmu_buses, sbase)
    changed = True
    while changed:
        changed = False
        for leave in list(cur):
            for add in switchable:
                if add in cur:
                    continue
                nxt = (cur - {leave}) | {add}
                if not connected(n, base_br, nxt, slack):
                    continue
                err = pf_vmag_mse(n, slack, vslack, base_br, nxt, zz, zi, zp, hours, pmu_buses, sbase)
                if err < cur_e - 1e-18:
                    cur, cur_e = nxt, err
                    changed = True
                    break
            if changed:
                break
    return cur_e, cur


def search_open(n_open, n, slack, vslack, base_br, switchable, zz, zi, zp, hours, pmu_buses, sbase, as_built):
    starts = []
    if as_built and len(as_built) == n_open:
        starts.append(set(as_built))
    starts.append(greedy_fill(n_open, set(), n, slack, vslack, base_br, switchable, zz, zi, zp, hours, pmu_buses, sbase))
    for sid in switchable:
        starts.append(
            greedy_fill(n_open, {sid}, n, slack, vslack, base_br, switchable, zz, zi, zp, hours, pmu_buses, sbase)
        )
    best = (1e18, None)
    seen = set()
    for st in starts:
        key = tuple(sorted(st))
        if len(st) != n_open or key in seen:
            continue
        seen.add(key)
        err, got = one_exchange(st, n, slack, vslack, base_br, switchable, zz, zi, zp, hours, pmu_buses, sbase)
        if err < best[0]:
            best = (err, tuple(sorted(got)))
    return best


def recover(root: Path) -> dict:
    case = json.loads((root / "case.json").read_text(encoding="utf-8"))
    net = json.loads((root / "network.json").read_text(encoding="utf-8"))
    nom_br = net["branches"]
    n = case["n_bus"]
    slack = case["slack_bus"]
    vslack = case["v_slack_pu"]
    pmu_buses = case["pmu_buses"]
    switchable = case["switchable_ids"]
    sbase = case["sbase_mva"]
    n_open = int(case["n_open_switches"])

    hours = []
    class_of = {}
    for h in range(24):
        inj = {int(r["bus"]): r for r in load_csv(root / "scada" / f"h{h:02d}_injections.csv")}
        for r in inj.values():
            class_of[int(r["bus"])] = r["class"]
        pmu = {}
        for bus in pmu_buses:
            row = load_csv(root / "pmu" / f"h{h:02d}" / f"pmu_bus{bus:02d}.csv")[0]
            pmu[bus] = row
        hours.append({"h": h, "inj": inj, "pmu": pmu})

    probe = [hours[i] for i in (8, 14)]
    zip_map = fit_zip_pmu(hours, pmu_buses, slack)
    zz, zi, zp = zip_from_classes(n, class_of, zip_map)
    br = nom_br
    as_built = net.get("as_built_open_ids") or []
    _err, open_ids_t = search_open(
        n_open, n, slack, vslack, br, switchable, zz, zi, zp, probe, pmu_buses, sbase, as_built
    )
    open_ids = list(open_ids_t)
    opens = set(open_ids)

    ang_res = {b: [] for b in pmu_buses if b != slack}
    for rec in hours:
        p0 = np.zeros(n)
        q0 = np.zeros(n)
        for b, row in rec["inj"].items():
            p0[b - 1] = float(row["p0_mw"]) / sbase
            q0[b - 1] = float(row["q0_mvar"]) / sbase
        v, th, ok = solve_nr(n, slack, vslack, br, opens, p0, q0, zz, zi, zp)
        if not ok:
            continue
        for b in pmu_buses:
            if b != slack:
                ang_res[b].append(float(rec["pmu"][b]["vang_rad"]) - th[b - 1])
    med_bias = {b: float(np.median(ang_res[b])) for b in ang_res}
    bad = sorted(med_bias, key=lambda b: abs(med_bias[b]), reverse=True)[:2]
    bias = {int(b): med_bias[b] for b in bad}

    hold = {int(r["bus"]): r for r in load_csv(root / "withheld_injections.csv")}
    p0 = np.zeros(n)
    q0 = np.zeros(n)
    for b, row in hold.items():
        p0[b - 1] = float(row["p0_mw"]) / sbase
        q0[b - 1] = float(row["q0_mvar"]) / sbase
        class_of[b] = row["class"]
    zz, zi, zp = zip_from_classes(n, class_of, zip_map)
    vh, thh, ok = solve_nr(n, slack, vslack, br, opens, p0, q0, zz, zi, zp)
    if not ok:
        raise RuntimeError("withheld NR failed")
    unmetered = [b for b in range(1, n + 1) if b not in pmu_buses]

    rec14 = hours[int(case["motor"]["train_hour_for_vth"])]
    p0t = np.zeros(n)
    q0t = np.zeros(n)
    for b, row in rec14["inj"].items():
        p0t[b - 1] = float(row["p0_mw"]) / sbase
        q0t[b - 1] = float(row["q0_mvar"]) / sbase
    vt, _, ok = solve_nr(n, slack, vslack, br, opens, p0t, q0t, zz, zi, zp)
    vth_train = float(vt[case["motor"]["train_bus"] - 1])
    vth_hold = float(vh[case["motor"]["hold_bus"] - 1])
    m = case["motor"]
    train = load_csv(root / "motor_train.csv")
    tm = np.array([float(r["tm_pu"]) for r in train])
    i_obs = np.array([float(r["i_a_pu"]) for r in train])
    dt = m["dt_s"]
    r1, x1, x2 = m["r1"], m["x1"], m["x2"]
    s0 = float(m["s0"])
    s_lo, s_hi = float(m["s_clip"][0]), float(m["s_clip"][1])
    best_hr = (1.35, 0.022)
    best_e = 1e18
    for h in np.linspace(0.8, 2.0, 25):
        for r2 in np.linspace(0.012, 0.035, 24):
            pred = sim_motor(vth_train, tm, float(h), r1, x1, float(r2), x2, dt, s0, s_lo, s_hi)
            e = float(np.mean((pred - i_obs) ** 2))
            if e < best_e:
                best_e = e
                best_hr = (float(h), float(r2))
    hold_mech = load_csv(root / "motor_hold_mech.csv")
    tm_h = np.array([float(r["tm_pu"]) for r in hold_mech])
    i_hold = sim_motor(vth_hold, tm_h, best_hr[0], r1, x1, best_hr[1], x2, dt, s0, s_lo, s_hi)

    return {
        "schema_version": case["schema_version"],
        "case_id": case["case_id"],
        "open_switch_ids": open_ids,
        "pmu_angle_bias_rad": {str(k): v for k, v in bias.items()},
        "zip": {cl: {"z": a, "i": b, "p": c} for cl, (a, b, c) in zip_map.items()},
        "withheld_vmag_pu": {str(b): float(vh[b - 1]) for b in unmetered},
        "withheld_vang_rad": {str(b): float(thh[b - 1]) for b in unmetered},
        "motor": {
            "h_s": best_hr[0],
            "r2": best_hr[1],
            "i_hold_pu": [float(x) for x in i_hold],
        },
    }
