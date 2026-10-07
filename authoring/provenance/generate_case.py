#!/usr/bin/env python3
"""Generate public feeder telemetry and sealed truth. Seed 20261005."""
from __future__ import annotations

import csv
import itertools
import json
import math
import random
import shutil
import string
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pf_nr import connected, solve_nr, with_scales, zip_scale

SEED = 20261005
ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / "environment" / "data"
TST = ROOT / "tests"

N = 22
SLACK = 1
VSLACK = 1.04
SBASE = 10.0
VBASE = 12.47

BRANCHES = []
for i in range(1, N):
    BRANCHES.append(
        {
            "id": f"L{i:02d}",
            "from": i,
            "to": i + 1,
            "r_pu": round(0.012 + 0.001 * (i % 5), 5),
            "x_pu": round(0.040 + 0.003 * (i % 5), 5),
            "b_pu": 0.002,
            "switchable": False,
        }
    )
# 20 comparable-Z ties. Cardinality is n_open_switches (8). C(20,8)=125970.
TIES = [
    (4, 7),
    (3, 8),
    (5, 9),
    (6, 11),
    (8, 12),
    (10, 15),
    (2, 9),
    (3, 12),
    (4, 13),
    (5, 14),
    (7, 15),
    (8, 17),
    (9, 18),
    (10, 19),
    (12, 20),
    (13, 21),
    (14, 22),
    (6, 16),
    (7, 18),
    (2, 6),
]
for k, (a, b) in enumerate(TIES):
    BRANCHES.append(
        {
            "id": "SW-" + string.ascii_uppercase[k],
            "from": a,
            "to": b,
            "r_pu": round(0.028 + 0.0012 * ((k * 5 + 3) % 17), 5),
            "x_pu": round(0.080 + 0.0030 * ((k * 5 + 3) % 17), 5),
            "b_pu": 0.001,
            "switchable": True,
        }
    )
BRANCHES.append(
    {
        "id": "L-914",
        "from": 9,
        "to": 14,
        "r_pu": 0.033,
        "x_pu": 0.092,
        "b_pu": 0.001,
        "switchable": False,
    }
)
BRANCHES.append(
    {
        "id": "L-1116",
        "from": 11,
        "to": 16,
        "r_pu": 0.031,
        "x_pu": 0.088,
        "b_pu": 0.001,
        "switchable": False,
    }
)

TRUE_OPEN = {"SW-B", "SW-D", "SW-F", "SW-G", "SW-I", "SW-O", "SW-S", "SW-T"}
AS_BUILT_OPEN = ["SW-A", "SW-C", "SW-E", "SW-H", "SW-J", "SW-K", "SW-L", "SW-N"]
# Slack metered. PMUs sit on true-open corridors so 1-swaps are observed.
PMU_BUSES = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20]
BAD_PMU = {7: 0.018, 8: -0.012}
TRUE_SCALES = {}
N_OPEN = len(TRUE_OPEN)
assert N_OPEN == 8
assert not (TRUE_OPEN & set(AS_BUILT_OPEN))
assert len([br for br in BRANCHES if br["switchable"]]) == 20

_cycle = ["res", "com", "ind"]
CLASS = [None, None] + [_cycle[(i - 2) % 3] for i in range(2, N + 1)]
P0_MW = [0, 0] + [round(0.45 + 0.04 * ((i * 5) % 13), 5) for i in range(2, N + 1)]
Q0_MVAR = [0, 0] + [round(0.16 + 0.015 * ((i * 3) % 11), 5) for i in range(2, N + 1)]
ZIP = {
    "res": (0.42, 0.28, 0.30),
    "com": (0.18, 0.22, 0.60),
    "ind": (0.08, 0.12, 0.80),
}

MOTOR_BUS_TRAIN = 3
MOTOR_BUS_HOLD = 21
MOTOR = {
    "h_s": 1.35,
    "r1": 0.028,
    "x1": 0.085,
    "r2": 0.022,
    "x2": 0.085,
    "dt_s": 0.004,
    "s0": 0.96,
    "s_clip": [1e-4, 0.99],
}


def zip_arrays():
    zz = np.zeros(N)
    zi = np.zeros(N)
    zp = np.ones(N)
    for b in range(2, N + 1):
        zz[b - 1], zi[b - 1], zp[b - 1] = ZIP[CLASS[b]]
    return zz, zi, zp


def inj(scale: float):
    p = np.zeros(N)
    q = np.zeros(N)
    for b in range(2, N + 1):
        p[b - 1] = P0_MW[b] / SBASE * scale
        q[b - 1] = Q0_MVAR[b] / SBASE * scale
    return p, q


def motor_te(v: float, s: float, m: dict) -> float:
    s = max(1e-4, min(0.999, s))
    r2os = m["r2"] / s
    den = (m["r1"] + r2os) ** 2 + (m["x1"] + m["x2"]) ** 2
    return 3.0 * v * v * r2os / den


def motor_i(v: float, s: float, m: dict) -> float:
    s = max(1e-4, min(0.999, s))
    r2os = m["r2"] / s
    zmag = math.hypot(m["r1"] + r2os, m["x1"] + m["x2"])
    return v / zmag


def simulate_motor(vth: float, tm: np.ndarray, m: dict) -> tuple[np.ndarray, np.ndarray]:
    s = m["s0"]
    slips = []
    amps = []
    dt = m["dt_s"]
    h = m["h_s"]
    for tload in tm:
        te = motor_te(vth, s, m)
        lo, hi = m["s_clip"]
        s = min(hi, max(lo, s + dt / h * (tload - te)))
        slips.append(s)
        amps.append(motor_i(vth, s, m))
    return np.array(slips), np.array(amps)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    rng = random.Random(SEED)
    np.random.seed(SEED)
    scaled = with_scales(BRANCHES, TRUE_SCALES)
    assert connected(N, scaled, TRUE_OPEN, SLACK)
    assert connected(N, scaled, set(AS_BUILT_OPEN), SLACK)
    zz, zi, zp = zip_arrays()
    pmu_root = PUB / "pmu"
    if pmu_root.exists():
        shutil.rmtree(pmu_root)
    scada_root = PUB / "scada"
    if scada_root.exists():
        shutil.rmtree(scada_root)

    hours = list(range(24))
    scales_h = [0.30 + 0.75 * (0.5 + 0.5 * math.sin((h - 6) / 24 * 2 * math.pi)) for h in hours]
    snapshots = []
    pmu_files = 0
    for h, sc in zip(hours, scales_h):
        p0, q0 = inj(sc)
        v, th, ok = solve_nr(N, SLACK, VSLACK, scaled, TRUE_OPEN, p0, q0, zz, zi, zp)
        if not ok:
            raise RuntimeError(f"NR failed hour {h}")
        snapshots.append({"hour": h, "v": v.tolist(), "th": th.tolist()})
        for bus in PMU_BUSES:
            bias = BAD_PMU.get(bus, 0.0)
            row = {
                "hour": h,
                "vmag_pu": round(v[bus - 1] + rng.gauss(0, 0.00035), 7),
                "vang_rad": round(th[bus - 1] + rng.gauss(0, 0.0004) + bias, 7),
                "p_inj_mw": round(
                    -p0[bus - 1] * zip_scale(v[bus - 1], zz[bus - 1], zi[bus - 1], zp[bus - 1]) * SBASE
                    + rng.gauss(0, 0.004),
                    5,
                ),
                "q_inj_mvar": round(
                    -q0[bus - 1] * zip_scale(v[bus - 1], zz[bus - 1], zi[bus - 1], zp[bus - 1]) * SBASE
                    + rng.gauss(0, 0.004),
                    5,
                ),
            }
            write_csv(
                PUB / "pmu" / f"h{h:02d}" / f"pmu_bus{bus:02d}.csv",
                [row],
                ["hour", "vmag_pu", "vang_rad", "p_inj_mw", "q_inj_mvar"],
            )
            pmu_files += 1
        scada_rows = []
        for b in range(2, N + 1):
            scada_rows.append(
                {
                    "bus": b,
                    "class": CLASS[b],
                    "p0_mw": round(P0_MW[b] * sc, 5),
                    "q0_mvar": round(Q0_MVAR[b] * sc, 5),
                }
            )
        write_csv(PUB / "scada" / f"h{h:02d}_injections.csv", scada_rows, ["bus", "class", "p0_mw", "q0_mvar"])

    hold_scale = 0.95
    p0h, q0h = inj(hold_scale)
    vh, thh, ok = solve_nr(N, SLACK, VSLACK, scaled, TRUE_OPEN, p0h, q0h, zz, zi, zp)
    if not ok:
        raise RuntimeError("NR withheld failed")
    hold_rows = []
    for b in range(2, N + 1):
        hold_rows.append(
            {
                "bus": b,
                "class": CLASS[b],
                "p0_mw": round(P0_MW[b] * hold_scale, 5),
                "q0_mvar": round(Q0_MVAR[b] * hold_scale, 5),
            }
        )
    write_csv(PUB / "withheld_injections.csv", hold_rows, ["bus", "class", "p0_mw", "q0_mvar"])
    unmetered = [b for b in range(1, N + 1) if b not in PMU_BUSES]

    t = np.arange(0.0, 4.0, MOTOR["dt_s"])
    tm_train = 0.22 + 0.05 * np.sin(2 * np.pi * t / 1.7)
    tm_hold = 0.30 + 0.04 * np.sin(2 * np.pi * t / 1.1)
    vth_train = float(snapshots[14]["v"][MOTOR_BUS_TRAIN - 1])
    vth_hold = float(vh[MOTOR_BUS_HOLD - 1])
    _, i_train = simulate_motor(vth_train, tm_train, MOTOR)
    _, i_hold = simulate_motor(vth_hold, tm_hold, MOTOR)
    train_rows = [
        {"t_s": round(float(tt), 6), "tm_pu": round(float(a), 6), "i_a_pu": round(float(b) + rng.gauss(0, 0.002), 6)}
        for tt, a, b in zip(t, tm_train, i_train)
    ]
    hold_mech = [{"t_s": round(float(tt), 6), "tm_pu": round(float(a), 6)} for tt, a in zip(t, tm_hold)]
    write_csv(PUB / "motor_train.csv", train_rows, ["t_s", "tm_pu", "i_a_pu"])
    write_csv(PUB / "motor_hold_mech.csv", hold_mech, ["t_s", "tm_pu"])

    case = {
        "case_id": "feeder-zip-se-01",
        "schema_version": "feeder-zip-se-v1",
        "sbase_mva": SBASE,
        "vbase_kv": VBASE,
        "slack_bus": SLACK,
        "v_slack_pu": VSLACK,
        "slack_angle_rad": 0.0,
        "n_bus": N,
        "pmu_buses": PMU_BUSES,
        "switchable_ids": [br["id"] for br in BRANCHES if br["switchable"]],
        "n_open_switches": N_OPEN,
        "nr_residual": 1e-10,
        "nr_max_iter": 40,
        "output_path": "/app/feeder_se.json",
        "motor": {
            "train_bus": MOTOR_BUS_TRAIN,
            "hold_bus": MOTOR_BUS_HOLD,
            "dt_s": MOTOR["dt_s"],
            "r1": MOTOR["r1"],
            "x1": MOTOR["x1"],
            "x2": MOTOR["x2"],
            "unknown": ["h_s", "r2"],
            "s0": MOTOR["s0"],
            "s_clip": MOTOR["s_clip"],
            "current_after_euler": True,
            "train_hour_for_vth": 14,
        },
    }
    network = {"branches": BRANCHES, "as_built_open_ids": AS_BUILT_OPEN}
    contract = {
        "power_flow": "polar Newton-Raphson. Load buses inject S = S0 (z V^2 + i V + p) with z+i+p=1 per class, same fractions on P and Q. Slack voltage and angle locked. Open switchable branch omitted from Ybus (series y and half charging). PQ mismatch residual max |dP,dQ| < nr_residual.",
        "pmu": "One hourly sample per file. vang_rad is on the slack frame; two channels have a constant angle bias. p_inj_mw is load convention (negative = consumption) including ZIP.",
        "motor": "Te = 3 Vth^2 (r2/s) / ((r1+r2/s)^2+(x1+x2)^2). I = Vth / hypot(r1+r2/s, x1+x2). Slip starts at case.motor.s0. Each step: s <- clip(s + dt_s/h_s*(Tm - Te), s_clip[0], s_clip[1]); then store I(s). Vth is algebraic |V| at train_bus for motor_train (hour train_hour_for_vth) and at hold_bus for motor_hold_mech, from the same ZIP power flow, constant over 4 s.",
        "zip_classes": ["res", "com", "ind"],
    }
    dump(PUB / "case.json", case)
    dump(PUB / "network.json", network)
    dump(PUB / "model_contract.json", contract)
    (PUB / "README.txt").write_text(
        "Industrial 12.47 kV feeder. Buses 1..n_bus, slack=1. PMU dumps under pmu/hXX/. "
        "SCADA nameplate injections under scada/. as_built_open_ids in network.json is the one-line the operators filed; it is not a measurement.\n"
        "ZIP class labels are on the SCADA rows. Switchable branch ids are those with switchable=true.\n",
        encoding="utf-8",
    )

    p_nom, q_nom = inj(hold_scale)
    v_as, _, ok_as = solve_nr(N, SLACK, VSLACK, scaled, set(AS_BUILT_OPEN), p_nom, q_nom, zz, zi, zp)
    zz1, zi1, zp1 = np.zeros(N), np.zeros(N), np.ones(N)
    v_p, _, ok_p = solve_nr(N, SLACK, VSLACK, scaled, TRUE_OPEN, p_nom, q_nom, zz1, zi1, zp1)
    def rmse_peak(est):
        dv = [abs(est[b - 1] - vh[b - 1]) for b in unmetered]
        return float(np.sqrt(np.mean(np.square(dv)))), float(max(dv))

    truth = {
        "case_id": "feeder-zip-se-01",
        "open_switch_ids": sorted(TRUE_OPEN),
        "pmu_angle_bias_rad": {str(k): v for k, v in BAD_PMU.items()},
        "zip": {k: {"z": a, "i": b, "p": c} for k, (a, b, c) in ZIP.items()},
        "unmetered_buses": unmetered,
        "withheld_vmag_pu": {str(b): vh[b - 1] for b in unmetered},
        "withheld_vang_rad": {str(b): thh[b - 1] for b in unmetered},
        "motor": {"h_s": MOTOR["h_s"], "r2": MOTOR["r2"], "i_hold_pu": i_hold.tolist()},
        "tolerances": {
            "zip_abs": 0.05,
            "bias_rad": 0.003,
            "vmag_rmse_pu": 0.003,
            "vmag_max_pu": 0.004,
            "vang_rmse_rad": 0.003,
            "motor_i_rmse": 0.022,
            "motor_i_heun_rmse": 0.025,
            "h_s": 0.12,
            "r2": 0.004,
        },
        "ablation": {
            "as_built": rmse_peak(v_as) if ok_as else None,
            "p_const": rmse_peak(v_p) if ok_p else None,
        },
    }
    dump(TST / "ground_truth.json", truth)
    dump(TST / "case.json", case)
    dump(TST / "network.json", network)
    (TST / "withheld_injections.csv").write_text((PUB / "withheld_injections.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (TST / "motor_hold_mech.csv").write_text((PUB / "motor_hold_mech.csv").read_text(encoding="utf-8"), encoding="utf-8")

    def pmu_vmag(hours):
        recs = []
        for h in hours:
            inj_rows = {int(r["bus"]): r for r in csv.DictReader((PUB / "scada" / f"h{h:02d}_injections.csv").open(encoding="utf-8"))}
            pmu = {}
            for b in PMU_BUSES:
                row = next(csv.DictReader((PUB / "pmu" / f"h{h:02d}" / f"pmu_bus{b:02d}.csv").open(encoding="utf-8")))
                pmu[b] = float(row["vmag_pu"])
            recs.append({"inj": inj_rows, "pmu": pmu})
        return recs

    def mse_open(opens, recs):
        acc = 0.0
        nobs = 0
        for rec in recs:
            p0 = np.zeros(N)
            q0 = np.zeros(N)
            for b, row in rec["inj"].items():
                p0[b - 1] = float(row["p0_mw"]) / SBASE
                q0[b - 1] = float(row["q0_mvar"]) / SBASE
            v, _, ok = solve_nr(N, SLACK, VSLACK, scaled, set(opens), p0, q0, zz, zi, zp)
            if not ok:
                return 1.0
            for b in PMU_BUSES:
                acc += (v[b - 1] - rec["pmu"][b]) ** 2
                nobs += 1
        return acc / max(nobs, 1)

    switchable = [br["id"] for br in BRANCHES if br["switchable"]]
    by_r = sorted(
        (br for br in BRANCHES if br["switchable"]),
        key=lambda br: (br["r_pu"], br["x_pu"], br["id"]),
    )
    lowest8 = {br["id"] for br in by_r[:8]}
    print("lowest8_r_pu", sorted(lowest8), "equals_true", lowest8 == TRUE_OPEN)
    assert lowest8 != TRUE_OPEN, "public r_pu rank must not equal TRUE_OPEN"
    rec_814 = pmu_vmag([8, 14])
    rec_24 = pmu_vmag(list(range(24)))
    closed_true = [s for s in switchable if s not in TRUE_OPEN]
    true_t = tuple(sorted(TRUE_OPEN))
    true_mse_814 = mse_open(true_t, rec_814)
    true_mse_24 = mse_open(true_t, rec_24)

    def ham_sets(dist):
        out = []
        for leave in itertools.combinations(sorted(TRUE_OPEN), dist):
            for add in itertools.combinations(closed_true, dist):
                cand = tuple(sorted((TRUE_OPEN - set(leave)) | set(add)))
                if not connected(N, scaled, set(cand), SLACK):
                    continue
                out.append(cand)
        return out

    ham1 = ham_sets(1)
    ham2 = ham_sets(2)
    scored_814 = [(mse_open(c, rec_814), c, "h1") for c in ham1]
    scored_814 += [(mse_open(c, rec_814), c, "h2") for c in ham2]
    scored_814.sort()
    nearest_814 = scored_814[0]
    scored_24 = [(mse_open(c, rec_24), c) for c in ham1]
    scored_24.sort()
    nearest_24 = scored_24[0]
    wins = 0
    for h in range(24):
        rec = pmu_vmag([h])
        if mse_open(true_t, rec) < mse_open(nearest_814[1], rec):
            wins += 1
    as_e = mse_open(tuple(sorted(AS_BUILT_OPEN)), rec_814)
    print("pmu_files", pmu_files, "ablation", truth["ablation"], "ok")
    print("n_switchable", len(switchable), "n_open", N_OPEN)
    print("true_mse_8_14", true_mse_814, "nearest_h1h2_8_14", nearest_814, "ratio814", nearest_814[0] / true_mse_814)
    print("true_mse_24", true_mse_24, "nearest_h1_24", nearest_24, "ratio24", nearest_24[0] / true_mse_24)
    print("as_built_mse_8_14", as_e, "true_wins_vs_nearest814", wins)
    for err, ids, tag in scored_814[:6]:
        print(" rank814", tag, ids, err, err / true_mse_814)


if __name__ == "__main__":
    main()
