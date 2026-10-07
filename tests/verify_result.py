#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from pf_gs import solve_gs


class VerificationError(AssertionError):
    pass


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise VerificationError(msg)


def finite(value, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} numeric")
    number = float(value)
    require(math.isfinite(number), f"{label} finite")
    return number


def load(path: Path) -> dict:
    require(path.is_file(), f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def zip_arrays(n: int, class_of: dict, zip_map: dict):
    zz = np.zeros(n)
    zi = np.zeros(n)
    zp = np.ones(n)
    for b, cl in class_of.items():
        z = zip_map[cl]
        zz[b - 1], zi[b - 1], zp[b - 1] = z["z"], z["i"], z["p"]
    return zz, zi, zp


def verify(result_path: Path, truth_path: Path, case_path: Path, net_path: Path, hold_path: Path) -> None:
    result = load(result_path)
    truth = load(truth_path)
    case = load(case_path)
    net = load(net_path)
    require(isinstance(result, dict), "result object")
    require(result.get("schema_version") == case["schema_version"], "schema_version")
    require(result.get("case_id") == case["case_id"], "case_id")
    got_open = set(result.get("open_switch_ids") or [])
    require(len(got_open) == int(case["n_open_switches"]), "open_switch_ids length")
    require(got_open <= set(case["switchable_ids"]), "open ids are switchable")
    require(got_open == set(truth["open_switch_ids"]), "open_switch_ids")
    tol = truth["tolerances"]
    bias = result.get("pmu_angle_bias_rad") or {}
    require(isinstance(bias, dict), "bias object")
    require(set(str(k) for k in bias) == set(truth["pmu_angle_bias_rad"]), "biased PMU set")
    for k, exp in truth["pmu_angle_bias_rad"].items():
        require(abs(finite(bias[k], "bias") - exp) <= tol["bias_rad"], f"bias bus {k}")
    zgot = result.get("zip") or {}
    for cl in ("res", "com", "ind"):
        require(cl in zgot, f"zip {cl}")
        s = finite(zgot[cl]["z"], "z") + finite(zgot[cl]["i"], "i") + finite(zgot[cl]["p"], "p")
        require(abs(s - 1.0) <= 1e-6, f"zip {cl} sum")
    um = [str(b) for b in truth["unmetered_buses"]]
    dv = [abs(finite(result["withheld_vmag_pu"][b], "vmag") - truth["withheld_vmag_pu"][b]) for b in um]
    da = [abs(finite(result["withheld_vang_rad"][b], "vang") - truth["withheld_vang_rad"][b]) for b in um]
    vrmse = math.sqrt(sum(x * x for x in dv) / len(dv))
    armse = math.sqrt(sum(x * x for x in da) / len(da))
    require(vrmse <= tol["vmag_rmse_pu"], "withheld |V| RMSE")
    require(max(dv) <= tol["vmag_max_pu"], "withheld |V| peak")
    require(armse <= tol["vang_rmse_rad"], "withheld angle RMSE")
    mot = result.get("motor") or {}
    require(abs(finite(mot.get("h_s"), "h_s") - truth["motor"]["h_s"]) <= tol["h_s"], "h_s")
    require(abs(finite(mot.get("r2"), "r2") - truth["motor"]["r2"]) <= tol["r2"], "r2")
    pred = mot.get("i_hold_pu") or []
    ref = truth["motor"]["i_hold_pu"]
    require(isinstance(pred, list), "motor series list")
    require(len(pred) == len(ref), "motor length")
    rmse = math.sqrt(sum((finite(a, "i") - b) ** 2 for a, b in zip(pred, ref)) / len(ref))
    require(rmse <= tol["motor_i_rmse"], "motor rmse")

    import csv

    from motor_heun import heun_current

    require(hold_path.is_file(), "withheld injections missing from verifier")
    hold_rows = list(csv.DictReader(hold_path.open(encoding="utf-8")))
    class_of = {int(r["bus"]): r["class"] for r in hold_rows}
    n = case["n_bus"]
    sbase = case["sbase_mva"]
    p0 = np.zeros(n)
    q0 = np.zeros(n)
    for r in hold_rows:
        b = int(r["bus"])
        p0[b - 1] = float(r["p0_mw"]) / sbase
        q0[b - 1] = float(r["q0_mvar"]) / sbase
    zz, zi, zp = zip_arrays(n, class_of, truth["zip"])
    mag, ang, ok = solve_gs(
        n,
        case["slack_bus"],
        case["v_slack_pu"],
        net["branches"],
        set(truth["open_switch_ids"]),
        p0,
        q0,
        zz,
        zi,
        zp,
    )
    require(ok, "gs did not converge")
    gs_dv = [abs(mag[int(b) - 1] - finite(result["withheld_vmag_pu"][b], "vmag")) for b in um]
    require(math.sqrt(sum(x * x for x in gs_dv) / len(gs_dv)) <= tol["vmag_rmse_pu"], "gs |V| RMSE vs artifact")
    mech = hold_path.parent / "motor_hold_mech.csv"
    require(mech.is_file(), "motor_hold_mech missing from verifier")
    tm = [float(r["tm_pu"]) for r in csv.DictReader(mech.open(encoding="utf-8"))]
    mm = case["motor"]
    vth = float(mag[int(mm["hold_bus"]) - 1])
    i_heun = heun_current(
        vth,
        tm,
        float(truth["motor"]["h_s"]),
        float(mm["r1"]),
        float(mm["x1"]),
        float(truth["motor"]["r2"]),
        float(mm["x2"]),
        float(mm["dt_s"]),
        float(mm["s0"]),
        float(mm["s_clip"][0]),
        float(mm["s_clip"][1]),
    )
    require(len(i_heun) == len(pred), "heun length")
    hrmse = math.sqrt(sum((finite(a, "i") - b) ** 2 for a, b in zip(pred, i_heun)) / len(pred))
    require(hrmse <= tol["motor_i_heun_rmse"], "heun motor rmse")


def _result_path(harbor: bool) -> Path:
    env = os.environ.get("RESULT_PATH")
    if env:
        p = Path(env)
        if p.is_file():
            head = p.read_text(encoding="utf-8")[:1200]
            if "feeder-zip-se" in head or harbor:
                return p
    if harbor:
        return Path("/app/feeder_se.json")
    return Path(__file__).resolve().parents[1] / "feeder_se.local.json"


def default_paths() -> tuple[Path, Path, Path, Path, Path]:
    harbor = Path("/tests/ground_truth.json").is_file()
    tests = Path("/tests") if harbor else Path(__file__).resolve().parent
    return (
        _result_path(harbor),
        tests / "ground_truth.json",
        tests / "case.json",
        tests / "network.json",
        tests / "withheld_injections.csv",
    )


def main(argv: list[str]) -> int:
    if len(argv) == 1:
        paths = default_paths()
    elif len(argv) == 6:
        paths = tuple(Path(a) for a in argv[1:])
    else:
        return 2
    try:
        verify(*paths)
    except Exception:
        print("verification failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
