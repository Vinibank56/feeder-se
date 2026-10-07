# Feeder ZIP switch-state audit

## Inputs

`/app/data/case.json`
`/app/data/network.json`
`/app/data/model_contract.json`
`/app/data/README.txt`
`/app/data/pmu/hXX/pmu_busYY.csv`
`/app/data/scada/hXX_injections.csv`
`/app/data/withheld_injections.csv`
`/app/data/motor_train.csv`
`/app/data/motor_hold_mech.csv`

## Deliverable

Write `/app/feeder_se.json`.

```json
{
  "schema_version": "feeder-zip-se-v1",
  "case_id": "<from case.json>",
  "open_switch_ids": ["<switchable id>"],
  "pmu_angle_bias_rad": {"<bus>": 0.0},
  "zip": {
    "res": {"z": 0.0, "i": 0.0, "p": 1.0},
    "com": {"z": 0.0, "i": 0.0, "p": 1.0},
    "ind": {"z": 0.0, "i": 0.0, "p": 1.0}
  },
  "withheld_vmag_pu": {"<unmetered bus>": 0.0},
  "withheld_vang_rad": {"<unmetered bus>": 0.0},
  "motor": {"h_s": 0.0, "r2": 0.0, "i_hold_pu": [0.0]}
}
```

`open_switch_ids` has exactly `n_open_switches` ids from `switchable_ids` (see `/app/data/case.json`). `pmu_angle_bias_rad` has one key per biased PMU (two channels). `withheld_*` has every bus that is not in `pmu_buses`. `i_hold_pu` has one sample per row of `/app/data/motor_hold_mech.csv`, same order.

## What is graded

Switch-status identity, the biased PMU set and angle offsets, ZIP rows that sum to one, the withheld unmetered voltage field (RMSE and peak |V| error), motor `h_s` and `r2`, and the withheld stator-current series.

## Notes

Slack angle is 0 on the `sbase_mva` / `vbase_kv` base in `/app/data/case.json`. An open switchable branch is omitted from Ybus (series admittance and half charging). ZIP uses the same (z, i, p) on P and Q.

Motor integration follows `/app/data/model_contract.json` and `/app/data/case.json` `motor`: slip starts at `s0`; each step is forward Euler `s <- clip(s + dt_s/h_s * (Tm - Te), s_clip)`; stator current is stored **after** that update; `Vth` is the solved |V| at `train_bus` (hour `train_hour_for_vth`) or `hold_bus` (withheld injections), held for the 4 s record.

You have 9000 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
