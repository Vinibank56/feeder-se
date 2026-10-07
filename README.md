# Feeder ZIP switch-state audit

## Difficulty

A commissioning engineer has to name which 8 of 20 switchable ties are open. The open-count is published (`n_open_switches` in `/app/data/case.json`); the identities are not. ZIP class fractions, two constant PMU angle biases, unmetered |V| on a withheld hour, and a second motor-start current are recovered from the same Ybus. Expected expert time is about 18 hours. Telemetry is seeded synthetic ZIP/Newton with PMU noise; the polar NR residual is public.

The hard coupling is topology × ZIP. C(20,8) connected subsets is large, and the 20 switchable series impedances sit in one band (`r_pu` 0.028–0.047). Sorting on `r_pu` does not recover the open set. Fitting ZIP on the as-built eight-id list in `/app/data/network.json` soaks switch error into the load model. Angle bias is a pair of constant offsets on the slack frame after that model is fixed. The withheld hour gives nameplate P0/Q0 only at unmetered buses.

Motor `s0` and post-Euler current sampling are in `/app/data/case.json`. Data provenance: `authoring/provenance/generate_case.py`, seed 20261005.

## Reference solution

`solution/identify.py` fits ZIP from PMU P versus |V|, then searches 8-subsets with a multi-start greedy fill (as-built, empty, and each switch forced in) followed by 1-exchange polish on hours 8 and 14. It takes the two largest median angle residuals as biased channels, solves withheld NR, then grids H and r2 on `motor_train.csv` using `s0` from `case.json`. Other search orders are allowed; the verifier does not import this code.

## Verification

Sealed generator: polar NR in `authoring/provenance/pf_nr.py`. Independent withheld |V|: Gauss–Seidel in `tests/pf_gs.py` with the sealed open set and sealed ZIP. Independent motor: Heun slip in `tests/motor_heun.py`. Seed 20261005.

`open_switch_ids` must have length `n_open_switches` (=8) and match the sealed set. Uniqueness is all Hamming-1 and Hamming-2 8-sets. On hours 8 and 14 the true set {SW-B, D, F, G, I, O, S, T} has PMU |V| MSE 1.26e-7. The nearest other 8-set is the 1-swap D→L at 3.10e-7 (2.46×). Over all 24 hours that same 1-swap is 2.35e-7 versus 1.20e-7 (1.96×); the true set is lower than that competitor on 22 of 24 hours. The eight lowest-`r_pu` switchable ids are not the true set.

| Gate | Threshold | Oracle / independent | Nearest measured null | Why |
| --- | --- | --- | --- | --- |
| `open_switch_ids` | exact {SW-B, D, F, G, I, O, S, T} | exact | 1-swap D→L 2.46× on h8+h14, 1.96× on 24 h | discrete; Hamming-1/2 rank |
| biased PMU set | exact {7, 8} | exact | — | 24-hour angle medians |
| bias | 0.003 rad | 4.6e-5 | 0 bias | PMU angle noise 4e-4 rad |
| withheld \|V\| RMSE | 0.003 pu | oracle 2.5e-5; GS vs NR 9.9e-7 | as-built 0.0523; P-const 0.0110 | below both nulls, above GS–NR |
| withheld \|V\| peak | 0.004 pu | oracle 2.8e-5; GS 9.9e-7 | as-built 0.0528; P-const 0.0112 | same |
| withheld angle RMSE | 0.003 rad | oracle 2.8e-5; GS ~1e-6 | as-built 0.0643; P-const 0.0120 | same buses |
| motor H, r2 | 0.12 s, 0.004 | on-grid Δ=0 | — | H step 0.05 s, r2 step 0.001 |
| motor I vs Euler GT | 0.022 pu | oracle 2.6e-4 | I before update ~0.034 | after-update sampling |
| motor I vs Heun | 0.025 pu | Heun vs GT 0.018 | — | method spread this 4 s record |

ZIP (z,i,p) is collinear on this |V| band and is only required to sum to 1. Reward is binary.
