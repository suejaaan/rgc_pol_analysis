# TE calibration of cc

Goal: recompute the calibration constant cc = P_TE / A_TE from thermal-equilibrium (TE) runs in
`data-d`, with P_TE the spin-1 Boltzmann polarization 2 sinh(x) / (1 + 2 cosh(x)), x = hν/kT
(≈ 0.067 % at 1.56 K, 5 T).

## Scripts

| Script | What it does |
|---|---|
| `te_survey.py` | Finds every TE run (label `TE`), checks whether the area reaches a plateau. Plot: `results/survey/all_te_runs.png` |
| `te_run_analysis.py` | One run: relaxation fits (1- and 2-exponential), which cc values the data can distinguish, averaged late-TE lineshape |
| `te_plateau_cc.py` | 2023-03-21 run: hour-binned, shape-weighted area, tail extrapolation, cc with systematics |
| `shape_cc_check.py` | Calibration-free cross-check: P from the horn asymmetry of high-P events, cc = P_shape / area |
| `shape_cc_survey.py` | The same check for every online-cc period in data-d (480 fits, 14 cores, ~5 min). Output: `results/shape_cc_survey/` |

## Findings

- `data-d` has 1,587 TE-labelled events in 8 runs. **None clearly reaches thermal equilibrium.**
  - 2022-06-30, 09-11, 11-11: 1–3 events.
  - 2022-09-07 (1.6 h), 2022-09-20 (4.8 h): still relaxing when the run ends; extrapolation
    cannot distinguish cc = −1.39 from −0.3 (2-exponential relaxation fits equally well).
  - 2023-02-27: microwaves on during the "TE" label → not TE.
  - 2023-03-22: area swings −40 → +70 ×10⁻³ → not TE.
  - 2023-03-21 (23.5 h, 1.6 K): the only near-plateau run. Tail extrapolation gives
    cc = −1.02 ± 0.17; taking the last 3 h as equilibrium reproduces the online −0.817 exactly.
- **The lineshape contradicts that TE value.** High-P events right before it give
  cc_shape ≈ −3.8 (online −3.67), and the operators returned to −3.5…−4.0 the next day.
  The most consistent reading: even the 23.5 h run had not reached equilibrium (very slow
  relaxation component), so it overestimates A_TE.
- Shape-method cc vs online cc: −1.46 vs −1.39 (Sep 2022), −3.82 vs −3.67 (Mar 2023).
  **The online cc values tested are consistent with the lineshape to ~5–10 %.**

## Lineshape cc for every period (`shape_cc_survey.py`)

| Category (lineshape cc / online cc) | Polarize events | Periods |
|---|---|---|
| consistent, 0.9–1.1 | 37,001 (56 %) | Sep 2022 → Mar 20 2023 (cc −1.34 … −1.42, −3.37 … −3.93) |
| online P too high, ratio ≈ 0.43 | 18,819 (29 %) | **Jul 21 → Sep 6 2022, cc = −14.5**: clean doublets, horn ratio ≈ 1.3 → P ≈ 0.27, online says ≈ 0.63 |
| online P too low, ratio ≈ 2.5 | 1,308 (2 %) | Mar 22 2023 (cc −1.12 / −1.0, set from the unfinished TE run); Jul 19–21 2022 (−3.70) |
| unreliable | 8,408 (13 %) | **Jun 23 → Jul 12 2022 (cc −5.247)**: online `fitsub` dominated by a curved background (no clean doublet; |pol| > 1 seen); short transients; low-P Mar 23 2023 |

## Conclusion

The TE runs logged in `data-d` cannot independently calibrate cc. From Sep 2022 on, the online
cc agrees with the calibration-free lineshape check to ~5 %. The summer-2022 periods do not:
the cc = −14.5 period (29 % of Polarize events) looks ~2.3× too high, and the June–early-July
online processing is unusable. These should be confirmed with the RGC target group before the
labels are corrected. Earlier rough TE comparisons that suggested 10–50× errors used
non-equilibrium TE events and should be disregarded.
