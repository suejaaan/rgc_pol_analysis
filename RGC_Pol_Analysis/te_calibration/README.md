# te_calibration

**Purpose:** recompute the calibration constant cc = P_TE / A_TE from the thermal-equilibrium (TE) runs in
`data-d`, independent of the hand-entered online cc. P_TE = 2 sinh x / (1 + 2 cosh x), x = hν/kT
(≈ 0.07 % at 1.6 K, 5 T).

**Scripts:**

| Script | What it does |
|---|---|
| `te_survey.py` | Finds every TE run and checks whether the area reaches a plateau → `results/survey/` |
| `te_run_analysis.py` | One run: 1- and 2-exponential relaxation fits; which cc values the data can tell apart |
| `te_plateau_cc.py` | The 2023-03-21 run: hour-binned, shape-weighted area, tail extrapolation, cc with systematics |
| `shape_cc_check.py`, `shape_cc_survey.py` | Early cc check from the horn asymmetry of the online `fitsub`, for every period. Superseded by `../lineshape_polarization/`, which fits the raw sweep. |

**Results:**
- **No usable TE run:** none of the 8 TE runs (1,587 events) clearly reaches equilibrium, so **TE cannot recalibrate cc here.**
  - Sep 7 and Sep 20, 2022: still relaxing when the run ends. cc anywhere from −0.3 to −1.4 fits equally well.
  - Feb 27 and Mar 22, 2023: not real TE. The microwaves were on, or the area swings.
  - Mar 21, 2023 (23.5 h): the only run near a plateau. It gives cc ≈ −1.0 ± 0.2, the source of the online −0.82 / −1.12. Polarized data right before and after point to −3.7, so this run had most likely not finished relaxing either.
- **First lineshape cross-check:** the shape check agreed with online cc within ~5–10 % from Sep 2022 to Mar 20, 2023. It flagged cc −14.5 (Jul 22 – Sep 6, 2022) as ~2.3× too high and Jun – early Jul 2022 (cc −5.25) as unusable.
- **Correction:** this survey also called Mar 22–23, 2023 "online P too low". `../lineshape_polarization/` later showed the lineshape is unreliable at low P and out of spin equilibrium, so that call is withdrawn.
- **Disregard older numbers:** earlier rough TE comparisons that suggested 10–50× errors used non-equilibrium events and double-counted files.
