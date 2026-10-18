# RGC_Pol_Analysis

Offline analysis of the CLAS12 RGC polarized-deuteron target NMR data, aimed at better
polarization extraction and physically valid Monte Carlo for ML training.

Builds on the `data_creation/` and `ml/` code in this repository (lineshape, Q-meter
circuit, signal generator). Raw data (`data_creation/data-d`) is not part of the repo.

## Contents

| Folder | What it does |
|---|---|
| `online_signal_inspection/` | Plots the online processing chain for selected events: raw Q-meter sweep, stored baseline, baseline-subtracted signal with the online wing polynomial, and the final `fitsub` that is summed into `area` (`pol = cc × area`). |
| `qmeter_dulya_fit/` | Fits the **raw sweep** directly with a Liverpool Q-meter circuit whose coil contains the Dulya deuteron susceptibility (χ″ lineshape, χ′ from Kramers–Kronig). No stored baseline, no polynomial. Extracts P with the **area method**: cc × Σ of the χ″-driven part of the fitted signal. |

## Running

Uses the miniforge environment (numpy, scipy, matplotlib, pyyaml):

```bash
python online_signal_inspection/plot_signal_comparison.py 0 250 500
cd qmeter_dulya_fit
python fit_events.py 0 250 500      # ~1-2 min/event; writes results/
python plot_results.py              # writes plots/ and plots/results_table.md
```

## Findings so far (run 2022-09-23_00-54-02, cc = −1.39)

| event | P online | P area (Q-meter + Dulya) | Δ\|P\| / \|P online\| |
|---|---|---|---|
| 0 | −0.1745 | −0.178 ± 0.002 | +2.0 % |
| 250 | −0.3190 | −0.318 ± 0.005 | −0.4 % |
| 500 | −0.3144 | −0.322 ± 0.003 | +2.4 % |

- With the same cc, the area from a physical Q-meter background agrees with the online
  baseline + wing-polynomial area to within 2.5 %, once only the χ″ (absorptive) part is
  summed. Summing the full output signal instead is ~15 % low, because the dispersive (χ′)
  part leaks a 1/f tail into the wings that the online polynomial treats as background.
- The background treatment therefore contributes little; the online polarization
  uncertainty is dominated by the calibration constant cc.

## Known issues in upstream inputs

- `data_creation/fitting/baseline_fits_*.yaml`: every Q-meter fit has at least one
  physically impossible parameter (negative L0, Rcoil, r, M, Cstray, …) because all
  21 parameters are fitted unbounded. These feed the RGC training-data generator.
- `data-d`: 133 of 343 run files are empty; `current_*` files are partial copies of
  run files; 0.3 % of events have `pol ≠ cc × area`; cc values are hand-entered online
  values that vary from −0.82 to −20 across periods.

## Next steps

1. Refit the data-d baselines with the bounded physical circuit (`qmeter_dulya_fit/model.py`).
2. TE-based cc recalibration per period from the 1,587 TE events.
3. Regenerate MC from physical parameters and retrain the ML models.
