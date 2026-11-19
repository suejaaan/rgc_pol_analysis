# multi_period_fit

**Purpose:**
- Check that the raw-sweep Q-meter + Dulya fit works across the whole run, not just one file:
  8 online-cc periods, both signs of P.
- Check that the fitted parameters regenerate realistic signals, which is the basis for Monte Carlo.

**Method:**
- `select_events.py`: 30 strong Polarize events, 2 with P > 0 and 2 with P < 0 per period, from
  Jul 2022 to Mar 2023.
- `fit_selected.py`: joint fit of the raw sweep, started from the online fitsub and the refitted
  baseline. P_area = online cc × Σχ″.
  - It then regenerates each sweep from the fit plus measured noise and runs the online chain on it
    (closure test). 14 cores, ~9 min.
- `refit_failures.py`: wider search for fits with rms > 80 µV.
- `plot_report.py`: `plots/1–5`, `plots/tables.md`.

**Results** (details: `MEETING_SUMMARY.md`):
- **Agreement with online (Sep 2022 – Mar 2023, 26 events):** P_area vs online has a median of −2.6 %; 19/26 are within ±5 %.
- **Fit residual:** ~30 µV on a ~2 V sweep.
- **Closure test:** the online chain on regenerated sweeps returns the online P within a median of 2.1 %. The fitted parameters are a valid basis for MC.
- **χ′ "leg":** the leg past the right pedestal is real χ′ dispersion. It flips sign with P.
- **Stable parameters:** η ≈ 0.11 and wq ≈ 82–83 kHz across periods. f0 shifts by period (32.677–32.700 MHz).
- **The cc −14.5 period is inconsistent:** mismatch −32 % to +65 %, and the lineshape gives P ≈ 0.2 where online says 0.5–1.0.

```bash
python select_events.py && python fit_selected.py && python refit_failures.py && python plot_report.py
```
