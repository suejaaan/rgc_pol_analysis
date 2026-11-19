# baseline_refit

**Purpose:** the upstream Q-meter baseline fits (`data_creation/fitting/baseline_fits_*.yaml`) fit all
21 circuit parameters without bounds. Every one of them has unphysical values (negative L₀, R_coil,
C_stray, …), and they feed the PT-NMR-ML training-data generator. This folder refits every baseline
sweep in `data-d` with the bounded physical circuit of `../qmeter_dulya_fit/model.py`.

**Method:**
- `collect_baselines.py`: finds the 107 unique baseline sweeps (deduplicated over run files and
  `current_*` copies) and maps every event to its baseline. Output in `cache/`, which is used by the
  other folders.
- `refit_baselines.py`: 9 bounded parameters, 200 random starts per baseline, 14 cores, ~5 min.
  A second pass re-seeds poor fits from their neighbours in time. 3 edge bins are masked.
- `plot_refit.py`: `plots/1_example_fits`, `2_old_vs_new`, `3_parameters_vs_time`, `4_fit_quality`.

**Results** (`results/baseline_fits_physical.yaml`):

| | upstream | this refit |
|---|---|---|
| fits with unphysical parameters | 64 / 64 | 0 / 107 |
| median rms (the 64 baselines fitted by both) | 66 µV | 20 µV |
| event-weighted median rms | — | 16 µV |

- **Poor fits:** 14 baselines, all from June–July 2022 and covering 1,772 events, still fit poorly (rms > 200 µV).
- **At a bound:** 23 baselines have at least one parameter at a bound.
- **Which file to use:** `baseline_fits_physical.yaml` is the final version; `_round1` is the first pass, kept for reference.

```bash
python collect_baselines.py && python refit_baselines.py && python plot_refit.py
```
