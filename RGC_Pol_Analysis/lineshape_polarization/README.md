# lineshape_polarization

**Purpose:** measure P without any calibration constant, and use it to check the online cc of every period.

**Method:**
- Fit the raw sweep with the Q-meter + Dulya model (`../qmeter_dulya_fit/model.py`), with η = 0.108 and wq = 82.5 kHz fixed.
- Take P from the ratio of the two spin-1 transitions: P = (r² − 1)/(r² + r + 1), with r = I₊/I₋.
- No cc, no area, no TE.
- Lineshape cc of a period = P_lineshape / online area.

**Assumption:** all three spin levels share one **spin temperature**. The area measures vector P
(∝ n₊ − n₋) in any state. The horn ratio measures tensor/vector polarization, and only converts to P
through that assumption.

## Results

**Verdict definitions.** Each period is judged on its ~6 strongest events. Fits with rms > 80 µV, and
signals weaker than 30 % of the period's strongest, are excluded.
- **Consistent:** both checks agree with online.
  - Lineshape P / online P is within ±12 %.
  - Area-method P / online P is within ±6 %.

  The online cc is fine within its usual ~5–10 % error. This is applied automatically in `compare_methods.py`.
- **Online wrong:** the lineshape is trustworthy for these events, and it rejects the online P. All of these must hold:
  - good fits;
  - lineshape \|P\| ≳ 0.3 on a stable plateau;
  - the area method reproduces the online area, so the signal extraction is fine;
  - the lineshape P differs from online by far more than 12 %, in the direction no known lineshape bias produces (lineshape lower than online);
  - fits with P fixed at the online value are clearly worse (rms ~2×, `hypotheses_online_wrong.png`).

  Only the cc can be wrong. The online P may also be physically impossible (> 0.55).
- **Unresolved:** the lineshape can't be trusted for these events, so it can't decide. Reasons:
  - low P (< 0.3);
  - possible non-equilibrium (see the build-up conflict below);
  - distorted signals that don't fit the model (400 mV channel).

  Online P may be right or wrong.
- **No signal / skipped:** no usable doublet, online processing unusable, or < 100 events.

The "online wrong" and "unresolved" labels are set by hand in `NOTES` in `compare_methods.py`, using these criteria.

| Verdict | Periods (online cc) | Polarize events |
|---|---|---|
| Consistent: lineshape and area agree with online within ~5–10 % | Sep 7, 2022 – Mar 19, 2023 (cc −1.34 … −1.57, −3.37 … −3.93). P from −0.51 to +0.55, median \|P\| 0.36, 80 % above 0.3 | 36,879 (56 %) |
| **Online wrong** | **−14.5, 5T channel** (Jul 28 – Sep 6, 2022, 12,198 events): online P ~3–4× too high, 36 % of events above 0.55. Likely the 400 mV-channel cc kept after switching back (−14.5 / −3.70 = 3.9). **−2.36** (Sep 6, 126 events): 1.76× too high | ~12,300 (19 %) |
| Unresolved | −14.5 and −15/−20 on the 400 mV channel (distorted signals); −1.122, −1.0, −4.0, −3.55 (Mar 22–23, 2023, low P; cc taken from an unfinished TE run) | ~8,000 (12 %) |
| No signal / skipped | −3.70; −5.25 (Jun – early Jul 2022, online processing unusable); periods with < 100 events | ~7,400 (11 %) |

## Conflict: lineshape vs area during build-ups

Along polarization build-ups at fixed cc, channel and stored baseline sweep:
- **Slow build-up (Sep 11, 2022):** the lineshape follows the area down to P ≈ 0.1, reading 0.04–0.10 high.
- **Fast build-ups (Sep 7, 2022; Mar 1, 2023):** the area grows ~9×, but the lineshape stays at 0.34–0.38. It only meets the area once the build-up slows.

Example, Mar 1, 16:25:
- **The area is small:** P_area = 0.043, and online gives 0.050.
- **The shape is lopsided:** the measured horns are 0.101 vs 0.160 mV, a ratio of 1.59 with noise of 4 µV. The model gives 1.08 for P = 0.05 and 1.51 for P = 0.38.
- **The fit prefers 0.38:** rms 9 µV, against 11 µV with P fixed at 0.05.
- **The fitter isn't the cause:** MC at known P = 0.05 and 0.385, generated through this event's circuit with its noise, is recovered as 0.048 ± 0.009 and 0.384 ± 0.007.
- **The area is right:** the fitted circuit (Cknob, phase) and the baseline sweep are unchanged from 16:25 to 18:35. With the gain fixed, the 9× area growth is real, so P really was ~0.04.

The likely explanation is that the tensor and vector polarization build up at different rates, so there is
no single spin temperature yet. The spin-temperature fit then matches the asymmetry and shrinks the
amplitude. An instrumental cause isn't excluded.

**Consequence:** lineshape P is a valid label only on **stable plateaus with |P| ≳ 0.3**. To test the
explanation, fit I₊ and I₋ with independent amplitudes, which gives vector and tensor polarization separately.

## Plots (`plots/`)

| File | What it shows |
|---|---|
| `lineshape_polarization.png` | Top: lineshape P of every fitted event vs time. Bottom: lineshape cc per period (log scale), labelled by the online cc. |
| `comparison_table.png` | Meeting table per period: online cc vs lineshape cc, \|P\| online / area method / lineshape, ratios, ΔP vs online, verdict colour. |
| `comparison_scatter.png` | Per event: area-method P and lineshape P vs online P, coloured by verdict. |
| `confirm_online_wrong.png`, `confirm_unresolved.png` | Problem events (measured χ″, area-normalized) with reference signals from consistent periods at the online P (red) and the lineshape P (blue). MC is used where no reference exists (P > 0.55). |
| `normalization_check.png` | Area normalization keeps shape differences: MC at P 0.09 vs 0.29 differs clearly, but two real "valid-period" events at online P 0.09 and 0.29 have the same shape. |
| `hypotheses_online_wrong.png`, `hypotheses_unresolved.png` | Each problem event in mV with two MC curves from its own fit, at P_online (red) and P_lineshape (blue), plus residuals and rms. Shows clearly that −14.5 and −2.36 reject the online P. |
| `buildup_test.png` | Left: lineshape P vs online P along 4 build-ups. Middle: horn ratio of the data vs MC expectation. Right: unnormalized signals along the Sep 7 build-up. |
| `buildup_example.png` | Two fast build-ups vs time: online P, our area P and lineshape P (top); the χ″ signals growing (bottom). |
| `lineshape_check_event.png` | The Mar 1 16:25 event: (1) model-free horn heights, (2) fits with P fixed at 0.05 and 0.38, (3) closure test of the fitter. |

## Scripts

| Script | Purpose |
|---|---|
| `run_lineshape.py` | Fit 6 high-signal events per period (14 cores, ~15 min) → `results/lineshape_fits.yaml` |
| `report_lineshape.py`, `compare_methods.py` | Tables and plots above |
| `confirm_shapes.py`, `confirm_hypotheses.py` | Problem-period checks |
| `buildup_test.py`, `buildup_example.py`, `check_lineshape_event.py` | Build-up test and the 16:25 event check |

```bash
python run_lineshape.py [cc ...]       # with cc values: rerun only those periods
python report_lineshape.py && python compare_methods.py
python confirm_hypotheses.py && python buildup_test.py && python buildup_example.py
python check_lineshape_event.py ["YYYY-MM-DD HH:MM:SS"]
```
