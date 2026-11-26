# label_dataset

**Purpose:** build polarization labels without any calibration constant (cc). Pick events on stable
high-polarization plateaus, fit them well, and label P from the lineshape. The fitted parameters later
feed the Monte Carlo (`../mc_generation/`, to come). Online cc and P are kept only as reference columns,
never used for a decision.

Stages 0–3 of the workflow live here. 

## Stage 0: event inventory (`stage0_inventory.py`, ~7 s)

- Reads every event once, using the same file list as `../baseline_refit/collect_baselines.py`, so the
  event indices match `event_baseline_map.csv`.
- Removes duplicate `current_*` copies.
- **Tune segment:** same channel, same stored baseline sweep, no gap over 6 h. The diode and phase tune
  voltages are constant inside every segment.
- **Candidate:** Polarize label, 5T channel, valid sweep, pol = cc × area, not a glitch.

| | Events |
|---|---|
| Unique events | 70,156 |
| Polarize | 63,072 |
| **Candidates** | **55,191**, in 55 segments (+24,407 / −30,782) |

Segments often span several online cc values with the same stored baseline. That is useful: inside a
segment, the area is proportional to P whatever cc was typed in.

## Stage 1: plateau selection (`stage1_plateaus.py`, ~3 s)

An event is on a plateau when, over the previous hour, all of these hold:
- the area changes by less than 3 % per hour;
- the scatter about the trend is below 2 %;
- the microwave frequency moved by less than 0.05 GHz;
- there are at least 8 events, all with the same sign;
- at least 2 h have passed since the segment start, a sign change or a microwave jump.

It also needs a **large signal**: |area| ≥ 50 % of the segment's largest |area| of that sign. This is a
relative cut, so no cc is involved. Plateaus shorter than 1 h are dropped. Up to 10 events per plateau
(spaced ≥ 30 min apart) and 50 per segment are picked.

| | |
|---|---|
| Plateaus | 116 (765 h) in 23 segments, 62 with + sign and 54 with − |
| **Picked events** | **732** (+367 / −365) |
| Period verdict of picks (reference only) | consistent 533 · online wrong 195 · unresolved 4 |

The |P| > 0.3 cut is **not** applied here. It is applied to the lineshape P after the Stage 2 fits. The
picks from the −14.5 period (true P ≈ 0.1–0.2) will most likely fall below it.

The two area levels in segment 78 (Jul 30–31) follow microwave-frequency tuning
(140.35 → 140.47 GHz). It is real DNP behaviour, not a data problem.

## Stage 2: high-quality fits (`stage2_fit.py`), pilot

- **Fit:** the full Q-meter + Dulya fit with all 15 parameters free, η and wq included.
  - P starts at 0.3 with the signal's sign, never at the online P.
  - Each event is refitted from P = 0.12 and 0.5 to check P stability.
  - Covariance and residual are saved.
- **Pilot:** one event per plateau (116 events), 68 min on 14 cores.
- **Gate:** 0/116 pass. The thresholds need revising, and the model has a real shape problem:
  - **Residual at the horns:** the residual has the same shape in every event, ±4–5 % of the signal peak at the horns (±75 kHz) and edges (±100 kHz). The powder grid accounts for < 0.8 %, so this is a lineshape-model mismatch (`plots/stage2_pilot_residuals.png`).
  - **η free is degenerate in some segments:** in segments 125 and 133, η runs to 0.2–0.23 and P scatters by 16–19 %. Elsewhere η = 0.108–0.115 and wq = 0.082–0.083, close to the fixed values used before.
  - **Bound flags:** most come from U and Cstray at their lower limits. The bound check needs to tell physical limits apart from fit problems.
  - **Noise reference:** the noise σ from bin differences sees only white noise, so the rms/noise cut is too strict. The wing rms is a better reference.
- **cc-free consistency where η is stable:** P_lineshape / area is constant within a segment to 1.4–4.6 %.
- **Lineshape vs online:** lineshape P is 5–22 % above online in the consistent periods, more at P ≈ 0.33.
- **cc −14.5 picks:** P_lineshape is 0.12–0.30 with large errors, and all fail. As expected, they are not labels.

**Model test (`model_variants.py`, 20 events, ~11 min):** η and wq fixed at 0.111 and 0.0825, plus a Gaussian
(inhomogeneous) broadening σ convolved with χ″.
- **Fit:** σ = 4–5 kHz in almost every event. The signal-region residual drops from 37.6 to 27.2 µV, and signal/wings from 2.27 to 1.83.
- **P barely moves:** |P| shifts by a median of −0.005 (range −0.026 … +0.003). The horn-shape model systematic is ~1–2 %.
- **Leftover structure:** oscillations of ±2–3 % of the peak remain between the horns and shoulders (75–100 kHz). The powder grid is ≤ 0.7 %, so the cause is still open (`plots/stage2_model_variants.png`).

**Circuit degeneracy (main issue):**
- **Several circuits per tune:** events in the same tune segment converge to different circuit solutions. In segment 125, U ≈ 2.85 gives P ≈ 0.26–0.30, while U ≈ 0.026 gives 0.36–0.40. Segments 119 and 133 show the same.
- **This dominates the label error:** the χ′/χ″ mixing of the circuit trades off against the horn ratio. The P change is up to 0.1–0.15, far more than the lineshape-model effect.
- **Fix: segment-joint fit (`segment_fit.py`).** All events of a tune segment share one circuit (Cknob, trim, L0, Rcoil, alpha, Cstray) and one Gaussian σ.
  - Per event, only U, phase, DC offset and the signal (A, P, f0, g) are free; η and wq are fixed.
  - Starts: good circuits from all segments, screened on 3 events. The best 3 get the full joint fit (sparse Jacobian), and stuck events are rescued from their neighbours.
- **Test on segments 112, 119, 125 and 133 (8 events each, 19 min):**
  - **Fit quality:** all four converge. The signal-region rms is 13–22 µV (single-event fits: ~35 µV), σ = 4.2–4.8 kHz, and U is the same for every event of a segment.
  - **cc-free consistency:** P / area is constant within each segment to **0.9–2.2 %**. For the single-event fits it was 4–40 %.
  - **Circuit systematic:** the best three starts agree on P within ~±0.01 (up to 0.02 in segment 112).
  - **Segments 125 and 133** end on the same circuit (same March tune). The wrong circuit families of the pilot (DC offset at ±50 V, P 0.26–0.30) are gone.
  - **Lineshape vs online:** the lineshape P / area is within −5 … +8 % of the online cc. The 5–22 % offset in the pilot was a circuit artefact.
  - **To watch:** g grows as P falls within a segment (0.001 → 0.015), and g sits at its lower bound at high P.
  - Plot: `plots/stage2_segment_test.png`.

### Full run (`stage2_full.py`, 66 min on 14 cores)

1. **Segment-joint fit:** 10 events in each of the 23 plateau segments give the shared circuit and σ (52 min).
2. **Event fits:** all 732 picked events are fitted with that circuit fixed, ~3 s each.
3. **Labels:** an event is a label if:
   - the fit is ok;
   - |P| ≥ 0.3;
   - the segment's P / online-area spread is ≤ 5 %.

| | |
|---|---|
| Fits ok | 732 / 732 |
| **Labels** | **484** (+226 / −258) from 16 segments, \|P\| 0.30–0.54 (median 0.36) |
| Errors | fit error on P ~0.005 (median); circuit systematic ~0.002 (spread over the best three circuit starts) |
| σ | 3.7–4.8 kHz in all consistent segments |
| Rejected | cc −14.5 segments: \|P\| 0.08–0.31, as expected. Segment 154 (unresolved): \|P\| 0.21. Segment 101: P/area spread 5.3 % |

Two things for Stage 3:
- **Sign asymmetry:** in segments with both signs, |P| / |area| differs between the two signs.
  - **Segment 96:** 7 % with the online area, but 0.5 % with our own fitted area, so the asymmetry is in the online area.
  - **Segment 101:** 10 % with both areas. This is the reason segment 101 fails the 5 % cut; each sign on its own is 0.4 % and 3.8 %.
  - **Effect of a per-sign cut:** judging each sign separately would add segment 101, giving 529 labels.
- **Slow drift:** within one sign, P/area drifts by 1–4 % over days, while U (gain) is constant to 0.1 %. This sets the label's statistical error. g sits at its lower bound in some segments and differs by sign in segment 96.

## Stage 3: label validation (`stage3_checks.py` ~75 min, `stage3_report.py`)

Six test segments (96, 103, 110, 119, 125, 133) are checked, still with no cc.

| Check | Result |
|---|---|
| **MC closure:** MC through each segment's own fitted circuit at known P = 0.28–0.56 (240 MC events), with white noise or the real residual added. Fitted from the data's own start circuits, never the truth. | Bias **−0.0006**, rms 0.003, flat in P, the same for both signs |
| **Model variation:** η ± 0.006, wq ± 0.0007, powder grid 128, g limit 1e-4 | Δ\|P\| 1–3 % (segment 133: 5 %). The g bound has no effect. Without the Gaussian, P shifts +0.006…+0.020, but the fit is ×2 worse, so it is excluded. |
| **Consistency:** P / area within segment and sign | event-to-event jitter 0.8 %, slow drift 1.5 % (max 3.7 %) |
| **Physics sanity** | \|P\| ≤ 0.539, none above 0.55. P changes smoothly: median 0.003/h, max 0.03/h. |

**Total error** (fit ⊕ consistency ⊕ circuit ⊕ model ⊕ closure, per label): median **3.6 %** relative (≈ ±0.014 in P), 90 % of labels below 5.0 %.
The largest terms are segment consistency and model, each ~2 %. See `plots/stage3_error_budget.png`.

**Open: sign vs the online cc.** Positive-P labels match the online cc (median 1.00). Negative-P labels read 9 % above it (median 1.09).
- **Not the fitter:** closure is symmetric in sign.
- **Not a fixed instrumental distortion:** the residual mirrors with the sign of P (`plots/stage3_sign_residuals.png`).
- **Remaining candidates:** the online area (segment 96: our own area is symmetric to 0.5 %), or a real non-equilibrium state after spin reversal (segment 101, 10 % with both areas; excluded).
- **Handling:** not added to the error; the 258 negative-P labels are flagged by their sign.

## Outputs

| File | Content |
|---|---|
| `results/events.csv` | One row per event: identity, label, channel, baseline, segment, online area/P/cc, tune, microwave, beam, temperature, dose, quality flags |
| `results/segments.csv` | One row per tune segment |
| `results/plateaus.csv` | One row per plateau: time span, sign, median area, spread, microwave frequency, beam-on fraction |
| `results/plateau_events.csv` | The 732 events for Stage 2 |
| `plots/stage0_segments.png` | Online area vs time, coloured by segment |
| `plots/stage1_plateaus.png` | One panel per segment: plateaus (green) and picked events (red) |
| `results/stage2/fits_pilot.yaml`, `npz/` | Stage 2 pilot fits: parameters, errors, covariance, residuals, gate |
| `results/stage2/segment_test.yaml`, `plots/stage2_segment_test.png` | Segment-joint test: P and P/area per event for the best three circuit starts vs the single-event pilot fits |
| `results/stage2/labels.csv` | **The labels**: one row per picked event, with P, errors, own parameters, gate flags and online reference columns |
| `results/stage2/segments.yaml`, `npz_full/` | Shared circuits per segment (best three starts); per-event sweep, model, parts, residual, covariance |
| `plots/stage2_labels.png` | Per segment: \|P\| and P / online area vs time, labels (blue) and rejected (red) |
| `results/stage3/labels_final.csv` | **Final labels** (484): P, total error and its components, parameters, reference online values |
| `results/stage3/error_budget.yaml`, `closure.yaml`, `variants.yaml` | Stage 3 numbers |
| `plots/stage3_error_budget.png` | MC closure, model variation per segment, error budget vs \|P\| |
| `plots/stage3_sign_residuals.png` | Median fit residual for +P and −P labels (mirror images) |
| `plots/stage2_pilot_residuals.png` | Stacked fit residuals of the pilot (systematic horn structure) and one example fit |

```bash
python stage0_inventory.py && python stage1_plateaus.py
python stage2_fit.py pilot            # single-event pilot (start circuits for the joint fit)
python stage2_full.py                 # segment-joint fit + all events -> results/stage2/labels.csv
python stage3_checks.py && python stage3_report.py   # closure, model variation -> results/stage3/labels_final.csv
```
