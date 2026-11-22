# label_dataset

**Purpose:** build polarization labels without any calibration constant (cc). Pick events on stable
high-polarization plateaus, fit them well, and label P from the lineshape. The fitted parameters later
feed the Monte Carlo (`../mc_generation/`, to come). Online cc and P are kept only as reference columns,
never used for a decision.

Stages 0–3 of the workflow live here. Stage 2 (high-quality fits) and Stage 3 (label validation) are to come.

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

## Outputs

| File | Content |
|---|---|
| `results/events.csv` | One row per event: identity, label, channel, baseline, segment, online area/P/cc, tune, microwave, beam, temperature, dose, quality flags |
| `results/segments.csv` | One row per tune segment |
| `results/plateaus.csv` | One row per plateau: time span, sign, median area, spread, microwave frequency, beam-on fraction |
| `results/plateau_events.csv` | The 732 events for Stage 2 |
| `plots/stage0_segments.png` | Online area vs time, coloured by segment |
| `plots/stage1_plateaus.png` | One panel per segment: plateaus (green) and picked events (red) |

```bash
python stage0_inventory.py && python stage1_plateaus.py
```
