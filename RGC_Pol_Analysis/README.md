# RGC_Pol_Analysis

Offline analysis of the CLAS12 RGC polarized-deuteron target NMR data. The goal: fit the raw NMR sweeps
with a physical model, generate realistic Monte Carlo from the fitted parameters, and train ML to extract
the polarization faster and more precisely than the traditional fits.

Builds on `data_creation/` and `ml/` in this repository. Raw data (`data_creation/data-d`) is not in the repo.
Each folder has its own README with purpose, method, results and run commands.

| Folder | Purpose | Main result |
|---|---|---|
| `online_signal_inspection/` | How the online P is made | P = hand-entered cc × summed area; no lineshape fit |
| `qmeter_dulya_fit/` | Physical raw-sweep model (Q-meter + Dulya χ in the coil) and area method | Matches online within 2.5 % at the same cc; `model.py` is used everywhere |
| `baseline_refit/` | Replace the unphysical upstream baseline fits | 0/107 unphysical (upstream 64/64); rms 66 → 20 µV |
| `multi_period_fit/` | Fit 30 events across 8 periods; regenerate signals | Median −2.6 % vs online; closure 2 %; parameters are a valid MC basis |
| `te_calibration/` | Recalibrate cc from TE runs | No TE run reaches equilibrium, so cc cannot be recalibrated from TE |
| `lineshape_polarization/` | P without cc, from the spin-1 lineshape; check every period | 56 % of events consistent (P −0.51…+0.55); cc −14.5 on the 5T channel and −2.36 (19 %) are wrong; lineshape conflicts with the area in fast build-ups, so it is valid only on stable plateaus, \|P\| ≳ 0.3 |

## Overall findings

- **Background is not the problem:** from Sep 2022 to Mar 2023 the online background treatment is good to a few %. The weak point is the hand-entered cc.
- **The cc −14.5 period is wrong on the 5T channel:** Jul 28 – Sep 6, 2022, 12,198 events. Online P is ~3–4× too high, and 36 % of events claim more than 55 %, which is impossible for deuterons. It is most likely the 400 mV-channel cc kept after switching channels. cc −2.36 (Sep 6) is 1.76× too high. These events are part of the PT-NMR-ML training input with the wrong cc.
- **Lineshape and area disagree in fast DNP build-ups:** the area grows ~9× at fixed tune while the lineshape P stays ~0.35. The likely cause is no common spin temperature yet. Lineshape labels must come from stable plateaus.
- **Unusable early data:** Jun – early Jul 2022 online processing is unusable. The 400 mV channel signals are distorted.
- **Upstream inputs have problems:**
  - `data-d`: 133 of 343 files are empty, and the `current_*` files partly duplicate run files.
  - Upstream baseline fits are unphysical.
  - The RGC generator scales the signal amplitude with the online cc.

## Next steps

1. Build polarization labels that don't rely on the online P or cc. Use the lineshape on stable high-P plateaus, and carry it to lower P within a plateau's fixed cc and tune.
2. Sample lineshape and circuit parameters, plus per-event background drift, from the raw-sweep fits.
3. Generate physical MC (signal inside the Q-meter via χ) and retrain the ML models.
