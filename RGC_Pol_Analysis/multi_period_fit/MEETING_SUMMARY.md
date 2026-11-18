# Raw-sweep Q-meter + Dulya fit vs online polarization — RGC deuteron, 30 events

## What was done

- **Events:** 30 strong-signal `Polarize` events from 8 online-cc periods (Jul 2022 – Mar 2023),
  2 with P > 0 and 2 with P < 0 per period where available (cc −3.93: only P > 0).
  The cc = −14.5 period (Jul 21 – Sep 6 2022) is included as a known problem period;
  June – early July 2022 (cc −5.25) is excluded (online processing unusable).
- **Fit:** the raw Q-meter sweep (no stored baseline, no wing polynomial) is fitted with a bounded,
  physical Liverpool Q-meter circuit whose coil contains the Dulya deuteron susceptibility
  (χ″ absorption lineshape, χ′ from Kramers–Kronig). 9 circuit + 6 lineshape parameters.
- **Polarization:** P_area = online cc × Σ(χ″-driven signal). Same cc as online, so the comparison
  isolates the background / signal extraction.
- **Regeneration:** each event's fitted parameters + measured white noise → synthetic raw sweep;
  the online chain (stored baseline, cubic wing polynomial, 25–75 % sum, × cc) is run on it.
- Runtime: ~9 min for 30 events on 14 cores (+12 min wider search for 6 initially failed fits).

## Results (good periods: 26 events, Sep 2022 – Mar 2023)

| | value |
|---|---|
| Fit residual (raw sweep, ~2 V scale) | median 32 µV, max 68 µV (signal peaks 1–3 mV) |
| P_area vs P_online | median −2.6 %; 19/26 within ±5 %, 4 within 5–8 %, 3 at ≈ 12 % |
| Absolute offset P_area − P_online | median −0.010, same sign for P > 0 and P < 0 |
| Closure (online chain on regenerated sweep) | median 2.1 %, max 7.5 % |
| Lineshape P (horn ratio, no cc) / P_online | median 1.08 (known +8 % shape-method bias) |

| period cc | n (+/−) | median mismatch | max \|mismatch\| |
|---|---|---|---|
| −1.39 | 4 (2/2) | 0.0 % | 2.9 % |
| −1.41 | 4 (2/2) | −1.7 % | 4.5 % |
| −1.42 | 4 (2/2) | −2.0 % | 7.6 % |
| −3.37 | 4 (2/2) | −2.4 % | 12.0 % |
| −3.65 | 4 (2/2) | −2.1 % | 12.2 % |
| −3.67 | 4 (2/2) | −6.6 % | 12.2 % |
| −3.93 | 2 (2/0) | −4.8 % | 5.1 % |
| **−14.5 ⚠** | 4 (2/2) | +28.9 % | 65.0 % |

## Take-aways

1. **From Sep 2022 on, a physical background model reproduces the online polarization to a few %,**
   for both polarization signs. The online background treatment is not the dominant error.
2. **The right-wing "leg" (signal below zero past the second pedestal) is real χ′ dispersion:**
   it flips sign with P in every period with both signs (plot 4, right). The online wing polynomial
   removes it as background; summing χ″ only gives the same area as online.
3. **Regenerated signals from fitted parameters reproduce the data** (residual ~30 µV) and the online
   chain run on them returns the online P within ~2 % → the fitted parameters are a valid basis
   for Monte Carlo generation.
4. **Lineshape parameters are stable across periods:** η ≈ 0.11, wq ≈ 82–83 kHz, g 0.02–0.045;
   f0 shifts by period (32.677 – 32.700 MHz).
5. **cc = −14.5 period is inconsistent:** area-method mismatches of −32 % to +65 %, lineshape P
   0.17–0.26 where online claims 0.5–1.0 (one event at +0.998), closure off by 22–37 %, lineshape fit
   at parameter bounds. Supports the earlier finding that cc = −14.5 is ~2.3× too large.

## Open points / questions for the group

- Was cc = −14.5 (Jul 21 – Sep 6 2022, ~18.8k Polarize events) known to be wrong? Relabel or exclude?
- The ≈ 12 % outliers in good periods (3 events) and the small negative offset (−0.01 in P)
  — background-shape systematics to understand before quoting an uncertainty.
- Noise in the regeneration is white; real residuals show structure near the horns (lineshape
  mismatch) — needs a better noise / lineshape model for the ML training set.

## Plots (`plots/`)

1. `1_polarization_vs_online.png` — fitted vs online P per event; mismatch per period (▲ P>0, ▼ P<0)
2. `2_regenerated_vs_original.png` — raw sweep vs regenerated; online-processed signal vs online chain on regenerated
3. `3_signal_decomposition_gallery.png` — χ″ (∝ P) and χ′ parts vs online fitsub
4. `4_closure_and_dispersion_test.png` — closure test; χ′ sign-flip test
5. `5_lineshape_parameters.png` — fitted η, g, wq, f0 per period
6. `tables.md` — per-period and per-event numbers
