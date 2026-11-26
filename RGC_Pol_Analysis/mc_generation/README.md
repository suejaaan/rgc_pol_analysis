# mc_generation

**Purpose:** generate realistic Monte Carlo NMR sweeps for ML training. They are built from the fitted parameters
of real events and labelled with a known P. No calibration constant (cc) is used anywhere: the signal size per
unit polarization comes from the cc-free lineshape labels (`../label_dataset/`).

## Stage 4: parameter library (`stage4_library.py`, ~10 s)

**One library record = one fitted real event.** After the extension below there are 2,316 records from 27 tune segments, 484 of them Stage 3 labels. Everything that is correlated stays together and must be **drawn together**:

| Per record | Source |
|---|---|
| Circuit (9) | shared Cknob, trim, L0, Rcoil, alpha, Cstray of the segment, plus the event's own U, phase, DC offset |
| σ | Gaussian broadening of the segment (3.7–4.8 kHz in label segments) |
| g, f0 | the event's own fit |
| k_seg = A(r+1)/P | amplitude per unit polarization of the segment; constant within a segment to 0.2–4 % |
| Noise | the event's fit residual minus the misfit template (512 bins) |
| η, wq | fixed, 0.111 and 0.0825 |

**Why records are drawn whole:**
- **Different circuits, different k:** the circuit is degenerate, so k_seg differs 70× between segments, because it belongs to that segment's circuit solution.
- **g and σ trade off:** for example, in the cc −14.5 segments, g sits at its bound and σ is smaller.

Mixing circuit, σ, g and k from different events would make unphysical combinations.

**Misfit template:** the median fit residual of the labels, in the signal frame (larger horn on the right), in units of the absorption peak.
- **Size:** up to ±5 % of the peak at the horns and edges.
- **Stable:** the same shape for |P| 0.30–0.36, 0.36–0.42 and 0.42–0.56.
- **Use:** it explains 62 % of the residual power at the horns. The MC adds template × peak, mirrored with the sign of P, so MC signals carry the same misfit as data.

**Noise:** the remainder of the residual is real, **coloured** noise.
- **Spectrum:** lag-1 autocorrelation 0.85, with slow wiggles dominating, plus spikes in the first and last bins.
- **Size:** rms 10 / 16 / 47 µV (10 / 50 / 90 % quantiles).
- **Use:** the MC adds real noise samples, not white noise.

**Quality classes:**
- `label` (484): all parameters trusted.
- `circuit` (248, mainly the cc −14.5 segments, |P| 0.08–0.43): the circuit and noise are usable; g and k are less certain.
- `extra` (1,349) and `extra_new_tune` (235), from the extension below: non-plateau events fitted with their segment's shared circuit. They add noise samples and per-event drift. Their P and g are not labels.

**Not used:** the stored-baseline refits (`../baseline_refit/`). They are a different circuit family, and no joint fit chose them.

## Stage 4 extension (`stage4_extend.py`, 91 min)

- **Part A: new tune states.** Every Stage 0 segment with ≥ 75 clean candidates and no shared circuit yet gets a segment-joint fit (16 segments). It uses the 10 strongest-signal events, starts from the good circuits of all label segments and the baseline refit, and estimates the amplitude start from the data.
- **Part B: extra records.** Up to 60 random non-picked events per segment (old and new) are fitted with the segment's shared circuit fixed. They add noise samples and per-event drift. Their P is not a label.

**Results:**
- **12 of 16 new segments failed:** 18, 34, 37–52 (Jun–Jul 2022) and 77.
  - σ ran to its 30 kHz bound.
  - The wing residual is 50–100 µV, against ~10 µV in good segments.
  - The circuit model does not describe that early hardware state with weak signals. The baseline refit also fit Jun–Jul poorly.
  - These segments are dropped.
- **4 new segments kept:** 95, 121, 124 and 150, which brings the library to **27 tune segments**.
  - Segment 121 has almost no signal, so its P and k are not meaningful. Use it only for its circuit and noise.
  - The new circuits lie close to existing tune clusters, so the gain in circuit diversity is small.
- **Extra records:** 1,584 of the 1,620 in kept segments pass (rms ≤ 1.5× the segment's median, in the signal region and in the wings).

| Library after the extension | |
|---|---|
| Records | **2,316**: label 484 · circuit 248 · extra 1,349 · extra_new_tune 235 |
| Tune segments | 27 |
| Noise samples | 2,316 (rms 9 / 14 / 46 µV; lag-1 autocorrelation 0.83) |

**Limit:** the MC can only draw from 27 tune states, most of them from Sep 2022 – Mar 2023. The Jun–Aug 2022 tunes
(before the 5T channel was stable) are missing, apart from the cc −14.5 segments.

## Stage 5: MC generator (`stage5_generate.py`) and realism check (`stage5_check.py`)

Per MC sweep:
1. **P:** drawn in [−0.65, +0.65], 50/50 sign. The |P| mixture is 12 % in 0–0.15, 23 % in 0.15–0.25, 55 % in 0.25–0.55 and 10 % in 0.55–0.65.
2. **Record:** one library record, drawn whole from the sweep's split. U, phase and DC offset are jittered by 0.5× their within-segment spread.
3. **Amplitude:** A = k_seg · P / (r(P) + 1), the tune's own signal size per unit P (cc-free).
4. **Sweep:** Q-meter(circuit, χ′ − iχ″) + misfit template × peak + real noise of the **same tune segment**. The noise is either:
   - a plain library sample (50 %), or
   - the segment's mean noise + a random-sign difference of two other samples / √2.

**The noise is mostly a systematic background pattern of each tune.** It must come from the same segment and must not be sign-flipped as a whole.

| Twin test noise (484 label events, MC from the same record + other noise) | Classifier AUC |
|---|---|
| Same segment, plain sample, no flip | 0.46 |
| Same segment, mean + random-sign difference | 0.43 |
| Same segment, whole sample sign-flipped | 0.66 (recognisable) |
| Noise from any segment | 0.74 (recognisable) |
| **Final generator (both allowed modes)** | **0.45 — indistinguishable from real** |

All single features have AUC ≤ 0.46: wing rms and autocorrelation, high-frequency noise, peak, area, horn ratio,
roughness, background shape. The twin difference is 0.50 × √2 × noise, so the residual is mostly reproducible
within a tune. Plot: `plots/stage5_check.png`.

**Signal segments:** only the 20 segments with a reliable k generate signals (16 label segments + 95, 124, 150, 154).
The cc −14.5 segments (k from low-P lineshape fits) and segment 121 (no signal) are left out.

**Splits are by circuit group.** Segments that ended on the same shared circuit stay together, and no circuit,
record or noise sample of a val/test tune appears in train.

| Split | Segments | Share |
|---|---|---|
| train | 93, 95, 97, 101, 103, 110, 112, 115, 116, 122, 123, 124, 125, 133, 150, 154 | 80 % |
| val | 105, 135 | 10 % |
| test | 96, 118, 119 | 10 % |

**Output** (`mc_data/<name>/`, gitignored): parquet shards of 50k rows in the upstream layout.
- Columns `"0"`…`"511"`: the sweep in volts (float32).
- Then: `P`, `Pzz`, `split`, `segment`, `record_id`, `quality`, `noise_id`, `noise_aug`, `A`, `k_seg`, `sigma`, `g`, `f0`, `signal_peak`, `area_chi2`, `noise_rms`, the 9 circuit parameters and `SNR`.
- `freq.npy` holds the common frequency grid; `manifest.yaml` records the settings and the shards.
- Speed: 10k sweeps in 19 s on 14 cores. 1M takes ~30 min and ~2.9 GB.

## Outputs

| File | Content |
|---|---|
| `library/segments_extra.yaml`, `extra_records.csv`, `npz_extra/` | Extension: joint fits of the new segments, extra records and their arrays |
| `library/param_library.npz` | Records: circuit, σ, g, f0, k_seg, A, P_fit, P_err, segment, time, quality, freq, noise, residual, peak; misfit template |
| `library/library_summary.yaml` | Counts, σ and k per segment, g and noise quantiles, template numbers |
| `mc_data/<name>/` | MC datasets: parquet shards, `freq.npy`, `manifest.yaml`, `check.yaml` (gitignored) |
| `plots/stage5_check.png` | Real vs MC twins: example sweeps, twin difference, feature distributions, classifier AUC |
| `plots/stage4_library.png` | Circuit states, σ per segment, g vs \|P\|, misfit template by \|P\|, residual vs noise example, noise spectrum |

```bash
python stage4_extend.py      # optional extension (~1.5 h)
python stage4_library.py     # builds the library (merges the extension if present)
python stage5_generate.py 10000 test10k && python stage5_check.py test10k
python stage5_generate.py 1000000 mc1m
```
