# qmeter_dulya_fit

**Purpose:** extract P from the **raw sweep** with a physical model, without the stored baseline or the wing polynomial.

**Model (`model.py`, used by all other folders):**
- Liverpool Q-meter circuit with the deuteron susceptibility inside the coil:
  L = L₀(1 + 4πη χ), χ = χ′ − iχ″.
- χ″ is the powder-averaged Dulya spin-1 lineshape, from `data_creation/physics/lineshape.py`.
- χ′ is its Kramers–Kronig (Hilbert) partner.
- 9 circuit parameters, all bounded to physical values: U, Cknob, trim, φ, DC, L₀, R_coil, α, C_stray.
- 6 signal parameters: A, P, f0, wq, η, g.

**Area method (`fit_events.py`):** P_area = online cc × Σ(χ″-driven part of the fitted signal).
It uses the same cc as online, so the comparison tests only the background and signal extraction.

**Results (3 events, run 2022-09-23, cc −1.39):**

| event | P online | P area | difference |
|---|---|---|---|
| 0 | −0.175 | −0.178 ± 0.002 | +2.0 % |
| 250 | −0.319 | −0.318 ± 0.005 | −0.4 % |
| 500 | −0.314 | −0.322 ± 0.003 | +2.4 % |

- **Online background is fine:** it agrees within 2.5 %, so the online background subtraction is not the main error. That leaves cc.
- **Sum χ″ only:** summing the full output, including χ′, comes out ~15 % low. The χ′ dispersion tail is what the online wing polynomial removes.

```bash
python fit_events.py 0 250 500    # ~1–2 min per event -> results/
python plot_results.py            # -> plots/1–3, plots/results_table.md
```
