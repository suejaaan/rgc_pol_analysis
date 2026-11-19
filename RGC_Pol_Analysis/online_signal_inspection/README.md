# online_signal_inspection

**Purpose:** see how the online RGC polarization is produced, before replacing any of it.

**What it does:** for 3 events of run 2022-09-23 (`data-test`), plots each step of the online chain:
raw Q-meter sweep → minus the stored baseline sweep → minus a cubic polynomial fitted to the wings →
`fitsub` → summed over the central 25–75 % of bins = `area` → `pol = cc × area`.

**Result:** the online P depends only on the summed area and a hand-entered calibration constant `cc`.
There is no lineshape fit, and every event in a period shares the same baseline sweep. Any error in
`cc` goes straight into P.

```bash
python plot_signal_comparison.py 0 250 500    # -> signal_comparison.png
```
