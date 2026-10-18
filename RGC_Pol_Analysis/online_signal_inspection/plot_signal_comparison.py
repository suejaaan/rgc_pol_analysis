import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA_FILE = REPO / "data_creation/data-test/2022-09-23_00-54-02__2022-09-23_14-13-02.txt"
OUT_PNG = Path(__file__).resolve().parent / "signal_comparison.png"
EVENTS = [0, 250, 500]
SUM_RANGE = (0.25, 0.75)


def load_events(path, indices):
    wanted = set(indices)
    events = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i in wanted and line.strip():
                events[i] = json.loads(line)
            if len(events) == len(wanted):
                break
    return [events[i] for i in indices]


def wing_mask(n, wings):
    idx = np.arange(n) / n
    return ((idx >= wings[0]) & (idx < wings[1])) | ((idx >= wings[2]) & (idx < wings[3]))


def main():
    indices = [int(a) for a in sys.argv[1:]] or EVENTS
    records = load_events(DATA_FILE, indices)

    fig, axes = plt.subplots(len(records), 3, figsize=(16, 4.2 * len(records)), squeeze=False)
    print(f"{'event':>5}  {'start_time':19}  {'label':9}  {'cc':>6}  {'area':>8}  {'pol':>8}  {'cc*area':>8}")

    for row, (idx, r) in zip(axes, zip(indices, records)):
        f = np.asarray(r["freq_list"])
        raw = np.asarray(r["phase"])
        baseline = np.asarray(r["baseline"])
        basesub = np.asarray(r["basesub"])
        fitcurve = np.asarray(r["fitcurve"])
        fitsub = np.asarray(r["fitsub"])
        n = len(f)
        wings = r["wings"]
        wmask = wing_mask(n, wings)
        lo, hi = int(SUM_RANGE[0] * n), int(SUM_RANGE[1] * n)
        cc, area, pol = r["cc"], r["area"], r["pol"]
        print(f"{idx:5d}  {r['start_time'][:19]}  {r['label']:9}  {cc:6.2f}  {area:8.4f}  {pol:+8.4f}  {cc * area:+8.4f}")

        ax = row[0]
        ax.plot(f, raw, color="tab:blue", lw=1.2, label="raw sweep (phase)")
        ax.plot(f, baseline, color="tab:gray", lw=1.2, ls="--", label=f"baseline ({Path(r['base_file']).name})")
        ax.set_title(f"evt {idx}: raw sweep vs baseline")
        ax.set_ylabel("Q-meter output [V]")

        ax = row[1]
        ax.plot(f, basesub, color="tab:purple", lw=1.2, label="basesub = raw − baseline")
        ax.plot(f, fitcurve, color="tab:red", lw=1.0, ls="--", label="wing poly fit")
        ax.scatter(f[wmask], basesub[wmask], s=4, color="tab:red", alpha=0.4, label="wing bins used for fit")
        ax.set_title("baseline-subtracted signal")

        ax = row[2]
        ax.plot(f, fitsub, color="tab:green", lw=1.2, label="fitsub = basesub − wing fit")
        ax.fill_between(f[lo:hi], 0, fitsub[lo:hi], color="tab:green", alpha=0.2, label=f"area sum region ({SUM_RANGE[0]:.2f}–{SUM_RANGE[1]:.2f})")
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set_title(f"final signal | area={area:.4f}  cc={cc}  →  P = {pol:+.2%}")

        for ax in row:
            ax.set_xlabel("frequency [MHz]")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"{DATA_FILE.name}  ({records[0]['label']}, {records[0]['channel']['name']})", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
