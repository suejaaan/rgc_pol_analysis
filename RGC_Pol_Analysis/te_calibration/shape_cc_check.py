"""Calibration-free cross-check of cc from the deuteron lineshape asymmetry.

For spin-1 with equal spin temperature, the ratio of the two transition intensities fixes P:
P = (r^2 - 1) / (r^2 + r + 1). Fitting the Dulya doublet (free amplitude) to the online `fitsub`
of strongly polarized events gives P_shape, and cc_shape = P_shape / area. At P >~ 0.3 this was
within ~5-10 % of the online value in both periods tested; at low P it is biased (see README).

Usage: python shape_cc_check.py            (periods defined in PERIODS)
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data_creation/data-d"
sys.path.insert(0, str(HERE.parent / "qmeter_dulya_fit"))
import model as M  # noqa: E402

PERIODS = {
    "2022-09-23 (data-test run)": ["2022-09-23_00-54-02__2022-09-23_14-13-02.txt"],
    "2023-03-15..17 (before TE)": ["2023-03-15_14-51-52__2023-03-16_03-47-29.txt",
                                   "2023-03-16_17-12-57__2023-03-17_06-38-24.txt"],
    "2023-03-23 (after TE)": ["2023-03-23_04-12-13__2023-03-23_13-56-26.txt"],
}
N_PER_FILE = 4
MIN_ABS_AREA = 0.03
LO = [-np.inf, -0.95, 32.6, 0.05, 0.0, 0.002]
HI = [np.inf, 0.95, 32.77, 0.15, 0.4, 0.08]


def shape_p(f, y):
    def mdl(s):
        return M.chi_components(f, *s)[0]

    best = None
    for p0 in (-0.6, -0.3, -0.1, 0.1, 0.3, 0.6):
        s0 = [1.0, p0, 32.682, 0.082, 0.09, 0.023]
        u = mdl(s0)
        s0[0] = np.dot(y, u) / np.dot(u, u)
        r = least_squares(lambda s: (mdl(s) - y) * 1e4, np.clip(s0, np.array(LO) + 1e-9, np.array(HI) - 1e-9),
                          bounds=(LO, HI), x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    return float(best.x[1])


def main():
    out = {}
    for name, files in PERIODS.items():
        rows = []
        for fname in files:
            recs = [json.loads(line) for line in open(DATA / fname)]
            pol = [r for r in recs if r["label"] == "Polarize" and abs(r["area"]) > MIN_ABS_AREA]
            for r in [pol[i] for i in np.linspace(0, len(pol) - 1, min(N_PER_FILE, len(pol))).astype(int)]:
                p = shape_p(np.asarray(r["freq_list"]), np.asarray(r["fitsub"]))
                rows.append({"time": r["start_time"][:19], "area": float(r["area"]), "cc_online": float(r["cc"]),
                             "P_online": float(r["pol"]), "P_shape": p, "cc_shape": p / float(r["area"])})
        ccs = np.array([x["cc_shape"] for x in rows])
        out[name] = {"events": rows, "cc_shape_median": float(np.median(ccs)),
                     "cc_shape_spread": float(np.std(ccs)), "cc_online_values": sorted({x["cc_online"] for x in rows})}
        print(f"{name:28s} cc_online {out[name]['cc_online_values']}  cc_shape median {np.median(ccs):+.2f} "
              f"(spread {np.std(ccs):.2f}, {len(rows)} events)")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "shape_cc_check.yaml").write_text(yaml.safe_dump(out, sort_keys=False))


if __name__ == "__main__":
    main()
