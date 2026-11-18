"""Visual check of the problem periods: does the signal shape follow the online P or the lineshape P?

For each "online wrong" / "unresolved" period, the strongest usable event is plotted (measured absorption
signal + Q-meter/Dulya fit) together with reference signals from consistent periods whose online P matches
(a) the online P and (b) the lineshape P of the problem event (same sign, within TOL). When no consistent-period
event has that P, a Monte Carlo reference is generated: Dulya chi'' at that P passed through a consistent-period
Q-meter circuit (eta, wq fixed; g from consistent fits).

All curves are the chi''-driven part of the Q-meter output, normalized to unit area and centred on f0, so only
the shape (horn asymmetry) is compared; amplitudes depend on cc / channel and are exactly what is in question.

Usage: python confirm_shapes.py      (fits missing reference events in parallel, cached in results/references/)
"""
import csv
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

import compare_methods as CM
import run_lineshape as RL
from run_lineshape import FS, M

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
REF_DIR = RES / "references"
PLOTS = HERE / "plots"
TOL = 0.02
MIN_REF_AREA_FRACTION = 0.0


def init_worker():
    RL.init_worker()
    FS.OUT = REF_DIR


def npz_path(r, folder):
    return folder / f"event_{r['file'][:19]}_{r['index']}.npz"


def consistent_pool(consistent_ccs):
    """All deduplicated Polarize events of the consistent periods (online P is trusted there)."""
    want, base_of = defaultdict(set), {}
    with open(RL.EVENT_MAP) as fh:
        for row in csv.DictReader(fh):
            if row["label"] == "Polarize":
                want[row["file"]].add(int(row["event_index"]))
                base_of[(row["file"], int(row["event_index"]))] = int(row["baseline_id"])
    pool = {}
    for fname, idxs in sorted(want.items()):
        with open(RL.DATA / fname) as fh:
            for i, line in enumerate(fh):
                if i not in idxs:
                    continue
                d = json.loads(line)
                if not any(abs(d["cc"] - c) < 1e-6 for c in consistent_ccs):
                    continue
                f = d["freq_list"]
                if d["channel"]["name"] not in RL.CHANNELS or not (32.29 < f[0] < 32.31 and 33.09 < f[-1] < 33.11):
                    continue
                if abs(d["cc"] * d["area"] - d["pol"]) > 1e-9 or not np.any(d["fitsub"]):
                    continue
                pool.setdefault(d["start_time"][:19], {
                    "file": fname, "index": i, "time": d["start_time"][:19], "cc": float(d["cc"]),
                    "pol": float(d["pol"]), "area": float(d["area"]), "baseline_id": base_of[(fname, i)],
                    "channel": d["channel"]["name"], "period_note": ""})
    return list(pool.values())


def pick_reference(pool, target, used):
    cands = [e for e in pool if np.sign(e["pol"]) == np.sign(target) and abs(e["pol"] - target) <= TOL
             and e["time"] not in used]
    if not cands:
        return None
    # closest P; among near-equal P prefer the latest (best-tuned) data
    return min(cands, key=lambda e: (round(abs(e["pol"] - target), 3), e["time"]))


def shape_from_npz(path, f0):
    z = np.load(path)
    f, raw, bg, absorb, disperse = z["f"], z["raw"], z["background"], z["absorb"], z["disperse"]
    data = raw - bg - disperse          # measured absorption: data minus fitted background and chi' part
    norm = np.sum(absorb)
    return f - f0, data / norm, absorb / norm


def mc_shape(P, ref_fit, g):
    c = [ref_fit["circuit"][n] for n in M.CIRCUIT_NAMES]
    s = dict(ref_fit["signal"], P=float(np.clip(P, -0.95, 0.95)), g=g, eta_q=RL.ETA_FIXED, wq=RL.WQ_FIXED)
    f = np.load(npz_path(ref_fit, RES))["f"]
    absorb = M.absorption_dispersion_parts(f, c, [s[n] for n in M.SIGNAL_NAMES])[0]
    return f - s["f0"], absorb / np.sum(absorb)


def main():
    REF_DIR.mkdir(parents=True, exist_ok=True)
    events, by = CM.load()
    rows = CM.period_rows(by)
    status = {r["cc"]: r["status"] for r in rows}
    consistent = [cc for cc, s in status.items() if s == "consistent"]
    problems = [r for r in rows if r["status"] in ("online wrong", "unresolved", "no result")]

    # representative event per problem period: strongest usable signal
    reps = []
    for r in problems:
        rs = [e for e in by[r["cc"]] if e["use"]] or by[r["cc"]]
        e = max(rs, key=lambda e: abs(e["area_online"]))
        reps.append((r, e))

    print("reading consistent-period events ...", flush=True)
    pool = consistent_pool(consistent)
    print(f"  {len(pool)} events; online |P| range {min(abs(e['pol']) for e in pool):.3f} – "
          f"{max(abs(e['pol']) for e in pool):.3f}", flush=True)

    cache_file = REF_DIR / "reference_fits.yaml"
    cache = {r["time"]: r for r in (yaml.safe_load(cache_file.read_text()) or [])} if cache_file.exists() else {}
    plan, used = [], set()
    for r, e in reps:
        refs = []
        for kind, target in (("online", e["P_online"]), ("lineshape", e["P_lineshape"])):
            ref = pick_reference(pool, target, used)
            if ref:
                used.add(ref["time"])
            refs.append((kind, target, ref))
        plan.append((r, e, refs))

    todo = {ref["time"]: ref for _, _, refs in plan for _, _, ref in refs if ref and ref["time"] not in cache}
    if todo:
        base = yaml.safe_load(FS.BASE_FITS.read_text())["baselines"]
        jobs = []
        for ref in todo.values():
            b = base.get(ref["baseline_id"])
            jobs.append((ref, [b["params"][n] for n in M.CIRCUIT_NAMES] if b else None))
        print(f"fitting {len(jobs)} reference events on {RL.N_WORKERS} cores ...", flush=True)
        with Pool(RL.N_WORKERS, initializer=init_worker) as p:
            for out in p.imap_unordered(FS.fit_one, jobs):
                cache[out["time"]] = out
                print(f"  {out['time']} cc {out['cc']:+.2f}  P_online {out['P_online']:+.3f}  "
                      f"P_ls {out['P_lineshape']:+.3f}  rms {out['fit_rms_V'] * 1e6:.0f} µV", flush=True)
        cache_file.write_text(yaml.safe_dump(sorted(cache.values(), key=lambda r: r["time"]), sort_keys=False))

    # MC references: circuit of the strongest consistent-period lineshape fit, g = median of consistent fits
    good = [e for e in events if status.get(e["cc"]) == "consistent" and e["use"]]
    mc_base = max(good, key=lambda e: abs(e["area_online"]))
    g_med = float(np.median([e["signal"]["g"] for e in good]))

    summary = []
    for group, title, fname in (
            (("online wrong",), "Periods where the online cc is inconsistent", "confirm_online_wrong.png"),
            (("unresolved", "no result"), "Unresolved periods (lineshape uncertain)", "confirm_unresolved.png")):
        sel = [p for p in plan if p[0]["status"] in group]
        ncol = min(3, len(sel))
        nrow = int(np.ceil(len(sel) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 5.4 * nrow), squeeze=False)
        for ax, (r, e, refs) in zip(axes.flat, sel):
            x, data, fit = shape_from_npz(npz_path(e, RES), e["signal"]["f0"])
            ax.plot(x * 1e3, data, ".", color="0.35", ms=3,
                    label=f"data  {e['time'][:16]}  ({e['channel'].replace('Deuteron', '')})")
            cc_ls = e["P_lineshape"] / e["area_online"]
            ax.plot(x * 1e3, fit, "-", color="k", lw=2,
                    label=f"fit:  P_lineshape = {e['P_lineshape']:+.2f} ± {e['P_lineshape_err']:.2f}, "
                          f"cc_ls = {cc_ls:+.2f}")
            ax.plot([], [], " ", label=f"online:  P = {e['P_online']:+.2f},  cc = {e['cc']:g}")
            for kind, target, ref in refs:
                col = "tab:red" if kind == "online" else "tab:blue"
                if ref:
                    rf = cache[ref["time"]]
                    xr, _, fr = shape_from_npz(npz_path(rf, REF_DIR), rf["signal"]["f0"])
                    lab = (f"ref (data) P = {rf['P_online']:+.2f}  [cc {rf['cc']:g}, {rf['time'][:10]}]"
                           f"  ≈ {kind} P")
                    src = f"data cc {rf['cc']:g} {rf['time'][:16]} P {rf['P_online']:+.3f}"
                else:
                    xr, fr = mc_shape(target, mc_base, g_med)
                    lab = f"ref (MC)   P = {target:+.2f}  = {kind} P"
                    src = f"MC P {target:+.3f}"
                ax.plot(xr * 1e3, fr, "--", color=col, lw=1.8, label=lab)
                summary.append({"period_cc": r["cc"], "event": e["time"], "reference_for": kind,
                                "target_P": float(target), "reference": src})
            ax.axhline(0, color="0.6", lw=0.6)
            ax.set_xlim(-330, 330)
            ax.set_title(f"cc {r['cc']:g}  ({r['dates']})  — {r['status']}", fontsize=11, weight="bold")
            ax.set_xlabel("f − f0  [kHz]")
            ax.set_ylabel("χ″ signal, normalized to unit area")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7.6, loc="upper left")
        for ax in list(axes.flat)[len(sel):]:
            ax.set_axis_off()
        fig.suptitle(f"{title}: signal shape vs reference shapes at the online P (red) and the lineshape P (blue)",
                     fontsize=13, weight="bold")
        fig.text(0.01, 0.003, "Data = raw sweep − fitted Q-meter background − fitted χ′ part. All curves normalized "
                 "to unit area and centred on the fitted f0. References: consistent-period events (Sep 2022 – "
                 f"Mar 2023) with online P within ±{TOL} of the target and the same sign; MC = Dulya χ″ at that P "
                 "through a consistent-period Q-meter circuit.", fontsize=8.5, color="0.3")
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        fig.savefig(PLOTS / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {PLOTS / fname}")
    (RES / "confirm_shapes_references.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    for s in summary:
        print(f"  cc {s['period_cc']:9.4f}  {s['reference_for']:9s} target {s['target_P']:+.3f}  -> {s['reference']}")


if __name__ == "__main__":
    main()
