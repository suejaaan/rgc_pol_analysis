"""Stage 4 extension: more tune states and more noise samples for the MC library (no cc anywhere).

Part A  new tune segments: every stage-0 segment (5T, clean candidates) with >= MIN_CAND candidates that has no
        shared circuit yet (mostly no plateaus: Jun-Jul 2022, build-ups). Segment-joint fit (label_dataset/
        segment_fit.py) on the N_EV events with the largest |online area| (strongest signal, spread in time),
        starts = good circuits of all label segments + the stored-baseline refit; amplitude start from the data.
Part B  extra records: up to N_PER_SEG random candidate events per segment (old and new), fitted with the
        segment's shared circuit and sigma fixed (stage2_full.fit_event). They add noise samples and per-event
        drift (U, phase, DC offset). Their P is not a label (not plateau-selected).
Quality of the new segments: signal-to-noise of the joint events (absorption peak / wing rms) is stored, since a
weak signal constrains the circuit family less.

Outputs: library/segments_extra.yaml, library/extra_records.csv, library/npz_extra/
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
LD = HERE.parent / "label_dataset"
sys.path.insert(0, str(HERE.parent / "qmeter_dulya_fit"))
sys.path.insert(0, str(LD))
import model as M  # noqa: E402
import segment_fit as SF  # noqa: E402
import stage2_full as S2F  # noqa: E402

LIB = HERE / "library"
MIN_CAND = 75
N_EV = 10
N_PER_SEG = 60
SEED = 7


def new_segment_setup(seg, cand, gstarts, base):
    c = cand[cand["segment"] == seg].sort_values("time")
    # strongest signals, spread in time: best |area| in each of N_EV time slices
    slices = np.array_split(c.index.to_numpy(), N_EV)
    evs = [c.loc[sl].loc[c.loc[sl]["area_online"].abs().idxmax()].to_dict() for sl in slices if len(sl)]
    starts = list(gstarts)
    b = base.get(int(c["baseline_id"].iloc[0]))
    if b:
        starts.append(("baseline refit", [b["params"][n] for n in M.CIRCUIT_NAMES]))
    sig = {e["time"]: {"A": None, "P": float(-np.sign(e["area_online"]) * 0.25), "f0": 32.685} for e in evs}
    return evs, starts, sig


def part_a(cand, done):
    pilot = yaml.safe_load((LD / "results/stage2/fits_pilot.yaml").read_text())
    base = yaml.safe_load(SF.FS.BASE_FITS.read_text())["baselines"]
    gstarts = SF.global_circuit_starts(pilot)
    n = cand.groupby("segment").size()
    segs = [int(s) for s in n[n >= MIN_CAND].index if s not in done]
    setup = {s: new_segment_setup(s, cand, gstarts, base) for s in segs}
    jobs = [(s, lab, c, setup[s][2], setup[s][0]) for s in segs for lab, c in setup[s][1]]
    print(f"part A: {len(segs)} new segments {segs}; screening {len(jobs)} pairs", flush=True)
    t0 = time.time()
    with Pool(SF.N_WORKERS) as pool:
        scr = pool.map(SF.screen, jobs, chunksize=1)
        jj = []
        for s in segs:
            ranked = sorted([x for x in scr if x[0] == s], key=lambda x: x[2])[:SF.N_JOINT]
            names = {lab for _, lab, _ in ranked}
            jj.append((s, [st for st in setup[s][1] if st[0] in names], setup[s][2], setup[s][0]))
        print(f"  screening done in {(time.time() - t0) / 60:.1f} min", flush=True)
        res = pool.map(SF.joint_segment, jj, chunksize=1)
    print(f"  joint fits done in {(time.time() - t0) / 60:.1f} min", flush=True)
    for r in res:
        b = r["solutions"][0]
        r["median_rms_signal_uV"] = float(np.median([e["rms_signal_uV"] for e in b["events"]]))
        r["median_rms_wings_uV"] = float(np.median([e["rms_wings_uV"] for e in b["events"]]))
    return res


def init_b():
    S2F.NPZ = LIB / "npz_extra"   # workers are fresh processes: set the output folder there


def part_b(cand, seg_res):
    rng = np.random.default_rng(SEED)
    jobs = []
    for s in seg_res:
        best = s["solutions"][0]
        sh = np.array([best["shared"][M.CIRCUIT_NAMES[k]] for k in SF.SHARED_C] + [best["shared"]["sigma_kHz"] / 1e3])
        jev = best["events"]
        tj = np.array([np.datetime64(e["time"]) for e in jev])
        c = cand[cand["segment"] == s["segment"]]
        pick = c.iloc[rng.choice(len(c), min(N_PER_SEG, len(c)), replace=False)]
        for ev in pick.to_dict("records"):
            starts = []
            for k in np.argsort(np.abs(tj - np.datetime64(ev["time"])))[:2]:
                e = jev[k]
                scale = ev["area_online"] / e["area_online"] if abs(e["area_online"]) > 1e-9 else 1.0
                starts.append(np.array([e["U"], e["phi_deg"], e["DC_offset"], e["A"] * scale, e["P"],
                                        e.get("f0", 32.685), max(e["g"], 0.002)]))
            ev.setdefault("plateau", -1)
            jobs.append((ev, sh, starts))
    print(f"part B: fitting {len(jobs)} extra events with the shared circuit fixed", flush=True)
    t0 = time.time()
    rows = []
    with Pool(SF.N_WORKERS, initializer=init_b) as pool:
        for k, r in enumerate(pool.imap_unordered(S2F.fit_event, jobs, chunksize=4), 1):
            rows.append(r)
            if k % 500 == 0 or k == len(jobs):
                print(f"  [{k}/{len(jobs)}] {(time.time() - t0) / 60:.1f} min", flush=True)
    return pd.DataFrame(rows)


def main():
    LIB.mkdir(exist_ok=True)
    ev = pd.read_csv(LD / "results/events.csv")
    cand = ev[ev["candidate"] & (ev["duplicate_of"].isna())].copy()
    old = yaml.safe_load((LD / "results/stage2/segments.yaml").read_text())
    done = {s["segment"] for s in old}
    t0 = time.time()
    extra_file = LIB / "segments_extra.yaml"
    if extra_file.exists():
        new = yaml.safe_load(extra_file.read_text())
        print(f"part A: reusing {extra_file}", flush=True)
    else:
        new = part_a(cand, done)
        extra_file.write_text(yaml.safe_dump(new, sort_keys=False))
    for r in new:
        b = r["solutions"][0]
        P = [abs(e["P"]) for e in b["events"]]
        print(f"  seg {r['segment']:3d}: rms signal {r['median_rms_signal_uV']:.1f} µV, wings {r['median_rms_wings_uV']:.1f} µV, "
              f"σ {b['shared']['sigma_kHz']:.1f} kHz, |P| {min(P):.2f}-{max(P):.2f}, start {b['start'][:28]}", flush=True)

    # part B over old + new segments, fitted with fit_event writing into npz_extra
    (LIB / "npz_extra").mkdir(exist_ok=True)
    picks = pd.read_csv(LD / "results/plateau_events.csv")
    used = set(zip(picks["file"], picks["index"]))
    cand_b = cand[[(f, i) not in used for f, i in zip(cand["file"], cand["index"])]]
    d = part_b(cand_b, old + new)
    d["new_segment"] = ~d["segment"].isin(done)
    d.to_csv(LIB / "extra_records.csv", index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min: {len(d)} extra records, "
          f"signal rms median {d['rms_signal_uV'].median():.1f} µV, wings {d['rms_wings_uV'].median():.1f} µV")


if __name__ == "__main__":
    main()
