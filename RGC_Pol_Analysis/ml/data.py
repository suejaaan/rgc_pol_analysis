"""Data loading and preprocessing for the RGC MC datasets (mc_generation/stage5_generate.py).

Splits are read from the `split` column (assigned by tune segment in the generator), never re-drawn at random:
val/test tunes are unseen in training.

Input modes (per trace, no global statistics, so nothing leaks between splits):
  raw      raw sweep centred and scaled by its own std (upstream train_cnn_v2.standardize_traces)
  wingsub  raw sweep minus a cubic polynomial fitted on the wings (outside the signal region), scaled by its
           own std: the signal shape with the Q-meter background removed, cc-free and model-free
  both     the two above as 2 channels (default)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

N_BINS = 512
SIGNAL_REGION = (32.48, 32.90)   # MHz, as in label_dataset
F0_NOMINAL = 32.685
META_COLS = ["P", "Pzz", "segment", "SNR", "signal_peak", "noise_rms", "quality"]


def read_split(data_dir, split, max_rows=None, seed=0):
    data_dir = Path(data_dir)
    man = yaml.safe_load((data_dir / "manifest.yaml").read_text())
    files = [data_dir / s["file"] for s in man["shards"] if s["file"].startswith(split + "_")]
    if not files:
        raise FileNotFoundError(f"no shards for split '{split}' in {data_dir}")
    cols = [str(i) for i in range(N_BINS)]
    parts, n = [], 0
    for f in files:
        df = pd.read_parquet(f, columns=cols + [c for c in META_COLS])
        parts.append(df)
        n += len(df)
        if max_rows and n >= max_rows:
            break
    df = pd.concat(parts, ignore_index=True)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    X = df[cols].to_numpy(np.float32)
    meta = df[[c for c in META_COLS if c in df]].reset_index(drop=True)
    return X, meta


def load_freq(data_dir):
    return np.load(Path(data_dir) / "freq.npy")


def _wing_projector(freq, deg=3):
    x = freq - F0_NOMINAL
    wing = (freq < SIGNAL_REGION[0]) | (freq > SIGNAL_REGION[1])
    V = np.vander(x, deg + 1)
    P = V @ np.linalg.pinv(V[wing])          # (N_BINS, n_wing): maps wing values to the fitted polynomial
    return P.astype(np.float64), wing


def preprocess(X, freq, mode="both", chunk=100_000):
    """X: (N, 512) raw sweeps [V] -> (N, C, 512) float32."""
    X = np.asarray(X, np.float64)
    out = []
    if mode in ("raw", "both"):
        c = X - X.mean(axis=1, keepdims=True)
        out.append(c / np.maximum(c.std(axis=1, keepdims=True), 1e-12))
    if mode in ("wingsub", "both"):
        P, wing = _wing_projector(freq)
        ws = np.empty_like(X)
        for i in range(0, len(X), chunk):
            xb = X[i:i + chunk]
            ws[i:i + chunk] = xb - xb[:, wing] @ P.T
        out.append(ws / np.maximum(ws.std(axis=1, keepdims=True), 1e-12))
    if not out:
        raise ValueError(f"unknown input mode {mode}")
    return np.stack(out, axis=1).astype(np.float32)


def n_channels(mode):
    return 2 if mode == "both" else 1
