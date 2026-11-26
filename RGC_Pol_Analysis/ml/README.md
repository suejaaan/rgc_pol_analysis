# ml

**Purpose:** train a network that reads P from a raw NMR sweep, on the cc-free MC of `../mc_generation/`, and
validate it on the real lineshape labels of `../label_dataset/`. The models are adapted from the upstream
`ml/` folder (`train_cnn_v2.py`, `pol_cnn.py`).

## What changed vs upstream

| | Upstream `ml/` | Here |
|---|---|---|
| Data | one parquet file, random train/val/test split | MC shards with a `split` column **by tune segment**; val and test tunes never appear in training |
| Target | P (and Q), from online-cc-scaled MC | P from the cc-free MC |
| Input | raw trace standardized (`train_cnn_v2`) | 2 channels: raw standardized + **wing-subtracted** (cubic background from the wings removed, i.e. the signal shape). `--input raw` reproduces upstream. |
| Output | P (and Q) | **P and σ_P** (log-variance head), Gaussian NLL + L1 loss after an L1 warm-up |
| Models | CNN v2, Inception-ResNet | the same two (`--model cnn` / `inception`); linear output, so negative P works |
| Evaluation | random test split | MC held-out tunes **and** real lineshape labels (`eval_real.py`); bias, rms, pulls, per \|P\| bin and per segment |

## Files

| File | Content |
|---|---|
| `data.py` | read MC shards by split, input preprocessing (`raw`, `wingsub`, `both`) |
| `models.py` | `PolarizationCNN` (from `train_cnn_v2.py`), `InceptionResNet` (from `pol_cnn.py`), both with P and log σ² heads |
| `train.py` | training: AdamW, cosine warm restarts, early stopping on val MAE, checkpoint/resume, bf16 on CUDA |
| `evaluate.py` | metrics and plots (pred vs true, bias/rms vs \|P\|, pull distribution, loss curves) |
| `eval_real.py` | `export`: real label sweeps → `data_real/real_labels.parquet` (once, on the analysis machine). `eval`: run a model on them, per split. |
| `run_desktop.sh` | example runs for the GPU desktop (main model + learning curve) |

**Real-data test:** the 484 labels carry the same tune split as the MC.
- **Test tunes (96, 118, 119): 93 labels.** Never seen in any form during training, so this is the real-data test.
- **Val tunes (105, 135): 74 labels.**
- **Train tunes: 317 labels.** These share circuit and noise with the training MC, but the sweeps themselves were never used.

## Run

```bash
# environment: python with torch, pandas, pyarrow, numpy, matplotlib, pyyaml
cd RGC_Pol_Analysis/ml
python train.py --data ../mc_generation/mc_data/mc1m --version cnn_both_1m          # main run
python eval_real.py eval --version cnn_both_1m                                     # real labels
python train.py --data ../mc_generation/mc_data/test10k --version smoke --epochs 2 --device cpu   # quick check
```

Options: `--model cnn|inception`, `--input both|raw|wingsub`, `--max-train N` (learning curve), `--epochs`,
`--batch-size`, `--lr`, `--patience`, `--device auto|cuda|mps|cpu`.

Outputs:
- `Models/<version>/`: `best_model.pth`, `checkpoint.pth` (resume), `config.yaml`.
- `Model_Performance/<version>/`:
  - `loss.csv`, `loss.png`;
  - `mc_val`, `mc_test`, `real_test`, `real_val`, `real_train`, each as `.png`, `_metrics.yaml` and `_predictions.csv`.

**Data on the desktop:** copy `../mc_generation/mc_data/mc1m/` (~2.9 GB) and `data_real/` (~2 MB). Both are
gitignored, because they contain real CLAS data (noise and sweeps). You can also regenerate the MC there with
`python stage5_generate.py 1000000 mc1m`, which needs `mc_generation/library/param_library.npz` (15 MB, also
gitignored).

**Mac note:** with the PyTorch 2.6 nightly in `torch_env`, the first validation pass on MPS can blow up
(a BatchNorm glitch). Training itself recovers, but use `--device cpu` for smoke tests on the Mac. CUDA is not affected.

## Smoke test (Mac, CPU, 8k MC sweeps, 3 epochs)
- The pipeline runs end to end.
- MC val MAE: 0.030.
- Real test tunes, 2 epochs: bias −0.004, rms 0.015.

These numbers only show that the plumbing works. They are not results.
