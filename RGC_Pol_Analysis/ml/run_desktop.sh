#!/usr/bin/env bash
# Example runs on the GPU desktop (RTX 5080). Run from RGC_Pol_Analysis/ml.
# Needs ../mc_generation/mc_data/mc1m/ and data_real/ copied from the analysis machine (both gitignored).
set -e
DATA=../mc_generation/mc_data/mc1m

# 1) main model: CNN, both input channels, all 800k training sweeps
python train.py --data $DATA --version cnn_both_1m --model cnn --input both --epochs 60
python eval_real.py eval --version cnn_both_1m

# 2) learning curve: does more MC help on unseen tunes?
for N in 50000 100000 300000; do
  python train.py --data $DATA --version cnn_both_${N} --max-train $N --epochs 60
  python eval_real.py eval --version cnn_both_${N}
done

# 3) input / model comparison
python train.py --data $DATA --version cnn_raw_1m --input raw --epochs 60       # upstream-style input
python eval_real.py eval --version cnn_raw_1m
python train.py --data $DATA --version inception_both_1m --model inception --epochs 60
python eval_real.py eval --version inception_both_1m
