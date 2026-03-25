#!/bin/bash
set -euo pipefail

GPUS="0,1,2,3,4,5,6,7"
ANALYZE_GPU="0"
MASTER_PORT=29600
SCRIPT="SuppExp2_depth_profile_robustness.py"

# 以 410M 为主模型做 Exp2；如需复核小模型，可把 pythia-70m 放开。
models=(
  "pythia-410m"
  # "pythia-70m"
)

# 围绕默认 M0=1024 和 K0=32 的最小 sweep。
latents=(512 1024 2048)
ks=(16 32 64)
thresholds="1.0,1.65,2.0,2.5"
cluster_ks="2,3,4,5"

BS=2
EPOCH=512
RESULTS_SUBDIR="pythia_results_pretrainedSAE_exp2"
SAVE_DIR="/data/hxt/P5_1/"
DATA_DIR="/data/hxt/P5_1/Dataset_text/"
MODEL_DIR="/data/hxt/P5_1/EleutherAI/"

# ---------- train ----------
for model in "${models[@]}"; do
  for n_latents in "${latents[@]}"; do
    for k in "${ks[@]}"; do
      echo "[Train] model=${model} n_latents=${n_latents} k=${k}"
      python "$SCRIPT" \
        --mode train \
        --gpus "$GPUS" \
        --master_port "$MASTER_PORT" \
        --data_dir "$DATA_DIR" \
        --model_dir "$MODEL_DIR" \
        --save_dir "$SAVE_DIR" \
        --results_subdir "$RESULTS_SUBDIR" \
        --model_name "$model" \
        --batch_size "$BS" \
        --epoch "$EPOCH" \
        --n_latents "$n_latents" \
        --k "$k"
      echo "[Done-Train] model=${model} n_latents=${n_latents} k=${k}"
      echo "--------------------------------------------"
    done
  done
done

# ---------- analyze ----------
for model in "${models[@]}"; do
  for n_latents in "${latents[@]}"; do
    for k in "${ks[@]}"; do
      echo "[Analyze] model=${model} n_latents=${n_latents} k=${k}"
      python "$SCRIPT" \
        --mode analyze \
        --gpus "$ANALYZE_GPU" \
        --data_dir "$DATA_DIR" \
        --model_dir "$MODEL_DIR" \
        --save_dir "$SAVE_DIR" \
        --results_subdir "$RESULTS_SUBDIR" \
        --model_name "$model" \
        --batch_size "$BS" \
        --n_latents "$n_latents" \
        --k "$k" \
        --thresholds "$thresholds" \
        --cluster_ks "$cluster_ks" \
        --default_threshold 1.65 \
        --default_cluster_k 3
      echo "[Done-Analyze] model=${model} n_latents=${n_latents} k=${k}"
      echo "--------------------------------------------"
    done
  done
done

# ---------- summarize ----------
python "$SCRIPT" \
  --mode summarize \
  --save_dir "$SAVE_DIR" \
  --results_subdir "$RESULTS_SUBDIR" \
  --summary_models "pythia-410m" \
  --thresholds "$thresholds" \
  --cluster_ks "$cluster_ks" \
  --ref_model "pythia-410m" \
  --ref_latents 1024 \
  --ref_k 32 \
  --ref_threshold 1.65 \
  --ref_cluster_k 3 \
  --summary_out_name "exp2_summary"
