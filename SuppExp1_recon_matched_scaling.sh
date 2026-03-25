#!/bin/bash
set -euo pipefail

GPUS="0,1,2,3,4,5,6,7"
MASTER_PORT=29601
SCRIPT="SuppExp1_recon_matched_scaling.py"

models=(
  "pythia-70m"
  "pythia-410m"
  "pythia-1.4b"
)

# 按实验1建议：围绕默认 M0 做 0.5x / 1x / 2x sweep。
# 这里先给出以 M0=1024 为中心的例子；如你们默认值不同，改这里即可。
latents=(512 1024 2048)

# 实验1起步先固定 K=32；如果后续发现不同模型的 EV 仍差太多，再扩 K sweep。
K=32
BS=2
EPOCH=512
RESULTS_SUBDIR="pythia_results_pretrainedSAE_exp1"
SAVE_DIR="/data/hxt/P5_1/"
DATA_DIR="/data/hxt/P5_1/Dataset_text/"
MODEL_DIR="/data/hxt/P5_1/EleutherAI/"

for model in "${models[@]}"; do
  for n_latents in "${latents[@]}"; do
    echo "[Train] model=${model} n_latents=${n_latents} k=${K}"
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
      --k "$K"
    echo "[Done] model=${model} n_latents=${n_latents}"
    echo "--------------------------------------------"
  done
done

echo "[Summary] Selecting EV-matched triplets"
python "$SCRIPT" \
  --mode summarize \
  --save_dir "$SAVE_DIR" \
  --results_subdir "$RESULTS_SUBDIR" \
  --summary_models "pythia-70m,pythia-410m,pythia-1.4b" \
  --summary_k "$K" \
  --summary_out_name "exp1_summary"
