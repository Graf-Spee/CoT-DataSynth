#!/usr/bin/env bash
set -euo pipefail

cd /home/hrh/CoT-DataSynth

PYTHON_BIN="${PYTHON_BIN:-/home/hrh/anaconda3/envs/verl-cot/bin/python}"

DRY_RUN="${DRY_RUN:---dry-run}"
DATASET="${DATASET:-gsm8k}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/hrh/COT/experiments}"
BASE_MODEL="${BASE_MODEL:-/data/pretrain_models/Qwen3.5-0.8B}"
GPU_IDS="${GPU_IDS:-0}"

SIZE_1K_SFT="${SIZE_1K_SFT:-/path/to/1k_sft.parquet}"
SIZE_5K_SFT="${SIZE_5K_SFT:-/path/to/5k_sft.parquet}"
SIZE_10K_SFT="${SIZE_10K_SFT:-/path/to/10k_sft.parquet}"
VAL_SFT="${VAL_SFT:-${SIZE_1K_SFT}}"

for item in "1k:${SIZE_1K_SFT}" "5k:${SIZE_5K_SFT}" "10k:${SIZE_10K_SFT}"; do
  bucket="${item%%:*}"
  data_path="${item#*:}"
  "${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
    --experiment-id exp7000_${DATASET}_size_${bucket} \
    --dataset "${DATASET}" \
    --base-model "${BASE_MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --stages metrics,sft,sft_eval,grpo,grpo_eval \
    --gpu-ids "${GPU_IDS}" \
    --data-variant size_${bucket} \
    --sft-data "${data_path}" \
    --sft-val-data "${VAL_SFT}" \
    --metrics-data "${data_path}" \
    --grpo-env TOTAL_EPOCHS=1
done
