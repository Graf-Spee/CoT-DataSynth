#!/usr/bin/env bash
set -euo pipefail

cd /home/hrh/CoT-DataSynth

PYTHON_BIN="${PYTHON_BIN:-/home/hrh/anaconda3/envs/verl-cot/bin/python}"

DRY_RUN="${DRY_RUN:---dry-run}"
DATASET="${DATASET:-gsm8k}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/hrh/COT/experiments}"
BASE_MODEL="${BASE_MODEL:-/data/pretrain_models/Qwen2.5-0.5B-Instruct}"
GPU_IDS="${GPU_IDS:-0}"

EASY_SFT="${EASY_SFT:-/path/to/easy_sft.parquet}"
MEDIUM_SFT="${MEDIUM_SFT:-/path/to/medium_sft.parquet}"
HARD_SFT="${HARD_SFT:-/path/to/hard_sft.parquet}"
VAL_SFT="${VAL_SFT:-${MEDIUM_SFT}}"

for item in "easy:${EASY_SFT}" "medium:${MEDIUM_SFT}" "hard:${HARD_SFT}"; do
  bucket="${item%%:*}"
  data_path="${item#*:}"
  "${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
    --experiment-id exp5000_${DATASET}_difficulty_${bucket} \
    --dataset "${DATASET}" \
    --base-model "${BASE_MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --stages metrics,sft,sft_eval,grpo,grpo_eval \
    --gpu-ids "${GPU_IDS}" \
    --data-variant difficulty_${bucket} \
    --sft-data "${data_path}" \
    --sft-val-data "${VAL_SFT}" \
    --metrics-data "${data_path}" \
    --grpo-env TOTAL_EPOCHS=1
done
