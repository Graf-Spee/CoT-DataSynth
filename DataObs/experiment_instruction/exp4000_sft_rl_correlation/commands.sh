#!/usr/bin/env bash
set -euo pipefail

cd /home/hrh/CoT-DataSynth

PYTHON_BIN="${PYTHON_BIN:-/home/hrh/anaconda3/envs/verl-cot/bin/python}"

DRY_RUN="${DRY_RUN:---dry-run}"
DATASET="${DATASET:-gsm8k}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/hrh/COT/experiments}"
BASE_MODEL="${BASE_MODEL:-/data/pretrain_models/Qwen3.5-0.8B}"
GPU_IDS="${GPU_IDS:-0}"

# Fill these with SFT checkpoints produced by other experiments.
SFT_A="${SFT_A:-/path/to/sft_a/checkpoint-last}"
SFT_B="${SFT_B:-/path/to/sft_b/checkpoint-last}"
SFT_C="${SFT_C:-/path/to/sft_c/checkpoint-last}"

for item in "a:${SFT_A}" "b:${SFT_B}" "c:${SFT_C}"; do
  name="${item%%:*}"
  ckpt="${item#*:}"
  "${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
    --experiment-id exp4000_${DATASET}_sft_${name}_to_rl \
    --dataset "${DATASET}" \
    --base-model "${BASE_MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --stages sft_eval,grpo,grpo_eval \
    --gpu-ids "${GPU_IDS}" \
    --sft-checkpoint "${ckpt}" \
    --grpo-env TOTAL_EPOCHS=1 \
    --grpo-env TEST_FREQ=5
done
