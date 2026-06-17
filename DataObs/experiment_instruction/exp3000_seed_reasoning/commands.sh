#!/usr/bin/env bash
set -euo pipefail

cd /home/hrh/CoT-DataSynth

PYTHON_BIN="${PYTHON_BIN:-/home/hrh/anaconda3/envs/verl-cot/bin/python}"

DRY_RUN="${DRY_RUN:---dry-run}"
DATASET="${DATASET:-gsm8k}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/hrh/COT/experiments}"
BASE_MODEL="${BASE_MODEL:-/data/pretrain_models/Qwen2.5-0.5B-Instruct}"
TEACHER_MODEL="${TEACHER_MODEL:-/data/pretrain_models/Qwen2.5-7B-Instruct}"
GPU_IDS="${GPU_IDS:-0}"

# Fill human/answer-only paths after confirming their source format.
HUMAN_REASONING_SFT="${HUMAN_REASONING_SFT:-/path/to/human_reasoning_sft.parquet}"
ANSWER_ONLY_SFT="${ANSWER_ONLY_SFT:-/path/to/answer_only_sft.parquet}"
VAL_SFT="${VAL_SFT:-${HUMAN_REASONING_SFT}}"

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp3000_${DATASET}_prompt_only_teacher_reasoning \
  --dataset "${DATASET}" \
  --base-model "${BASE_MODEL}" \
  --teacher-model "${TEACHER_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --stages distill,metrics,sft,sft_eval,grpo,grpo_eval \
  --gpu-ids "${GPU_IDS}" \
  --teacher-num-samples 4 \
  --teacher-temperature 0.7 \
  --teacher-do-sample \
  --data-variant prompt_only_teacher_reasoning \
  --reasoning-source teacher \
  --grpo-env TOTAL_EPOCHS=1

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp3000_${DATASET}_human_reasoning \
  --dataset "${DATASET}" \
  --base-model "${BASE_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --stages metrics,sft,sft_eval,grpo,grpo_eval \
  --gpu-ids "${GPU_IDS}" \
  --data-variant human_reasoning \
  --reasoning-source human \
  --sft-data "${HUMAN_REASONING_SFT}" \
  --sft-val-data "${VAL_SFT}" \
  --metrics-data "${HUMAN_REASONING_SFT}" \
  --grpo-env TOTAL_EPOCHS=1

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp3000_${DATASET}_answer_only \
  --dataset "${DATASET}" \
  --base-model "${BASE_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --stages metrics,sft,sft_eval,grpo,grpo_eval \
  --gpu-ids "${GPU_IDS}" \
  --data-variant answer_only \
  --reasoning-source none \
  --sft-data "${ANSWER_ONLY_SFT}" \
  --sft-val-data "${VAL_SFT}" \
  --metrics-data "${ANSWER_ONLY_SFT}" \
  --grpo-env TOTAL_EPOCHS=1
