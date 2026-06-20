#!/usr/bin/env bash
set -euo pipefail

cd /home/hrh/CoT-DataSynth

PYTHON_BIN="${PYTHON_BIN:-/home/hrh/anaconda3/envs/verl-cot/bin/python}"

source DataObs/experiment_instruction/_qwen_family_defaults.sh
set_qwen_family_defaults

DRY_RUN="${DRY_RUN---dry-run}"
DATASET="${DATASET:-gsm8k}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/hrh/COT/experiments}"
GPU_IDS="${GPU_IDS:-0}"

# Fill human/answer-only paths after confirming their source format.
HUMAN_REASONING_SFT="${HUMAN_REASONING_SFT:-/path/to/human_reasoning_sft.parquet}"
ANSWER_ONLY_SFT="${ANSWER_ONLY_SFT:-/path/to/answer_only_sft.parquet}"
VAL_SFT="${VAL_SFT:-${HUMAN_REASONING_SFT}}"

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp3000_${DATASET}_${FAMILY_TAG}_prompt_only_teacher_reasoning \
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
  --experiment-id exp3000_${DATASET}_${FAMILY_TAG}_human_reasoning \
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
  --experiment-id exp3000_${DATASET}_${FAMILY_TAG}_answer_only \
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
