#!/usr/bin/env bash
set -euo pipefail

cd /home/hrh/CoT-DataSynth

PYTHON_BIN="${PYTHON_BIN:-/home/hrh/anaconda3/envs/verl-cot/bin/python}"

source DataObs/experiment_instruction/_qwen_family_defaults.sh
set_qwen_family_defaults

DRY_RUN="${DRY_RUN---dry-run}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/hrh/COT/experiments}"
GPU_IDS="${GPU_IDS:-0}"
SMOKE_NUM_ROWS="${SMOKE_NUM_ROWS:-8}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-512}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-2}"

# Purpose: verify distill prompt formatting, reward filtering, and output parquet schema.
# To actually run, use: DRY_RUN="" bash DataObs/experiment_instruction/exp1000_smoke/commands.sh

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp1000_${FAMILY_TAG}_gsm8k_distill_smoke \
  --dataset gsm8k \
  --base-model "${BASE_MODEL}" \
  --teacher-model "${TEACHER_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --stages distill \
  --gpu-ids "${GPU_IDS}" \
  --smoke-num-rows "${SMOKE_NUM_ROWS}" \
  --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --teacher-batch-size "${TEACHER_BATCH_SIZE}"

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp1000_${FAMILY_TAG}_math500_distill_smoke \
  --dataset math-500 \
  --base-model "${BASE_MODEL}" \
  --teacher-model "${TEACHER_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --stages distill \
  --gpu-ids "${GPU_IDS}" \
  --smoke-num-rows "${SMOKE_NUM_ROWS}" \
  --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --teacher-batch-size "${TEACHER_BATCH_SIZE}"

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp1000_${FAMILY_TAG}_arc_distill_smoke \
  --dataset arc-challenge \
  --base-model "${BASE_MODEL}" \
  --teacher-model "${TEACHER_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --stages distill \
  --gpu-ids "${GPU_IDS}" \
  --smoke-num-rows "${SMOKE_NUM_ROWS}" \
  --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --teacher-batch-size "${TEACHER_BATCH_SIZE}"

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp1000_${FAMILY_TAG}_strategyqa_distill_smoke \
  --dataset strategyQA \
  --base-model "${BASE_MODEL}" \
  --teacher-model "${TEACHER_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --stages distill \
  --gpu-ids "${GPU_IDS}" \
  --smoke-num-rows "${SMOKE_NUM_ROWS}" \
  --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --teacher-batch-size "${TEACHER_BATCH_SIZE}"

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp1000_${FAMILY_TAG}_mbpp_distill_smoke \
  --dataset mbpp \
  --base-model "${BASE_MODEL}" \
  --teacher-model "${TEACHER_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --stages distill \
  --gpu-ids "${GPU_IDS}" \
  --smoke-num-rows "${SMOKE_NUM_ROWS}" \
  --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --teacher-batch-size "${TEACHER_BATCH_SIZE}"
