#!/usr/bin/env bash
set -euo pipefail

cd /home/hrh/CoT-DataSynth

PYTHON_BIN="${PYTHON_BIN:-/home/hrh/anaconda3/envs/verl-cot/bin/python}"

DRY_RUN="${DRY_RUN:---dry-run}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/hrh/COT/experiments}"
BASE_MODEL="${BASE_MODEL:-/data/pretrain_models/Qwen3.5-0.8B}"
TEACHER_MODEL="${TEACHER_MODEL:-/data/pretrain_models/Qwen3.5-0.8B}"
GPU_IDS="${GPU_IDS:-0}"
SMOKE_NUM_ROWS="${SMOKE_NUM_ROWS:-8}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-512}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-2}"

# Purpose: verify distill prompt formatting, reward filtering, and output parquet schema.
# To actually run, use: DRY_RUN="" bash DataObs/experiment_instruction/exp1000_smoke/commands.sh

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp1000_gsm8k_distill_smoke \
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
  --experiment-id exp1000_math500_distill_smoke \
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
  --experiment-id exp1000_arc_distill_smoke \
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
  --experiment-id exp1000_strategyqa_distill_smoke \
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
  --experiment-id exp1000_mbpp_distill_smoke \
  --dataset mbpp \
  --base-model "${BASE_MODEL}" \
  --teacher-model "${TEACHER_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --stages distill \
  --gpu-ids "${GPU_IDS}" \
  --smoke-num-rows "${SMOKE_NUM_ROWS}" \
  --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --teacher-batch-size "${TEACHER_BATCH_SIZE}"
