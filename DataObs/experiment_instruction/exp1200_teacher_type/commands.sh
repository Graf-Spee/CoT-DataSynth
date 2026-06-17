#!/usr/bin/env bash
set -euo pipefail

cd /home/hrh/CoT-DataSynth

PYTHON_BIN="${PYTHON_BIN:-/home/hrh/anaconda3/envs/verl-cot/bin/python}"

DRY_RUN="${DRY_RUN:---dry-run}"
DATASET="${DATASET:-gsm8k}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/hrh/COT/experiments}"
BASE_MODEL="${BASE_MODEL:-/data/pretrain_models/Qwen2.5-0.5B-Instruct}"
GPU_IDS="${GPU_IDS:-0}"
BIG_GPU_IDS="${BIG_GPU_IDS:-0,1}"
TEACHER_NUM_SAMPLES="${TEACHER_NUM_SAMPLES:-4}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.7}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-1024}"
STAGES="${STAGES:-distill,metrics,sft,sft_eval,grpo,grpo_eval}"

# Purpose: compare ordinary instruct teachers vs reasoning/cross-family/task-specialist teachers.
# Keep these results separate from exp1100 teacher-size curve.

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp1200_${DATASET}_reasoning_deepseek_r1_qwen7b_t${TEACHER_TEMPERATURE}_n${TEACHER_NUM_SAMPLES} \
  --dataset "${DATASET}" \
  --base-model "${BASE_MODEL}" \
  --teacher-model /data/pretrain_models/DeepSeek-R1-Distill-Qwen-7B \
  --output-dir "${OUTPUT_DIR}" \
  --stages "${STAGES}" \
  --gpu-ids "${GPU_IDS}" \
  --teacher-num-samples "${TEACHER_NUM_SAMPLES}" \
  --teacher-temperature "${TEACHER_TEMPERATURE}" \
  --teacher-do-sample \
  --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --data-variant reasoning_deepseek_r1_qwen7b \
  --reasoning-source teacher \
  --grpo-env TOTAL_EPOCHS=1

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp1200_${DATASET}_reasoning_qwq32b_t${TEACHER_TEMPERATURE}_n${TEACHER_NUM_SAMPLES} \
  --dataset "${DATASET}" \
  --base-model "${BASE_MODEL}" \
  --teacher-model /data/pretrain_models/QwQ-32B \
  --output-dir "${OUTPUT_DIR}" \
  --stages "${STAGES}" \
  --gpu-ids "${BIG_GPU_IDS}" \
  --teacher-num-samples "${TEACHER_NUM_SAMPLES}" \
  --teacher-temperature "${TEACHER_TEMPERATURE}" \
  --teacher-do-sample \
  --teacher-tensor-parallel-size 2 \
  --teacher-batch-size 16 \
  --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --data-variant reasoning_qwq32b \
  --reasoning-source teacher \
  --grpo-env TOTAL_EPOCHS=1

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp1200_${DATASET}_cross_llama3_1_8b_t${TEACHER_TEMPERATURE}_n${TEACHER_NUM_SAMPLES} \
  --dataset "${DATASET}" \
  --base-model "${BASE_MODEL}" \
  --teacher-model /data/pretrain_models/Llama-3.1-8B-Instruct \
  --output-dir "${OUTPUT_DIR}" \
  --stages "${STAGES}" \
  --gpu-ids "${GPU_IDS}" \
  --teacher-num-samples "${TEACHER_NUM_SAMPLES}" \
  --teacher-temperature "${TEACHER_TEMPERATURE}" \
  --teacher-do-sample \
  --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --data-variant cross_llama3_1_8b \
  --reasoning-source teacher \
  --grpo-env TOTAL_EPOCHS=1

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp1200_${DATASET}_cross_gemma2_9b_t${TEACHER_TEMPERATURE}_n${TEACHER_NUM_SAMPLES} \
  --dataset "${DATASET}" \
  --base-model "${BASE_MODEL}" \
  --teacher-model /data/pretrain_models/gemma-2-9b-it \
  --output-dir "${OUTPUT_DIR}" \
  --stages "${STAGES}" \
  --gpu-ids "${GPU_IDS}" \
  --teacher-num-samples "${TEACHER_NUM_SAMPLES}" \
  --teacher-temperature "${TEACHER_TEMPERATURE}" \
  --teacher-do-sample \
  --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --data-variant cross_gemma2_9b \
  --reasoning-source teacher \
  --grpo-env TOTAL_EPOCHS=1
