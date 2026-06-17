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
XL_GPU_IDS="${XL_GPU_IDS:-0,1,2,3}"
TEACHER_NUM_SAMPLES="${TEACHER_NUM_SAMPLES:-4}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.7}"
TEACHER_TOP_P="${TEACHER_TOP_P:-0.95}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-64}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-1024}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"
GRPO_TOTAL_EPOCHS="${GRPO_TOTAL_EPOCHS:-1}"
STAGES="${STAGES:-distill,metrics,sft,sft_eval,grpo,grpo_eval}"

# Main teacher-size curve. Keep this same-family Qwen2.5-Instruct curve separate
# from reasoning/cross-family teachers.
# To actually run, use: DRY_RUN="" bash DataObs/experiment_instruction/exp1100_teacher_size/commands.sh

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp1100_${DATASET}_teacher_self_qwen2_5_0_5b_t${TEACHER_TEMPERATURE}_n${TEACHER_NUM_SAMPLES} \
  --dataset "${DATASET}" \
  --base-model "${BASE_MODEL}" \
  --teacher-model /data/pretrain_models/Qwen2.5-0.5B-Instruct \
  --output-dir "${OUTPUT_DIR}" \
  --stages "${STAGES}" \
  --gpu-ids "${GPU_IDS}" \
  --teacher-num-samples "${TEACHER_NUM_SAMPLES}" \
  --teacher-temperature "${TEACHER_TEMPERATURE}" \
  --teacher-top-p "${TEACHER_TOP_P}" \
  --teacher-do-sample \
  --teacher-batch-size "${TEACHER_BATCH_SIZE}" \
  --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --teacher-tensor-parallel-size 1 \
  --data-variant teacher_self_qwen2_5_0_5b \
  --reasoning-source teacher \
  --sft-epochs "${SFT_EPOCHS}" \
  --eval-arg data.batch_size=8 \
  --grpo-env TOTAL_EPOCHS="${GRPO_TOTAL_EPOCHS}"

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp1100_${DATASET}_teacher_qwen2_5_3b_t${TEACHER_TEMPERATURE}_n${TEACHER_NUM_SAMPLES} \
  --dataset "${DATASET}" \
  --base-model "${BASE_MODEL}" \
  --teacher-model /data/pretrain_models/Qwen2.5-3B-Instruct \
  --output-dir "${OUTPUT_DIR}" \
  --stages "${STAGES}" \
  --gpu-ids "${GPU_IDS}" \
  --teacher-num-samples "${TEACHER_NUM_SAMPLES}" \
  --teacher-temperature "${TEACHER_TEMPERATURE}" \
  --teacher-top-p "${TEACHER_TOP_P}" \
  --teacher-do-sample \
  --teacher-batch-size "${TEACHER_BATCH_SIZE}" \
  --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --teacher-tensor-parallel-size 1 \
  --data-variant teacher_qwen2_5_3b \
  --reasoning-source teacher \
  --sft-epochs "${SFT_EPOCHS}" \
  --eval-arg data.batch_size=8 \
  --grpo-env TOTAL_EPOCHS="${GRPO_TOTAL_EPOCHS}"

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp1100_${DATASET}_teacher_qwen2_5_7b_t${TEACHER_TEMPERATURE}_n${TEACHER_NUM_SAMPLES} \
  --dataset "${DATASET}" \
  --base-model "${BASE_MODEL}" \
  --teacher-model /data/pretrain_models/Qwen2.5-7B-Instruct \
  --output-dir "${OUTPUT_DIR}" \
  --stages "${STAGES}" \
  --gpu-ids "${GPU_IDS}" \
  --teacher-num-samples "${TEACHER_NUM_SAMPLES}" \
  --teacher-temperature "${TEACHER_TEMPERATURE}" \
  --teacher-top-p "${TEACHER_TOP_P}" \
  --teacher-do-sample \
  --teacher-batch-size "${TEACHER_BATCH_SIZE}" \
  --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --teacher-tensor-parallel-size 1 \
  --data-variant teacher_qwen2_5_7b \
  --reasoning-source teacher \
  --sft-epochs "${SFT_EPOCHS}" \
  --eval-arg data.batch_size=8 \
  --grpo-env TOTAL_EPOCHS="${GRPO_TOTAL_EPOCHS}"

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp1100_${DATASET}_teacher_qwen2_5_32b_t${TEACHER_TEMPERATURE}_n${TEACHER_NUM_SAMPLES} \
  --dataset "${DATASET}" \
  --base-model "${BASE_MODEL}" \
  --teacher-model /data/pretrain_models/Qwen2.5-32B-Instruct \
  --output-dir "${OUTPUT_DIR}" \
  --stages "${STAGES}" \
  --gpu-ids "${BIG_GPU_IDS}" \
  --teacher-num-samples "${TEACHER_NUM_SAMPLES}" \
  --teacher-temperature "${TEACHER_TEMPERATURE}" \
  --teacher-top-p "${TEACHER_TOP_P}" \
  --teacher-do-sample \
  --teacher-batch-size 16 \
  --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --teacher-tensor-parallel-size 2 \
  --data-variant teacher_qwen2_5_32b \
  --reasoning-source teacher \
  --sft-epochs "${SFT_EPOCHS}" \
  --eval-arg data.batch_size=8 \
  --grpo-env TOTAL_EPOCHS="${GRPO_TOTAL_EPOCHS}"

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp1100_${DATASET}_teacher_qwen2_5_72b_t${TEACHER_TEMPERATURE}_n${TEACHER_NUM_SAMPLES} \
  --dataset "${DATASET}" \
  --base-model "${BASE_MODEL}" \
  --teacher-model /data/pretrain_models/Qwen2.5-72B-Instruct \
  --output-dir "${OUTPUT_DIR}" \
  --stages "${STAGES}" \
  --gpu-ids "${XL_GPU_IDS}" \
  --teacher-num-samples "${TEACHER_NUM_SAMPLES}" \
  --teacher-temperature "${TEACHER_TEMPERATURE}" \
  --teacher-top-p "${TEACHER_TOP_P}" \
  --teacher-do-sample \
  --teacher-batch-size 8 \
  --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --teacher-tensor-parallel-size 4 \
  --data-variant teacher_qwen2_5_72b \
  --reasoning-source teacher \
  --sft-epochs "${SFT_EPOCHS}" \
  --eval-arg data.batch_size=8 \
  --grpo-env TOTAL_EPOCHS="${GRPO_TOTAL_EPOCHS}"
