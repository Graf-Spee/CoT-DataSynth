#!/usr/bin/env bash

cd /home/hrh/CoT-DataSynth

PYTHON_BIN="${PYTHON_BIN:-/home/hrh/anaconda3/envs/verl-cot/bin/python}"

DRY_RUN="${DRY_RUN---dry-run}"
DATASET="${DATASET:-gsm8k}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/hrh/COT/experiments}"
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

run_teacher() {
  local family="$1"
  local tag="$2"
  local model_path="$3"
  local gpu_ids="$4"
  local tp_size="$5"
  local batch_size="$6"

  "${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
    --experiment-id exp1100_${DATASET}_${family}_teacher_${tag}_t${TEACHER_TEMPERATURE}_n${TEACHER_NUM_SAMPLES} \
    --dataset "${DATASET}" \
    --base-model "${BASE_MODEL}" \
    --teacher-model "${model_path}" \
    --output-dir "${OUTPUT_DIR}" \
    --stages "${STAGES}" \
    --gpu-ids "${gpu_ids}" \
    --teacher-num-samples "${TEACHER_NUM_SAMPLES}" \
    --teacher-temperature "${TEACHER_TEMPERATURE}" \
    --teacher-top-p "${TEACHER_TOP_P}" \
    --teacher-do-sample \
    --teacher-batch-size "${batch_size}" \
    --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
    --teacher-tensor-parallel-size "${tp_size}" \
    --data-variant ${family}_teacher_${tag} \
    --reasoning-source teacher \
    --sft-epochs "${SFT_EPOCHS}" \
    --eval-arg data.batch_size=8 \
    --grpo-env TOTAL_EPOCHS="${GRPO_TOTAL_EPOCHS}"
}
