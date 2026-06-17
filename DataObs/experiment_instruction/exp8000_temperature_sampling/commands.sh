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
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-1024}"
STAGES="${STAGES:-distill,metrics,sft,sft_eval,grpo,grpo_eval}"

# Temperature x number-of-samples sweep. Keep teacher, seed data, student, and train config fixed.

for temp in 0.0 0.3 0.7 1.0; do
  for n in 1 4 8; do
    sample_flag=""
    if [ "${n}" -gt 1 ] || [ "${temp}" != "0.0" ]; then
      sample_flag="--teacher-do-sample"
    fi
    "${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
      --experiment-id exp8000_${DATASET}_t${temp}_n${n}_filter_on \
      --dataset "${DATASET}" \
      --base-model "${BASE_MODEL}" \
      --teacher-model "${TEACHER_MODEL}" \
      --output-dir "${OUTPUT_DIR}" \
      --stages "${STAGES}" \
      --gpu-ids "${GPU_IDS}" \
      --teacher-temperature "${temp}" \
      --teacher-num-samples "${n}" \
      ${sample_flag} \
      --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
      --data-variant temp_${temp}_n_${n}_filter_on \
      --reasoning-source teacher \
      --grpo-env TOTAL_EPOCHS=1
  done
done

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp8000_${DATASET}_t0.7_n4_filter_off \
  --dataset "${DATASET}" \
  --base-model "${BASE_MODEL}" \
  --teacher-model "${TEACHER_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --stages "${STAGES}" \
  --gpu-ids "${GPU_IDS}" \
  --teacher-temperature 0.7 \
  --teacher-num-samples 4 \
  --teacher-do-sample \
  --disable-teacher-filter \
  --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --data-variant temp_0.7_n_4_filter_off \
  --reasoning-source teacher \
  --grpo-env TOTAL_EPOCHS=1
