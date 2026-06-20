#!/usr/bin/env bash
set -euo pipefail

cd /home/hrh/CoT-DataSynth

PYTHON_BIN="${PYTHON_BIN:-/home/hrh/anaconda3/envs/verl-cot/bin/python}"

SET_TEACHER_MODEL=0
source DataObs/experiment_instruction/_qwen_family_defaults.sh
set_qwen_family_defaults

DRY_RUN="${DRY_RUN---dry-run}"
DATASET="${DATASET:-gsm8k}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/hrh/COT/experiments}"
GPU_IDS="${GPU_IDS:-0}"
SIMILARITY_TYPE="${SIMILARITY_TYPE:-jaccard}"

LOW_SFT="${LOW_SFT:-/path/to/low_diversity_sft.parquet}"
MID_SFT="${MID_SFT:-/path/to/mid_diversity_sft.parquet}"
HIGH_SFT="${HIGH_SFT:-/path/to/high_diversity_sft.parquet}"
VAL_SFT="${VAL_SFT:-${MID_SFT}}"

for item in "low:${LOW_SFT}" "mid:${MID_SFT}" "high:${HIGH_SFT}"; do
  bucket="${item%%:*}"
  data_path="${item#*:}"
  "${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
    --experiment-id exp6000_${DATASET}_${FAMILY_TAG}_diversity_${bucket} \
    --dataset "${DATASET}" \
    --base-model "${BASE_MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --stages metrics,sft,sft_eval,grpo,grpo_eval \
    --gpu-ids "${GPU_IDS}" \
    --data-variant diversity_${bucket} \
    --sft-data "${data_path}" \
    --sft-val-data "${VAL_SFT}" \
    --metrics-data "${data_path}" \
    --similarity-type "${SIMILARITY_TYPE}" \
    --grpo-env TOTAL_EPOCHS=1
done
