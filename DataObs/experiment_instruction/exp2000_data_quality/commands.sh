#!/usr/bin/env bash
set -euo pipefail

cd /home/hrh/CoT-DataSynth

PYTHON_BIN="${PYTHON_BIN:-/home/hrh/anaconda3/envs/verl-cot/bin/python}"

DRY_RUN="${DRY_RUN:---dry-run}"
DATASET="${DATASET:-gsm8k}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/hrh/COT/experiments}"
BASE_MODEL="${BASE_MODEL:-/data/pretrain_models/Qwen2.5-0.5B-Instruct}"
GPU_IDS="${GPU_IDS:-0}"

# Fill these with distilled parquet files produced by exp1100/exp1200 or manual filters.
QUALITY_LOW_SFT="${QUALITY_LOW_SFT:-/path/to/quality_low_sft.parquet}"
QUALITY_MID_SFT="${QUALITY_MID_SFT:-/path/to/quality_mid_sft.parquet}"
QUALITY_HIGH_SFT="${QUALITY_HIGH_SFT:-/path/to/quality_high_sft.parquet}"
VAL_SFT="${VAL_SFT:-${QUALITY_HIGH_SFT}}"

for variant in low mid high; do
  data_var="QUALITY_${variant^^}_SFT"
  data_path="${!data_var}"
  "${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
    --experiment-id exp2000_${DATASET}_quality_${variant} \
    --dataset "${DATASET}" \
    --base-model "${BASE_MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --stages metrics,sft,sft_eval,grpo,grpo_eval \
    --gpu-ids "${GPU_IDS}" \
    --data-variant quality_${variant} \
    --sft-data "${data_path}" \
    --sft-val-data "${VAL_SFT}" \
    --metrics-data "${data_path}" \
    --metrics-model "${BASE_MODEL}" \
    --n-splits 10 \
    --similarity-type jaccard \
    --grpo-env TOTAL_EPOCHS=1
done
