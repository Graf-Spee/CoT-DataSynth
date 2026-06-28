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
if [ -z "${SEED_INPUT:-}" ]; then
  case "${DATASET}" in
    gsm8k) SEED_INPUT="/data/open_datasets/GSM8K/main/train-00000-of-00001.parquet" ;;
    math|math-500) SEED_INPUT="/data/open_datasets/MATH/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet" ;;
    arc-challenge|ai2_arc) SEED_INPUT="/data/open_datasets/ai2_arc/ARC-Challenge/train-00000-of-00001.parquet" ;;
    aqua_rat) SEED_INPUT="/data/open_datasets/aqua_rat/raw/train-00000-of-00001.parquet" ;;
    strategyQA|strategyqa) SEED_INPUT="/data/open_datasets/StrategyQA/data/train-00000-of-00001-506370352f622815.parquet" ;;
    commonsenseQA|commonsenseqa) SEED_INPUT="/data/open_datasets/CommonsenseQA/data/train-00000-of-00001.parquet" ;;
    numinamath) SEED_INPUT="/data/open_datasets/NuminaMath-CoT/data/train-00000-of-00005.parquet" ;;
    mbpp) SEED_INPUT="/data/open_datasets/mbpp/sanitized/train-00000-of-00001.parquet" ;;
    *) echo "[ERROR] Please set SEED_INPUT for DATASET=${DATASET}" >&2; exit 1 ;;
  esac
fi
SPLIT_DIR="${SPLIT_DIR:-${OUTPUT_DIR}/_prepared/exp3100_${DATASET}_${FAMILY_TAG}_seed_split}"
SFT_RATIO="${SFT_RATIO:-0.5}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TEACHER_NUM_SAMPLES="${TEACHER_NUM_SAMPLES:-4}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.7}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-1024}"
STAGES="${STAGES:-distill,metrics,sft,sft_eval,grpo,grpo_eval}"

# Setting A: SFT train prompt == RL train prompt.
# Setting B: SFT train prompt != RL train prompt.
# The split step always runs because it is cheap and records the exact prompt partition.

"${PYTHON_BIN}" DataObs/scripts/split_seed_prompts.py \
  --input "${SEED_INPUT}" \
  --output-dir "${SPLIT_DIR}" \
  --sft-ratio "${SFT_RATIO}" \
  --seed "${SPLIT_SEED}" \
  --prefix "${DATASET}"

SFT_SEED="${SPLIT_DIR}/${DATASET}_sft_seed.parquet"
RL_DISJOINT="${SPLIT_DIR}/${DATASET}_rl_train.parquet"
RL_OVERLAP="${SPLIT_DIR}/${DATASET}_overlap_train.parquet"

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp3100_${DATASET}_${FAMILY_TAG}_settingA_sft_eq_rl_prompt \
  --dataset "${DATASET}" \
  --base-model "${BASE_MODEL}" \
  --teacher-model "${TEACHER_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --stages "${STAGES}" \
  --gpu-ids "${GPU_IDS}" \
  --distill-input "${SFT_SEED}" \
  --rl-train-from-distill-kept \
  --data-variant sft_eq_rl_prompt \
  --reasoning-source teacher \
  --teacher-num-samples "${TEACHER_NUM_SAMPLES}" \
  --teacher-temperature "${TEACHER_TEMPERATURE}" \
  --teacher-do-sample \
  --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --grpo-env TOTAL_EPOCHS=1

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline.py ${DRY_RUN} \
  --experiment-id exp3100_${DATASET}_${FAMILY_TAG}_settingB_sft_ne_rl_prompt \
  --dataset "${DATASET}" \
  --base-model "${BASE_MODEL}" \
  --teacher-model "${TEACHER_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --stages "${STAGES}" \
  --gpu-ids "${GPU_IDS}" \
  --distill-input "${SFT_SEED}" \
  --rl-train-data "${RL_DISJOINT}" \
  --data-variant sft_ne_rl_prompt \
  --reasoning-source teacher \
  --teacher-num-samples "${TEACHER_NUM_SAMPLES}" \
  --teacher-temperature "${TEACHER_TEMPERATURE}" \
  --teacher-do-sample \
  --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --grpo-env TOTAL_EPOCHS=1
