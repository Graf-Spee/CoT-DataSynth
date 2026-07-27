#!/usr/bin/env bash
set -euo pipefail

cd /home/hrh/CoT-DataSynth

PYTHON_BIN="${PYTHON_BIN:-/home/hrh/anaconda3/envs/verl-torch280/bin/python}"
PYTHON_BIN_DIR="$(dirname "${PYTHON_BIN}")"
export PATH="${PYTHON_BIN_DIR}:${PATH}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

SET_TEACHER_MODEL=0
source DataObs/experiment_instruction/_qwen_family_defaults.sh
set_qwen_family_defaults

DATASET="${DATASET:-gsm8k}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/hrh/COT/experiments}"
GPU_IDS="${GPU_IDS:-4}"
DISTILL_GPU_IDS="${DISTILL_GPU_IDS:-${GPU_IDS}}"
TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-${GPU_IDS}}"
EVAL_GPU_IDS="${EVAL_GPU_IDS:-${TRAIN_GPU_IDS}}"
SIMILARITY_TYPE="${SIMILARITY_TYPE:-jaccard}"

BASE_MODEL="${BASE_MODEL:-/data/pretrain_models/Qwen3-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-/data/pretrain_models/Qwen3-14B}"
DISTILL_INPUT="${DISTILL_INPUT:-/data/open_datasets/GSM8K/main/train-00000-of-00001.parquet}"

POOL_NUM_REPHRASES="${POOL_NUM_REPHRASES:-16}"
MAX_SUBSET_N="${MAX_SUBSET_N:-8}"
SAMPLE_COUNTS="${SAMPLE_COUNTS:-1,2,4,8}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-8}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-1024}"
REPHRASE_MAX_NEW_TOKENS="${REPHRASE_MAX_NEW_TOKENS:-512}"
TEACHER_GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.65}"

RUN_TAG="${RUN_TAG:-qrephrase_teacher14b_pool_n${MAX_SUBSET_N}}"
POOL_EXPERIMENT_ID="exp7100_${DATASET}_${FAMILY_TAG}_${RUN_TAG}"
SUBSET_DIR="${OUTPUT_DIR}/exp7100_question_rephrasing_same_prompt_subsets"
SUBSET_PREFIX="${DATASET}_${FAMILY_TAG}_${RUN_TAG}_same_prompt"
POOL_SFT="${OUTPUT_DIR}/${POOL_EXPERIMENT_ID}/distill/filtered_sft.parquet"
VAL_SFT="${SUBSET_DIR}/${SUBSET_PREFIX}_n1.parquet"

RUN_LOG="${OUTPUT_DIR}/exp7100_question_rephrasing_run.log"
mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "[$(date '+%F %T')] Starting question rephrasing experiment"
echo "  dataset=${DATASET}"
echo "  base_model=${BASE_MODEL}"
echo "  teacher_model=${TEACHER_MODEL}"
echo "  distill_gpu_ids=${DISTILL_GPU_IDS}"
echo "  train_gpu_ids=${TRAIN_GPU_IDS}"
echo "  pool_experiment_id=${POOL_EXPERIMENT_ID}"
echo "  subset_dir=${SUBSET_DIR}"

"${PYTHON_BIN}" DataObs/scripts/experiment_pipeline_vllm.py \
  --experiment-id "${POOL_EXPERIMENT_ID}" \
  --dataset "${DATASET}" \
  --base-model "${BASE_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --stages distill \
  --gpu-ids "${DISTILL_GPU_IDS}" \
  --data-variant "${RUN_TAG}" \
  --reasoning-source teacher \
  --distill-method question_rephrasing \
  --distill-input "${DISTILL_INPUT}" \
  --teacher-model "${TEACHER_MODEL}" \
  --teacher-num-samples "${POOL_NUM_REPHRASES}" \
  --teacher-temperature 0.7 \
  --teacher-top-p 0.95 \
  --teacher-batch-size "${TEACHER_BATCH_SIZE}" \
  --teacher-max-new-tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --teacher-tensor-parallel-size 1 \
  --teacher-gpu-memory-utilization "${TEACHER_GPU_MEMORY_UTILIZATION}" \
  --teacher-correct-threshold 0.99 \
  --teacher-do-sample \
  --rephrase-max-new-tokens "${REPHRASE_MAX_NEW_TOKENS}"

"${PYTHON_BIN}" DataObs/tools/make_same_prompt_sample_subsets.py \
  --input "${POOL_SFT}" \
  --output-dir "${SUBSET_DIR}" \
  --prefix "${SUBSET_PREFIX}" \
  --group-key source_index \
  --sample-counts "${SAMPLE_COUNTS}" \
  --seed 42

for n in 1 2 4 8; do
  subset_path="${SUBSET_DIR}/${SUBSET_PREFIX}_n${n}.parquet"
  if [ ! -f "${subset_path}" ]; then
    echo "[ERROR] Missing subset for n=${n}: ${subset_path}" >&2
    exit 1
  fi

  "${PYTHON_BIN}" DataObs/scripts/experiment_pipeline_vllm.py \
    --experiment-id "exp7100_${DATASET}_${FAMILY_TAG}_${RUN_TAG}_same_prompt_n${n}" \
    --dataset "${DATASET}" \
    --base-model "${BASE_MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --stages metrics,sft,sft_eval,grpo,grpo_eval \
    --gpu-ids "${TRAIN_GPU_IDS}" \
    --eval-gpu-ids "${EVAL_GPU_IDS}" \
    --data-variant "${RUN_TAG}_same_prompt_n${n}" \
    --reasoning-source teacher \
    --sft-data "${subset_path}" \
    --sft-val-data "${VAL_SFT}" \
    --metrics-data "${subset_path}" \
    --similarity-type "${SIMILARITY_TYPE}" \
    --sft-arg data.train_batch_size=4 \
    --sft-arg data.micro_batch_size_per_gpu=2 \
    --sft-arg data.max_length=2048 \
    --eval-temperature 0.0 \
    --eval-top-p 1.0 \
    --eval-max-response-len 16384 \
    --eval-batch-size 8 \
    --eval-gpu-memory-utilization 0.5 \
    --eval-tensor-parallel-size 1 \
    --grpo-env TOTAL_EPOCHS=1 \
    --grpo-env TRAIN_BATCH_SIZE=32 \
    --grpo-env PPO_MINI_BATCH_SIZE=32 \
    --grpo-env PPO_MICRO_BATCH_SIZE_PER_GPU=4 \
    --grpo-env LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8 \
    --grpo-env ROLLOUT_N=4 \
    --grpo-env MAX_PROMPT_LEN=512 \
    --grpo-env MAX_RESPONSE_LEN=2048 \
    --grpo-env LR=5e-7 \
    --grpo-env KL_COEF=0.001 \
    --grpo-env GPU_MEMORY_UTILIZATION=0.6 \
    --grpo-env VLLM_MAX_NUM_SEQS=64 \
    --grpo-env VLLM_MAX_NUM_BATCHED_TOKENS=4096 \
    --grpo-env SAVE_FREQ=10 \
    --grpo-env TEST_FREQ=5
done

echo "[$(date '+%F %T')] Finished question rephrasing experiment"
