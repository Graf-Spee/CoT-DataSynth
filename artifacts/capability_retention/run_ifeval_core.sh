#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/hrh/CoT-DataSynth
PYTHON=/home/hrh/anaconda3/envs/verl-torch280/bin/python
GPU_IDS=${GPU_IDS:-1}
BASE_MODEL=/data/pretrain_models/Qwen3-4B
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/artifacts/capability_retention/ifeval_core}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_V1=0
export NLTK_DATA=${REPO_ROOT}/artifacts/nltk_data

run_eval() {
  local name=$1
  local model_path=$2
  local out_dir=${OUT_ROOT}/${name}

  mkdir -p "${out_dir}/logs"
  echo "[RUN] ${name}"
  echo "[RUN] model_path=${model_path}"
  echo "[RUN] out_dir=${out_dir}"

  "${PYTHON}" "${REPO_ROOT}/DataObs/lib/evaluation/eval_vllm.py" \
    --model-path "${model_path}" \
    --base-model "${BASE_MODEL}" \
    --dataset ifeval \
    --output-dir "${out_dir}" \
    --gpu-ids "${GPU_IDS}" \
    --temperature 0 \
    --top-p 1 \
    --top-k -1 \
    --max-response-len 4096 \
    --batch-size 8 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --tensor-parallel-size 1 \
    --seed 42 2>&1 | tee "${out_dir}/logs/run.log"
}

run_eval "base_qwen3_4b" \
  "${BASE_MODEL}"

run_eval "easy_sft" \
  "/data/hrh/COT/experiments/exp5000_gsm8k_qwen3_difficulty_easy_distill/eval/sft/merged_model"

run_eval "easy_grpo" \
  "/data/hrh/COT/experiments/exp5000_gsm8k_qwen3_difficulty_easy_distill/grpo/global_step_233/actor/merged_model"
