#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_filter_smoke_teacher_max_tokens_sweep.sh
#   FORCE_RERUN=1 ./scripts/run_filter_smoke_teacher_max_tokens_sweep.sh
#   DRY_RUN=1 ./scripts/run_filter_smoke_teacher_max_tokens_sweep.sh
#
# Optional environment overrides:
#   GPU_ID, OUTPUT_DIR, LOG_DIR, PYTHON_BIN, DATASET, BASE_MODEL, TEACHER_MODEL

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_ID="${GPU_ID:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/hrh/COT/experiments}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/filter_smoke_teacher_max_tokens_sweep}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

DATASET="${DATASET:-gsm8k}"
BASE_MODEL="${BASE_MODEL:-/data/pretrain_models/Qwen3-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-/data/pretrain_models/Qwen3-8B}"

mkdir -p "$LOG_DIR"

run_pipeline() {
  local token_budget="$1"
  local method="$2"
  local experiment_id="$3"
  shift 3

  local done_file="$LOG_DIR/${experiment_id}.done"
  local log_file="$LOG_DIR/${experiment_id}.log"
  local cmd_file="$LOG_DIR/${experiment_id}.cmd"

  if [[ -f "$done_file" && "$FORCE_RERUN" != "1" && "$FORCE_RERUN" != "true" ]]; then
    echo "[SKIP] $experiment_id"
    return
  fi

  local cmd=(
    "$PYTHON_BIN" DataObs/scripts/experiment_pipeline_vllm.py
    --experiment-id "$experiment_id"
    --dataset "$DATASET"
    --base-model "$BASE_MODEL"
    --teacher-model "$TEACHER_MODEL"
    --output-dir "$OUTPUT_DIR"
    --stages distill
    --distill-method "$method"
    --gpu-ids "$GPU_ID"
    --teacher-temperature 0.7
    --teacher-top-p 0.95
    --teacher-do-sample
    --teacher-batch-size 8
    --teacher-max-new-tokens "$token_budget"
    --teacher-tensor-parallel-size 1
    --teacher-gpu-memory-utilization 0.65
  )
  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
    cmd+=(--dry-run)
  fi
  cmd+=("$@")

  printf "%q " "${cmd[@]}" > "$cmd_file"
  printf "\n" >> "$cmd_file"
  echo "[RUN] $experiment_id"
  "${cmd[@]}" >"$log_file" 2>&1
  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN DONE] $experiment_id"
  else
    date -Is > "$done_file"
    echo "[DONE] $experiment_id"
  fi
}

for token_budget in 2048 4096 8192; do
  run_pipeline \
    "$token_budget" \
    teacher_correctness_filter \
    "smoke20_gsm8k_teacher_filter_qwen3_8b_t${token_budget}_gpu${GPU_ID}" \
    --teacher-num-samples 1 \
    --smoke-num-rows 20 \
    --sft-epochs 1 \
    --sft-arg data.train_batch_size=1 \
    --sft-arg data.micro_batch_size_per_gpu=1 \
    --sft-arg data.max_length=2048 \
    --eval-batch-size 8 \
    --eval-gpu-memory-utilization 0.75

  run_pipeline \
    "$token_budget" \
    question_rephrasing \
    "smoke_gsm8k_question_rephrasing_qwen3_8b_n4_t${token_budget}_gpu${GPU_ID}" \
    --teacher-num-samples 4 \
    --disable-teacher-filter \
    --smoke-num-rows 5 \
    --sft-epochs 1 \
    --sft-arg data.train_batch_size=1 \
    --sft-arg data.micro_batch_size_per_gpu=1 \
    --sft-arg data.max_length=2048 \
    --eval-batch-size 8 \
    --eval-gpu-memory-utilization 0.75

  run_pipeline \
    "$token_budget" \
    answer_augmentation \
    "smoke_gsm8k_answer_augmentation_qwen3_8b_n4_t${token_budget}_gpu${GPU_ID}" \
    --teacher-num-samples 4 \
    --answer-aug-use-original-metamath-prompt \
    --smoke-num-rows 5 \
    --sft-epochs 1 \
    --sft-arg data.train_batch_size=4 \
    --sft-arg data.micro_batch_size_per_gpu=2 \
    --sft-arg data.max_length=2048 \
    --eval-batch-size 8 \
    --eval-gpu-memory-utilization 0.75

  run_pipeline \
    "$token_budget" \
    question_augmentation \
    "smoke_gsm8k_question_augmentation_qwen3_8b_n4_t${token_budget}_gpu${GPU_ID}" \
    --teacher-num-samples 4 \
    --disable-teacher-filter \
    --smoke-num-rows 5 \
    --sft-epochs 1 \
    --sft-arg data.train_batch_size=4 \
    --sft-arg data.micro_batch_size_per_gpu=2 \
    --sft-arg data.max_length=2048 \
    --eval-batch-size 8 \
    --eval-gpu-memory-utilization 0.75

  run_pipeline \
    "$token_budget" \
    reverse_thinking \
    "smoke20_gsm8k_reverse_thinking_qwen3_8b_n1_t${token_budget}_gpu${GPU_ID}" \
    --teacher-num-samples 1 \
    --disable-teacher-filter \
    --smoke-num-rows 20 \
    --sft-epochs 1 \
    --sft-arg data.train_batch_size=1 \
    --sft-arg data.micro_batch_size_per_gpu=1 \
    --sft-arg data.max_length=3072 \
    --eval-batch-size 8 \
    --eval-gpu-memory-utilization 0.75
done

echo "All filter smoke runs completed. Logs: $LOG_DIR"
