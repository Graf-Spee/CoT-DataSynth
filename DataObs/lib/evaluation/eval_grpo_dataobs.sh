#!/bin/bash
# One-click evaluation for DataObs GRPO outputs.

set -euo pipefail

usage() {
    cat <<EOF
Usage:
  bash $0 <grpo_output_dir> <base_model> [dataset_name] [eval_output_dir] [gpu_ids]

Example:
  bash $0 \\
    /data/hrh/COT/GSM8K/grpo/split_0 \\
    /data/pretrain_models/Qwen2.5-0.5B-Instruct \\
    gsm8k \\
    /data/hrh/COT/GSM8K/eval/grpo_split_0 \\
    0,1

Notes:
  - If latest GRPO actor is LoRA, this script merges it into actor/merged_model first.
  - Set STEP=174 to evaluate a specific global_step_174.
  - Set FORCE_MERGE=1 to rebuild actor/merged_model.
  - Set DRY_RUN=1 to print commands without running merge/eval.
EOF
}

if [ "$#" -lt 2 ]; then
    usage
    exit 1
fi

GRPO_DIR="$1"
BASE_MODEL="$2"
DATASET_NAME="${3:-gsm8k}"
EVAL_OUTPUT_DIR="${4:-}"
GPU_IDS="${5:-0}"

# The GRPO actor merge happens before eval_dataobs.sh is called, so the
# visible-device mapping must be set here as well.
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAOBS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_DIR="$(cd "${DATAOBS_DIR}/.." && pwd)"
MERGE_SCRIPT="${DATAOBS_DIR}/lib/model_ops/merge_lora_qwen.py"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_dataobs.sh"

if [ ! -d "$GRPO_DIR" ]; then
    echo "[ERROR] GRPO output dir not found: $GRPO_DIR"
    exit 1
fi

if [ ! -d "$BASE_MODEL" ]; then
    echo "[ERROR] Base model dir not found: $BASE_MODEL"
    exit 1
fi

if [ ! -f "$MERGE_SCRIPT" ]; then
    echo "[ERROR] Merge script not found: $MERGE_SCRIPT"
    exit 1
fi

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "[ERROR] Eval script not found: $EVAL_SCRIPT"
    exit 1
fi

if [ -n "${STEP:-}" ]; then
    STEP_ID="$STEP"
elif [ -f "${GRPO_DIR}/latest_checkpointed_iteration.txt" ]; then
    STEP_ID="$(cat "${GRPO_DIR}/latest_checkpointed_iteration.txt")"
else
    STEP_ID="$(find "$GRPO_DIR" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null \
        | sed 's|.*/global_step_||' \
        | sort -n \
        | tail -n 1)"
fi

if [ -z "$STEP_ID" ]; then
    echo "[ERROR] Could not find latest checkpoint under: $GRPO_DIR"
    exit 1
fi

ACTOR_DIR="${GRPO_DIR}/global_step_${STEP_ID}/actor"
ACTOR_HF_DIR="${ACTOR_DIR}/huggingface"
ACTOR_LORA_DIR="${ACTOR_DIR}/lora_adapter"
MERGED_MODEL_DIR="${ACTOR_DIR}/merged_model"

if [ -z "$EVAL_OUTPUT_DIR" ]; then
    EVAL_OUTPUT_DIR="${GRPO_DIR}/eval_${DATASET_NAME}_step_${STEP_ID}"
fi

is_full_hf_model_dir() {
    local path="$1"
    [ -d "$path" ] || return 1
    [ -f "$path/config.json" ] || return 1
    [ -f "$path/model.safetensors" ] || [ -f "$path/pytorch_model.bin" ] || [ -f "$path/model.safetensors.index.json" ] || compgen -G "$path/*.safetensors" >/dev/null
}

if [ ! -d "$ACTOR_DIR" ]; then
    echo "[ERROR] Actor checkpoint dir not found: $ACTOR_DIR"
    exit 1
fi

if [ -d "$ACTOR_LORA_DIR" ] && [ -f "${ACTOR_LORA_DIR}/adapter_model.safetensors" ]; then
    if is_full_hf_model_dir "$MERGED_MODEL_DIR" && [ "${FORCE_MERGE:-0}" != "1" ]; then
        MODEL_FOR_EVAL="$MERGED_MODEL_DIR"
        echo "[INFO] Reusing merged GRPO model: $MODEL_FOR_EVAL"
    else
        if [ ! -d "$ACTOR_HF_DIR" ]; then
            echo "[ERROR] Tokenizer/config dir not found: $ACTOR_HF_DIR"
            exit 1
        fi

        echo "[INFO] Merging GRPO LoRA adapter..."
        echo "[INFO]   base:      $BASE_MODEL"
        echo "[INFO]   lora:      $ACTOR_LORA_DIR"
        echo "[INFO]   tokenizer: $ACTOR_HF_DIR"
        echo "[INFO]   output:    $MERGED_MODEL_DIR"

        merge_cmd=(
            python3 "$MERGE_SCRIPT"
            --base "$BASE_MODEL"
            --lora "$ACTOR_LORA_DIR"
            --tokenizer "$ACTOR_HF_DIR"
            --output "$MERGED_MODEL_DIR"
        )

        if [ "${DRY_RUN:-0}" = "1" ]; then
            printf '[DRY_RUN] Merge command:'
            printf ' %q' "${merge_cmd[@]}"
            printf '\n'
        else
            "${merge_cmd[@]}"
        fi

        MODEL_FOR_EVAL="$MERGED_MODEL_DIR"
    fi
elif is_full_hf_model_dir "$ACTOR_HF_DIR"; then
    MODEL_FOR_EVAL="$ACTOR_HF_DIR"
    echo "[INFO] No LoRA adapter found; using actor HuggingFace dir: $MODEL_FOR_EVAL"
else
    echo "[ERROR] Could not find evaluable actor model."
    echo "[ERROR] Expected LoRA adapter at: $ACTOR_LORA_DIR"
    echo "[ERROR] Or full HF model at: $ACTOR_HF_DIR"
    exit 1
fi

echo "=========================================="
echo "DataObs GRPO Evaluation"
echo "  GRPO dir:      $GRPO_DIR"
echo "  Step:          $STEP_ID"
echo "  Model:         $MODEL_FOR_EVAL"
echo "  Base model:    $BASE_MODEL"
echo "  Dataset:       $DATASET_NAME"
echo "  Eval output:   $EVAL_OUTPUT_DIR"
echo "  GPUs:          $GPU_IDS"
echo "=========================================="

eval_cmd=(
    bash "$EVAL_SCRIPT"
    "$MODEL_FOR_EVAL"
    "$BASE_MODEL"
    "$DATASET_NAME"
    "$EVAL_OUTPUT_DIR"
    "$GPU_IDS"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '[DRY_RUN] Eval command:'
    printf ' %q' "${eval_cmd[@]}"
    printf '\n'
    exit 0
fi

"${eval_cmd[@]}"
