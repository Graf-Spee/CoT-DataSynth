#!/bin/bash
# Run generation + evaluation for multiple datasets with scripts/eval.sh.

set -euo pipefail

usage() {
    cat <<EOF
Usage:
  bash $0 <gpu_ids> [checkpoint_path] [dataset_csv] [other eval.sh configs...]

Examples:
  # Evaluate the configured/base model on a small default suite.
  BASE_MODEL=/data/pretrain_models/Qwen2.5-0.5B-Instruct MODEL_NAME=Qwen2.5-0.5B-Instruct \\
    bash $0 0

  # Evaluate a checkpoint on selected datasets.
  bash $0 0,1 /data/hjw/outputs/Qwen2.5-0.5B-gsm8k-sft/checkpoint-last gsm8k,math-500,arc-challenge

Environment:
  DATASETS="gsm8k,math-500,arc-challenge"  # alternative to dataset_csv
  STOP_ON_ERROR=0                           # continue after failures
  DRY_RUN=1                                 # print commands only
  PROJECT_DIR=/path/to/output_root          # eval outputs go under \$PROJECT_DIR/evals

Default datasets:
  gsm8k,math-500,arc-challenge,aqua_rat,commonsenseQA,strategyQA
EOF
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

GPU_IDS="$1"
shift 1

CHECKPOINT_PATH_ARG=""
if [ "$#" -gt 0 ] && [[ "$1" == */* ]]; then
    CHECKPOINT_PATH_ARG="$1"
    shift 1
fi

DATASET_CSV="${DATASETS:-gsm8k,math-500,arc-challenge,aqua_rat,commonsenseQA,strategyQA}"
if [ "$#" -gt 0 ] && [[ "$1" != *=* ]]; then
    DATASET_CSV="$1"
    shift 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval.sh"

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "[ERROR] eval.sh not found: $EVAL_SCRIPT"
    exit 1
fi

SUMMARY_DIR="${SUMMARY_DIR:-${REPO_DIR}/outputs/eval_suite_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$SUMMARY_DIR"
SUMMARY_FILE="${SUMMARY_DIR}/summary.tsv"
printf 'dataset\tstatus\teval_dir\tmetric_tail\n' > "$SUMMARY_FILE"

IFS=',' read -r -a DATASET_LIST <<< "$DATASET_CSV"

normalize_dataset() {
    local raw="$1"
    local lowered
    lowered="$(echo "$raw" | tr '[:upper:]' '[:lower:]')"
    case "$lowered" in
        arc | arc- | ai2_arc | ai2-arc | arc_challenge | arc-challenge)
            echo "arc-challenge"
            ;;
        aqua | aqua-rat | aqua_rat)
            echo "aqua_rat"
            ;;
        commonsenseqa | csqa)
            echo "commonsenseQA"
            ;;
        math500 | math_500 | math-500)
            echo "math-500"
            ;;
        mbpp-plus | mbpp_plus | mbppplus)
            echo "mbppplus"
            ;;
        humaneval-plus | humaneval_plus | human-eval-plus | humanevalplus)
            echo "humanevalplus"
            ;;
        strategyqa | strategy_qa | strategy-QA)
            echo "strategyQA"
            ;;
        numinamath | numinamath-cot | numinamath_cot)
            echo "numinamath"
            ;;
        *)
            echo "$raw"
            ;;
    esac
}

echo "=========================================="
echo "Eval Suite"
echo "  GPUs:       $GPU_IDS"
echo "  Checkpoint: ${CHECKPOINT_PATH_ARG:-<BASE_MODEL/config>}"
echo "  Datasets:   ${DATASET_LIST[*]}"
echo "  Summary:    $SUMMARY_FILE"
echo "=========================================="

for dataset in "${DATASET_LIST[@]}"; do
    raw_dataset="$(echo "$dataset" | xargs)"
    [ -n "$raw_dataset" ] || continue
    dataset="$(normalize_dataset "$raw_dataset")"
    eval_dir="${SUMMARY_DIR}/${dataset}"

    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "Dataset: $dataset"
    if [ "$dataset" != "$raw_dataset" ]; then
        echo "Alias: $raw_dataset -> $dataset"
    fi
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    cmd=(bash "$EVAL_SCRIPT" "$dataset" "$GPU_IDS")
    if [ -n "$CHECKPOINT_PATH_ARG" ]; then
        cmd+=("$CHECKPOINT_PATH_ARG")
    fi
    cmd+=("$@")

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[DRY_RUN] TARGET_EVAL_DIR=%q' "$eval_dir"
        printf ' %q' "${cmd[@]}"
        printf '\n'
        printf '%s\t%s\t%s\t%s\n' "$dataset" "dry_run" "" "" >> "$SUMMARY_FILE"
        continue
    fi

    status="ok"
    if ! TARGET_EVAL_DIR="$eval_dir" "${cmd[@]}"; then
        status="failed"
        if [ "${STOP_ON_ERROR:-0}" = "1" ]; then
            printf '%s\t%s\t%s\t%s\n' "$dataset" "$status" "" "" >> "$SUMMARY_FILE"
            exit 1
        fi
    fi

    metric_tail=""
    if [ -n "$eval_dir" ] && [ -f "${eval_dir}/logs/evaluation.log" ]; then
        metric_tail="$(grep -E "(test_score|pass@|accuracy|reward)" "${eval_dir}/logs/evaluation.log" 2>/dev/null | tail -5 | tr '\n' ';' || true)"
    fi

    printf '%s\t%s\t%s\t%s\n' "$dataset" "$status" "$eval_dir" "$metric_tail" >> "$SUMMARY_FILE"
done

echo ""
echo "=========================================="
echo "Eval suite finished"
echo "Summary: $SUMMARY_FILE"
echo "=========================================="
cat "$SUMMARY_FILE"
