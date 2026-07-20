#!/usr/bin/env bash
# Rename eval outputs for the experiments selected for vLLM eval reruns.
#
# Default mode is dry-run. Pass --apply to actually rename:
#   bash scripts/rename_rerun_experiments_old.sh --apply

set -euo pipefail

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
elif [[ "${1:-}" == "--dry-run" || "${1:-}" == "" ]]; then
  APPLY=0
else
  echo "Usage: bash $0 [--dry-run|--apply]" >&2
  exit 2
fi

EXPERIMENTS=(
  "/data/hrh/COT/experiments/exp1100_gsm8k_qwen3_teacher_qwen3_1_7b_t0.7_n4"
  "/data/hrh/COT/experiments/exp1100_gsm8k_qwen3_teacher_qwen3_4b_t0.7_n4"
  "/data/hrh/COT/experiments/exp1100_gsm8k_qwen3_teacher_qwen3_8b_t0.7_n4"
  "/data/hrh/COT/experiments/exp1200_gsm8k_qwen3_teacher_type_deepseek_r1_qwen32b_t0.7_n4"
  "/data/hrh/COT/experiments/exp1200_gsm8k_qwen3_teacher_type_qwen3_14b_t0.7_n4"
  "/data/hrh/COT/experiments/exp1200_math500_qwen3_teacher_type_qwen3_14b_t0.7_n4"
  "/data/hrh/COT/experiments/exp3100_gsm8k_qwen3_settingA_sft_eq_rl_prompt"
  "/data/hrh/COT/experiments/exp3100_gsm8k_qwen3_settingB_sft_ne_rl_prompt"
  "/data/hrh/COT/experiments/exp5000_gsm8k_qwen3_difficulty_easy_distill"
  "/data/hrh/COT/experiments/exp5000_gsm8k_qwen3_difficulty_hard_distill"
  "/data/hrh/COT/experiments/exp5000_gsm8k_qwen3_difficulty_medium_distill"
  "/data/hrh/COT/experiments/exp7000_gsm8k_qwen3_teacher14b_pool_n8_same_prompt_n1"
  "/data/hrh/COT/experiments/exp7000_gsm8k_qwen3_teacher14b_pool_n8_same_prompt_n2"
  "/data/hrh/COT/experiments/exp7000_gsm8k_qwen3_teacher14b_pool_n8_same_prompt_n4"
  "/data/hrh/COT/experiments/exp7000_gsm8k_qwen3_teacher14b_pool_n8_same_prompt_n8"
  "/data/hrh/COT/experiments/pipeline_gsm8k_qwen3_4b_0620"
)

echo "Mode: $([[ "$APPLY" == "1" ]] && echo apply || echo dry-run)"
echo "Experiments: ${#EXPERIMENTS[@]}"

for exp_dir in "${EXPERIMENTS[@]}"; do
  if [[ ! -d "$exp_dir" ]]; then
    echo "[ERROR] experiment directory not found: $exp_dir" >&2
    exit 1
  fi

  renames=(
    "${exp_dir}/eval:${exp_dir}/eval_old"
    "${exp_dir}/manifest.json:${exp_dir}/manifest_old.json"
    "${exp_dir}/results.json:${exp_dir}/results_old.json"
  )

  for pair in "${renames[@]}"; do
    src="${pair%%:*}"
    dst="${pair#*:}"

    if [[ -e "$dst" && ! -e "$src" ]]; then
      echo "[SKIP] already renamed: $dst"
      continue
    fi

    if [[ ! -e "$src" ]]; then
      echo "[SKIP] source not found: $src"
      continue
    fi

    if [[ -e "$dst" ]]; then
      echo "[ERROR] destination already exists, refusing to overwrite: $dst" >&2
      exit 1
    fi

    if [[ "$APPLY" == "1" ]]; then
      echo "[MOVE] $src -> $dst"
      mv "$src" "$dst"
    else
      echo "[DRY-RUN] mv \"$src\" \"$dst\""
    fi
  done
done

echo "Done."
