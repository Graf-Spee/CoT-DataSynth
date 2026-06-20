#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_teacher_size_common.sh"

BASE_MODEL="${BASE_MODEL:-/data/pretrain_models/Qwen2.5-3B-Instruct}"
FAMILY="qwen2_5"

# Qwen2.5 teacher-size curve from local /data/pretrain_models.
# To actually run: DRY_RUN="" bash DataObs/experiment_instruction/exp1100_teacher_size/commands_qwen2_5.sh
run_teacher "${FAMILY}" qwen2_5_0_5b /data/pretrain_models/Qwen2.5-0.5B-Instruct "${GPU_IDS}" 1 "${TEACHER_BATCH_SIZE}"
run_teacher "${FAMILY}" qwen2_5_3b /data/pretrain_models/Qwen2.5-3B-Instruct "${GPU_IDS}" 1 "${TEACHER_BATCH_SIZE}"
run_teacher "${FAMILY}" qwen2_5_7b /data/pretrain_models/Qwen2.5-7B-Instruct "${GPU_IDS}" 1 "${TEACHER_BATCH_SIZE}"
run_teacher "${FAMILY}" qwen2_5_32b /data/pretrain_models/Qwen2.5-32B-Instruct "${BIG_GPU_IDS}" 2 16
run_teacher "${FAMILY}" qwen2_5_72b /data/pretrain_models/Qwen2.5-72B-Instruct "${XL_GPU_IDS}" 4 8
