#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_teacher_size_common.sh"

BASE_MODEL="${BASE_MODEL:-/data/pretrain_models/Qwen3.5-4B}"
FAMILY="qwen3_5"

# Qwen3.5 teacher-size curve from local /data/pretrain_models.
# To actually run: DRY_RUN="" bash DataObs/experiment_instruction/exp1100_teacher_size/commands_qwen3_5.sh
run_teacher "${FAMILY}" qwen3_5_0_8b /data/pretrain_models/Qwen3.5-0.8B "${GPU_IDS}" 1 "${TEACHER_BATCH_SIZE}"
run_teacher "${FAMILY}" qwen3_5_2b /data/pretrain_models/Qwen3.5-2B "${GPU_IDS}" 1 "${TEACHER_BATCH_SIZE}"
run_teacher "${FAMILY}" qwen3_5_4b /data/pretrain_models/Qwen3.5-4B "${GPU_IDS}" 1 "${TEACHER_BATCH_SIZE}"
run_teacher "${FAMILY}" qwen3_5_9b /data/pretrain_models/Qwen3.5-9B "${GPU_IDS}" 1 32
run_teacher "${FAMILY}" qwen3_5_27b /data/pretrain_models/Qwen3.5-27B "${BIG_GPU_IDS}" 2 16
run_teacher "${FAMILY}" qwen3_5_35b_a3b /data/pretrain_models/Qwen3.5-35B-A3B "${XL_GPU_IDS}" 4 8
