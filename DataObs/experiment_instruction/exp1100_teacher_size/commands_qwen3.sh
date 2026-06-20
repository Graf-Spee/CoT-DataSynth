#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_teacher_size_common.sh"

BASE_MODEL="${BASE_MODEL:-/data/pretrain_models/Qwen3-4B}"
FAMILY="qwen3"

# Qwen3 teacher-size curve from local /data/pretrain_models.
# Excludes embedding/reranker/VL/image assets and the 235B GPTQ checkpoint by default.
# To actually run: DRY_RUN="" bash DataObs/experiment_instruction/exp1100_teacher_size/commands_qwen3.sh
run_teacher "${FAMILY}" qwen3_1_7b /data/pretrain_models/Qwen3-1.7B "${GPU_IDS}" 1 "${TEACHER_BATCH_SIZE}"
run_teacher "${FAMILY}" qwen3_4b /data/pretrain_models/Qwen3-4B "${GPU_IDS}" 1 "${TEACHER_BATCH_SIZE}"
run_teacher "${FAMILY}" qwen3_8b /data/pretrain_models/Qwen3-8B "${GPU_IDS}" 1 32
run_teacher "${FAMILY}" qwen3_14b /data/pretrain_models/Qwen3-14B "${BIG_GPU_IDS}" 2 16
run_teacher "${FAMILY}" qwen3_30b_a3b /data/pretrain_models/Qwen3-30B-A3B "${BIG_GPU_IDS}" 2 16
run_teacher "${FAMILY}" qwen3_32b /data/pretrain_models/Qwen3-32B "${XL_GPU_IDS}" 4 8
