#!/bin/bash
# SFT training script for DataObs

# =========================== Load User Configs =========================
# Find & Load Config File
# Precedence: 1. Env CONFIG_FILE  2. ../config/bash_config.env

# CONFIG_DIR: 使用绝对路径或从环境变量读取
if [ -z "$CONFIG_DIR" ]; then
    CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config"
fi

CONFIG_FILE="${CONFIG_FILE:-${CONFIG_DIR}/bash_config.env}"

if [ -f "$CONFIG_FILE" ]; then
    echo "[INFO] Loading config from: $CONFIG_FILE"
    source "$CONFIG_FILE"
else
    echo "[WARNING] Config file not found. Using default values."
fi

# fallback
PROJECT_DIR="${PROJECT_DIR:-$HOME/CoT-Data-verl}"
STORE_DIR="${STORE_DIR:-/data/hjw}"
# =======================================================================


# ============================ Configurations ===========================
MODEL_NAME="Qwen2.5-0.5B-Instruct"
DATA_NAME="gsm8k"
LOG_NAME="${MODEL_NAME}--${DATA_NAME}--sft--$(date +%Y%m%d-%H%M%S)"
# =======================================================================

if [ "$#" -lt 6 ]; then
    echo "Usage: bash $0 <base_model> <data_path> <val_data_path> <output_dir> <gpu_ids> <num_epochs> [other_configs...]"
    echo "Example: bash $0 <base_model> <data_path> <output_dir> 0 15"
    exit 1
fi

MODEL_ID="$1"
DATA_DIR="$2"
VAL_DATA_DIR="$3"
SAVE_PATH="$4"
gpu_ids=$5
num_epochs=$6

LOG_PATH="${SAVE_PATH}"

# Shift the arguments so $@ refers to the rest
shift 6

# Limit the visible GPUs
export CUDA_VISIBLE_DEVICES=$gpu_ids

# 自动计算GPU数量（将逗号转为空格后计数）
nproc_per_node=$(echo $gpu_ids | tr ',' ' ' | wc -w)

echo "=========================================="
echo "Using GPUs: $gpu_ids"
echo "Auto-calculated nproc_per_node: $nproc_per_node"
echo "=========================================="

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

export WANDB_DIR=${LOG_PATH}
export WANDB_RUN_ID="exp-${MODEL_NAME}-${DATA_NAME}-sft"

SFT_ARGS=" \
data.train_files=${DATA_DIR} \
data.val_files=${VAL_DATA_DIR} \
data.prompt_key=question \
data.response_key=answer \
data.micro_batch_size_per_gpu=2 \
data.max_length=16384 \
data.train_batch_size=8 \
optim.lr=5e-6 \
model.partial_pretrain=${MODEL_ID} \
model.trust_remote_code=true \
model.enable_gradient_checkpointing=false \
model.use_liger=true \
model.fsdp_config.model_dtype=bf16 \
trainer.default_local_dir=${SAVE_PATH} \
trainer.project_name=${DATA_NAME}-sft \
trainer.experiment_name=${LOG_NAME} \
trainer.total_epochs=$num_epochs \
trainer.logger=["console","wandb"] \
trainer.save_freq=10000 \
trainer.test_freq=10000 \
trainer.default_hdfs_dir=null"

SFT_ARGS="${SFT_ARGS} \
ulysses_sequence_parallel_size=2 \
use_remove_padding=true "

# SFT_ARGS="${SFT_ARGS} \
# model.fsdp_config.cpu_offload=true \
# model.fsdp_config.offload_params=true \
# model.strategy='fsdp'"

# SFT_ARGS="${SFT_ARGS} \
# data.prompt_dict_keys=['content'] \
# +data.response_dict_keys=['content'] "

SFT_ARGS="${SFT_ARGS} \
model.lora_rank=8 \
model.lora_alpha=16 "

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
     --config-path=${CONFIG_DIR} \
     --config-name=sft_trainer \
     ${SFT_ARGS} $@
