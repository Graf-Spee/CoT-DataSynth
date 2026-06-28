#!/bin/bash
# DataObs 专用评估脚本
# 用法: bash eval_dataobs.sh <checkpoint_path> <base_model> <dataset_name> <output_path> <gpu_id>

if [ "$#" -lt 5 ]; then
    echo "Usage: bash $0 <checkpoint_path> <base_model> <dataset_name> <output_path> <gpu_id>"
    echo "Example: bash $0 /data/hrh/COT/GSM8K/training/split_0/global_step_2 /data/pretrain_models/Qwen2.5-0.5B-Instruct /data/open_datasets/GSM8K/main/test-00000-of-00001.parquet /data/hrh/COT/GSM8K/eval/split_0 0"
    exit 1
fi

CHECKPOINT_PATH="$1"
BASE_MODEL="$2"
DATA_NAME="$3"
EVAL_OUTPUT_DIR="$4"
GPU_ID="${5:-0}"
CALC_MAJ=1
IS_BFCL=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAOBS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_REPO_DIR="$(cd "${DATAOBS_DIR}/.." && pwd)"
REPO_DIR="$SCRIPT_REPO_DIR"
PROMPT_TEMPLATE_METHOD="${PROMPT_TEMPLATE_METHOD:-zeroshot}"

# =========================== Load User Configs =========================
# Find & Load Config File
# Precedence: 1. Env CONFIG_FILE  2. ../config/bash_config.env

# CONFIG_DIR: 使用绝对路径或从环境变量读取
if [ -z "$CONFIG_DIR" ]; then
    CONFIG_DIR="${REPO_DIR}/config"
fi

CONFIG_FILE="${CONFIG_FILE:-${CONFIG_DIR}/bash_config.env}"

if [ -f "$CONFIG_FILE" ]; then
    echo "[INFO] Loading config from: $CONFIG_FILE"
    source "$CONFIG_FILE"
else
    echo "[WARNING] Config file not found. Using default values."
fi

# bash_config.env may contain a stale REPO_DIR from another checkout. Evaluation
# paths must follow the script location so reward functions are loaded locally.
REPO_DIR="$SCRIPT_REPO_DIR"

# fallback
PROJECT_DIR="${PROJECT_DIR:-$HOME/CoT-Data-verl}"
STORE_DIR="${STORE_DIR:-/data/hjw}"
# =======================================================================

shopt -s nocasematch    # Enable caseless match
case $DATA_NAME in
    "ai2_arc" | "ai2-arc" | "arc-challenge")
        REWARD_FUNCTION_PATH="${REPO_DIR}/verl/utils/reward_score/multiple_choice.py"
        EVAL_DATA="/data/open_datasets/ai2_arc/ARC-Challenge/test-00000-of-00001.parquet"
        echo "[INFO] Load config for ARC-Challenge: Success!"
        ;;
    "aqua_rat" | "aqua-rat")
        REWARD_FUNCTION_PATH="${REPO_DIR}/verl/utils/reward_score/multiple_choice.py"
        EVAL_DATA="/data/open_datasets/aqua_rat/raw/test-00000-of-00001.parquet"
        echo "[INFO] Load config for AQuA-RAT: Success!"
        ;;
    "commonsenseQA")
        REWARD_FUNCTION_PATH="${REPO_DIR}/verl/utils/reward_score/multiple_choice.py"
        EVAL_DATA="/data/open_datasets/CommonsenseQA/data/validation-00000-of-00001.parquet"
        echo "[INFO] Load config for CommonsenseQA: Success!"
        ;;
    "gsm8k")
        REWARD_FUNCTION_PATH="${REPO_DIR}/verl/utils/reward_score/gsm8k.py"
        EVAL_DATA="/data/open_datasets/GSM8K/main/test-00000-of-00001.parquet"
        echo "[INFO] Load config for GSM8K: Success!"
        ;;
    "humaneval")
        REWARD_FUNCTION_PATH="${REPO_DIR}/verl/utils/reward_score/mbpp.py"
        EVAL_DATA="/data/open_datasets/humaneval/openai_humaneval/test-00000-of-00001.parquet"
        CALC_MAJ=0
        echo "[INFO] Load config for HumanEval: Success!"
        ;;
    "humanevalplus" | "human-eval-plus")
        REWARD_FUNCTION_PATH="${REPO_DIR}/verl/utils/reward_score/mbpp.py"
        EVAL_DATA="/data/open_datasets/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet"
        CALC_MAJ=0
        echo "[INFO] Load config for HumanEvalPlus: Success!"
        ;;
    "math")
        REWARD_FUNCTION_PATH="${REPO_DIR}/verl/utils/reward_score/math_verify.py"
        EVAL_DATA="/data/open_datasets/MATH/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet"
        echo "[INFO] Load config for MATH: Success!"
        ;;
    "math-500" | "math-cot")
        REWARD_FUNCTION_PATH="${REPO_DIR}/verl/utils/reward_score/math_verify.py"
        EVAL_DATA="/data/open_datasets/MATH-500/test.parquet"
        echo "[INFO] Load config for MATH-500: Success!"
        ;;
    "mbpp")
        REWARD_FUNCTION_PATH="${REPO_DIR}/verl/utils/reward_score/mbpp.py"
        EVAL_DATA="/data/open_datasets/mbpp/sanitized/test-00000-of-00001.parquet"
        CALC_MAJ=0
        echo "[INFO] Load config for MBPP: Success!"
        ;;
    "mbppplus" | "mbpp-plus")
        REWARD_FUNCTION_PATH="${REPO_DIR}/verl/utils/reward_score/mbpp.py"
        EVAL_DATA="/data/open_datasets/mbppplus/data/test-00000-of-00001-d5781c9c51e02795.parquet"
        CALC_MAJ=0
        echo "[INFO] Load config for MBPPPlus: Success!"
        ;;
    "numinamath" | "numinamath-CoT")
        REWARD_FUNCTION_PATH="${REPO_DIR}/verl/utils/reward_score/math_verify.py"
        EVAL_DATA="/data/open_datasets/NuminaMath-CoT/data/test-00000-of-00001.parquet"
        echo "[INFO] Load config for NuminaMath-CoT: Success!"
        ;;
    "strategyQA")
        REWARD_FUNCTION_PATH="${REPO_DIR}/verl/utils/reward_score/truefalse.py"
        EVAL_DATA="/data/open_datasets/StrategyQA/data/test-00000-of-00001-bae602f3ee37f4ca.parquet"
        echo "[INFO] Load config for StrategyQA: Success!"
        ;;
    "bfcl")
        IS_BFCL=1
        echo "[INFO] Load config for BFCL: Success!"
        ;;
    *)
        # Default: unknown dataset
        echo "[ERROR] Unsupported dataset $DATA_NAME."
        echo "Supported datasets: ai2_arc, aqua_rat, commonsenseQA, gsm8k, humaneval, humanevalplus, math, math-500, mbpp, mbppplus, numinamath, strategyQA, bfcl"
        exit 1
        ;;
esac
shopt -u nocasematch    # Disable caseless match

# 配置
export CUDA_VISIBLE_DEVICES=$GPU_ID
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

n_gpus_per_node=$(echo $GPU_ID | tr ',' ' ' | wc -w)
tp_size=$n_gpus_per_node

echo "=========================================="
echo "DataObs Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Base Model: $BASE_MODEL"
echo "GPU: $GPU_ID"
echo "Config dir: $CONFIG_DIR"
echo "Eval data: $EVAL_DATA"
echo "=========================================="

# 检查 checkpoint 是否存在
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

# 检查 config 目录是否存在
if [ ! -d "$CONFIG_DIR" ]; then
    echo "[ERROR] Config directory not found: $CONFIG_DIR"
    exit 1
fi

# 如果 checkpoint 中有 adapter_model.safetensors，说明是 LoRA 模型，需要 merge
if [ -f "$CHECKPOINT_PATH/adapter_model.safetensors" ]; then
    MERGED_MODEL_PATH="${EVAL_OUTPUT_DIR}/merged_model"
    mkdir -p ${MERGED_MODEL_PATH}

    if [ -f "${MERGED_MODEL_PATH}/config.json" ] && (ls "${MERGED_MODEL_PATH}"/*.safetensors >/dev/null 2>&1 || [ -f "${MERGED_MODEL_PATH}/pytorch_model.bin" ] || [ -f "${MERGED_MODEL_PATH}/model.safetensors.index.json" ]); then
        MODEL_PATH=${MERGED_MODEL_PATH}
        echo "[INFO] Reusing existing merged model: $MODEL_PATH"
    else
        echo "[INFO] Detected LoRA adapter, merging with base model: $BASE_MODEL"

        # 调用 merge 脚本
        MERGE_SCRIPT="${DATAOBS_DIR}/lib/model_ops/merge_lora_qwen.py"

        python3 ${MERGE_SCRIPT} \
            --base ${BASE_MODEL} \
            --lora ${CHECKPOINT_PATH} \
            --tokenizer ${CHECKPOINT_PATH} \
            --output ${MERGED_MODEL_PATH}

        if [ $? -ne 0 ]; then
            echo "[ERROR] LoRA merge failed!"
            exit 1
        fi

        MODEL_PATH=${MERGED_MODEL_PATH}
        echo "[INFO] LoRA merge completed, using merged model: $MODEL_PATH"
    fi
else
    echo "[INFO] Using checkpoint as full model: $CHECKPOINT_PATH"
    MODEL_PATH="$CHECKPOINT_PATH"
fi

# BFCL 评测入口：直接调用专用脚本，绕过 verl main_generation/main_eval
if [ "$IS_BFCL" -eq 1 ]; then
    echo "[INFO] BFCL mode detected, bypassing verl generation/evaluation."
    BFCL_EVAL_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_bfcl_dataobs.py"
    if [ ! -f "$BFCL_EVAL_SCRIPT" ]; then
        echo "[ERROR] BFCL evaluation script not found: $BFCL_EVAL_SCRIPT"
        exit 1
    fi
    python3 "$BFCL_EVAL_SCRIPT" \
        "$MODEL_PATH" \
        "$BASE_MODEL" \
        "$DATA_NAME" \
        "$EVAL_OUTPUT_DIR" \
        "$GPU_ID"
    exit $?
fi

# Stage 1: Generation
mkdir -p ${EVAL_OUTPUT_DIR}/{generated,logs}
GENERATION_OUTPUT="${EVAL_OUTPUT_DIR}/generated/responses.parquet"
PREPARED_EVAL_DATA="${EVAL_OUTPUT_DIR}/generated/prepared_eval.parquet"

echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Stage 1: Generation"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

python3 "${DATAOBS_DIR}/lib/evaluation/prepare_eval_data.py" \
    --dataset "$DATA_NAME" \
    --input "$EVAL_DATA" \
    --output "$PREPARED_EVAL_DATA" \
    --method "$PROMPT_TEMPLATE_METHOD"

python3 -m verl.trainer.main_generation \
    --config-path=${CONFIG_DIR} \
    --config-name=generation \
    model.path=${MODEL_PATH} \
    model.no_chat=false \
    data.path=${PREPARED_EVAL_DATA} \
    data.output_path=${GENERATION_OUTPUT} \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.batch_size=32 \
    rollout.temperature=0.6 \
    rollout.seed=42 \
    rollout.prompt_length=512 \
    rollout.response_length=1024 \
    rollout.gpu_memory_utilization=0.8 \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=1 \
    trainer.device=cuda \
    ray_init.num_cpus=48 \
    2>&1 | tee ${EVAL_OUTPUT_DIR}/logs/generation.log

if [ ${PIPESTATUS[0]} -ne 0 ] || [ ! -f "${GENERATION_OUTPUT}" ]; then
    echo "[ERROR] Generation Failed!"
    exit 1
fi

echo "[INFO] Generation Done!"

# Stage 2: Evaluation
EVALUATION_OUTPUT="${EVAL_OUTPUT_DIR}/generated/responses_labeled.json"

echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Stage 2: Evaluation"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

EVAL_ARGS="\
data.path=${GENERATION_OUTPUT} \
data.output_path=${EVALUATION_OUTPUT} \
data.response_key=responses \
data.data_source_key=data_source \
data.reward_model_key=reward_model \
custom_reward_function.path=${REWARD_FUNCTION_PATH} \
ray_init.num_cpus=48"

if [[ "$CALC_MAJ" == 0 ]]; then
    EVAL_ARGS="${EVAL_ARGS} custom_reward_function.calc_maj=false"
fi

python3 -m verl.trainer.main_eval \
    --config-path=${CONFIG_DIR} \
    --config-name=evaluation \
    ${EVAL_ARGS} \
    2>&1 | tee ${EVAL_OUTPUT_DIR}/logs/evaluation.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] Evaluation Failed!"
    exit 1
fi

echo "[INFO] Evaluation Done!"

# 提取准确率
echo ""
echo "=========================================="
echo "Evaluation Results:"
echo "=========================================="
grep -E "(test_score|pass@|accuracy|reward)" ${EVAL_OUTPUT_DIR}/logs/evaluation.log | tail -10 || echo "See ${EVAL_OUTPUT_DIR}/logs/evaluation.log for details"

echo ""
echo "Output directory: ${EVAL_OUTPUT_DIR}"
