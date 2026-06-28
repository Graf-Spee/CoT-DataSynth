#!/bin/bash
# Run GRPO from an SFT checkpoint.

set -euo pipefail

# =========================== Load User Configs =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAOBS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_REPO_DIR="$(cd "${DATAOBS_DIR}/.." && pwd)"
REPO_DIR="$SCRIPT_REPO_DIR"

if [ -z "${CONFIG_DIR:-}" ]; then
    CONFIG_DIR="${REPO_DIR}/config"
fi

CONFIG_FILE="${CONFIG_FILE:-${CONFIG_DIR}/bash_config.env}"

if [ -f "$CONFIG_FILE" ]; then
    echo "[INFO] Loading config from: $CONFIG_FILE"
    source "$CONFIG_FILE"
else
    echo "[WARNING] Config file not found. Using default values."
fi

PROJECT_DIR="${PROJECT_DIR:-$HOME/CoT-Data-verl}"
STORE_DIR="${STORE_DIR:-/data/hjw}"

REWARD_ROUTER="${REWARD_ROUTER:-${SCRIPT_REPO_DIR}/verl/utils/reward_score/reward_fn_router.py}"

usage() {
    cat <<EOF
Usage:
  bash $0 <sft_checkpoint_or_split_dir> <train_parquet> <val_parquet> <gpu_ids> [output_dir] [base_model] [other_configs...]

Examples:
  # Use a DataObs split SFT output directory. The script will pick checkpoint-last or the latest global_step_*.
  bash $0 /data/hrh/COT/GSM8K/training/split_0 \\
    /data/open_datasets/GSM8K/main/train-00000-of-00001.parquet \\
    /data/open_datasets/GSM8K/main/test-00000-of-00001.parquet \\
    0,1,2,3 \\
    /data/hrh/COT/GSM8K/grpo/split_0 \\
    /data/pretrain_models/Qwen2.5-0.5B-Instruct

  # Pass the exact merged/full HF checkpoint.
  bash $0 /path/to/sft/checkpoint-last /path/to/train.parquet /path/to/test.parquet 0

Notes:
  - RL parquet should be verl RL format by default: prompt, data_source, reward_model.
  - If the SFT checkpoint is a LoRA adapter, pass base_model or set BASE_MODEL.
  - Use DRY_RUN=1 to print the command without launching training.
EOF
}

if [ "$#" -lt 4 ]; then
    usage
    exit 1
fi

SFT_INPUT="$1"
TRAIN_DATA="$2"
VAL_DATA="$3"
GPU_IDS="$4"
shift 4

DATA_NAME="${DATA_NAME:-$(basename "$(dirname "$TRAIN_DATA")")}"
MODEL_NAME="${MODEL_NAME:-$(basename "$SFT_INPUT")}"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"

OUTPUT_DIR="${GRPO_OUTPUT_DIR:-${STORE_DIR}/outputs/${MODEL_NAME}--${DATA_NAME}--grpo}"
if [ "$#" -gt 0 ] && [[ "$1" != *=* ]]; then
    OUTPUT_DIR="$1"
    shift 1
fi

BASE_MODEL="${BASE_MODEL:-}"
if [ "$#" -gt 0 ] && [[ "$1" != *=* ]]; then
    BASE_MODEL="$1"
    shift 1
fi

is_hf_or_lora_model_dir() {
    local path="$1"
    [ -d "$path" ] || return 1

    if [ -f "$path/adapter_model.safetensors" ]; then
        return 0
    fi

    if [ -f "$path/config.json" ] && \
       { [ -f "$path/model.safetensors" ] || [ -f "$path/pytorch_model.bin" ] || [ -f "$path/model.safetensors.index.json" ] || compgen -G "$path/*.safetensors" >/dev/null; }; then
        return 0
    fi

    return 1
}

resolve_sft_path() {
    local input="$1"

    if [ ! -d "$input" ]; then
        echo "[ERROR] SFT path not found: $input" >&2
        return 1
    fi

    if is_hf_or_lora_model_dir "$input"; then
        echo "$input"
        return 0
    fi

    if [ -d "$input/checkpoint-last" ] && is_hf_or_lora_model_dir "$input/checkpoint-last"; then
        echo "$input/checkpoint-last"
        return 0
    fi

    local latest=""
    local latest_step=-1
    local dir
    local name
    local step
    shopt -s nullglob
    for dir in "$input"/global_step_*; do
        [ -d "$dir" ] || continue
        is_hf_or_lora_model_dir "$dir" || continue
        name="$(basename "$dir")"
        step="${name#global_step_}"
        if [[ "$step" =~ ^[0-9]+$ ]] && [ "$step" -gt "$latest_step" ]; then
            latest="$dir"
            latest_step="$step"
        fi
    done
    shopt -u nullglob

    if [ -n "$latest" ]; then
        echo "$latest"
        return 0
    fi

    echo "[ERROR] Could not find an HF/LoRA checkpoint under: $input" >&2
    echo "[ERROR] Expected direct model files, checkpoint-last, or global_step_*." >&2
    return 1
}

STRICT_PATH_CHECK="${STRICT_PATH_CHECK:-1}"
if [ "$STRICT_PATH_CHECK" = "1" ]; then
    [ -f "$TRAIN_DATA" ] || { echo "[ERROR] Train parquet not found: $TRAIN_DATA"; exit 1; }
    [ -f "$VAL_DATA" ] || { echo "[ERROR] Val parquet not found: $VAL_DATA"; exit 1; }
    [ -d "$CONFIG_DIR" ] || { echo "[ERROR] Config dir not found: $CONFIG_DIR"; exit 1; }
    [ -f "$REWARD_ROUTER" ] || { echo "[ERROR] Reward router not found: $REWARD_ROUTER"; exit 1; }
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
N_GPUS_PER_NODE="$(echo "$GPU_IDS" | tr ',' ' ' | wc -w)"

SFT_MODEL_PATH="$(resolve_sft_path "$SFT_INPUT")"
MODEL_PATH="$SFT_MODEL_PATH"

if [ -f "$SFT_MODEL_PATH/adapter_model.safetensors" ]; then
    if [ -z "$BASE_MODEL" ]; then
        echo "[ERROR] Detected LoRA SFT checkpoint but base_model was not provided."
        echo "[ERROR] Pass base_model as the sixth argument or set BASE_MODEL=/path/to/base_model."
        exit 1
    fi

    MERGED_MODEL_PATH="${MERGED_MODEL_PATH:-${OUTPUT_DIR}/merged_sft_model}"
    if is_hf_or_lora_model_dir "$MERGED_MODEL_PATH" && [ ! -f "$MERGED_MODEL_PATH/adapter_model.safetensors" ]; then
        echo "[INFO] Reusing merged SFT model: $MERGED_MODEL_PATH"
    elif [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[DRY_RUN] Detected LoRA SFT checkpoint. Would merge with base model: $BASE_MODEL"
        echo "[DRY_RUN] Merged model path would be: $MERGED_MODEL_PATH"
    else
        echo "[INFO] Detected LoRA SFT checkpoint. Merging with base model: $BASE_MODEL"
        python3 "${DATAOBS_DIR}/lib/model_ops/merge_lora_qwen.py" \
            --base "$BASE_MODEL" \
            --lora "$SFT_MODEL_PATH" \
            --tokenizer "$SFT_MODEL_PATH" \
            --output "$MERGED_MODEL_PATH"
    fi
    MODEL_PATH="$MERGED_MODEL_PATH"
fi

TP_SIZE="${TP_SIZE:-1}"
if [ $((N_GPUS_PER_NODE % TP_SIZE)) -ne 0 ]; then
    echo "[WARNING] N_GPUS_PER_NODE($N_GPUS_PER_NODE) is not divisible by TP_SIZE($TP_SIZE). Falling back to TP_SIZE=1."
    TP_SIZE=1
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_DIR}/wandb_logs}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-exp-${MODEL_NAME}-${DATA_NAME}-grpo}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-32}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-64}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
LR="${LR:-1e-6}"
KL_COEF="${KL_COEF:-0.001}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
PROMPT_KEY="${PROMPT_KEY:-prompt}"
REWARD_FN_KEY="${REWARD_FN_KEY:-data_source}"
RL_LORA_RANK="${RL_LORA_RANK:-32}"
RL_LORA_ALPHA="${RL_LORA_ALPHA:-32}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-5}"
PROJECT_NAME="${PROJECT_NAME:-${DATA_NAME}_grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_NAME}--${DATA_NAME}--grpo--${RUN_STAMP}}"

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR"

echo "=========================================="
echo "GRPO From SFT Setup"
echo "  GPUs: $GPU_IDS (count: $N_GPUS_PER_NODE)"
echo "  SFT input: $SFT_INPUT"
echo "  Model path: $MODEL_PATH"
echo "  Train data: $TRAIN_DATA"
echo "  Val data: $VAL_DATA"
echo "  Output dir: $OUTPUT_DIR"
echo "  Prompt key: $PROMPT_KEY"
echo "  Reward fn key: $REWARD_FN_KEY"
echo "  Epochs: $TOTAL_EPOCHS"
echo "  Rollout n: $ROLLOUT_N"
echo "  vLLM max num seqs: $VLLM_MAX_NUM_SEQS"
echo "  vLLM GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo "=========================================="

cmd=(
    python3 -m verl.trainer.main_ppo
    --config-path="$CONFIG_DIR"
    --config-name=ppo_trainer
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_DATA"
    data.val_files="$VAL_DATA"
    data.train_batch_size="$TRAIN_BATCH_SIZE"
    data.max_prompt_length="$MAX_PROMPT_LEN"
    data.max_response_length="$MAX_RESPONSE_LEN"
    data.filter_overlong_prompts=True
    data.truncation=error
    data.prompt_key="$PROMPT_KEY"
    data.reward_fn_key="$REWARD_FN_KEY"
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.actor.optim.lr="$LR"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef="$KL_COEF"
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE"
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION"
    actor_rollout_ref.rollout.max_num_batched_tokens="$VLLM_MAX_NUM_BATCHED_TOKENS"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs="$VLLM_MAX_NUM_SEQS"
    actor_rollout_ref.rollout.n="$ROLLOUT_N"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.model.lora_rank="$RL_LORA_RANK"
    actor_rollout_ref.model.lora_alpha="$RL_LORA_ALPHA"
    actor_rollout_ref.model.target_modules=all-linear
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    trainer.critic_warmup=0
    'trainer.logger=["console","wandb"]'
    trainer.project_name="$PROJECT_NAME"
    trainer.experiment_name="$EXPERIMENT_NAME"
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE"
    trainer.nnodes=1
    trainer.save_freq="$SAVE_FREQ"
    trainer.test_freq="$TEST_FREQ"
    trainer.total_epochs="$TOTAL_EPOCHS"
    trainer.default_local_dir="$OUTPUT_DIR"
    custom_reward_function.path="$REWARD_ROUTER"
    custom_reward_function.name=compute_score_router
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '[DRY_RUN] Command:'
    printf ' %q' "${cmd[@]}" "$@"
    printf '\n'
    exit 0
fi

"${cmd[@]}" "$@"
