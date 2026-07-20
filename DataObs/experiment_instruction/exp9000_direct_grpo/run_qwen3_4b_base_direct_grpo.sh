#!/usr/bin/env bash
set -euo pipefail

cd /home/hrh/CoT-DataSynth

EXP_ID="${EXP_ID:-exp9000_gsm8k_qwen3_4b_base_direct_grpo}"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
EXP_DIR="${EXP_DIR:-/data/hrh/COT/experiments/${EXP_ID}}"
SOURCE_EXP="${SOURCE_EXP:-/data/hrh/COT/experiments/exp1100_gsm8k_qwen3_teacher_qwen3_4b_t0.7_n4}"
BASE_MODEL="${BASE_MODEL:-/data/pretrain_models/Qwen3-4B}"
TRAIN_DATA="${TRAIN_DATA:-${SOURCE_EXP}/rl_data/train_prepared.parquet}"
VAL_DATA="${VAL_DATA:-${SOURCE_EXP}/rl_data/val_prepared.parquet}"
TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-3}"
EVAL_GPU_IDS="${EVAL_GPU_IDS:-3}"
EVAL_TP_SIZE="${EVAL_TP_SIZE:-1}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrh/anaconda3/envs/verl-torch280/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

GRPO_DIR="${GRPO_DIR:-${EXP_DIR}/grpo}"
EVAL_DIR="${EVAL_DIR:-${EXP_DIR}/eval/grpo}"
LOG_DIR="${LOG_DIR:-${EXP_DIR}/logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/direct_grpo_run.log}"
PID_FILE="${PID_FILE:-${EXP_DIR}/direct_grpo.pid}"
COMMANDS_JSONL="${COMMANDS_JSONL:-${EXP_DIR}/commands.jsonl}"
SESSION_NAME="${SESSION_NAME:-${EXP_ID}}"

mkdir -p "$GRPO_DIR" "$EVAL_DIR" "$LOG_DIR"

if [[ "${1:-run}" == "start" ]]; then
    if command -v tmux >/dev/null 2>&1; then
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo "[ERROR] Existing tmux session appears active: $SESSION_NAME"
            exit 1
        fi
        tmux new-session -d -s "$SESSION_NAME" "cd /home/hrh/CoT-DataSynth && exec \"$SCRIPT_PATH\" run >>\"$RUN_LOG\" 2>&1"
        tmux list-panes -t "$SESSION_NAME" -F "#{pane_pid}" >"$PID_FILE"
        echo "[INFO] Started ${EXP_ID} in tmux session: $SESSION_NAME"
        echo "[INFO] Pane pid: $(cat "$PID_FILE")"
        echo "[INFO] Log: $RUN_LOG"
        echo "[INFO] Output: $EXP_DIR"
        exit 0
    fi

    if [[ -f "$PID_FILE" ]]; then
        old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
            echo "[ERROR] Existing run appears active: pid=$old_pid"
            echo "[ERROR] PID file: $PID_FILE"
            exit 1
        fi
    fi
    nohup "$0" run >"$RUN_LOG" 2>&1 &
    echo "$!" >"$PID_FILE"
    echo "[INFO] Started ${EXP_ID}: pid=$(cat "$PID_FILE")"
    echo "[INFO] Log: $RUN_LOG"
    echo "[INFO] Output: $EXP_DIR"
    exit 0
fi

append_command_record() {
    local stage="$1"
    shift
    "$PYTHON_BIN" - "$COMMANDS_JSONL" "$stage" "$@" <<'PY'
import json
import sys
import time

path = sys.argv[1]
stage = sys.argv[2]
cmd = sys.argv[3:]
record = {
    "stage": stage,
    "time": time.strftime("%Y%m%d-%H%M%S"),
    "cmd": cmd,
    "cmd_str": " ".join(cmd),
    "manual_launcher": "DataObs/experiment_instruction/exp9000_direct_grpo/run_qwen3_4b_base_direct_grpo.sh",
}
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
PY
}

write_result_record() {
    local stage="$1"
    local returncode="$2"
    local elapsed_sec="$3"
    "$PYTHON_BIN" - "$COMMANDS_JSONL" "$stage" "$returncode" "$elapsed_sec" <<'PY'
import json
import sys
import time

path = sys.argv[1]
stage = sys.argv[2]
returncode = int(sys.argv[3])
elapsed_sec = float(sys.argv[4])
record = {
    "stage": stage,
    "returncode": returncode,
    "elapsed_sec": elapsed_sec,
    "time": time.strftime("%Y%m%d-%H%M%S"),
}
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
PY
}

elapsed_since() {
    local start_ts="$1"
    "$PYTHON_BIN" - "$start_ts" <<'PY'
import sys
import time
print(f"{time.time() - float(sys.argv[1]):.3f}")
PY
}

cat >"${EXP_DIR}/manifest.json" <<JSON
{
  "experiment_id": "${EXP_ID}",
  "purpose": "Qwen3-4B base direct GRPO control for SFT+GRPO comparison",
  "base_model": "${BASE_MODEL}",
  "source_experiment_for_rl_data": "${SOURCE_EXP}",
  "train_data": "${TRAIN_DATA}",
  "val_data": "${VAL_DATA}",
  "grpo_dir": "${GRPO_DIR}",
  "eval_dir": "${EVAL_DIR}",
  "train_gpu_ids": "${TRAIN_GPU_IDS}",
  "eval_gpu_ids": "${EVAL_GPU_IDS}",
  "eval_tensor_parallel_size": "${EVAL_TP_SIZE}"
}
JSON

export MODEL_NAME="${MODEL_NAME:-${EXP_ID}-grpo}"
export DATA_NAME="${DATA_NAME:-gsm8k}"

# Matched to the successful exp1100 Qwen3-4B SFT->GRPO run on 2026-06-30.
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-2048}"
export LR="${LR:-5e-7}"
export KL_COEF="${KL_COEF:-0.001}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TEST_FREQ="${TEST_FREQ:-5}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-/home/hrh/CoT-DataSynth/wandb_logs}"

grpo_cmd=(
    bash /home/hrh/CoT-DataSynth/DataObs/lib/training/grpo_from_sft.sh
    "$BASE_MODEL"
    "$TRAIN_DATA"
    "$VAL_DATA"
    "$TRAIN_GPU_IDS"
    "$GRPO_DIR"
    "$BASE_MODEL"
)

append_command_record grpo "${grpo_cmd[@]}"
start_ts="$("$PYTHON_BIN" - <<'PY'
import time
print(time.time())
PY
)"
set +e
"${grpo_cmd[@]}" 2>&1 | tee "${LOG_DIR}/grpo.log"
rc=${PIPESTATUS[0]}
set -e
write_result_record grpo "$rc" "$(elapsed_since "$start_ts")"
if [[ "$rc" -ne 0 ]]; then
    exit "$rc"
fi

if [[ -f "${GRPO_DIR}/latest_checkpointed_iteration.txt" ]]; then
    step_id="$(tr -d '[:space:]' <"${GRPO_DIR}/latest_checkpointed_iteration.txt")"
else
    step_id="$(find "$GRPO_DIR" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sed 's|.*/global_step_||' | sort -n | tail -n 1)"
fi

if [[ -z "$step_id" ]]; then
    echo "[ERROR] Could not find latest GRPO checkpoint under: $GRPO_DIR"
    exit 1
fi

actor_dir="${GRPO_DIR}/global_step_${step_id}/actor"
actor_lora_dir="${actor_dir}/lora_adapter"
actor_hf_dir="${actor_dir}/huggingface"
merged_model_dir="${actor_dir}/merged_model"

merge_cmd=(
    "$PYTHON_BIN" /home/hrh/CoT-DataSynth/DataObs/lib/model_ops/merge_lora_qwen.py
    --base "$BASE_MODEL"
    --lora "$actor_lora_dir"
    --tokenizer "$actor_hf_dir"
    --output "$merged_model_dir"
)

append_command_record grpo_merge "${merge_cmd[@]}"
start_ts="$("$PYTHON_BIN" - <<'PY'
import time
print(time.time())
PY
)"
set +e
CUDA_VISIBLE_DEVICES="$EVAL_GPU_IDS" "${merge_cmd[@]}" 2>&1 | tee "${LOG_DIR}/grpo_merge.log"
rc=${PIPESTATUS[0]}
set -e
write_result_record grpo_merge "$rc" "$(elapsed_since "$start_ts")"
if [[ "$rc" -ne 0 ]]; then
    exit "$rc"
fi

eval_cmd=(
    "$PYTHON_BIN" /home/hrh/CoT-DataSynth/DataObs/lib/evaluation/eval_vllm.py
    --model-path "$merged_model_dir"
    --base-model "$BASE_MODEL"
    --dataset gsm8k
    --output-dir "$EVAL_DIR"
    --gpu-ids "$EVAL_GPU_IDS"
    --num-samples 1
    --temperature 0.0
    --top-p 1.0
    --max-response-len 16384
    --batch-size 8
    --gpu-memory-utilization 0.8
    --tensor-parallel-size "$EVAL_TP_SIZE"
    --prompt-template-method zeroshot
    --seed 42
)

append_command_record grpo_eval "${eval_cmd[@]}"
start_ts="$("$PYTHON_BIN" - <<'PY'
import time
print(time.time())
PY
)"
set +e
"${eval_cmd[@]}" 2>&1 | tee "${LOG_DIR}/grpo_eval.log"
rc=${PIPESTATUS[0]}
set -e
write_result_record grpo_eval "$rc" "$(elapsed_since "$start_ts")"
exit "$rc"
