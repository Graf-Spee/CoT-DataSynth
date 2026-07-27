# Exp9100 Cross-Domain SFT+RL Commands

These commands test SFT+RL gains across math, code, commonsense, and agent-style evaluation.

Common environment:

```bash
cd /home/hrh/CoT-DataSynth

export PATH=/home/hrh/anaconda3/envs/verl-torch280/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0

PYTHON=/home/hrh/anaconda3/envs/verl-torch280/bin/python
```

## Math: Train On MATH, Evaluate On AIME25

`aime25` is currently supported by `DataObs/lib/evaluation/eval_vllm.py` as an eval dataset, but not by the distill/GRPO training defaults. Train on `math`, then evaluate both SFT and GRPO checkpoints on `aime25`.

```bash
${PYTHON} DataObs/scripts/experiment_pipeline_vllm.py \
  --experiment-id exp9100_math_qwen3_4b_teacher14b_n4_sft_rl_for_aime25 \
  --dataset math \
  --base-model /data/pretrain_models/Qwen3-4B \
  --teacher-model /data/pretrain_models/Qwen3-14B \
  --output-dir /data/hrh/COT/experiments \
  --stages distill,metrics,sft,grpo \
  --gpu-ids 4 \
  --distill-method answer_augmentation \
  --teacher-num-samples 4 \
  --teacher-temperature 0.7 \
  --teacher-top-p 0.95 \
  --teacher-do-sample \
  --teacher-batch-size 8 \
  --teacher-max-new-tokens 1024 \
  --teacher-tensor-parallel-size 1 \
  --teacher-gpu-memory-utilization 0.65 \
  --sft-epochs 1 \
  --sft-arg data.train_batch_size=4 \
  --sft-arg data.micro_batch_size_per_gpu=2 \
  --sft-arg data.max_length=2048 \
  --grpo-env TOTAL_EPOCHS=1 \
  --grpo-env TRAIN_BATCH_SIZE=32 \
  --grpo-env PPO_MINI_BATCH_SIZE=32 \
  --grpo-env PPO_MICRO_BATCH_SIZE_PER_GPU=4 \
  --grpo-env LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8 \
  --grpo-env ROLLOUT_N=4 \
  --grpo-env MAX_PROMPT_LEN=512 \
  --grpo-env MAX_RESPONSE_LEN=2048 \
  --grpo-env LR=5e-7 \
  --grpo-env KL_COEF=0.001 \
  --grpo-env GPU_MEMORY_UTILIZATION=0.6 \
  --grpo-env VLLM_MAX_NUM_SEQS=64 \
  --grpo-env VLLM_MAX_NUM_BATCHED_TOKENS=4096 \
  --grpo-env SAVE_FREQ=10 \
  --grpo-env TEST_FREQ=5
```

```bash
${PYTHON} DataObs/scripts/experiment_pipeline_vllm.py \
  --experiment-id exp9100_aime25_eval_from_math_qwen3_4b_teacher14b_n4_sft_rl \
  --dataset aime25 \
  --base-model /data/pretrain_models/Qwen3-4B \
  --output-dir /data/hrh/COT/experiments \
  --stages sft_eval,grpo_eval \
  --gpu-ids 4 \
  --eval-gpu-ids 4 \
  --sft-output-dir /data/hrh/COT/experiments/exp9100_math_qwen3_4b_teacher14b_n4_sft_rl_for_aime25/sft \
  --grpo-output-dir /data/hrh/COT/experiments/exp9100_math_qwen3_4b_teacher14b_n4_sft_rl_for_aime25/grpo \
  --sft-eval-output-dir /data/hrh/COT/experiments/exp9100_math_qwen3_4b_teacher14b_n4_sft_rl_for_aime25/eval/aime25_sft \
  --grpo-eval-output-dir /data/hrh/COT/experiments/exp9100_math_qwen3_4b_teacher14b_n4_sft_rl_for_aime25/eval/aime25_grpo \
  --eval-temperature 0.0 \
  --eval-top-p 1.0 \
  --eval-max-response-len 16384 \
  --eval-batch-size 8 \
  --eval-gpu-memory-utilization 0.5 \
  --eval-tensor-parallel-size 1
```

## Code: MBPP+

```bash
${PYTHON} DataObs/scripts/experiment_pipeline_vllm.py \
  --experiment-id exp9100_mbppplus_qwen3_4b_teacher14b_n4_sft_rl \
  --dataset mbppplus \
  --base-model /data/pretrain_models/Qwen3-4B \
  --teacher-model /data/pretrain_models/Qwen3-14B \
  --output-dir /data/hrh/COT/experiments \
  --stages distill,metrics,sft,sft_eval,grpo,grpo_eval \
  --gpu-ids 4 \
  --distill-method answer_augmentation \
  --teacher-num-samples 4 \
  --teacher-temperature 0.7 \
  --teacher-top-p 0.95 \
  --teacher-do-sample \
  --teacher-batch-size 8 \
  --teacher-max-new-tokens 2048 \
  --teacher-tensor-parallel-size 1 \
  --teacher-gpu-memory-utilization 0.65 \
  --sft-epochs 1 \
  --sft-arg data.train_batch_size=4 \
  --sft-arg data.micro_batch_size_per_gpu=2 \
  --sft-arg data.max_length=4096 \
  --eval-temperature 0.0 \
  --eval-top-p 1.0 \
  --eval-max-response-len 4096 \
  --eval-batch-size 4 \
  --eval-gpu-memory-utilization 0.5 \
  --eval-tensor-parallel-size 1 \
  --grpo-env TOTAL_EPOCHS=1 \
  --grpo-env TRAIN_BATCH_SIZE=32 \
  --grpo-env PPO_MINI_BATCH_SIZE=32 \
  --grpo-env PPO_MICRO_BATCH_SIZE_PER_GPU=4 \
  --grpo-env LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8 \
  --grpo-env ROLLOUT_N=4 \
  --grpo-env MAX_PROMPT_LEN=1024 \
  --grpo-env MAX_RESPONSE_LEN=4096 \
  --grpo-env LR=5e-7 \
  --grpo-env KL_COEF=0.001 \
  --grpo-env GPU_MEMORY_UTILIZATION=0.6 \
  --grpo-env VLLM_MAX_NUM_SEQS=64 \
  --grpo-env VLLM_MAX_NUM_BATCHED_TOKENS=4096 \
  --grpo-env SAVE_FREQ=10 \
  --grpo-env TEST_FREQ=5
```

## Commonsense: CommonsenseQA

```bash
${PYTHON} DataObs/scripts/experiment_pipeline_vllm.py \
  --experiment-id exp9100_commonsenseqa_qwen3_4b_teacher14b_n4_sft_rl \
  --dataset commonsenseQA \
  --base-model /data/pretrain_models/Qwen3-4B \
  --teacher-model /data/pretrain_models/Qwen3-14B \
  --output-dir /data/hrh/COT/experiments \
  --stages distill,metrics,sft,sft_eval,grpo,grpo_eval \
  --gpu-ids 4 \
  --distill-method answer_augmentation \
  --teacher-num-samples 4 \
  --teacher-temperature 0.7 \
  --teacher-top-p 0.95 \
  --teacher-do-sample \
  --teacher-batch-size 8 \
  --teacher-max-new-tokens 1024 \
  --teacher-tensor-parallel-size 1 \
  --teacher-gpu-memory-utilization 0.65 \
  --sft-epochs 1 \
  --sft-arg data.train_batch_size=4 \
  --sft-arg data.micro_batch_size_per_gpu=2 \
  --sft-arg data.max_length=2048 \
  --eval-temperature 0.0 \
  --eval-top-p 1.0 \
  --eval-max-response-len 4096 \
  --eval-batch-size 8 \
  --eval-gpu-memory-utilization 0.5 \
  --eval-tensor-parallel-size 1 \
  --grpo-env TOTAL_EPOCHS=1 \
  --grpo-env TRAIN_BATCH_SIZE=32 \
  --grpo-env PPO_MINI_BATCH_SIZE=32 \
  --grpo-env PPO_MICRO_BATCH_SIZE_PER_GPU=4 \
  --grpo-env LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8 \
  --grpo-env ROLLOUT_N=4 \
  --grpo-env MAX_PROMPT_LEN=512 \
  --grpo-env MAX_RESPONSE_LEN=2048 \
  --grpo-env LR=5e-7 \
  --grpo-env KL_COEF=0.001 \
  --grpo-env GPU_MEMORY_UTILIZATION=0.6 \
  --grpo-env VLLM_MAX_NUM_SEQS=64 \
  --grpo-env VLLM_MAX_NUM_BATCHED_TOKENS=4096 \
  --grpo-env SAVE_FREQ=10 \
  --grpo-env TEST_FREQ=5
```

## Agent: BFCL Eval

BFCL is not currently wired into `experiment_pipeline_vllm.py` as a distill/SFT/GRPO training dataset. The current repo path also lacks the BFCL evaluator dependency:

```text
gorilla/berkeley-function-call-leaderboard
```

After that dependency is available, evaluate an SFT checkpoint:

```bash
BASE_MODEL=/data/pretrain_models/Qwen3-4B \
MODEL_NAME=exp9100_bfcl_sft_eval \
TARGET_EVAL_DIR=/data/hrh/COT/experiments/exp9100_bfcl_eval/sft \
bash scripts/eval.sh bfcl 4 /path/to/sft/checkpoint
```

Evaluate a GRPO checkpoint:

```bash
BASE_MODEL=/data/pretrain_models/Qwen3-4B \
MODEL_NAME=exp9100_bfcl_grpo_eval \
TARGET_EVAL_DIR=/data/hrh/COT/experiments/exp9100_bfcl_eval/grpo \
bash scripts/eval.sh bfcl 4 /path/to/grpo/global_step_xxx/actor/merged_model
```
