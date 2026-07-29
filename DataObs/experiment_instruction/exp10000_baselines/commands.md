# Exp10000 Baselines

These commands test baselines across different datasets.

List of all baselines:
- ZeroShot
    - [x] Implemented
    - [x] Experiment Done
- ZeroShot CoT
    - [x] Implemented
    - [x] Experiment Done
- Vanilla Distill (Teacher Correctness Filter)
    - [x] Implemented
    - [ ] Experiment Done
- Heuristics: Question Rephrasing
    - [x] Implemented
    - [ ] Experiment Done
- Heuristics: Question Augmentation (Backward Questions Generation)
    - [x] Implemented
    - [ ] Experiment Done
- Heuristics: Answer Augmentation
    - [x] Implemented
    - [ ] Experiment Done
- Heuristics: Reverse Thinking Augmentation
    - [x] Implemented
    - [ ] Experiment Done
- Self Instruct
    - [ ] Implemented
    - [ ] Experiment Done
- CoT-Self-Instruct
    - [ ] Implemented
    - [ ] Experiment Done
- Agent-Self-Instruct
    - [ ] Implemented
    - [ ] Experiment Done
- Iterative SFT
    - [ ] Implemented
    - [ ] Experiment Done
- active learning style(uncertainty sampling/entropy-based selection)
    - [ ] Implemented
    - [ ] Experiment Done

## Common Environment & Clarifications

```bash
cd /home/hrh/CoT-DataSynth

export PATH=/home/hrh/anaconda3/envs/verl-torch280/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0

PYTHON=/home/hrh/anaconda3/envs/verl-torch280/bin/python
```

`VLLM_USE_FLASHINFER_SAMPLER=0` disables vLLM v1's FlashInfer top-k/top-p sampler and falls back to the PyTorch-native sampler. Keep it in the common environment to match existing experiment instructions and reduce sampler-backend differences across baseline runs.

For ZeroShot and ZeroShot CoT, current eval runs deterministic greedy decoding (`temperature=0.0`), so this top-k/top-p sampler switch does not change their decoding path.

Length policy for distillation baselines:
- Use `--teacher-max-new-tokens 8192`.
- Do not set method-specific length knobs; let the pipeline's teacher-max override fill them.
- Use `--sft-arg data.max_length=12288` to allow 8192-token generations plus prompt/formatting margin without going to 16384.

Important notes:
- Change the dataset & gpu_ids on demand.
- For answer augmentation, do not pass `--answer-aug-use-original-metamath-prompt` for non-mathematics datasets (i.e. all except gsm8k / MATH / NuminaMath).
- Number of samples is fixed at 4 except for:
    - Vanilla Distill (Teacher Filter): fixed at 1.
    - Reverse Thinking: does not apply (1 forward-backward chain per seed data).
- `--teacher-do-sample` is always passed, otherwise temperature & top-p do not apply.

## ZeroShot

```bash
python DataObs/tools/run_all_eval.py --gpu-ids 2 --prompt-template-method plain --output-root /data/hrh/COT/baselines/ZeroShot
```

## ZeroShot CoT

```bash
python DataObs/tools/run_all_eval.py --gpu-ids 2 --prompt-template-method zeroshot --enable-thinking
```

## Vanilla Distill (Teacher Correctness Filter)

```bash
${PYTHON} DataObs/scripts/experiment_pipeline_vllm.py \
  --experiment-id exp10000_gsm8k_teacher_filter_qwen3_14b_t8192_sft_grpo \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen3-4B \
  --teacher-model /data/pretrain_models/Qwen3-14B \
  --output-dir /data/hrh/COT/experiments \
  --stages distill,metrics,sft,sft_eval,grpo,grpo_eval \
  --distill-method teacher_correctness_filter \
  --gpu-ids 1 \
  --teacher-num-samples 1 \
  --teacher-temperature 0.7 \
  --teacher-top-p 0.95 \
  --teacher-do-sample \
  --teacher-batch-size 8 \
  --teacher-max-new-tokens 8192 \
  --teacher-tensor-parallel-size 1 \
  --teacher-gpu-memory-utilization 0.65 \
  --sft-epochs 1 \
  --sft-arg data.train_batch_size=4 \
  --sft-arg data.micro_batch_size_per_gpu=2 \
  --sft-arg data.max_length=12288 \
  --eval-batch-size 8 \
  --eval-gpu-memory-utilization 0.9 \
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

## Heuristics: Question Rephrasing

```bash
${PYTHON} DataObs/scripts/experiment_pipeline_vllm.py \
  --experiment-id exp10000_gsm8k_question_rephrasing_qwen3_14b_n4_t8192_sft_grpo \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen3-4B \
  --teacher-model /data/pretrain_models/Qwen3-14B \
  --output-dir /data/hrh/COT/experiments \
  --stages distill,metrics,sft,sft_eval,grpo,grpo_eval \
  --distill-method question_rephrasing \
  --gpu-ids 1 \
  --teacher-num-samples 4 \
  --teacher-temperature 0.7 \
  --teacher-top-p 0.95 \
  --teacher-do-sample \
  --teacher-batch-size 8 \
  --teacher-max-new-tokens 8192 \
  --teacher-tensor-parallel-size 1 \
  --teacher-gpu-memory-utilization 0.65 \
  --sft-epochs 1 \
  --sft-arg data.train_batch_size=4 \
  --sft-arg data.micro_batch_size_per_gpu=2 \
  --sft-arg data.max_length=12288 \
  --eval-batch-size 8 \
  --eval-gpu-memory-utilization 0.9 \
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

## Heuristics: Question Augmentation (Backward Questions Generation)

```bash
${PYTHON} DataObs/scripts/experiment_pipeline_vllm.py \
  --experiment-id exp10000_gsm8k_question_augmentation_qwen3_14b_n4_t8192_sft_grpo \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen3-4B \
  --teacher-model /data/pretrain_models/Qwen3-14B \
  --output-dir /data/hrh/COT/experiments \
  --stages distill,metrics,sft,sft_eval,grpo,grpo_eval \
  --distill-method question_augmentation \
  --gpu-ids 1 \
  --teacher-num-samples 4 \
  --teacher-temperature 0.7 \
  --teacher-top-p 0.95 \
  --teacher-do-sample \
  --teacher-batch-size 8 \
  --teacher-max-new-tokens 8192 \
  --teacher-tensor-parallel-size 1 \
  --teacher-gpu-memory-utilization 0.65 \
  --sft-epochs 1 \
  --sft-arg data.train_batch_size=4 \
  --sft-arg data.micro_batch_size_per_gpu=2 \
  --sft-arg data.max_length=12288 \
  --eval-batch-size 8 \
  --eval-gpu-memory-utilization 0.9 \
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

## Heuristics: Answer Augmentation

```bash
${PYTHON} DataObs/scripts/experiment_pipeline_vllm.py \
  --experiment-id exp10000_gsm8k_answer_augmentation_qwen3_14b_n4_t8192_sft_grpo \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen3-4B \
  --teacher-model /data/pretrain_models/Qwen3-14B \
  --output-dir /data/hrh/COT/experiments \
  --stages distill,metrics,sft,sft_eval,grpo,grpo_eval \
  --distill-method answer_augmentation \
  --gpu-ids 1 \
  --teacher-num-samples 4 \
  --teacher-temperature 0.7 \
  --teacher-top-p 0.95 \
  --teacher-do-sample \
  --teacher-batch-size 8 \
  --teacher-max-new-tokens 8192 \
  --teacher-tensor-parallel-size 1 \
  --teacher-gpu-memory-utilization 0.65 \
  --answer-aug-use-original-metamath-prompt \
  --sft-epochs 1 \
  --sft-arg data.train_batch_size=4 \
  --sft-arg data.micro_batch_size_per_gpu=2 \
  --sft-arg data.max_length=12288 \
  --eval-batch-size 8 \
  --eval-gpu-memory-utilization 0.9 \
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

## Heuristics: Reverse Thinking Augmentation

```bash
${PYTHON} DataObs/scripts/experiment_pipeline_vllm.py \
  --experiment-id exp10000_gsm8k_reverse_thinking_qwen3_14b_n1_t8192_sft_grpo \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen3-4B \
  --teacher-model /data/pretrain_models/Qwen3-14B \
  --output-dir /data/hrh/COT/experiments \
  --stages distill,metrics,sft,sft_eval,grpo,grpo_eval \
  --distill-method reverse_thinking \
  --gpu-ids 1 \
  --teacher-num-samples 1 \
  --teacher-temperature 0.7 \
  --teacher-top-p 0.95 \
  --teacher-do-sample \
  --teacher-batch-size 8 \
  --teacher-max-new-tokens 8192 \
  --teacher-tensor-parallel-size 1 \
  --teacher-gpu-memory-utilization 0.65 \
  --sft-epochs 1 \
  --sft-arg data.train_batch_size=4 \
  --sft-arg data.micro_batch_size_per_gpu=2 \
  --sft-arg data.max_length=12288 \
  --eval-batch-size 8 \
  --eval-gpu-memory-utilization 0.9 \
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

## Self Instruct

Not Implemented Yet

## CoT-Self-Instruct

Not Implemented Yet

## Agent-Self-Instruct

Not Implemented Yet

## Iterative SFT

Not Implemented Yet

## active learning style(uncertainty sampling/entropy-based selection)

Not Implemented Yet
