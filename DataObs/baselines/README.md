# Baselines

此处记录使用的所有 baseline 方法。特别注意：现在我们做的测试中，蒸馏环节 **使用了** teacher correctness filter

ZeroShot: 使用 `run_all_eval.py` + plain + no thinking 评测。命令：

```bash
python DataObs/tools/run_all_eval.py --gpu-ids 2 --prompt-template-method plain --output-root /data/hrh/COT/baselines/ZeroShot
```

ZeroShot CoT: 使用 `run_all_eval.py` + zeroshot + thinking 评测。命令：

```bash
python DataObs/tools/run_all_eval.py --gpu-ids 2 --prompt-template-method zeroshot --enable-thinking
```

Vanilla Distill: 只对蒸馏结果加 Teacher Correctness Filter

Heuristics: Data Augmentation 方法，参考 The Quest for Efficient Reasoning: A Data-Centric Benchmark to CoT Distillation
- question rephrasing
- question augmentation
    - 实际上实现的是 backward questions generation
- answer augmentation
- reverse thinking augmentation

使用的命令：（现在是 smoke 命令）
- smoke 进度：
    - [x] question rephrasing
    - [x] answer aug
    - [x] question aug
    - [-] reverse thinking
 
python DataObs/scripts/experiment_pipeline_vllm.py \
    --experiment-id test_smoke_gsm8k_question_rephrasing_qwen3_8b_n4 \
    --dataset gsm8k \
    --base-model /data/pretrain_models/Qwen3-4B \
    --teacher-model /data/pretrain_models/Qwen3-8B \
    --output-dir /data/hrh/COT/experiments \
    --stages distill,metrics,sft,sft_eval \
    --distill-method question_rephrasing \
    --gpu-ids 1 \
    --teacher-num-samples 4 \
    --teacher-temperature 0.7 \
    --teacher-top-p 0.95 \
    --teacher-do-sample \
    --teacher-batch-size 8 \
    --teacher-max-new-tokens 1024 \
    --teacher-tensor-parallel-size 1 \
    --teacher-gpu-memory-utilization 0.65 \
    --disable-teacher-filter \
    --smoke-num-rows 5 \
    --sft-epochs 1 \
    --sft-arg data.train_batch_size=4 \
    --sft-arg data.micro_batch_size_per_gpu=2 \
    --sft-arg data.max_length=2048 \
    --eval-batch-size 8

python DataObs/scripts/experiment_pipeline_vllm.py \
    --experiment-id test_smoke_gsm8k_answer_augmentation_qwen3_8b_n4 \
    --dataset gsm8k \
    --base-model /data/pretrain_models/Qwen3-4B \
    --teacher-model /data/pretrain_models/Qwen3-8B \
    --output-dir /data/hrh/COT/experiments \
    --stages distill,metrics,sft,sft_eval \
    --distill-method answer_augmentation \
    --gpu-ids 0 \
    --teacher-num-samples 4 \
    --teacher-temperature 0.7 \
    --teacher-top-p 0.95 \
    --teacher-do-sample \
    --teacher-batch-size 8 \
    --teacher-max-new-tokens 1024 \
    --teacher-tensor-parallel-size 1 \
    --teacher-gpu-memory-utilization 0.65 \
    --disable-teacher-filter \
    --smoke-num-rows 5 \
    --sft-epochs 1 \
    --sft-arg data.train_batch_size=4 \
    --sft-arg data.micro_batch_size_per_gpu=2 \
    --sft-arg data.max_length=2048 \
    --eval-batch-size 8

python DataObs/scripts/experiment_pipeline_vllm.py \
    --experiment-id test_smoke_gsm8k_question_augmentation_qwen3_8b_n4 \
    --dataset gsm8k \
    --base-model /data/pretrain_models/Qwen3-4B \
    --teacher-model /data/pretrain_models/Qwen3-8B \
    --output-dir /data/hrh/COT/experiments \
    --stages distill,metrics,sft,sft_eval \
    --distill-method question_augmentation \
    --gpu-ids 1 \
    --teacher-num-samples 4 \
    --teacher-temperature 0.7 \
    --teacher-top-p 0.95 \
    --teacher-do-sample \
    --teacher-batch-size 8 \
    --teacher-max-new-tokens 1024 \
    --teacher-tensor-parallel-size 1 \
    --teacher-gpu-memory-utilization 0.65 \
    --disable-teacher-filter \
    --smoke-num-rows 5 \
    --sft-epochs 1 \
    --sft-arg data.train_batch_size=4 \
    --sft-arg data.micro_batch_size_per_gpu=2 \
    --sft-arg data.max_length=2048 \
    --eval-batch-size 8

python DataObs/scripts/experiment_pipeline_vllm.py \
    --experiment-id test_smoke_gsm8k_reverse_thinking_qwen3_8b_n1 \
    --dataset gsm8k \
    --base-model /data/pretrain_models/Qwen3-4B \
    --teacher-model /data/pretrain_models/Qwen3-8B \
    --output-dir /data/hrh/COT/experiments \
    --stages distill,metrics,sft,sft_eval \
    --distill-method reverse_thinking \
    --gpu-ids 0 \
    --teacher-num-samples 1 \
    --teacher-temperature 0.7 \
    --teacher-top-p 0.95 \
    --teacher-do-sample \
    --teacher-batch-size 8 \
    --teacher-max-new-tokens 1024 \
    --teacher-tensor-parallel-size 1 \
    --teacher-gpu-memory-utilization 0.65 \
    --disable-teacher-filter \
    --smoke-num-rows 5 \
    --backward-question-max-new-tokens 1024 \
    --consistency-max-new-tokens 1024 \
    --sft-epochs 1 \
    --sft-arg data.train_batch_size=4 \
    --sft-arg data.micro_batch_size_per_gpu=2 \
    --sft-arg data.max_length=3072 \
    --eval-batch-size 8


最终使用的参数参考这个：
python DataObs/scripts/experiment_pipeline_vllm.py \
    --experiment-id exp1100_gsm8k_qwen3_teacher_qwen3_8b_t0.7_n4 \
    --dataset gsm8k \
    --base-model /data/pretrain_models/Qwen3-4B \
    --teacher-model /data/pretrain_models/Qwen3-8B \
    --output-dir /data/hrh/COT/experiments \
    --stages grpo,grpo_eval \
    --gpu-ids 1 \
    --teacher-num-samples 4 \
    --teacher-temperature 0.7 \
    --teacher-top-p 0.95 \
    --teacher-do-sample \
    --teacher-batch-size 8 \
    --teacher-max-new-tokens 1024 \
    --teacher-tensor-parallel-size 1 \
    --teacher-gpu-memory-utilization 0.65 \
    --disable-teacher-filter \
    --sft-epochs 1 \
    --sft-arg data.train_batch_size=4 \
    --sft-arg data.micro_batch_size_per_gpu=2 \
    --sft-arg data.max_length=2048 \
    --eval-arg data.batch_size=8 \
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


Self Instruct

CoT-Self-Instruct