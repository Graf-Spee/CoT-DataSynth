# 训练资源开销参考

本文记录 DataObs 里 SFT、GRPO、Eval 的显存、时间和存储开销。这里的数字用于排期和选 GPU，不是硬保证；实际开销会受 GPU 型号、样本长度、vLLM 版本、batch size、验证频率影响。

## 当前脚本默认参数

| 阶段 | 入口 | 关键默认值 |
| --- | --- | --- |
| SFT | `DataObs/lib/training/sft_dataobs.sh` | LoRA rank 8，bf16，FSDP2，`data.train_batch_size=8`，`data.micro_batch_size_per_gpu=2`，`data.max_length=16384`，单卡自动把 `ulysses_sequence_parallel_size` 降为 1 |
| GRPO | `DataObs/lib/training/grpo_from_sft.sh` | LoRA rank 32，`TRAIN_BATCH_SIZE=128`，`PPO_MICRO_BATCH_SIZE_PER_GPU=32`，`LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=64`，`ROLLOUT_N=8`，`MAX_RESPONSE_LEN=1024`，`GPU_MEMORY_UTILIZATION=0.75`，`TEST_FREQ=5` |
| Eval | `scripts/eval.sh` | `N_SAMPLES=1`，`MAX_PROMPT_LEN=512`，`MAX_RESPONSE_LEN=1024`，`BATCH_SIZE=32` |

## 本地已观测基准

实验：`pipeline_check_gsm8k_qwen7b`

| 项目 | 观测结果 |
| --- | --- |
| Student | `/data/pretrain_models/Qwen2.5-0.5B-Instruct` |
| Dataset | GSM8K |
| SFT | 8 条 smoke 数据，2 卡，1 step，约 15 秒；样本太少，不适合外推完整训练 |
| SFT Eval | GSM8K test，约 819 秒 |
| GRPO | 2 卡，`TOTAL_EPOCHS=1`，`ROLLOUT_N=8`，共 58 steps |
| GRPO 普通 step | 约 56-60 秒 |
| GRPO 验证 step | `TEST_FREQ=5` 时约 105 秒 |
| GRPO 显存 | `perf/max_memory_allocated_gb` 最高约 38.8 GB，`perf/max_memory_reserved_gb` 最高约 55.1 GB |

注意：`reserved` 比 `allocated` 更接近 `nvidia-smi` 看到的显存压力。你的 GRPO OOM 是在默认参数下，2 张 80GB 卡也可能因为 actor backward + vLLM/Ray/PyTorch 缓存 + 碎片化顶到峰值。

## 模型存储大小

本地 `/data/pretrain_models` 下模型目录大小：

| 模型 | 磁盘大小 | 备注 |
| --- | ---: | --- |
| Qwen2.5-0.5B-Instruct | 954 MB | 当前主要 student |
| Qwen2.5-3B-Instruct | 5.8 GB | 中等 student |
| Qwen2.5-7B-Instruct | 15 GB | 常用 teacher/student |
| Qwen2.5-32B-Instruct | 62 GB | 大 teacher/student |
| Qwen2.5-72B-Instruct | 136 GB | teacher 规模 |
| DeepSeek-R1-Distill-Qwen-1.5B | 3.4 GB | reasoning 候选 |
| DeepSeek-R1-Distill-Qwen-7B | 15 GB | reasoning teacher |
| DeepSeek-R1-Distill-Qwen-32B | 62 GB | reasoning teacher |

LoRA adapter 通常很小；merge 后模型大小接近 base model。GRPO 输出可能包含 actor checkpoint、LoRA adapter、optimizer state、merged model，会比 SFT 输出增长更快。

## SFT 显存和时间估算

假设：LoRA SFT、bf16、`micro_batch_size_per_gpu=2`、`train_batch_size=8`、`max_length=16384`。如果实际样本很长，显存会明显上升。

| Base model | 最低可尝试 | 推荐配置 | GSM8K 1 epoch 粗略时间 | 备注 |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | 1 x 24 GB | 1 x 40 GB 或 2 x 40 GB | 0.5-2 小时 | 单卡可跑 |
| DeepSeek-R1-Distill-Qwen-1.5B | 1 x 40 GB | 1 x 80 GB 或 2 x 40 GB | 1-3 小时 | hidden size 更大 |
| Qwen2.5-3B | 1 x 80 GB 或 2 x 40 GB | 2 x 80 GB | 2-5 小时 | OOM 时先降 micro batch |
| Qwen2.5-7B / DeepSeek-R1-Distill-Qwen-7B | 2 x 80 GB 或 4 x 40 GB | 4 x 80 GB | 4-10 小时 | 长 CoT 建议开 gradient checkpointing |
| Qwen2.5-32B / DeepSeek-R1-Distill-Qwen-32B | 4-8 x 80 GB | 8 x 80 GB | 1-2 天 | 40GB 卡风险较高 |
| Qwen2.5-72B | 8 x 80 GB 起 | 多机或 8+ x 80 GB | 多天 | 不建议常规消融使用 |

SFT 时间估算：

```text
steps_per_epoch = ceil(num_training_rows / data.train_batch_size)
wall_time ~= startup_time + steps_per_epoch * seconds_per_step
```

更可靠的办法：先跑 20-50 个真实 step，再从日志外推。

## GRPO 显存和时间估算

GRPO 比 SFT 贵很多，因为同时涉及 actor、ref、rollout/vLLM，并且 `ROLLOUT_N=8` 会把每个 prompt 的采样数量放大。

| Base model | 最低可尝试 | 推荐配置 | GSM8K 1 epoch 粗略时间 | 备注 |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | 2 x 80 GB；2 x 40 GB 需要降参数 | 2 x 80 GB | 1-1.5 小时 | 本地默认参数 reserved 约 55GB/GPU |
| DeepSeek-R1-Distill-Qwen-1.5B | 2 x 80 GB | 4 x 80 GB | 2-4 小时 | 建议先减小 micro batch |
| Qwen2.5-3B | 4 x 80 GB | 4-8 x 80 GB | 4-8 小时 | 可考虑 `TP_SIZE=2` |
| Qwen2.5-7B / DeepSeek-R1-Distill-Qwen-7B | 4-8 x 80 GB | 8 x 80 GB | 8-18 小时 | 40GB 卡需要大幅降参数 |
| Qwen2.5-32B / DeepSeek-R1-Distill-Qwen-32B | 8 x 80 GB | 多机 80GB | 1-3 天 | 需要仔细调 tensor parallel/vLLM |
| Qwen2.5-72B | 多机 80GB | 多机 80GB | 多天 | 不建议常规消融 |

GRPO 时间估算：

```text
steps = ceil(num_training_rows / TRAIN_BATCH_SIZE) * TOTAL_EPOCHS
total_time ~= 普通 step 数 * normal_step_time + 验证 step 数 * validation_step_time + startup
```

当前 0.5B 本地观测：58 steps，普通 step 约 56-60 秒，`TEST_FREQ=5` 的验证 step 约 105 秒，完整 1 epoch 约 65-90 分钟。

## Eval 开销

| 模型 | 常见 GPU 配置 | 备注 |
| --- | --- | --- |
| 0.5B-1.5B | 1 x 24-40 GB | 本地 GSM8K eval 2 卡约 819 秒 |
| 3B-7B | 1 x 80 GB 或 2 x 40 GB | OOM 时降 `BATCH_SIZE` |
| 32B | 4-8 x 80 GB | tensor parallel 要能整除 GPU 数 |
| 72B | 8 x 80 GB 或多机 | 谨慎使用 pass@n |

Eval 主要放大项：

- `N_SAMPLES`：pass@n 会近似按倍数增加生成量。
- `MAX_RESPONSE_LEN`：越大越吃 KV cache 和时间。
- 数据集大小：GSM8K 中等；MATH/code 任务通常更慢。

## OOM 时优先调参

SFT：

| 参数 | 作用 |
| --- | --- |
| `--sft-arg data.micro_batch_size_per_gpu=1` | 首先降低 activation 显存 |
| `--sft-arg data.train_batch_size=4` | 降低 global batch |
| `--sft-arg data.max_length=4096` | 如果不需要 16k，上限降低很有效 |
| `--sft-arg model.enable_gradient_checkpointing=true` | 省显存但变慢 |

GRPO：

| 参数 | 作用 |
| --- | --- |
| `--grpo-env PPO_MICRO_BATCH_SIZE_PER_GPU=16` | 降低 actor update 显存 |
| `--grpo-env LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=32` | 降低 old/ref logprob 显存 |
| `--grpo-env VLLM_MAX_NUM_SEQS=128` | 降低 rollout 并发显存 |
| `--grpo-env TEST_FREQ=20` | 减少验证峰值和耗时 |
| `--grpo-env MAX_RESPONSE_LEN=512` | 降低生成 token 和 KV cache |
| `--grpo-env ROLLOUT_N=4` | 最大幅度降低显存/时间，但会改变采样强度 |

如果日志提示碎片化，例如：

```text
reserved by PyTorch but unallocated
```

可以在命令里加：

```bash
--grpo-env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 如何统计新实验

看 SFT：

```bash
tail -f /data/hrh/COT/experiments/<exp_id>/logs/sft.log
```

看 GRPO 显存和时间：

```bash
grep -E "perf/max_memory|timing_s/step|Total steps|Training Progress" \
  /data/hrh/COT/experiments/<exp_id>/logs/grpo.log
```

关键字段：

| 字段 | 含义 |
| --- | --- |
| `perf/max_memory_allocated_gb` | PyTorch 实际分配峰值 |
| `perf/max_memory_reserved_gb` | CUDA reserved 峰值，更接近 OOM 压力 |
| `timing_s/step` | 单步耗时 |
| `timing_s/gen` | rollout 生成耗时 |
| `timing_s/update_actor` | actor 更新耗时 |
| `response_length/max` | 是否顶到 `MAX_RESPONSE_LEN` |
| `response_length/clip_ratio` | 被 response 上限截断的比例 |
