# DataObs 里的 CoT-Self-Instruct Baseline

这个目录放的是 `CoT-Self-Instruct` 在 `DataObs` 里的当前正式 baseline 实现。

目标不是做一个“像 CoT-Self-Instruct 的新方法”，而是：

- 尽量保留论文 / 公共实现的主流程
- 只做本地运行所需的最小适配
  - `远程 API -> 本地 Qwen + vLLM`
  - `原始输出 -> DataObs parquet / distill 输出`

## 当前结论

截至 **2026-08-23**，这版 baseline 已经是一个**正式可用**的 `DataObs` baseline：

- `reasoning + Answer-Consistency` 跑通
- `instruction + RIP` 跑通
- `code + code_tests` 跑通
- 已接入 `DataObs/scripts/experiment_pipeline_vllm.py`
- 已能稳定产出：
  - `filtered_sft.parquet`
  - `filtered_sft.candidates.parquet`
  - `filtered_sft.summary.json`

更准确地说，它现在属于：

- **论文主流程等价迁移 + 本地最小适配版**
- 不是“逐字逐组件完全复刻版”

## 这个 baseline 在做什么

`CoT-Self-Instruct` 在这里是一个**数据构造 baseline**。

- 对 `reasoning / verifiable tasks`
  - 先生成新的 reasoning 题
  - 再用 `Answer-Consistency` 做过滤
- 对 `instruction-following / non-verifiable tasks`
  - 先生成新的 instruction prompts
  - 再用 `RIP` 做过滤
- 对 `code tasks`
  - 直接复用代码数据集的 seed prompt
  - 用本地 teacher 生成代码解答
  - 再用 `code_tests` 打分过滤
- 最后把保留下来的样本整理成 `DataObs` 可直接训练的 SFT 数据

## 当前流程

主入口是 `cot_self_instruct.py`。

执行流程：

1. 读取 seed tasks
2. 根据 `--cot-task-mode` 选择分支
3. reasoning 分支：
   - `generate_cot_tasks.py`
   - `filter_answer_consistency.py`
4. instruction 分支：
   - `generate_instruction_tasks.py`
   - `filter_rip.py`
5. code 分支：
   - `generate_code_tasks.py`
   - `filter_code_tests.py`
6. `prepare_for_finetuning.py`
7. 写出最终产物

## 目录说明

### `cot_self_instruct.py`

主入口脚本，负责串起整个 baseline。

### `templates.py`

放 prompt 模板，当前包含：

- `REASONING_PROMPT_TEMPLATE`
- `INSTRUCTION_PROMPT_TEMPLATE`
- `ANSWER_CONSISTENCY_SOLVE_TEMPLATE`
- `RIP_RESPONSE_TEMPLATE`
- `RIP_SCORE_TEMPLATE`

### `generate_cot_tasks.py`

reasoning 分支的 synthetic task 生成器。

主要做：

- 从 seed tasks 抽 few-shot examples
- 生成 reasoning questions
- 解析：
  - `[New Question Begin] ... [New Question End]`
  - `[Final Answer to New Question Begin] ... [Final Answer to New Question End]`

### `generate_instruction_tasks.py`

instruction 分支的 synthetic prompt 生成器。

主要做：

- 从 seed tasks 抽 instruction few-shot examples
- 生成 instruction-following prompts
- 解析：
  - `[New Prompt Begin] ... [New Prompt End]`

当前额外加了一层**本地最小补强过滤**：

- 尽量去掉明显像数学问答 / 求值题的候选
- 尽量去掉明显依赖缺失上下文的伪 instruction prompt

说明：

- 论文 / 第三方实现的方向，是生成可直接用于 instruction-following 的 synthetic prompt
- 第三方实现主要依赖 seed 质量、模板、聚类来把分布拉正
- 当前本地实现多加了一层轻量过滤，目的是让生成结果更接近可直接训练的 instruction prompts

### `filter_answer_consistency.py`

reasoning 分支的过滤模块。

逻辑：

1. 对每个 synthetic question 重复求解 `K` 次
2. 提取每次最终答案
3. 做 majority vote
4. 若多数答案与 `target_answer` 一致，则通过

### `filter_rip.py`

instruction 分支的过滤模块。

逻辑：

1. 对每个 instruction prompt 采样多个 response
2. 用本地 judge prompt 对每个 response 打分
3. 取该 prompt 的最小 response score 作为保守分数
4. 若最小分数达到阈值，则通过
5. 同时选分数最高的 response 作为最终 SFT completion

说明：

- 当前 `RIP` 是本地 `LM-as-a-judge` 适配版
- 它比第三方实现里用长度之类 proxy 的 placeholder 更接近论文里“多次采样 + 评分过滤”的主旨
- 但它还不是独立 reward model 服务版 `RIP`

### `generate_code_tasks.py`

code 分支的任务准备模块。

主要做：

- 从已经 `prepare_eval_data` 过的 seed parquet 里直接读取代码 prompt
- 保留对应的 `reward_model.ground_truth` 作为测试用例
- 把 `mbpp / mbpp+ / humaneval / humaneval+` 这类数据转成 `cot-self-instruct` 的统一中间格式

说明：

- 这里没有再走 generic instruction prompt 生成
- 这是一个**有意的最小改动**
- 目的是直接和 `self-instruct / Vanilla Distill` 的 `code_tests` 思路对齐

### `filter_code_tests.py`

code 分支的过滤模块。

逻辑：

1. 对每个代码任务 prompt 生成 `N` 个候选解答
2. 用 `verl/utils/reward_score/mbpp.py` 的 `code_tests` 逻辑打分
3. 选择分数最高的候选
4. 只有分数达到阈值的样本才保留

说明：

- 这一步不走 `Answer-Consistency`
- 也不走 `RIP`
- 因为代码任务本身是可验证的，所以直接走测试过滤更合理

### `prepare_for_finetuning.py`

把通过过滤的样本整理成 `DataObs` 的 SFT parquet。

### `common.py`

放通用工具函数，包括：

- 读 seed 数据
- 初始化本地 teacher / vLLM
- 批量生成
- 文本抽取与规范化
- 写出 parquet / candidates / summary

## 如何执行

建议按三种场景理解：

- `reasoning 正式实验`
  - 适合 `gsm8k / math / numinamath / arc / csqa / strategyqa` 这类推理数据集
  - 直接走 `experiment_pipeline_vllm.py`
  - 一次串 `distill,metrics,sft,sft_eval,grpo,grpo_eval`
- `code 正式实验`
  - 适合 `mbpp / mbpp+ / humaneval / humaneval+`
  - 自动走 `code` 分支
  - 核心是 `teacher generation + code_tests filter`
- `instruction 分支验证`
  - 先只跑 `distill`
  - 重点检查生成出的 prompt 是否像真正 instruction-following 数据，以及 `RIP` 是否正常过滤
- `单脚本 smoke`
  - 直接跑 `cot_self_instruct.py`
  - 更适合只验证 step 1 / step 2 本身

### reasoning 正式实验命令

```bash
python DataObs/scripts/experiment_pipeline_vllm.py \
  --experiment-id exp_cot_self_instruct_gsm8k_qwen3_4b_teacher4b_sft_rl \
  --dataset gsm8k \
  --base-model /home/hrh/data/pretrain_models/Qwen3-4B \
  --teacher-model /home/hrh/data/pretrain_models/Qwen3-4B \
  --output-dir /home/hrh/data/COT/experiments \
  --stages distill,metrics,sft,sft_eval,grpo,grpo_eval \
  --distill-method cot_self_instruct \
  --gpu-ids 5 \
  --teacher-batch-size 1 \
  --teacher-tensor-parallel-size 1 \
  --teacher-gpu-memory-utilization 0.8 \
  --cot-task-mode reasoning \
  --cot-num-instructions 100 \
  --cot-num-fewshot-examples 2 \
  --cot-max-new-tokens 2048 \
  --consistency-k 16 \
  --consistency-max-new-tokens 1024 \
  --consistency-temperature 0.6 \
  --sft-epochs 1 \
  --sft-arg data.train_batch_size=4 \
  --sft-arg data.micro_batch_size_per_gpu=1 \
  --sft-arg data.max_length=2048 \
  --sft-arg model.enable_gradient_checkpointing=true \
  --eval-batch-size 8 \
  --eval-gpu-memory-utilization 0.7 \
  --grpo-env TOTAL_EPOCHS=1 \
  --grpo-env TRAIN_BATCH_SIZE=32 \
  --grpo-env PPO_MINI_BATCH_SIZE=32 \
  --grpo-env PPO_MICRO_BATCH_SIZE_PER_GPU=4 \
  --grpo-env LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8 \
  --grpo-env ROLLOUT_N=4 \
  --grpo-env MAX_PROMPT_LEN=512 \
  --grpo-env MAX_RESPONSE_LEN=4096 \
  --grpo-env LR=5e-7 \
  --grpo-env KL_COEF=0.001 \
  --grpo-env GPU_MEMORY_UTILIZATION=0.6 \
  --grpo-env VLLM_MAX_NUM_SEQS=64 \
  --grpo-env VLLM_MAX_NUM_BATCHED_TOKENS=6144 \
  --grpo-env SAVE_FREQ=10 \
  --grpo-env TEST_FREQ=5
```

### code 正式实验命令

```bash
python DataObs/scripts/experiment_pipeline_vllm.py \
  --experiment-id exp_cot_self_instruct_mbpp_qwen3_4b_teacher4b_sft_rl \
  --dataset mbpp \
  --base-model /home/hrh/data/pretrain_models/Qwen3-4B \
  --teacher-model /home/hrh/data/pretrain_models/Qwen3-4B \
  --output-dir /home/hrh/data/COT/experiments \
  --stages distill,metrics,sft,sft_eval,grpo,grpo_eval \
  --distill-method cot_self_instruct \
  --gpu-ids 0,1 \
  --teacher-batch-size 1 \
  --teacher-tensor-parallel-size 2 \
  --teacher-gpu-memory-utilization 0.8 \
  --cot-task-mode code \
  --code-num-samples 1 \
  --code-max-new-tokens 2048 \
  --code-temperature 0.7 \
  --code-top-p 0.95 \
  --sft-epochs 1 \
  --sft-arg data.train_batch_size=4 \
  --sft-arg data.micro_batch_size_per_gpu=1 \
  --sft-arg data.max_length=2048 \
  --sft-arg model.enable_gradient_checkpointing=true \
  --eval-batch-size 8 \
  --eval-gpu-memory-utilization 0.7 \
  --grpo-env TOTAL_EPOCHS=1 \
  --grpo-env TRAIN_BATCH_SIZE=32 \
  --grpo-env PPO_MINI_BATCH_SIZE=32 \
  --grpo-env PPO_MICRO_BATCH_SIZE_PER_GPU=4 \
  --grpo-env LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8 \
  --grpo-env ROLLOUT_N=4 \
  --grpo-env MAX_PROMPT_LEN=512 \
  --grpo-env MAX_RESPONSE_LEN=4096 \
  --grpo-env LR=5e-7 \
  --grpo-env KL_COEF=0.001 \
  --grpo-env GPU_MEMORY_UTILIZATION=0.6 \
  --grpo-env VLLM_MAX_NUM_SEQS=64 \
  --grpo-env VLLM_MAX_NUM_BATCHED_TOKENS=6144 \
  --grpo-env SAVE_FREQ=10 \
  --grpo-env TEST_FREQ=5
```

说明：

- `auto` 模式下，`mbpp / mbpp+ / humaneval / humaneval+` 现在也会自动走 `code`
- 如果你想显式写清楚，保留 `--cot-task-mode code` 更稳

### instruction 分支 distill 验证命令

```bash
/home/hrh/miniconda3/envs/verl-cu124/bin/python DataObs/scripts/experiment_pipeline_vllm.py \
  --experiment-id exp_cot_self_instruct_instruction_distill_check \
  --dataset gsm8k \
  --base-model /home/hrh/data/pretrain_models/Qwen3-4B \
  --teacher-model /home/hrh/data/pretrain_models/Qwen3-4B \
  --output-dir /home/hrh/data/COT/experiments \
  --stages distill \
  --distill-method cot_self_instruct \
  --gpu-ids 5 \
  --teacher-batch-size 1 \
  --teacher-tensor-parallel-size 1 \
  --teacher-gpu-memory-utilization 0.8 \
  --cot-task-mode instruction \
  --instruction-num-instructions 100 \
  --instruction-num-fewshot-examples 2 \
  --instruction-max-new-tokens 2048 \
  --instruction-temperature 0.8 \
  --instruction-top-p 0.9 \
  --rip-num-responses 32 \
  --rip-response-max-new-tokens 512 \
  --rip-response-temperature 1.0 \
  --rip-response-top-p 1.0 \
  --rip-score-max-new-tokens 32
```

### 单脚本 instruction smoke 命令

```bash
CUDA_VISIBLE_DEVICES=5 VLLM_TARGET_DEVICE=cuda VLLM_LOGGING_LEVEL=DEBUG \
/home/hrh/miniconda3/envs/verl-cu124/bin/python \
/home/hrh/CoT-DataSynth/DataObs/baselines/cot-self-instruct/cot_self_instruct.py \
  --dataset gsm8k \
  --input-file /home/hrh/open_datasets/GSM8K/main/train-00000-of-00001.parquet \
  --output-file /home/hrh/tmp/cot_self_instruct_instruction_smoke_v2.parquet \
  --log-file /home/hrh/tmp/cot_self_instruct_instruction_smoke_v2.log \
  --model-id /data2/pretrain_models/Qwen3-4B \
  --gpu-ids 5 \
  --gen-batch-size 1 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --instruction-num-fewshot-examples 2 \
  --instruction-num-instructions 2 \
  --instruction-max-new-tokens 512 \
  --rip-num-responses 2 \
  --rip-response-max-new-tokens 256 \
  --rip-score-max-new-tokens 32 \
  --smoke-num-rows 2 \
  --cot-task-mode instruction
```

## 关键参数

### 基础参数

- `--dataset`
- `--input-file`
- `--output-file`
- `--model-id`
- `--gpu-ids`

### reasoning 分支参数

- `--cot-task-mode reasoning`
- `--cot-num-instructions`
- `--cot-num-fewshot-examples`
- `--cot-max-new-tokens`
- `--cot-temperature`
- `--cot-top-p`

### instruction 分支参数

- `--cot-task-mode instruction`
- `--instruction-num-instructions`
- `--instruction-num-fewshot-examples`
- `--instruction-max-new-tokens`
- `--instruction-temperature`
- `--instruction-top-p`

### Answer-Consistency 参数

- `--consistency-k`
- `--consistency-max-new-tokens`
- `--consistency-temperature`
- `--consistency-top-p`
- `--consistency-min-vote-ratio`

### RIP 参数

- `--rip-num-responses`
- `--rip-response-max-new-tokens`
- `--rip-response-temperature`
- `--rip-response-top-p`
- `--rip-score-max-new-tokens`
- `--rip-min-score`

### 本地 vLLM 参数

- `--gen-batch-size`
- `--tensor-parallel-size`
- `--gpu-memory-utilization`
- `--max-model-len`

## 当前和论文的差距

当前主要剩这几类本地适配：

1. `instruction bootstrapping / instance generation` 这部分的工程参数可以更参考 `Self-Instruct`
2. `consistency / RIP` 这部分继续以 `CoT-Self-Instruct` 为准
3. `RIP` 用本地 Qwen 做 `LLM-as-a-judge`
4. 少量 prompt 仍保留了模型接入所需的最小适配
5. 部分运行参数按本地显存和速度做了调整

所以当前可以算作：

- 一个可用的 `CoT-Self-Instruct` baseline
- 一个尽量贴近论文流程的本地迁移版
