# DataObs 里的 Self-Instruct Baseline

这个目录是把官方 `yizhongw/self-instruct` 四步流程逐步迁到 `DataObs` 里的实现位置。

现在的原则不是“做一个像 Self-Instruct 的方法”，而是：

- 尽量保留官方四步结构
- 尽量保留官方 prompt / 解析 / 过滤思路
- 只在两类地方做必要适配：
  - `OpenAI API -> 本地 teacher / vLLM`
  - `官方中间格式 -> DataObs 可直接消费的 parquet 输出`

目前它已经接到了 DataObs 的 experiment pipeline，可以通过 `--distill-method self_instruct` 直接走进 DataObs 的 distill 阶段；如果你后面还要接 `sft_eval / grpo_eval` 的 vLLM 版评测，直接用 `DataObs/scripts/experiment_pipeline_vllm.py` 更方便。

## 官方四步到底在做什么

如果只看官方 repo 的名字，`step 1 / 2 / 3 / 4` 很容易有点抽象。

其实它们合起来做的事情很简单：

- 目标是自动生成一份可以拿去做 SFT 的训练数据
- 整体逻辑是“先造任务，再判断任务类型，再给任务造样本，最后清洗整理”

可以按下面理解：

### step 1: `bootstrap_instructions`

这一步的作用是：

- 先生成一批新的 `instruction`

也就是：

- 给模型一些 seed tasks
- 让模型模仿这些任务风格
- 再自己提出更多新任务

这一步的输出还不是最终训练样本，它产出的只是“任务描述”。

例如它更像是在生成：

- “判断一句话的情感类别”
- “把一句英文翻译成中文”
- “根据段落回答问题”

所以 step 1 解决的问题是：

- 训练数据里的“任务”从哪里来

### step 2: `identify_clf_or_not`

这一步的作用是：

- 判断 step 1 生成出来的 instruction 是不是 classification task

之所以要单独做这一步，是因为官方 Self-Instruct 在后面生成实例时：

- classification task 用一套 prompt
- non-classification task 用另一套 prompt

例如：

- 分类任务：情感分类、真假判断、多选题、类别判断
- 非分类任务：翻译、摘要、改写、开放式问答、数学解题

所以 step 2 解决的问题是：

- 这条 instruction 后面应该按哪种方式生成样本

### step 3: `generate_instances`

这一步的作用是：

- 给每条 instruction 生成具体的训练样本

也就是把“任务描述”变成真正的：

- `instruction`
- `input`
- `output`

比如 step 1 可能只生成了一句：

- “判断下面句子的情感是 positive 还是 negative”

这还不能直接拿去训练。

step 3 会继续把它补成类似：

- `Input: This movie is wonderful.`
- `Output: positive`

或者对于非分类任务：

- `Instruction: 把下面句子翻译成中文`
- `Input: Hello world`
- `Output: 你好，世界`

所以 step 3 解决的问题是：

- 怎么把任务变成真正可训练的数据样本

### step 4: `prepare_for_finetuning`

这一步的作用是：

- 清洗 step 3 生成出来的样本
- 去掉无效、重复、异常的数据
- 最后整理成 finetuning 可以直接使用的格式

所以 step 4 解决的问题是：

- 怎么把前面自动生成的中间结果，变成最终训练集

### 一句话总结

官方四步可以概括成：

1. step 1：先造任务
2. step 2：判断任务类型
3. step 3：给任务造具体样本
4. step 4：清洗并整理成训练数据

放到 DataObs 语境里理解，就是：

- `Self-Instruct baseline` 的主要职责不是训练模型本身
- 而是作为一种“数据生成 / 数据蒸馏方法”，为后续 SFT 提供训练数据

## 目录里的文件分别做什么

### `self_instruct.py`

这是 Self-Instruct baseline 的主入口脚本。

它现在负责串起整个本地四步流程：

1. 读取 seed tasks
2. `bootstrap_instructions.py`
3. `identify_clf_or_not.py`
4. `generate_instances.py`
5. `prepare_for_finetuning.py`
6. 写出 DataObs 风格的输出文件

也就是说，实验 pipeline 最终调到的就是它。

## 如何执行

最常用的是通过 experiment pipeline 调它。

推荐分三种跑法理解：

- `正式实验`
  - 直接走 `experiment_pipeline_vllm.py`
  - 一次串 `distill,metrics,sft,sft_eval,grpo,grpo_eval`
- `baseline 回归 / 烟雾测试`
  - 还是走 `experiment_pipeline_vllm.py`
  - 但先只跑 `distill`
  - 再加 `--smoke-num-rows 5`
- `单独调 Self-Instruct 四步`
  - 直接跑 `DataObs/baselines/self-instruct/self_instruct.py`
  - 更适合只看 step 1 -> step 4 本身有没有正常产数据

完整实验命令可以写成：

```bash
python DataObs/scripts/experiment_pipeline_vllm.py \
    --experiment-id exp_self_instruct_gsm8k_qwen3_4b_full \
    --dataset gsm8k \
    --base-model /home/hrh/data/pretrain_models/Qwen3-4B \
    --teacher-model /home/hrh/data/pretrain_models/Qwen3-4B \
    --output-dir /home/hrh/data/COT/experiments \
    --stages distill,metrics,sft,sft_eval,grpo,grpo_eval \
    --distill-method self_instruct \
    --gpu-ids 6 \
    --teacher-batch-size 1 \
    --teacher-max-new-tokens 256 \
    --teacher-tensor-parallel-size 1 \
    --teacher-gpu-memory-utilization 0.8 \
    --bootstrap-num-instructions 100 \
    --sft-epochs 1 \
    --sft-arg data.train_batch_size=4 \
    --sft-arg data.micro_batch_size_per_gpu=2 \
    --sft-arg data.max_length=2048 \
    --eval-batch-size 8 \
    --eval-gpu-memory-utilization 0.5 \
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

如果只是先检查 baseline 本身有没有接通，再把 `--stages` 改成 `distill`，并加上 `--smoke-num-rows 5` 会更稳。

另外要注意：

- 这里显式写了 `--bootstrap-num-instructions 100`
- 这个值现在和官方 step 1 parser 默认值对齐
- 但正式实验时，它通常还需要结合“最终 step 4 保留多少训练样本”再单独调整
- `--teacher-batch-size / --teacher-tensor-parallel-size / --teacher-gpu-memory-utilization`
  - 这几项主要是本地硬件运行参数
  - 它们不是 Self-Instruct 方法本身的官方超参数
  - 所以这几项优先按显存和机器情况调，不要把它们理解成“官方方法设定”

如果你只想单独调这个 baseline 主脚本，也可以直接跑：

```bash
python DataObs/baselines/self-instruct/self_instruct.py \
    --dataset gsm8k \
    --input-file /path/to/prepared_input.parquet \
    --output-file /tmp/self_instruct_output.parquet \
    --model-id /home/hrh/data/pretrain_models/Qwen3-4B \
    --gpu-ids 6 \
    --gen-batch-size 1 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --smoke-num-rows 5
```

如果你只想先做 distill 阶段回归，推荐先用这种最小命令：

```bash
python DataObs/scripts/experiment_pipeline_vllm.py \
    --experiment-id test_self_instruct_distill_smoke \
    --dataset gsm8k \
    --base-model /home/hrh/data/pretrain_models/Qwen3-4B \
    --teacher-model /home/hrh/data/pretrain_models/Qwen3-4B \
    --output-dir /home/hrh/data/COT/experiments \
    --stages distill \
    --distill-method self_instruct \
    --gpu-ids 6 \
    --teacher-batch-size 1 \
    --teacher-max-new-tokens 256 \
    --teacher-tensor-parallel-size 1 \
    --teacher-gpu-memory-utilization 0.8 \
    --bootstrap-num-instructions 100 \
    --smoke-num-rows 5
```

跑完以后，主要看：

- `distill/filtered_sft.parquet`
- `distill/filtered_sft.candidates.parquet`
- `distill/filtered_sft.summary.json`

这里也补一句很重要的口径：

- 对 baseline 做公平比较时，更应该对齐的是 `step 4` 最终保留下来的训练数据规模
- 不一定要机械要求所有方法的某个中间参数名字或者中间产物数量完全一样

## 参数说明

下面只写这套 baseline 里最重要、最常用的参数。

### 通用入口参数

- `--dataset`
  - 数据集名字，例如 `gsm8k`、`math-500`
  - 主要用于 DataObs 侧的数据准备和最终输出标记
- `--input-file`
  - 输入 seed tasks 路径
  - 可以是 `.jsonl` 或 `.parquet`
- `--output-file`
  - 最终输出 parquet 路径
  - 同目录还会写出 `*.candidates.parquet` 和 `*.summary.json`
- `--model-id`
  - 本地 teacher / generator 模型路径
  - 例如 `/home/hrh/data/pretrain_models/Qwen3-4B`
- `--gpu-ids`
  - 使用哪张 GPU，例如 `6`
- `--gen-batch-size`
  - 通用本地 vLLM 生成 batch size
  - 现在默认是 `5`，这样更接近官方 step 2 / step 3 的 `request_batch_size=5`
- `--tensor-parallel-size`
  - vLLM tensor parallel size
- `--gpu-memory-utilization`
  - vLLM 显存占用比例
- `--smoke-num-rows`
  - 只取前多少条 seed task 做 smoke
  - 调试时非常有用

### step 1: bootstrap 参数

- `--bootstrap-num-instructions`
  - 目标要 bootstrap 出多少条 instruction
  - 对应官方 step 1 里的 `--num_instructions_to_generate`
  - 现在默认值已经对齐官方 parser，默认是 `100`
  - 但官方 `generate_instructions.sh` 在全量生成时会显式传更大的值，所以正式实验时建议不要依赖默认值，而是显式指定
- `--bootstrap-num-prompt-instructions`
  - 每次 bootstrap prompt 里放多少条 few-shot instruction
- `--bootstrap-request-batch-size`
  - step 1 一次并行发多少个 bootstrap prompt
  - 现在默认是 `5`，这项已经和官方 step 1 的 `request_batch_size=5` 对齐
- `--bootstrap-max-new-tokens`
  - step 1 单次生成的最大 token 数
  - 现在默认是 `1024`，这项已经和官方 step 1 默认值对齐
- `--bootstrap-temperature`
  - step 1 的采样温度
  - 现在默认是 `0.7`，这项已经和官方 step 1 默认值对齐
- `--bootstrap-top-p`
  - step 1 的 top-p
  - 现在默认是 `0.5`，这项已经和官方 step 1 默认值对齐
- `--bootstrap-similarity-threshold`
  - 相似度过滤阈值
- `--bootstrap-use-clf-seed-tasks-only`
  - 只用 classification seed tasks 做 bootstrap
- `--bootstrap-allow-stub-fallback`
  - 仅调试用
  - 开启后 step 1 失败时允许退回 seed task

### step 2: classification 参数

- `--classification-max-new-tokens`
  - step 2 判断 `Yes/No` 时允许生成的最大 token 数
  - 现在默认是 `3`，这已经贴回官方 repo 的默认写法
- `--classification-request-batch-size`
  - step 2 一次并行判断多少条 instruction
  - 现在默认是 `5`，对应官方 `request_batch_size=5`
- `--classification-allow-heuristic-fallback`
  - 仅调试用
  - 模型判别结果不可解析时，允许退回启发式判断

### step 3: instance generation 参数

- `--instance-request-batch-size`
  - step 3 一次并行生成多少条 instruction 的实例
  - 现在默认是 `5`，对应官方 `request_batch_size=5`
- `--instance-max-instances-to-generate`
  - 官方 step 3 的停止边界会用到这个数
  - 例如 stop 在 `Example 6`
  - 现在默认是 `5`，这项已经和官方 step 3 默认值对齐
- `--instance-max-new-tokens-clf`
  - classification task 的 step 3 最大生成 token
- `--instance-max-new-tokens-gen`
  - non-classification task 的 step 3 最大生成 token

### 当前参数里哪些基本只是兼容保留

下面这些参数目前还在入口里，但对这套四步逻辑不是最核心的：

- `--temperature`
- `--top-p`
- `--max-new-tokens`
- `--correct-threshold`
- `--do-sample`
- `--disable-teacher-filter`

它们主要是为了和 DataObs 现有 distill 接口保持一致，不是 Self-Instruct 四步里最关键的控制参数。

### 作为 baseline 时，最值得先盯住的两个规模参数

如果你后面要在 `/home/hrh/data/open_datasets` 上批量跑，最值得先看的通常是这两个：

- `--bootstrap-num-instructions`
  - step 1 最终想保留多少条 instruction
  - 对应官方 `--num_instructions_to_generate`
- `--instance-max-instances-to-generate`
  - step 3 每条 instruction 最多生成多少个 instance
  - 对应官方 step 3 的 `max_instances_to_generate`

它们分别控制：

- “先造多少条任务描述”
- “每条任务描述最多再展开多少条训练样本”

但真正做 baseline 对齐时，不要只盯这两个数本身，而要继续看：

- step 4 最终到底保留了多少条可训练数据
- 这个数量和其他 baseline 是否处在可比较区间

### `common.py`

这是这一套 baseline 的公共工具文件。

现在主要负责：

- 从 `.jsonl` 或 `.parquet` 读取 seed tasks
- 从 DataObs 的 `prompt` 字段里提取 instruction
- 从 `answer` / `output` / `response` / `reward_model.ground_truth` 里提取答案
- 构造本地 `LocalTeacher`
- 做 batched vLLM 生成
- 写出：
  - `output_file`
  - `*.candidates.parquet`
  - `*.summary.json`

如果你想先理解“DataObs 输入是怎么被转成 Self-Instruct 中间格式”的，建议先看这个文件。

### `templates.py`

这个文件专门放 prompt 模板。

现在里面主要是：

- step 1 的 bootstrap prompt 前缀
- step 2 的 classification 判别模板
- step 3 的 classification / generation instance 模板


### `bootstrap_instructions.py`

这个文件对应官方 step 1：`bootstrap_instructions.py`。

它现在做的事情是：

- 从 seed tasks 里抽 instruction
- 按官方风格拼 bootstrap prompt
- 从 seed instructions 和已生成 machine instructions 里采样 few-shot 示例
- 调本地 teacher / vLLM 生成候选 instruction
- 按官方风格解析编号任务
- 用相似度过滤掉和已有 instruction 太像的候选
- 给每条保留下来的数据记录调试信息，例如：
  - `bootstrap_prompt`
  - `bootstrap_raw_generation`
  - `bootstrap_most_similar`
  - `bootstrap_error`


### `identify_clf_or_not.py`

这个文件对应官方 step 2：`identify_if_classification.py` / `is_clf_or_not` 这一步。

它现在负责：

- 给 instruction 标注 `is_classification`
- 优先走官方 classification 模板
- 尝试用本地 teacher 做判别
- 记录：
  - `classification_prompt`
  - `classification_raw_generation`
  - `classification_detection_method`
  - `classification_detection_reason`


### `generate_instances.py`

这个文件对应官方 step 3：`generate_instances.py`。

它现在负责：

- 根据 instruction 是否是 classification task 走不同 prompt
- 使用官方 step 3 的 generation / classification 模板语义
- 在最外层加一层 Qwen chat wrapper，让本地 chat model 更稳定按官方格式输出
- 调本地 teacher / vLLM 真正生成实例
- 按官方脚本的职责边界，主要保留：
  - `instruction`
  - `raw_instances`
  - `instance_metadata`
  - `instruction_metadata`
  - `most_similar`
  - `avg_similarity_score`

也就是说，这一步现在默认不再承担“把 raw text 解析成最终训练样本”的主职责。

那部分工作已经继续放在 step 4。


### `prepare_for_finetuning.py`

这个文件对应官方 step 4：`prepare_for_finetuning.py`。

它现在负责：

- 解析 raw instance text
- 拆成 `(instruction, input, output)` 三元组
- 过滤无效实例
- 去重
- 重组为 DataObs 当前 SFT 训练能直接消费的行
- 同时尽量保留官方中间字段，便于后面继续对齐

现在产出的关键字段包括：

- `question`
- `answer`
- `gold_answer`
- `prompt`
- `completion`
- `instruction`
- `instance_input`
- `instance_output`
- `augmentation_method`
- `generation_method`
- `teacher_filter_passed`
- `prepare_parser`
- `parsed_instance_count`
- `raw_instances`

所以这一步本质上就是：

- 前三步尽量贴官方 Self-Instruct 中间逻辑
- 最后一步把结果接回 DataObs 的 parquet 训练格式


### `__init__.py`

只是一个包标记文件，本身没有业务逻辑。

