# experiment_pipeline_vllm 中接入 alternative distill baseline 的修改思路

## 背景

`DataObs/scripts/experiment_pipeline_vllm.py` 继承 `DataObs/scripts/experiment_pipeline.py` 里的 `Pipeline`，主要只重写了 `stage_sft_eval()` 和 `stage_grpo_eval()`，因此 alternative baseline 的主要接入点应放在父类 `Pipeline.stage_distill()`。vLLM 版本只需要同步新增 CLI 参数，不需要单独重写数据生成逻辑。

当前 `stage_distill()` 固定调用：

```text
DataObs/lib/data_process/cot_distill_teacher_filter.py
```

目标是把它改成可选择的数据生成方法，并让原来的 teacher correctness filter 与四个 heuristic baseline 并列。

## 新增参数

建议新增：

```text
--distill-method
```

取值：

```text
teacher_correctness_filter
question_rephrasing
answer_augmentation
question_augmentation
reverse_thinking
```

默认值：

```text
teacher_correctness_filter
```

注意：不要叫 `teacher_distill`，因为这个名字不够区分其核心行为。这里的基线更准确地说是 teacher 生成 CoT 后用 correctness reward 过滤，所以命名为 `teacher_correctness_filter`。

## teacher_correctness_filter 的约束

`teacher_correctness_filter` 必须强制 `num_cots = 1`。

原因：如果 teacher correctness filter 允许 `num_cots > 1`，它就变成“同一个问题采样多条 teacher answer，并保留正确答案”，这和 `answer_augmentation` 的行为高度重合。为了让 baseline 之间语义清晰：

- `teacher_correctness_filter`：每个 seed prompt 只生成 1 条 teacher CoT，然后用 correctness filter 保留或丢弃。
- `answer_augmentation`：每个 seed prompt 生成多条 answer/CoT candidates，用 correctness filter 保留正确样本。

实现上建议：

- 在 `stage_distill()` 或参数校验阶段检查：

```text
if distill_method == "teacher_correctness_filter" and teacher_num_samples != 1:
    error
```

也可以更强硬地直接忽略 `--teacher-num-samples` 并传 `--num-cots 1`，但显式报错更不容易误跑实验。

## stage_distill 的 dispatch 设计

把现在硬编码构造 `cot_distill_teacher_filter.py` 命令的逻辑拆成 method dispatch。

建议新增一个内部 mapping：

```text
teacher_correctness_filter -> DataObs/lib/data_process/cot_distill_teacher_filter.py
question_rephrasing        -> DataObs/baselines/heuristics/question_rephrasing.py
answer_augmentation        -> DataObs/baselines/heuristics/answer_augmentation.py
question_augmentation      -> DataObs/baselines/heuristics/question_augmentation.py
reverse_thinking           -> DataObs/baselines/heuristics/reverse_thinking_augmentation.py
```

公共参数可统一传：

```text
--dataset
--input-file
--output-file
--model-id
--gpu-ids
--temperature
--top-p
--gen-batch-size
--max-new-tokens
--tensor-parallel-size
--gpu-memory-utilization
--correct-threshold
--smoke-num-rows
--trust-remote-code
```

其中 `--dataset` 使用：

```text
self.args.distill_dataset or self.args.dataset
```

输入仍走现有逻辑：

```text
self.distill_input_path()
prepare_dataset_data(..., "distill_input")
```

输出仍写到：

```text
self.distill_output
```

这样 `stage_sft()` 不需要改，仍默认读取 `self.distill_output` 作为 SFT 数据。

## method-specific 参数映射

建议最小实现如下。

### teacher_correctness_filter

调用：

```text
cot_distill_teacher_filter.py
```

参数：

```text
--num-cots 1
```

并禁止：

```text
--teacher-num-samples != 1
```

### answer_augmentation

调用：

```text
answer_augmentation.py
```

参数：

```text
--num-augmented-answers <teacher_num_samples>
```

当 `teacher_num_samples > 1` 时，继续沿用现有逻辑自动启用 `--do-sample`。

### question_rephrasing

调用：

```text
question_rephrasing.py
```

建议先映射：

```text
--num-rephrases <teacher_num_samples>
--num-cots 1
```

如果之后需要更细控制，再新增：

```text
--heuristic-num-cots
--rephrase-max-new-tokens
```

### question_augmentation

调用：

```text
question_augmentation.py
```

当前实现是 backward-question generation，不是真正的新题生成。参数映射：

```text
--num-backward-questions <teacher_num_samples>
```

该方法输出的 SFT 训练目标是 backward question 本身，即：

```text
question = Generate the inverse question based on seed question and answer
answer   = generated backward question
```

这个行为目前保留，不在本次 pipeline 接入中修改。

该方法只支持有 backward-question ICL prompt 的数据集；code datasets 禁用。

### reverse_thinking

调用：

```text
reverse_thinking_augmentation.py
```

建议保持每条 seed 一个 backward question，不把 `teacher_num_samples` 映射进去，除非后续明确要多样本 reverse thinking。

可新增参数：

```text
--backward-question-max-new-tokens
--consistency-max-new-tokens
```

当前 `reverse_thinking_augmentation.py` 会把通过一致性检查的样本展开为三类 SFT 训练目标：

```text
forward_reasoning
backward_question
backward_reasoning
```

其中 `backward_question` 也是一个 SFT 训练目标，即“给定 seed question 和答案，生成 backward question”。这个行为与参考代码中的完整 reverse thinking 训练一致，目前保留，不在本次 pipeline 接入中修改。

## parser 修改位置

需要同步修改两个入口的 parser：

```text
DataObs/scripts/experiment_pipeline.py
DataObs/scripts/experiment_pipeline_vllm.py
```

原因：`experiment_pipeline_vllm.py` 复制了一份 `parse_args()`，没有复用父脚本的 parser。如果只改 `experiment_pipeline.py`，vLLM 入口无法识别新参数。

建议新增参数：

```text
--distill-method
--rephrase-max-new-tokens
--backward-question-max-new-tokens
--consistency-max-new-tokens
```

是否新增 `--heuristic-num-cots` 可以后置；最小实现里 question rephrasing 的 CoT 数固定为 1。

## manifest 和 results

`write_manifest()` 中建议记录：

```text
distill_method
```

`stage_distill()` 完成后，`results.json` 中建议记录：

```text
distill_method
distill_output
distill_candidates
distill_summary
```

现有 candidates/summary 命名规则可以保留：

```text
<distill_output stem>.candidates.parquet
<distill_output stem>.summary.json
```

## RL/GRPO 数据流保持不变

当前已有：

```text
--rl-train-from-distill-kept
```

它的逻辑是：读取 distill output 的 `source_index`，回到原始 seed parquet 中选出被保留的原始 prompt 作为 RL train。

本次接入 alternative baseline 时，RL/GRPO 逻辑完全不变：

```text
if args.rl_train_from_distill_kept:
    仍使用现有 source_index 回选逻辑
else:
    仍使用 args.rl_train_data 或 dataset 默认 rl_train
```

也就是说，所有 baseline 方法生成的 distill output 只作为 SFT 训练数据使用，不新增任何“从 distill output prompts 构造 GRPO train data”的逻辑，也不新增 `--rl-train-from-distill-output-prompts` 之类的参数。

后果需要在实验解释中明确：如果某个 baseline 产生了新 prompt，例如 `question_rephrasing`、`question_augmentation` 或 `reverse_thinking` 中的 backward question，这些新 prompt 当前只参与 SFT，不参与 GRPO。GRPO 阶段仍沿用 pipeline 既有数据选择机制。

## 数据集支持和失败策略

`question_augmentation` 与 `reverse_thinking` 当前依赖 backward-question ICL prompt，不支持所有数据集。

建议先不在 pipeline 里复制完整支持矩阵，而是让底层脚本报错，并把错误写到 stage log。可选增强是提前在 pipeline 里给出更友好的错误信息。

当前需要注意：

- code datasets 对 backward-question 方法禁用。
- `aqua_rat` 和 `numinamath` 当前没有启用 backward-question ICL prompt。

## 最小实现顺序

1. 在两个 parser 中增加 `--distill-method` 和少量 method-specific token 参数。
2. 在 `write_manifest()` 中记录 `distill_method`。
3. 重构 `Pipeline.stage_distill()` 为 method dispatch。
4. 对 `teacher_correctness_filter` 强制 `teacher_num_samples == 1`。
5. 保持 `stage_grpo()` 和 `--rl-train-from-distill-kept` 的现有行为，不为 alternative baseline 增加新的 RL 数据构造逻辑。
