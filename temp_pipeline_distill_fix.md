# experiment_pipeline_vllm distill baseline 修复审阅稿

## 背景结论

当前 `DataObs/scripts/experiment_pipeline.py` 的 `stage_distill()` 在进入 distill 脚本前，会调用 `prepare_dataset_data(..., "distill_input")`。这个 helper 实际走的是 eval 数据准备逻辑：

```text
DataObs/lib/evaluation/prepare_eval_data.py
```

如果输入 raw parquet 没有 `prompt/reward_model/data_source` 三列，它会调用：

```text
verl/utils/eval/apply_prompt_template.py
```

也就是说，distill 阶段拿到的不是原始问题，而是已经套过 eval prompt template 的问题。随后 heuristic 脚本里的 `load_input_rows()` 又从 `prompt` 的最后一个 user message 里取 `question`，导致 question rephrasing、reverse thinking、question augmentation、answer augmentation 都可能把 eval prompt 当成原始问题使用。

以 GSM8K 为例，原始问题 `<question>` 会先变成：

```text
<question>
Please reason step by step, and put your final answer within \boxed{}.
```

然后再被 heuristic baseline 塞进自己的 prompt 中。这是当前 smoke 中 teacher 没有按预期工作的主要原因。

## 问题 1：distill 阶段复用了 eval prompt template

### 具体问题

`stage_distill()` 当前使用：

```text
self.prepare_dataset_data(distill_input_source, "distill_input")
```

该函数内部调用 `prepare_parquet()`，而 `prepare_parquet()` 固定调用 eval helper：

```text
DataObs/lib/evaluation/prepare_eval_data.py
```

这会把 raw dataset 转成 eval prompt，而不是 distill prompt。

### 影响

所有 distill method 都可能拿到带 eval 指令的 `question`：

- `teacher_correctness_filter` 会在 eval prompt 后再追加 distill reasoning suffix。
- `answer_augmentation` 会基于 templated question 采样 CoT。
- `question_rephrasing` 会要求 teacher rephrase 一个已经包含 CoT 指令的问题。
- `question_augmentation` / `reverse_thinking` 会把 CoT 指令混入 backward-question prompt。

这会污染 teacher 输入，也会污染最终 SFT 数据里的 `question` 字段。

### 修改方案

新增 distill 专用数据准备 helper，例如：

```text
DataObs/lib/data_process/prepare_distill_data.py
```

它的职责类似 `apply_prompt_template.py`，但输出目标是 distill seed 数据，而不是 eval prompt 数据。

建议输出列：

```text
prompt        # [{"role": "user", "content": <distill_question>}]
reward_model  # {"ground_truth": <gold_answer>}
data_source
raw_question  # optional，但建议保留
```

其中 `<distill_question>` 是适合 baseline 使用的干净问题文本：

- `gsm8k`: raw `question`
- `math` / `math-500` / `numinamath`: raw `problem`
- `arc-challenge`: `question + choices`
- `commonsenseqa`: `question + A/B/C/D/E choices`
- `aqua_rat`: `question + options`
- `strategyqa`: raw yes/no question，是否加 `Yes or no:` 需要和 heuristic prompt 约定统一
- code datasets: 保留 code task 必要上下文；但 reverse/question backward 类方法本来不支持 code

然后在 pipeline 中新增 `prepare_distill_data()` 调用路径。`distill` stage 使用 distill helper，`eval` stage 继续使用 eval helper。

### 验收点

用 GSM8K raw parquet 跑 `--stages distill --smoke-num-rows 5` 后，distill 输入中的 `question` 不应包含：

```text
Please reason step by step
\boxed{}
Answer:
```

最终 distill 输出的 `question/original_question/backward_question` 等字段也不应继承 eval prompt 指令，除非该字段本身就是某个 baseline 明确生成的训练任务 prompt。

## 问题 3：teacher_correctness_filter 也应 follow 干净 distill 输入

### 具体问题

`DataObs/lib/data_process/cot_distill_teacher_filter.py` 当前 `_prepare_generation_rows()` 从 input `prompt` 中取 question，并调用 `_append_reasoning_suffix()` 给最后一个 user message 追加 distill instruction。

如果 input `prompt` 已经是 eval prompt，就会变成：

```text
<question>
<eval reasoning instruction>

<distill reasoning instruction>
```

### 影响

teacher correctness filter 作为最基础的 CoT distill baseline，也不是在原始问题上生成 teacher CoT。

### 修改方案

pipeline 给 teacher filter 传 distill helper 输出的干净 `prompt`，然后现有 `_append_reasoning_suffix()` 继续追加 distill instruction。

### 验收点

GSM8K teacher filter 的生成 prompt 中只出现一次 reasoning/final-answer 指令，并且不是 eval template 的指令。

## 问题 4：answer_augmentation 当前默认不是 original MetaMath prompt，且 pipeline 没暴露开关

### 具体问题

`DataObs/baselines/heuristics/answer_augmentation.py` 支持：

```text
--use-original-metamath-prompt
```

但 `experiment_pipeline.py` / `experiment_pipeline_vllm.py` 没有暴露对应参数，也不会传给 answer augmentation。

当前默认调用的是：

```text
build_reasoning_prompt(dataset, question)
```

而旧版 `../Baseline-Heuristics-CoT/rephrase_questions.py` 的 answer augmentation 使用的是短 two-shot MetaMath prompt：

```text
Q: <question>
A: Let's think step by step.
```

### 影响

如果实验目标是复现 original answer augmentation baseline，当前 pipeline 默认 setting 不一致。

### 修改方案

新增 pipeline 参数：

```text
--answer-aug-use-original-metamath-prompt
```

当 `--distill-method answer_augmentation` 且该参数开启时，向脚本追加：

```text
--use-original-metamath-prompt
```

是否默认开启需要实验语义决定：

- 若目标是贴近 `Baseline-Heuristics-CoT/rephrase_questions.py`，默认应开启。
- 若目标是 DataObs dataset-specific distill，默认可保持关闭，但 experiment id/manifest 里必须明确记录。

修改时明确列出哪些数据集有能够通过这个参数打开的 prompt（需要对 pipeline 支持的全部数据集运行 eval），哪些没有（例如参考代码不支持的 coding 数据集）。

### 验收点

`commands.jsonl` 中能看到 answer augmentation 是否传入 `--use-original-metamath-prompt`，summary 中的 `used_original_metamath_prompt` 与命令一致。

## 问题 5：question_rephrasing 的 `--num-cots` 没在 pipeline 中暴露

### 具体问题

`DataObs/baselines/heuristics/question_rephrasing.py` 支持两个数量参数：

```text
--num-rephrases
--num-cots
```

当前 pipeline 只把：

```text
--teacher-num-samples -> --num-rephrases
```

并固定：

```text
--num-cots 1
```

### 影响

无法通过 pipeline 运行“每个 rephrased question 采多条 CoT”的 setting。

### 修改方案

新增参数，例如：

```text
--rephrase-num-cots
```

默认值保持 `1`，以兼容当前行为。`question_rephrasing` dispatch 中改为：

```text
--num-rephrases <teacher_num_samples>
--num-cots <rephrase_num_cots>
```

`do_sample` 自动开启逻辑也要考虑：

```text
teacher_num_samples > 1 or rephrase_num_cots > 1
```

### 验收点

dry run 时命令能正确传 `--num-cots`；当 `--rephrase-num-cots > 1` 且用户没传 `--teacher-do-sample` 时，pipeline 自动追加 `--do-sample` 或显式报错。

## 问题 6：`--dtype` 没在 pipeline 中暴露

### 具体问题

teacher filter 和 heuristic common args 都支持：

```text
--dtype auto|float16|bfloat16|float32
```

但 pipeline 只暴露了：

```text
--teacher-tensor-parallel-size
--teacher-gpu-memory-utilization
```

没有 `--teacher-dtype`。

### 影响

无法通过 pipeline 控制 vLLM teacher load dtype。遇到显存、兼容性或性能问题时，只能绕开 pipeline 直接跑底层脚本。

### 修改方案

在两个入口 parser 中新增：

```text
--teacher-dtype
```

取值：

```text
auto,float16,bfloat16,float32
```

在 `stage_distill()` 公共参数中传：

```text
--dtype <teacher_dtype>
```

### 验收点

dry run 命令包含 `--dtype`；底层 summary 的 generation/config 能记录 dtype。如果当前 summary 未记录 dtype，也应补上。

## 问题 8：reverse thinking 固定每个 seed 一个 backward question，未暴露多样本 reverse setting

### 具体问题

当前 reverse thinking 脚本内部每个生成阶段都固定 `n=1`：

```text
backward question n=1
forward reasoning n=1
backward reasoning n=1
consistency n=1
```

pipeline 也没有把 `--teacher-num-samples` 映射到 reverse thinking。

### 影响

这符合之前“reverse thinking 只生成一个思维过程”的最小实现，但如果希望评估采样版 reverse thinking，就无法通过 pipeline 控制数量。

### 修改方案

在文档和 manifest 中明确 `teacher_num_samples` 对 reverse thinking 无效。

**当前不做多样本，文档中明确说明无效，然后传 teacher num samples 时报错**

### 验收点

当前最小实现下，用户传 `--teacher-num-samples > 1` 且 `--distill-method reverse_thinking` 时，pipeline 应该报错，避免误以为生效。

## 问题 9：distill prompt 与 eval prompt 的 final-answer 格式不一致需要显式记录

### 具体问题

eval prompt template 和 heuristic baseline prompt 使用的答案格式并不完全一致。例如 GSM8K：

- eval zeroshot: 要求 final answer within `\boxed{}`
- heuristic baseline `gen_reasoning_prompt["GSM8K"]`: 要求 `"The answer is: $\boxed{[ANS]}$"`
- teacher filter `REASONING_SUFFIX["gsm8k"]`: 要求 `#### <number>`

### 修改方案

确认一下当前 gsm8k reward function 的默认行为，如果是解析 boxed 那么明确提出（需要修改 teacher filter 的 reasoning suffix）。

### 验收点

抽样检查 candidates 中 teacher answer 的 final-answer 格式，确认能被对应 reward function 解析。

## 问题 10：pipeline manifest 对 method-specific setting 记录不够显式

### 具体问题

pipeline 的 manifest 会记录 `args`，但对 method-specific 语义没有结构化字段。例如：

- answer augmentation 是否用了 original MetaMath prompt
- question rephrasing 的 `num_rephrases` 和 `num_cots`
- reverse thinking 是否固定 `n=1`
- question augmentation 到底是 synthetic question 还是 backward-question
- distill 输入是 eval-prepared 还是 distill-prepared

### 影响

后续看 experiment id 或 manifest 时，不容易判断实验真实 setting，容易把不同 baseline 混在一起比较。

### 修改方案

在 `stage_distill()` 后写入结构化 `self.results["distill_config"]`，或在 manifest 中新增：

```text
distill_config:
  method:
  input_prepare_mode:
  prompt_source:
  question_source:
  method_specific:
```

示例：

```text
method: question_rephrasing
input_prepare_mode: distill_raw
question_source: raw_question
method_specific:
  num_rephrases: 4
  num_cots_per_rephrase: 1
```

### 验收点

每次 pipeline 结束后，不看命令行也能从 manifest/results 判断 distill baseline 的完整语义。

## 建议修复顺序

1. 先新增 distill 专用 prepare helper，并让 `stage_distill()` 使用它。
3. 暴露低风险参数：`--teacher-dtype`、`--answer-aug-use-original-metamath-prompt`、`--rephrase-num-cots`。
5. 补 manifest/results 中的 distill structured config。
6. 最后做 GSM8K smoke，检查 prompt 文本、候选数据、summary 和 SFT 数据字段。

具体修复操作根据上面的具体问题来做（删除了一些不需要改的点，因此会缺少几点），最后的 smoke 需要 gpu 资源，待修复完成后交给我执行。

## 不建议的修法

不建议只在 `load_input_rows()` 里用字符串删除：

```text
Please reason step by step...
\boxed{}
Answer:
```

原因：

- 不同 dataset 的 eval template 不同，字符串规则很快会变脆。
- MCQ choices/code tests 等数据不是简单删除 suffix 就能恢复。
- 已经 prepared 的 parquet 可能丢掉了原始字段，无法可靠还原。

正确方向是让 distill 阶段从 raw data 构造自己的输入格式，而不是复用 eval prompt。

## 额外：backward question 相关

backward question 的 reasoning 是否保留要看情况：
- 作为“生成 backward question 任务”时：应当保留，可以指导 student 如何生成 backward question。
- 作为“回答 backward question 任务”时：应当去除，以避免污染模型输入。

因此这部分还需要再做一些处理。此外，每个 method 具体发送给 teacher 和 student 的 prompt 也需要检查。这部分不跟着上面一起修复，但要在修复完成后作为”下一步内容“提示给我。



# 现在要做的

1. temp_pipeline_distill_fix.md:83
  strategyqa 是否加 Yes or no: 需要和 heuristic prompt 统一。
  现状：我现在的 distill helper 保留 raw question，不主动加 Yes or no:。最好加上以和 eval / student prompt 保持一致。

2. temp_pipeline_distill_fix.md:170
  answer_augmentation 的 original MetaMath prompt 是否默认开启需要实验语义决定。
  现状：我实现为默认关闭，通过 --answer-aug-use-original-metamath-prompt 显式开启，并记录到 manifest/results。最好做成在不开启时报一个 warning（pipeline 层报，不然运行时看不到报错）。

3. temp_pipeline_distill_fix.md:175
  “明确列出哪些数据集能打开 original MetaMath prompt，哪些不能”。
  现状：我代码里禁止 code datasets 使用该开关，非 code datasets 允许。但还没有按文档说的“对 pipeline 支持的全部数据集
  运行 eval/smoke”逐个验证，所以这仍是后续验收项。
  
  **这里的关键问题是：哪些数据集有原生的 original prompt，哪些数据集没有但是可以借用别的数据集的 prompt，哪些数据集没有且不能借用 / 不支持？此外，对不支持的数据集，它们现在用的是什么 prompt？** 对所有的 method 回答这个问题！
  还有，这个 metamath prompt 在参考代码中在哪？我没找到。

4. temp_pipeline_distill_fix.md:309
  distill prompt 和 eval prompt 的 final-answer 格式不一致。
  现状：我查了 GSM8K reward，当前 compute_score() 用的是 OC-boxed，没有 \boxed{} 时会 fallback 到最后一个数字。因此
  teacher filter 的 #### <number> 大概率可被解析，但它不是和 eval prompt 完全一致的格式。最好把 teacher filter 改掉，变成 \boxed{}。

7. temp_pipeline_distill_fix.md:398
  backward question reasoning 保留/去除，以及逐 method 检查 teacher/student prompt。
  现状：这是明确的下一步，没有在刚才那轮修复里处理。尤其要看 question_augmentation 和 reverse_thinking 输出给 SFT 的
  三类 row：生成 backward question、回答原问题、回答 backward question，分别是否带了不该出现在 student input 里的
  reasoning。


1. question rephrasing 的 SFT 数据严重错误。
  路径：/data/hrh/COT/experiments/test_smoke_gsm8k_question_rephrasing_qwen3_8b_n4/distill/filtered_sft.parquet
  question 字段 20/20 都包含 <think>，也就是把 teacher 的重写推理过程当成了“改写后的问题”。这会导致 student 的输入不
  是数学题，而是一大段“我该如何 rephrase”的推理。answer 也 20/20 带 <think>。这说明 clean_rephrase() 仍没有正确从
  Qwen thinking 输出中抽取最终改写问题。
  这里需要改掉：保证 question rephrasing 的清洗去掉 sft input 的 think，answer 保留 think 作为思维链训练素材。具体的清洗规则看下面。

2. backward question 相关的清洗仍然坏，和你文档里新增的点一致。
  question_augmentation candidates 中：
  - raw_backward_question: 20/20 含 <think>
  - backward_question: 19/20 含 <think>
  - 最终 SFT answer: 19/20 含 <think>

  reverse_thinking candidates 中：
  - raw_backward_question/backward_question/backward_reasoning/consistency_reasoning: 5/5 都含 <think>
  - 最终 SFT backward_question: 3/3 含 <think>
  - 最终 SFT question: 1/3 含 <think>

  这里最危险的是 reverse thinking 的 backward_reasoning 样本：它把“带 reasoning 的 backward question”作为要回答的输
  入，正好是你说的“回答 backward question 任务时应去除 reasoning”的污染。


## 完整的清洗（按照类别）

- teacher filter 和 answer augmentation: 不清洗，保留 think 用于训练 cot
- question rephrasing：
  - 生成 rephrased question：清洗，只保留最终的问题，如果问题中含有答案也要一并去掉（先检查一下现在输出的问题会不会含有答案），这个清洗需要在把 question 喂给 teacher 做 reasoning 前做，避免污染 teacher 的回答
    - 去答案可以复用 backward question 的，但要先修，让它能够去除 math / yesno 答案
  - 回答 rephrased question：不清洗，保留 think
- question augmentation：保留 think（目的是生成 backward question），可以选择去掉 answer（如果参考代码去掉了，参考代码看 /home/hrh/Baseline-Heuristics-CoT）
- reverse thinking：
  - forward reasoning：不清洗
  - backward question：同 question augmentation
  - backward reasoning：作为输入的 question 需要按照 question rephrasing 的标准清洗，且要在喂给 teacher 前清洗；输出的 reasoning 不清洗



1. 统一各数据集的 distill/student prompt 约定
      - strategyqa 的 distill input 不应只是 raw question，建议追加 Yes or no:，和 eval/student prompt、heuristic prompt
        语义统一。

      - 对 pipeline 支持的全部数据集列一个 prompt 支持矩阵：每个 method 在每个 dataset 上用的是原生 baseline prompt、借
        用 prompt、DataObs fallback prompt，还是不支持。

      - 需要覆盖：teacher_filter、answer_augmentation、question_rephrasing、question_augmentation、reverse_thinking。

  2. 明确 original MetaMath prompt 的语义和默认行为
      - 参考位置是 /home/hrh/Baseline-Heuristics-CoT/rephrase_questions.py:18，build_answer_prompt() 返回两条 math few-
        shot + Q: <question>\nA: Let's think step by step.。

      - 当前实现默认关闭 --answer-aug-use-original-metamath-prompt 是合理的 DataObs 取向，但如果目标是复现 original
        baseline，就应默认开启或至少 pipeline 层 warning。

      - 需要明确哪些 dataset 能“原生”使用该 prompt。严格看，它是 GSM8K/math word problem 风格；对 MCQ、StrategyQA、code
        都不是原生 prompt。当前“非 code 都允许”过宽，至少需要 warning 或支持矩阵说明。

  3. 统一 final-answer 格式，尤其 GSM8K teacher filter
      - 当前不一致：
          - eval: \boxed{}
          - heuristic gen_reasoning_prompt["GSM8K"]: The answer is: $\boxed{[ANS]}$
          - teacher filter REASONING_SUFFIX["gsm8k"]: #### <number>

      - 修改点：把 teacher filter 的 GSM8K suffix 改成 boxed 风格，或至少显式记录并验证 reward 能稳定解析。
      - 验证点：抽样检查 candidates 的 final answer 格式和 teacher_score 是否一致。

  4. 建立统一的 teacher output 清洗策略，但按用途区分
      - teacher_filter / answer_augmentation：SFT answer 不清洗，保留 <think> 作为 CoT 训练素材。
      - question_rephrasing：
          - rephrase 阶段输出必须清洗成干净问题，不能把 <think> 放进 SFT input，也不能把污染问题再喂给 teacher 做
            reasoning。

          - 要检查 rephrased question 是否携带答案；如果携带，需要去掉。
          - 回答 rephrased question 的 teacher reasoning 不清洗，保留 <think>。

      - question_augmentation：
          - 如果训练目标是“生成 backward question”，SFT answer 可以保留生成过程 reasoning。
          - 但需要决定最终可复用的 backward_question 字段是否去掉答案标注。建议保留 raw/generation response，同时另存
            clean backward question。

      - reverse_thinking：
          - forward_reasoning answer 不清洗。
          - backward_question 生成任务的 answer 可按 question augmentation 策略处理。
          - backward_reasoning 的 input question 必须清洗：去 <think>、去生成推理、去 backward question 自己的答案标注，
            并且要在喂给 teacher 做 backward reasoning 前完成。

          - backward reasoning 的输出不清洗。

  5. 修复 backward question / rephrase 的具体 parser
      - clean_rephrase() 需要支持 Qwen thinking 输出：去掉 <think>...</think>，从 OUTPUT: / final field / 最后一条问题中
        抽取干净 question。

      - remove_backward_answer() 需要支持 MCQ、math、yes/no，不只是 The correct answer is (A).。
      - _strip_output_prefix() 也要更稳：当前只找 OUTPUT: 后全部内容，遇到 thinking block、多段 OUTPUT:、后续解释时容易
        保留过多文本。

  6. 逐 method 检查 teacher prompt 和 student prompt
      - teacher prompt 已经需要列出并验证。
      - student prompt 也要看最终 SFT question/answer：
          - answer augmentation：原问题 -> CoT answer
          - question rephrasing：干净 rephrased question -> CoT answer
          - question augmentation：生成 backward question task -> backward question generation response
          - reverse thinking 三类 row：原问题 reasoning、生成 backward question、回答 backward question

      - 验收标准：任何作为 student input 的字段都不能带不该出现的 reasoning 或答案泄漏。

  7. 验收 smoke / 全数据集验证
      - GSM8K smoke：检查 question/original_question/backward_question 不继承 eval prompt。
      - 检查 SFT input 不含 <think>，只允许 SFT output 在策略允许的位置含 <think>。
      - 检查 commands.jsonl、summary、manifest/results 能反映 method-specific config。
      - 对 pipeline 支持的全部 dataset 做至少 prompt 构造 / smoke 级验证，确认支持矩阵不是纸面结论。


  8. backward question prompt 对 GSM8K 有 MCQ wording 污染。
     当前 prompt 说 “same number of inverse answer choices / four answer choices”，但 GSM8K ICL 是开放数学题。这会误导
     teacher，把有问题的几行做成按问题类型填充，分为mcp（保留原本的prompt！）/ math / yesno三类，coding 和 aquarat / numinamath 仍然保持不支持 backward question。

  9. teacher 输出用于 scoring 和用于 SFT 可能需要分开。
     即使 SFT answer 保留 <think>，reward scoring 最好基于可见 final answer 或经过一致规则抽取的文本，否则数字 fallback
     可能被 reasoning 中的中间数字干扰。这部分只需要验证一下当前是否符合预期，如果有问题就在修改完后提出来，不要直接基于这点修改代码。

  10. train-time <think> 与 eval-time enable_thinking=false 的策略需要明确。
     如果训练保留 <think>，但 eval 禁用 thinking，要确认这是预期训练目标，而不是训练/推理格式不一致。
     确认一下当前 eval 的默认行为是不是禁用了 thinking，并在修改后汇报。