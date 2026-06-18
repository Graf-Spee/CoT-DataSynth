# 实验准备 TODO 与现有代码缺口

本文档用于把当前想跑的蒸馏、SFT、RL、DataObs 相关实验拆成可执行 TODO，并记录当前代码已经具备的能力与还没有做到的部分。目标不是一次性跑所有组合，而是先保证每条实验链路可以稳定复现、记录和汇总。

## 0. 总体实验主线

建议把实验统一拆成下面四层，不要直接枚举所有组合：

1. 数据层：seed data、teacher distilled data、不同 quality/difficulty/diversity/size/temperature/n_samples 的数据版本。
2. SFT 层：固定 student/base model 与训练超参，在每个数据版本上训 SFT。
3. RL 层：从每个 SFT checkpoint 启动 GRPO/RL，记录最终表现和训练动态。
4. 分析层：把数据指标、SFT 表现、RL 表现合并，做相关性和消融比较。

每一个实验 run 至少需要记录：

- `experiment_id`
- `student_model`
- `teacher_model`
- `dataset_name`
- `data_variant`
- `seed_data_source`
- `reasoning_source`: `none` / `human` / `teacher` / `self`
- `teacher_temperature`
- `teacher_num_samples`
- `teacher_filter`
- `train_size`
- `sft_checkpoint`
- `sft_eval_metrics`
- `rl_checkpoint`
- `rl_eval_metrics`
- `rl_training_metrics`: entropy、reward curve、validation curve、convergence step、训练步数、失败原因
- `data_metrics`: quality、difficulty、diversity、entropy、PPL、IFD、size、coverage

## 1. P0：先保证链路能跑

- [ ] 修复 DataObs 训练阶段参数 bug：`DataObs/scripts/data_obs_pipeline.py` 调 `run_all_trainings(..., eval_data_path=...)`，但 `TrainingPipeline.run_all_trainings` 没有 `eval_data_path` 参数，当前会 TypeError。
- [ ] 修复 DataObs evaluation 参数语义：`DataObs/lib/evaluation/eval_dataobs.sh` 第 3 个参数要的是 dataset name，如 `gsm8k` / `math-500`，但 DataObs pipeline 当前用 `eval_data_path` 调 `run_evaluation`，会导致 dataset case 匹配失败。
- [ ] 修复 DataObs parallel pipeline 调用签名：`TrainingPipelineParallel.prepare_training_configs` 的签名和 `data_obs_pipeline.py` 调用方式不一致，`--parallel` 当前高概率不能正常跑。
- [ ] 明确 SFT 数据格式转换：`DataObs/lib/training/sft_dataobs.sh` 使用 `question`/`answer` 列；RL/eval 使用 `prompt`/`data_source`/`reward_model` 列。需要为每个数据版本保存两份或一个统一转换脚本。
- [ ] 增加 run manifest：每次蒸馏、SFT、RL、eval 都写一个 `manifest.json`，记录模型、数据、超参、代码 commit、输出路径。
- [ ] 增加实验总表 `experiments.csv` 或 `runs.jsonl`，后续分析不要从目录名猜实验条件。
- [ ] 为每个入口加 smoke test：小数据 8-32 条，确认 distill -> SFT -> eval -> GRPO -> eval 全链路可完成。

验收标准：

- [ ] 单个 `gsm8k` 或 `commonsenseQA` 小样本数据能跑完 distill/SFT/eval。
- [ ] 一个 SFT checkpoint 能成功启动 `DataObs/lib/training/grpo_from_sft.sh`。
- [ ] 一个 GRPO 输出能用 `DataObs/lib/evaluation/eval_grpo_dataobs.sh` 或 `scripts/eval.sh` 评测。
- [ ] 结果表能自动合并 SFT accuracy、RL accuracy、RL entropy/收敛信息、数据指标。

## 2. 已有代码能力

### 2.1 评测

已有 `scripts/eval.sh`，支持：

- `arc-challenge`
- `aqua_rat`
- `commonsenseQA`
- `gsm8k`
- `livecodebench`
- `humaneval`
- `humanevalplus`
- `math`
- `math-500`
- `mbpp`
- `mbppplus`
- `numinamath`
- `strategyQA`
- `bfcl`

普通数据集流程是先生成 `generated/responses.parquet`，再用 `verl/trainer/main_eval.py` 和对应 `verl/utils/reward_score/*.py` 打分。BFCL 单独走 `scripts/eval_bfcl_dataobs.py`。

已有 `DataObs/lib/evaluation/summarize_eval_results.py` 可以从 `responses_labeled.json/parquet` 汇总：

- example accuracy / pass@1
- response accuracy
- mean reward
- 错例展示

### 2.2 SFT

已有：

- `DataObs/lib/training/sft_dataobs.sh`
- `DataObs/lib/training/sft_dataobs_lora.sh`
- `scripts/sft.sh`

DataObs 版本默认使用：

- `data.prompt_key=question`
- `data.response_key=answer`
- LoRA rank 8
- 默认 15 epoch

这适合训练 teacher distilled CoT 数据，但不直接适配原始 RL parquet 的 `prompt`/`reward_model` 格式。

### 2.3 RL / GRPO

已有 `DataObs/lib/training/grpo_from_sft.sh`，可以从 SFT checkpoint 启动 GRPO：

- 自动解析 `checkpoint-last` 或最新 `global_step_*`
- 支持 LoRA SFT checkpoint merge
- 使用 `verl.trainer.main_ppo`
- 默认 `algorithm.adv_estimator=grpo`
- 默认 rollout n=8
- 使用 `verl/utils/reward_score/reward_fn_router.py`

当前 reward router 支持：

- `openai/gsm8k`
- `math`
- `math-500`
- `numinamath`
- `aqua_rat`
- `ai2_arc`
- `strategyQA`

注意：router 当前不支持 `commonsenseQA`、`mbpp`、`livecodebench`、`humaneval` 等作为 RL 训练 reward source，除非数据里的 `data_source` 已经映射到现有 key 或补 router。

### 2.4 教师蒸馏

已有 `DataObs/lib/data_process/cot_distill_teacher_filter.py`：

- vLLM 本地 teacher 生成
- 支持 temperature、top-p、num_cots
- 支持 teacher correctness filter
- 输出 SFT 格式：`question`、`answer`、`gold_answer`

当前已扩展支持：

- `arc-challenge` / `ai2_arc`
- `aqua_rat`
- `commonsenseqa`
- `gsm8k`
- `math` / `math-500` / `numinamath`
- `strategyqa`
- `mbpp` / `mbppplus`
- `humaneval` / `humanevalplus`
- `livecodebench`

不支持 `bfcl`，因为 BFCL 不是普通 parquet + reward function 评测链路。

注意：代码类数据集的 teacher filter 会执行生成代码对应的测试，速度更慢，也更依赖执行环境；大规模蒸馏前必须先 `--smoke-num-rows 8/32`。

### 2.5 数据指标 / DataObs

已有 `DataObs/scripts/data_obs_pipeline.py`，能做：

- 数据 split
- statistics
- quality
- diversity
- entropy
- 可选 PPL/IFD
- SFT training
- optional evaluation
- correlation analysis

已有指标说明在 `DataObs/docs/metrics.md`。

当前可以回答部分问题：

- 数据质量和 SFT 表现的相关性
- diversity/entropy/PPL/IFD 和 SFT 表现的相关性
- split 级别数据指标与训练表现的相关性

但还不能完整回答：

- SFT 表现和 RL 表现相关性
- RL entropy、收敛 step、RL gain 与数据指标相关性
- teacher size / temperature / num_samples 对蒸馏数据质量和后续训练的系统影响

## 3. 你的实验问题如何落成 TODO

### 3.1 Teacher size / 自蒸馏 / 大教师是否有意义

问题：

- 自蒸馏有没有意义？
- 更大的教师有没有意义？
- 自己模型蒸馏是否有意义？

TODO：

- [ ] 定义 student model，例如固定 `Qwen2.5-0.5B-Instruct`。
- [ ] 定义 teacher 列表：`0.5B/self`、`7B`、`32B`、更大模型。
- [ ] 对同一 seed dataset、同一 prompt 模板、同一 temperature、同一 num_samples 生成蒸馏数据。
- [ ] 保存 teacher 通过率、保留样本数、平均长度、答案正确率、重复率。
- [ ] 每个 teacher 版本训练一个 SFT。
- [ ] 每个 SFT 做统一 eval。
- [ ] 每个 SFT 启动统一 GRPO。
- [ ] 计算 `SFT gain = SFT_score - base_score`。
- [ ] 计算 `RL gain = RL_score - SFT_score`。
- [ ] 比较 teacher size 与 SFT/RL gain 的关系。

当前缺口：

- [x] `cot_distill_teacher_filter.py` 已扩展到目标主数据集。
- [ ] 缺少 teacher model registry。
- [ ] 缺少批量蒸馏调度脚本。
- [ ] 缺少 distill manifest 和统一结果表。

### 3.2 不同蒸馏数据质量与 SFT 的关系

问题：

- 怎么从后验判断蒸馏数据有没有意义？

建议指标：

- teacher filter pass rate
- verifier reward
- response length
- format validity
- answer coverage
- duplicate rate
- diversity
- entropy
- PPL/IFD
- student perplexity on distilled answers
- SFT 后 test accuracy
- SFT 相对 base 的 gain

TODO：

- [ ] 对每个蒸馏数据版本运行 DataObs 指标。
- [ ] 训练 SFT 并评测。
- [ ] 合并 `data_metrics_summary.csv` 和 SFT eval 结果。
- [ ] 做 Spearman/Pearson correlation。
- [ ] 对 top/bottom 数据样本做人眼抽查。

当前缺口：

- [ ] DataObs 目前更偏 split 级分析，还没有“跨实验数据版本”的总表分析脚本。
- [ ] 缺少蒸馏数据质量专用指标，如 teacher pass rate、verifier score distribution、format error rate。

### 3.3 Seed data 是 prompt 还是 human reasoning

问题：

- Reasoning 是否对 RL 有帮助？

对照组：

- A: prompt-only seed，teacher 生成 reasoning。
- B: human-written reasoning seed，直接 SFT。
- C: human reasoning seed，再 teacher rewrite/augment。
- D: no-CoT answer-only SFT。

TODO：

- [ ] 定义统一数据 schema：`question`、`answer`、`reasoning_source`、`gold_answer`、`data_source`、`reward_model`。
- [ ] 写 prompt-only -> SFT 格式转换。
- [ ] 写 human reasoning -> SFT 格式转换。
- [ ] 每组训练 SFT。
- [ ] 每组从 SFT 启 RL。
- [ ] 比较 SFT score、RL final score、RL gain、RL convergence step、RL entropy。

当前缺口：

- [ ] 当前 SFT 脚本能训 `question`/`answer`，但没有系统区分 answer-only、human reasoning、teacher reasoning。
- [ ] 当前结果记录没有保存 `reasoning_source`。

### 3.4 SFT 表现和 RL 表现相关性

问题：

- 最强 SFT 是否最适合 RL？

TODO：

- [ ] 对每个 SFT checkpoint 记录统一 eval 分数。
- [ ] 用相同 RL 数据、相同 RL 超参从每个 SFT checkpoint 启 GRPO。
- [ ] 记录 RL final score、best score、gain、收敛 step、entropy 曲线。
- [ ] 计算 SFT score vs RL final score。
- [ ] 计算 SFT score vs RL gain。
- [ ] 计算 SFT score vs convergence speed。

当前缺口：

- [ ] 还没有自动批量从 DataObs SFT outputs 启动 GRPO 的 pipeline。
- [ ] 还没有 GRPO 训练日志解析器，尤其是 entropy、best step、收敛 step。
- [ ] `DataObs` analysis 当前只收集 `training_results.json` 的数值字段，缺少 RL 结果整合。

### 3.5 不同难度数据集对 SFT/RL 的影响

问题：

- 难度不同的数据是否带来不同 SFT/RL gain？

建议难度指标：

- teacher pass rate 低表示难
- base/student pass rate 低表示难
- IFD 高表示难
- PPL 高可能表示分布外或噪声
- verifier score distribution

TODO：

- [ ] 按 difficulty quantile 切分 seed data / distilled data：easy / medium / hard。
- [ ] 保持每个 bucket size 一致。
- [ ] 每个 bucket 训练 SFT。
- [ ] 每个 bucket 启 RL。
- [ ] 比较 SFT gain、RL gain、entropy、收敛 step。

当前缺口：

- [ ] DataObs 有 IFD/PPL，但没有内置按指标分桶生成数据集的脚本。
- [ ] 缺少 base model 对 seed data 的 difficulty 预评估流程。

### 3.6 不同 diversity 对 SFT/RL 的影响

问题：

- 数据是否越 diverse 越好？

TODO：

- [ ] 先固定 dataset size。
- [ ] 基于 embedding/similarity 做低/中/高 diversity 子集。
- [ ] 每个 diversity bucket 保持 difficulty 和 size 尽量接近。
- [ ] 训练 SFT 与 RL。
- [ ] 记录 RL entropy、convergence step、final score。

当前缺口：

- [ ] DataObs 能算 diversity，但没有做 diversity-controlled subset selection。
- [ ] 当前 diversity 默认采样计算，实验前要确认采样比例是否稳定。

### 3.7 不同蒸馏数据 size 的影响

问题：

- 规模重要还是覆盖度重要？

对照设计：

- size: 1k / 5k / 10k / 50k
- coverage/diversity: low / medium / high
- 训练 token budget 尽量对齐或明确记录不对齐

TODO：

- [ ] 生成 nested subset：1k 是 5k 的子集，5k 是 10k 的子集，减少随机噪声。
- [ ] 生成 coverage-controlled subset：同 size 下最大化 coverage/diversity。
- [ ] 每个 size 训练 SFT。
- [ ] 每个 SFT 启 RL。
- [ ] 比较 size gain 和 coverage gain。

当前缺口：

- [ ] 没有 subset builder。
- [ ] 没有 token budget 对齐逻辑。

### 3.8 蒸馏 temperature 和采样数量

问题：

- 蒸馏方法的多样性是否越多越好？

建议实验矩阵：

- temperature: 0.0 / 0.3 / 0.7 / 1.0
- num_samples: 1 / 4 / 8 / 16
- filter: on / off 或 threshold 0.99 / 0.5

TODO：

- [ ] 对同一 teacher、同一 seed data 批量生成数据。
- [ ] 记录候选数、保留数、pass rate、重复率、平均长度。
- [ ] 固定最终训练 size 后抽样训练，避免高 `num_samples` 只是数据更多。
- [ ] 另做不固定 size 的实验，观察 size + diversity 的联合增益。
- [ ] 比较 SFT/RL 表现和 generation diversity。

当前缺口：

- [ ] `cot_distill_teacher_filter.py` 支持 temperature/num_cots，但缺少批量 sweep 脚本。
- [ ] 缺少候选级输出保存；当前只保存过滤后的 SFT rows，不利于分析 filter 前分布。

## 4. 推荐实验目录结构

建议统一为：

```text
/data/hrh/COT/experiments/
  <experiment_group>/
    manifest.json
    distill/
      <data_variant>/
        candidates.parquet
        filtered_sft.parquet
        rl_train.parquet
        manifest.json
        metrics.json
    sft/
      <data_variant>/
        checkpoint-last/
        logs/
        eval/
        manifest.json
    rl/
      <data_variant>/
        checkpoints/
        logs/
        eval/
        manifest.json
    analysis/
      runs.csv
      data_metrics.csv
      sft_results.csv
      rl_results.csv
      correlations.csv
      plots/
```

## 5. 推荐优先级

### Phase A：跑通最小闭环

- [ ] 选一个主任务：建议 `gsm8k` 或 `math-500`。
- [ ] 选一个 student：建议 0.5B。
- [ ] 选两个 teacher：self/0.5B 和 7B。
- [ ] 每个 teacher 只生成 100-500 条。
- [ ] 跑 SFT。
- [ ] 跑 eval。
- [ ] 从 SFT 跑 GRPO。
- [ ] 跑 RL eval。
- [ ] 汇总到一张表。

### Phase B：补齐 teacher size 实验

- [ ] teacher: self / 7B / 32B / larger。
- [ ] 固定 seed data、size、temperature、num_samples。
- [ ] 输出 SFT/RL gain。

### Phase C：补齐数据属性实验

- [ ] difficulty buckets。
- [ ] diversity buckets。
- [ ] size buckets。
- [ ] prompt-only vs human reasoning vs teacher reasoning。

### Phase D：补齐蒸馏策略实验

- [ ] temperature sweep。
- [ ] num_samples sweep。
- [ ] filter threshold sweep。

## 6. 需要新增或修改的代码清单

- [x] 扩展 `cot_distill_teacher_filter.py`，支持 `gsm8k`、`math`、`arc-challenge`、`strategyQA` 等主任务。
- [ ] `scripts/build_sft_dataset.py`，把 seed/distilled/human reasoning 统一转成 `question`/`answer`。
- [ ] `scripts/build_rl_dataset.py`，把对应数据转成 `prompt`/`data_source`/`reward_model`。
- [ ] `scripts/run_distill_sweep.sh`，批量跑 teacher、temperature、num_samples。
- [ ] `scripts/run_sft_sweep.sh`，批量跑每个 data variant 的 SFT。
- [ ] `scripts/run_grpo_sweep.sh`，批量从 SFT checkpoint 启 RL。
- [ ] `scripts/collect_experiment_results.py`，合并 distill metrics、SFT eval、RL eval、RL logs。
- [ ] `scripts/parse_grpo_logs.py`，抽取 entropy、reward、validation score、best step、convergence step。
- [ ] `scripts/make_controlled_subsets.py`，按 size/difficulty/diversity 生成控制变量子集。
- [ ] 修复 DataObs pipeline 的 eval/train/parallel 参数问题。
- [ ] 扩展 `reward_fn_router.py`，至少补齐你要做 RL 的数据源。

## 7. 当前最重要的风险

- 数据格式不统一：SFT、RL、eval 三套入口使用不同列名。
- DataObs pipeline 当前有参数 bug，不能直接依赖它一键完成训练+评测。
- 蒸馏脚本已支持主数据集；剩余风险是代码类 filter 会执行测试、BFCL 不走这个链路、RL reward router 仍需单独补齐。
- RL 结果没有标准化解析，无法回答 entropy、收敛 step、SFT-RL 相关性。
- 没有 manifest，后续容易无法追踪某个 checkpoint 对应哪个 teacher/data/temperature。
- 大 teacher 蒸馏成本高，必须先用 smoke subset 验证格式和评测链路。

## 8. 最小可执行命令草案

先做 smoke distill：

```bash
cd /home/hrh/CoT-DataSynth
python DataObs/lib/data_process/cot_distill_teacher_filter.py \
  --dataset commonsenseqa \
  --input-file /data/open_datasets/CommonsenseQA/data/validation-processed.parquet \
  --output-file /tmp/csqa_distill_smoke.parquet \
  --model-id /path/to/teacher \
  --gpu-ids 0 \
  --num-cots 1 \
  --temperature 0.0 \
  --smoke-num-rows 32
```

跑 smoke SFT：

```bash
cd /home/hrh/CoT-DataSynth
bash DataObs/lib/training/sft_dataobs.sh \
  /path/to/student \
  /tmp/csqa_distill_smoke.parquet \
  /tmp/csqa_distill_smoke.parquet \
  /tmp/csqa_sft_smoke \
  0 \
  1
```

跑统一 eval：

```bash
cd /home/hrh/CoT-DataSynth/scripts
bash eval.sh commonsenseQA 0 /tmp/csqa_sft_smoke/checkpoint-last
```

从 SFT 启 GRPO 前，需要确认该任务在 `reward_fn_router.py` 中被支持；否则先选 `gsm8k` / `math-500` / `aqua_rat` / `ai2_arc` / `strategyQA`。
