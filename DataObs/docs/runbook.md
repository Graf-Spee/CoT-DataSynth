# Experiment Runbook

本文档对应 `DataObs/experiment_todo.md` 中的研究问题，说明每类问题应该跑哪些脚本、推荐 stages、输入数据要求和结果位置。



1.评估数据集难度：
如果你只是想估计原始数据难度，不跑 exp5000，可以不传：

python DataObs/tools/estimate_dataset_difficulty.py \
  --dataset gsm8k \
  --model-id /data/pretrain_models/Qwen2.5-7B-Instruct \
  --gpu-ids 0 \
  --num-attempts 10 \
  --output-dir /data/hrh/COT/difficulty/gsm8k_qwen2_5_7b


统一入口：

```bash
cd /home/hrh/CoT-DataSynth
python3 DataObs/scripts/experiment_pipeline.py --help
```

Teacher registry：

```text
DataObs/config/teacher_registry.json
```

可直接执行的实验命令目录：

```text
DataObs/experiment_instruction/
```

每个 `exp1xxx_*` 目录下面都有 `commands.sh`。默认是 dry-run；确认命令无误后用 `DRY_RUN="" bash ...` 真跑。

所有实验都会写入：

```text
<output_dir>/<experiment_id>/
├── manifest.json
├── commands.jsonl
├── results.json
├── logs/
├── sft/
├── grpo/
└── eval/
```

先用 `--dry-run` 检查命令是否正确，再去掉 `--dry-run` 真跑。

## -1. 运行前需要你确认

这些确认项会影响 observation 结论，建议在正式 sweep 前定下来：

- 主数据集优先级：建议先 `gsm8k`，再 `math-500`，然后 `arc-challenge/strategyQA`；代码类单独排期。
- GPU 分配：`GPU_IDS` 给 0.5B/3B/7B，`BIG_GPU_IDS` 给 32B，`XL_GPU_IDS` 给 72B；需要确认哪些卡空闲。
- teacher size 主线是否先只跑 `self/7B/32B`：建议先跑这三个，趋势稳定后补 `3B/72B`。
- 固定训练预算方式：size 实验要确认是固定 epoch，还是固定 optimizer steps / token budget。
- human reasoning 数据来源：如果要做 exp3000，需要你指定 human reasoning parquet 或原始数据路径。
- difficulty/diversity/size 子集文件路径：exp5000/6000/7000 需要先构造这些 parquet。
- GRPO 是否每组都跑：为了省算力，可以先跑 `distill,metrics,sft,sft_eval`，筛出有希望的组再跑 `grpo,grpo_eval`。
- 代码类 RL 是否现在纳入：`mbpp/mbppplus/humaneval/humanevalplus` 已接 router，但训练会慢；`livecodebench` 还需要补 router。

## -0. Recipes 索引
- `exp1000_smoke`: 先确认 distill 格式、reward filter、输出 parquet。
- `exp1100_teacher_size`: Qwen2.5-Instruct 同系列 teacher size curve，回答“更大 teacher 是否有意义 / 自蒸馏是否有意义”。
- `exp1200_teacher_type`: reasoning teacher、跨系列 teacher、任务专家 teacher 对照。
- `exp2000_data_quality`: 不同蒸馏数据质量和 SFT/RL 表现的关系。
- `exp3000_seed_reasoning`: prompt-only teacher reasoning vs human reasoning vs answer-only。
- `exp3100_sft_rl_prompt_overlap`: setting A/B，比较 SFT train prompt 是否和 RL train prompt 相同。
- `exp4000_sft_rl_correlation`: 最强 SFT 是否最适合 RL。
- `exp5000_difficulty`: 不同难度数据对 SFT/RL 的影响。
- `exp6000_diversity`: 不同 diversity 数据对 SFT/RL/entropy/收敛的影响。
- `exp7000_size`: 数据规模和覆盖度影响。
- `exp8000_temperature_sampling`: temperature、num_samples、filter on/off 对表现的影响。

建议运行顺序：

1. `exp1000_smoke`：每个目标 dataset 先确认 distill 能跑。
2. `exp1100_teacher_size`：先只跑 `self/7B/32B`，确认 teacher size 主趋势。
3. `exp8000_temperature_sampling`：固定 7B teacher，找稳定的 temperature / num_samples。
4. `exp2000_data_quality`：用已经产出的蒸馏数据做 quality vs SFT/RL。
5. `exp3000_seed_reasoning`：等 human reasoning 数据路径确认后再跑。
6. `exp5000/6000/7000`：等 controlled subsets 生成后再跑。
7. `exp4000_sft_rl_correlation`：等多组 SFT checkpoint 产出后统一跑。

Observation 到 recipe 的对应关系：

| Observation 问题 | 主 recipe | 运行前需要确认 |
|---|---|---|
| 自蒸馏有没有意义 | `exp1100_teacher_size` | `self_base` 还是 `self_sft`；默认先用 `self_base` |
| 更大的 teacher 有没有意义 | `exp1100_teacher_size` | 先跑 `self/7B/32B` 还是全跑到 `72B` |
| 不同 teacher 类型是否有意义 | `exp1200_teacher_type` | reasoning/cross-family/task-specialist 是否和主曲线分开报告 |
| 蒸馏数据质量和 SFT 的关系 | `exp2000_data_quality` | low/mid/high quality parquet 如何定义 |
| Seed prompt vs human reasoning | `exp3000_seed_reasoning` | human reasoning / answer-only 数据路径 |
| SFT cold-start 是否能迁移到新 RL prompt | `exp3100_sft_rl_prompt_overlap` | seed split ratio；是否固定 SFT/RL 训练 token budget |
| 最强 SFT 是否最适合 RL | `exp4000_sft_rl_correlation` | SFT checkpoint 列表 |
| difficulty 对 SFT/RL 的影响 | `exp5000_difficulty` | easy/medium/hard 子集构造方法 |
| diversity 对 SFT/RL/entropy 的影响 | `exp6000_diversity` | low/mid/high diversity 子集构造方法 |
| size vs coverage | `exp7000_size` | nested subset 路径；是否固定 token budget |
| temperature / num_samples / filter | `exp8000_temperature_sampling` | teacher 固定为哪个；是否跑 filter off |

## 0. Smoke Test

目的：先确认已有 SFT checkpoint 可以评测，并且可以启动 GRPO。

```bash
python3 DataObs/scripts/experiment_pipeline.py \
  --dry-run \
  --experiment-id smoke_gsm8k_sft_grpo \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen2.5-0.5B-Instruct \
  --output-dir /data/hrh/COT/experiments \
  --stages sft_eval,grpo \
  --gpu-ids 0 \
  --sft-checkpoint /data/hjw/outputs/Qwen2.5-0.5B-gsm8k-sft/checkpoint-last \
  --eval-arg data.batch_size=8 \
  --grpo-env TOTAL_EPOCHS=1 \
  --grpo-env TEST_FREQ=5
```

已有 GRPO checkpoint 只做评测：

```bash
python3 DataObs/scripts/experiment_pipeline.py \
  --dry-run \
  --experiment-id smoke_gsm8k_grpo_eval_step174 \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen2.5-0.5B-Instruct \
  --output-dir /data/hrh/COT/experiments \
  --stages grpo_eval \
  --gpu-ids 4 \
  --grpo-output-dir /data/hrh/COT/GSM8K/grpo/split_0 \
  --grpo-step 174
```

注意：GRPO 评测请传 `--grpo-output-dir /.../grpo/split_0`，不要把 `global_step_174` 根目录直接传给 `eval.sh`。pipeline 会调用 `DataObs/lib/evaluation/eval_grpo_dataobs.sh` 自动找 `global_step_174/actor/merged_model`。

## 1. Teacher Size / 自蒸馏 / 大教师是否有意义

问题：

- 自蒸馏有没有意义？
- 更大的 teacher 是否带来增益？
- student 自己蒸馏是否有意义？

推荐变量：

- teacher: `0.5B/self`, `7B`, `32B`, larger
- 固定 student、seed data、temperature、num samples、训练超参

当前推荐 teacher 选择：

- 主结论只用同系列 size curve：`Qwen2.5-0.5B-Instruct`、`Qwen2.5-3B-Instruct`、`Qwen2.5-7B-Instruct`、`Qwen2.5-32B-Instruct`、`Qwen2.5-72B-Instruct`。
- 原因：它们都是 Qwen2.5 Instruct，chat template/tokenizer family/训练风格相对一致，更适合回答“teacher size 是否有用”。
- 自蒸馏默认先定义为 `self_base`: 用 student base model `/data/pretrain_models/Qwen2.5-0.5B-Instruct` 作为 teacher。以后如果用某个 SFT checkpoint 自己生成，实验名写 `self_sft`，不要和 `self_base` 混在一起。
- 跨系列 teacher 只做 robustness：`Llama-3.1-8B-Instruct`、`Llama-3.3-70B-Instruct`、`gemma-2-9b-it`、`gemma-2-27b-it`。
- reasoning teacher 单独做一组：`DeepSeek-R1-Distill-Qwen-7B/32B`、`QwQ-32B`。这回答的是“reasoning teacher 是否有额外收益”，不和普通 size curve 合并。
- 任务专家 teacher 单独做：math 用 `Qwen2.5-Math-7B` / `deepseek-math-7b-instruct`，code 用 `Qwen2.5-Coder-7B-Instruct` / `Qwen2.5-Coder-32B-Instruct`。

每个 teacher 跑一组：

```bash
python3 DataObs/scripts/experiment_pipeline.py \
  --experiment-id gsm8k_teacher_7b_t07_n4 \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen2.5-0.5B-Instruct \
  --teacher-model /data/pretrain_models/Qwen2.5-7B-Instruct \
  --output-dir /data/hrh/COT/experiments \
  --stages distill,metrics,sft,sft_eval,grpo,grpo_eval \
  --gpu-ids 0 \
  --distill-dataset gsm8k \
  --distill-input /data/open_datasets/GSM8K/train.parquet \
  --teacher-num-samples 4 \
  --teacher-temperature 0.7 \
  --sft-epochs 1 \
  --eval-arg data.batch_size=8 \
  --grpo-env TOTAL_EPOCHS=1
```

当前 `DataObs/lib/data_process/cot_distill_teacher_filter.py` 已支持普通 parquet 评测链路里的主数据集：`arc-challenge`、`aqua_rat`、`commonsenseqa`、`gsm8k`、`math`、`math-500`、`numinamath`、`strategyqa`、`mbpp`、`mbppplus`、`humaneval`、`humanevalplus`、`livecodebench`。`bfcl` 是单独评测链路，暂不走这个 CoT distill filter。

如果不传 `--distill-input`，pipeline 会按 `--dataset` 使用默认 seed parquet，例如 `gsm8k -> /data/open_datasets/GSM8K/train.parquet`。建议正式实验前先加 `--smoke-num-rows 32` 跑小样本。

对于已有蒸馏数据：

```bash
python3 DataObs/scripts/experiment_pipeline.py \
  --experiment-id gsm8k_teacher_7b_existing_data \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen2.5-0.5B-Instruct \
  --output-dir /data/hrh/COT/experiments \
  --stages metrics,sft,sft_eval,grpo,grpo_eval \
  --gpu-ids 0 \
  --sft-data /path/to/teacher_7b_sft.parquet \
  --sft-val-data /path/to/teacher_7b_sft_val.parquet \
  --sft-epochs 1 \
  --grpo-env TOTAL_EPOCHS=1
```

看结果：

- `eval/sft/generated/responses_labeled.metrics.json`
- `eval/grpo/generated/responses_labeled.metrics.json`
- `manifest.json` 中记录 teacher 和参数

## 2. 不同蒸馏数据质量和 SFT 的关系

问题：

- 哪些数据质量指标能解释 SFT 表现？
- 从后验怎么看蒸馏是否有意义？

每个数据版本跑：

```bash
python3 DataObs/scripts/experiment_pipeline.py \
  --experiment-id gsm8k_quality_teacher7b_v1 \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen2.5-0.5B-Instruct \
  --output-dir /data/hrh/COT/experiments \
  --stages metrics,sft,sft_eval \
  --gpu-ids 0 \
  --sft-data /path/to/distilled_sft.parquet \
  --sft-val-data /path/to/distilled_val.parquet \
  --metrics-data /path/to/distilled_sft.parquet \
  --metrics-model /data/pretrain_models/Qwen2.5-0.5B-Instruct \
  --n-splits 10 \
  --similarity-type jaccard
```

产出：

- `dataobs/<experiment_id>/data_metrics_summary.csv`
- `eval/sft/generated/responses_labeled.metrics.json`

后续分析时合并每个 experiment 的 data metrics 和 SFT eval metrics。

## 3. Seed Data 是 Prompt 还是 Human Reasoning

问题：

- human reasoning 是否对 SFT/RL 有帮助？
- prompt-only 经 teacher 生成 reasoning 是否优于直接 human reasoning？

对照组建议：

- `prompt_only_teacher_reasoning`
- `human_reasoning`
- `answer_only`
- `human_reasoning_teacher_rewrite`

已有 SFT 格式数据时，每组跑：

```bash
python3 DataObs/scripts/experiment_pipeline.py \
  --experiment-id gsm8k_human_reasoning \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen2.5-0.5B-Instruct \
  --output-dir /data/hrh/COT/experiments \
  --stages metrics,sft,sft_eval,grpo,grpo_eval \
  --gpu-ids 0 \
  --data-variant human_reasoning \
  --reasoning-source human \
  --sft-data /path/to/human_reasoning_sft.parquet \
  --sft-val-data /path/to/human_reasoning_val.parquet \
  --grpo-env TOTAL_EPOCHS=1
```

输入要求：

- SFT 数据：`question`, `answer`
- RL 数据：默认用对应 dataset 的 RL parquet；如需自定义，传 `--rl-train-data` 和 `--rl-val-data`

## 4. SFT 表现和 RL 表现相关性

问题：

- 最强 SFT 是否最适合 RL？
- SFT score 高是否意味着 RL final score / RL gain 高？

对每个 SFT checkpoint 跑：

```bash
python3 DataObs/scripts/experiment_pipeline.py \
  --experiment-id gsm8k_sftA_to_rl \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen2.5-0.5B-Instruct \
  --output-dir /data/hrh/COT/experiments \
  --stages sft_eval,grpo,grpo_eval \
  --gpu-ids 0 \
  --sft-checkpoint /path/to/sft_checkpoint \
  --grpo-env TOTAL_EPOCHS=1 \
  --grpo-env TEST_FREQ=5
```

如果 GRPO 已经跑完，只补评测：

```bash
python3 DataObs/scripts/experiment_pipeline.py \
  --experiment-id gsm8k_sftA_rl_eval \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen2.5-0.5B-Instruct \
  --output-dir /data/hrh/COT/experiments \
  --stages grpo_eval \
  --gpu-ids 4 \
  --grpo-output-dir /path/to/grpo_output \
  --grpo-step 174
```

需要比较：

- SFT `test_score`
- RL `test_score`
- `RL gain = RL score - SFT score`
- GRPO log 中 reward / entropy / validation curve

## 4.1 SFT train prompt 和 RL train prompt 是否相同

问题：

- setting A: `SFT train prompt == RL train prompt`，看 SFT cold-start 对同一训练 prompt 分布上的 RL 是否有帮助。
- setting B: `SFT train prompt != RL train prompt`，看 SFT 学到的 reasoning 是否能迁移到新的 RL prompt。

直接跑 recipe：

```bash
DRY_RUN="" bash DataObs/experiment_instruction/exp3100_sft_rl_prompt_overlap/commands.sh
```

默认会先把 seed parquet 稳定切成两份：

```text
<OUTPUT_DIR>/_prepared/exp3100_<dataset>_seed_split/
├── <dataset>_sft_seed.parquet
├── <dataset>_rl_train.parquet
├── <dataset>_overlap_train.parquet
└── <dataset>_split_manifest.json
```

然后跑两组：

- `exp3100_<dataset>_settingA_sft_eq_rl_prompt`
  - distill input: `<dataset>_sft_seed.parquet`
  - RL train: pipeline 从 `distill/filtered_sft.parquet` 的 `source_index` 回捞得到 `rl_data/train_from_distill_kept.parquet`
  - 含义：实际进入 SFT 的 prompt 和 RL 训练 prompt 相同。这样即使 teacher filter 丢了一部分样本，A 仍然是严格 same-prompt。
- `exp3100_<dataset>_settingB_sft_ne_rl_prompt`
  - distill input: `<dataset>_sft_seed.parquet`
  - RL train: `<dataset>_rl_train.parquet`
  - 含义：SFT/RL 训练 prompt 不重叠。

可调参数：

```bash
DATASET=gsm8k
SEED_INPUT=/data/open_datasets/GSM8K/train.parquet
SFT_RATIO=0.5
SPLIT_SEED=42
STAGES=distill,metrics,sft,sft_eval,grpo,grpo_eval
```

解释结果时不要把这两组混在一起：

- A 主要看 same-prompt cold-start 是否让 RL 更快、更稳、更高。
- B 主要看 reasoning SFT 是否迁移到新 prompt。
- 两组都必须用同一个 eval test set，例如 GSM8K test。

## 5. 不同 Difficulty 的数据对 SFT/RL 的影响

问题：

- easy / medium / hard 数据是否对 SFT/RL 有不同影响？

先准备三个 SFT 数据文件：

```text
/path/to/easy_sft.parquet
/path/to/medium_sft.parquet
/path/to/hard_sft.parquet
```

每个 bucket 跑：

```bash
python3 DataObs/scripts/experiment_pipeline.py \
  --experiment-id gsm8k_difficulty_easy \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen2.5-0.5B-Instruct \
  --output-dir /data/hrh/COT/experiments \
  --stages metrics,sft,sft_eval,grpo,grpo_eval \
  --gpu-ids 0 \
  --data-variant difficulty_easy \
  --sft-data /path/to/easy_sft.parquet \
  --sft-val-data /path/to/easy_val.parquet \
  --metrics-data /path/to/easy_sft.parquet \
  --grpo-env TOTAL_EPOCHS=1
```

当前 pipeline 不负责自动分桶；分桶脚本需要后续补 `make_controlled_subsets.py`。

## 6. 不同 Diversity 的数据对 SFT/RL 的影响

问题：

- diversity 是否越高越好？

先准备控制 size 后的三个数据：

```text
low_diversity_sft.parquet
mid_diversity_sft.parquet
high_diversity_sft.parquet
```

每组跑：

```bash
python3 DataObs/scripts/experiment_pipeline.py \
  --experiment-id gsm8k_diversity_high \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen2.5-0.5B-Instruct \
  --output-dir /data/hrh/COT/experiments \
  --stages metrics,sft,sft_eval,grpo,grpo_eval \
  --gpu-ids 0 \
  --data-variant diversity_high \
  --sft-data /path/to/high_diversity_sft.parquet \
  --sft-val-data /path/to/high_diversity_val.parquet \
  --metrics-data /path/to/high_diversity_sft.parquet \
  --similarity-type jaccard \
  --grpo-env TOTAL_EPOCHS=1
```

可替换 `--similarity-type`：

- `jaccard`
- `cosine`
- `ngram`
- `rouge`
- `bleu`

## 7. 不同 Size 的蒸馏数据对表现的影响

问题：

- 规模重要还是覆盖度重要？

建议先准备 nested subset：

```text
1k_sft.parquet
5k_sft.parquet
10k_sft.parquet
```

每个 size 跑：

```bash
python3 DataObs/scripts/experiment_pipeline.py \
  --experiment-id gsm8k_size_1k \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen2.5-0.5B-Instruct \
  --output-dir /data/hrh/COT/experiments \
  --stages metrics,sft,sft_eval,grpo,grpo_eval \
  --gpu-ids 0 \
  --data-variant size_1k \
  --sft-data /path/to/1k_sft.parquet \
  --sft-val-data /path/to/1k_val.parquet \
  --metrics-data /path/to/1k_sft.parquet \
  --grpo-env TOTAL_EPOCHS=1
```

注意：如果要严格比较 size，需要记录训练 token budget；否则 size 变大和训练步数变多会混在一起。

## 8. 蒸馏 Temperature 和 Sampling Number

问题：

- 多样化采样是否越多越好？

对同一 teacher 和 seed data sweep：

```bash
python3 DataObs/scripts/experiment_pipeline.py \
  --experiment-id csqa_t07_n4 \
  --dataset commonsenseQA \
  --base-model /data/pretrain_models/Qwen2.5-0.5B-Instruct \
  --teacher-model /data/pretrain_models/Qwen2.5-7B-Instruct \
  --output-dir /data/hrh/COT/experiments \
  --stages distill,metrics,sft,sft_eval \
  --gpu-ids 0 \
  --distill-dataset commonsenseqa \
  --distill-input /data/open_datasets/CommonsenseQA/data/validation-processed.parquet \
  --teacher-temperature 0.7 \
  --teacher-num-samples 4 \
  --teacher-do-sample \
  --sft-epochs 1
```

建议 sweep：

- temperature: `0.0`, `0.3`, `0.7`, `1.0`
- num samples: `1`, `4`, `8`, `16`
- filter: on/off

filter off 的命令加：

```bash
--disable-teacher-filter
```

这个 sweep 现在也可以直接换成 `gsm8k` / `math` / `arc-challenge` / `strategyQA` 等数据集。例如 GSM8K：

```bash
python3 DataObs/scripts/experiment_pipeline.py \
  --experiment-id gsm8k_t07_n4 \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen2.5-0.5B-Instruct \
  --teacher-model /data/pretrain_models/Qwen2.5-7B-Instruct \
  --output-dir /data/hrh/COT/experiments \
  --stages distill,metrics,sft,sft_eval \
  --gpu-ids 0 \
  --teacher-temperature 0.7 \
  --teacher-num-samples 4 \
  --teacher-do-sample \
  --smoke-num-rows 32 \
  --sft-epochs 1
```

## 9. 常见参数

只打印命令不执行：

```bash
--dry-run
```

给 eval.sh 传 Hydra 参数：

```bash
--eval-arg data.batch_size=8
--eval-arg rollout.max_num_seqs=64
```

给 GRPO 脚本传环境变量：

```bash
--grpo-env TOTAL_EPOCHS=1
--grpo-env TEST_FREQ=5
--grpo-env ROLLOUT_N=8
```

给 SFT 脚本传 Hydra 参数：

```bash
--sft-arg trainer.total_epochs=1
--sft-arg data.train_batch_size=8
```

指定不同 GPU：

```bash
--sft-gpu-ids 0
--eval-gpu-ids 4
--grpo-gpu-ids 0,1
```

## 10. 当前还需要补的能力

- 自动生成 difficulty/diversity/size controlled subsets
- 自动汇总所有 experiment 的 metrics 到一个总表
- 解析 GRPO training log 中的 entropy、best step、convergence step

## 11. 当前 Seed Data 怎么做

现在的 seed data 主要就是 `/data/open_datasets/...` 下的 processed parquet。它们不是 SFT 的 `question/answer` 格式，而是 eval/RL 格式：

- `prompt`: chat message list，teacher/generation/RL 都从这里读题。
- `reward_model.ground_truth`: 规则 reward 用的标准答案或测试用例。
- `data_source`: reward/router 和后续统计用的数据源名。

蒸馏流程是：seed parquet 的 `prompt` 给 teacher，teacher 生成 CoT/代码答案，`reward_model.ground_truth` 用对应 reward function 过滤，最后输出 SFT parquet：`question`、`answer`、`gold_answer`、`dataset`、`data_source`、`teacher_score`。

因此：

- prompt-only seed：直接用这些 processed parquet 跑 `distill`。
- human reasoning seed：需要已经有 `question`/`answer` 的 SFT parquet，可直接从 `metrics,sft,sft_eval,...` 开始。
- RL seed：默认还是用原始 processed parquet；当前 pipeline 不会自动把 distilled SFT 数据反转成 RL 数据。

## 12. Pipeline 会保留哪些文件

假设命令里有：

```bash
--output-dir /data/hrh/COT/experiments
--experiment-id exp1100_gsm8k_teacher_qwen2_5_7b_t0.7_n4
```

那么主目录是：

```text
/data/hrh/COT/experiments/exp1100_gsm8k_teacher_qwen2_5_7b_t0.7_n4/
```

### 12.1 Pipeline 自己写的追踪文件

```text
manifest.json
commands.jsonl
results.json
logs/
```

- `manifest.json`: 实验静态配置。包含 `experiment_id`、`dataset`、`base_model`、`teacher_model`、`data_variant`、`reasoning_source`、`stages`、所有命令参数、默认输出路径。
- `commands.jsonl`: 每个 stage 的实际命令记录。stage 开始时写 `cmd/cmd_str/env/log_path/dry_run`，stage 结束后写 `returncode/elapsed_sec/log_path`。复现实验时优先看这个。
- `results.json`: pipeline 逐 stage 更新的产物路径索引，例如 `distill_output`、`metrics_dir`、`sft_dir`、`sft_eval_dir`、`grpo_dir`、`grpo_eval_dir`。
- `logs/<stage>.log`: pipeline 捕获的子命令 stdout/stderr。失败时先看对应 stage log，比如 `logs/distill.log`、`logs/sft.log`、`logs/grpo.log`。

dry-run 时也会写 `manifest.json` 和 `commands.jsonl`，但不会产生真正的 distill/SFT/eval/GRPO 产物。

`exp3100_sft_rl_prompt_overlap` 还会在 pipeline 之外先写一份 seed split manifest：

```text
<OUTPUT_DIR>/_prepared/exp3100_<dataset>_seed_split/<dataset>_split_manifest.json
```

里面记录 `num_rows_sft_seed`、`num_rows_rl_train`、`prompt_overlap_sft_rl=0` 等切分信息。

setting A 如果使用 `--rl-train-from-distill-kept`，还会额外生成：

```text
rl_data/train_from_distill_kept.parquet
```

这是从 distill 实际保留下来的 `source_index` 回捞出的 RL train parquet，用来保证 `SFT train prompt == RL train prompt`。

### 12.2 `distill` stage 产物

默认路径：

```text
distill/
├── filtered_sft.parquet
├── filtered_sft.candidates.parquet
└── filtered_sft.summary.json
```

`filtered_sft.parquet` 是 teacher 蒸馏后给 SFT 用的数据，主要列：

- `question`: 原始 user prompt 文本。
- `answer`: teacher 生成的 CoT/代码答案。
- `gold_answer`: 原始 `reward_model.ground_truth`。
- `dataset`: 规范化后的 dataset 名。
- `data_source`: 原始数据源。
- `source_index`: seed parquet 里的行号。
- `teacher_score`: reward function 给 teacher answer 的分数。
- `teacher_filter_passed`: 是否通过 teacher correctness filter。

如果没加 `--disable-teacher-filter`，默认只保留 `teacher_score >= teacher_correct_threshold` 的样本。

`filtered_sft.candidates.parquet` 会保留所有 teacher candidates，包括没通过 filter 的样本。主要列：

- `source_index`: 原 seed prompt 行号。
- `sample_index`: 同一个 prompt 下第几个 candidate。
- `answer`: teacher candidate。
- `teacher_score`: reward 分数。
- `teacher_filter_passed`: 是否通过 filter。
- `answer_char_len`: candidate 长度。

`filtered_sft.summary.json` 是蒸馏质量摘要。无论后续 SFT/RL 是否失败，只要 distill 成功就能看这些数：

- `teacher_pass_rate`: 全部 candidates 中通过 reward filter 的比例。
- `kept_rate`: 最终写入 `filtered_sft.parquet` 的比例。filter 开启时通常等于 `teacher_pass_rate`；filter off 时会是 1.0。
- `prompt_pass_rate`: 至少有一个 candidate 通过 filter 的 prompt 比例。
- `num_candidates` / `num_passed_candidates` / `num_kept`
- `score.mean/min/max/p25/p50/p75`
- `answer_length_chars.mean_all_candidates` / `mean_kept`
- `empty_answer_rate`
- `elapsed_sec`

### 12.3 `metrics` stage 产物

默认路径：

```text
dataobs/
├── splits/
│   ├── split_0.parquet
│   └── ...
├── metrics/
│   ├── split_0_metrics.json
│   └── ...
└── data_metrics_summary.csv
```

- `splits/split_*.parquet`: DataObs 把输入 SFT 数据切成 `n_splits` 份后的数据。
- `metrics/split_*_metrics.json`: 每个 split 的 statistics、quality、diversity、entropy，以及可选 PPL/IFD。
- `data_metrics_summary.csv`: 所有 split 的指标汇总表。后续做 quality/diversity/difficulty 和 SFT/RL 表现相关性时主要读这个。

当前 pipeline 调用 DataObs 时用了 `--skip_training --skip_analysis`，所以这里主要产出数据指标，不会在 DataObs 内部重复训练。

### 12.4 `sft` stage 产物

默认路径：

```text
sft/
├── global_step_*/
│   ├── adapter_model.safetensors 或 model*.safetensors
│   ├── config.json
│   ├── tokenizer_config.json
│   └── ...
└── wandb/offline logs 可能也会写在这里
```

`DataObs/lib/training/sft_dataobs.sh` 会把 `trainer.default_local_dir` 设成 `sft/`。checkpoint 通常是 `sft/global_step_*`，pipeline 会用 `resolve_sft_checkpoint()` 自动找：

1. 显式传入的 `--sft-checkpoint`
2. `sft/` 本身如果已经是 HF/LoRA checkpoint
3. `sft/checkpoint-last`
4. 最新的 `sft/global_step_*`

SFT 训练日志在：

```text
logs/sft.log
```

### 12.5 `sft_eval` stage 产物

默认路径：

```text
eval/sft/
├── generated/
│   ├── responses.parquet
│   ├── responses_labeled.json
│   └── responses_labeled.metrics.json
└── logs/
    ├── generation.log
    └── evaluation.log
```

- `generated/responses.parquet`: eval 数据 + 新增 `responses` 列。`responses` 是模型生成答案列表。
- `generated/responses_labeled.json`: 每条样本的 reward 打分、是否正确、预测结果等标签。
- `generated/responses_labeled.metrics.json`: 汇总指标，重点看 `test_score`、`accuracy`、`pass@1/mean` 等。
- `logs/generation.log`: 生成阶段日志。
- `logs/evaluation.log`: 打分阶段日志，也会打印指标。

Pipeline 外层还会有：

```text
logs/sft_eval.log
```

这是 `eval.sh` 整体日志；具体生成/评测日志在 `eval/sft/logs/`。

### 12.6 `grpo` stage 产物

默认路径：

```text
grpo/
├── global_step_*/
│   ├── actor/
│   │   └── merged_model/
│   └── ...
└── ...
```

`DataObs/lib/training/grpo_from_sft.sh` 会把 RL checkpoint 写到 `grpo/`。评测 GRPO 时不要直接把 `global_step_*` 根目录传给 `eval.sh`；pipeline 的 `grpo_eval` 会调用 `DataObs/lib/evaluation/eval_grpo_dataobs.sh` 自动找：

```text
grpo/global_step_*/actor/merged_model
```

GRPO 训练日志在：

```text
logs/grpo.log
```

后续要解析 entropy、reward curve、validation curve、best step、convergence step，主要从这个 log 或 GRPO 自己的 logger 输出里抽取。

### 12.7 `grpo_eval` stage 产物

默认路径：

```text
eval/grpo/
├── generated/
│   ├── responses.parquet
│   ├── responses_labeled.json
│   └── responses_labeled.metrics.json
└── logs/
    ├── generation.log
    └── evaluation.log
```

含义和 `sft_eval` 一样，只是被评测模型换成 GRPO checkpoint 的 `actor/merged_model`。

Pipeline 外层日志：

```text
logs/grpo_eval.log
```

### 12.8 BFCL 特殊产物

如果 dataset 是 `bfcl`，不走普通 parquet reward function。输出会在 eval 目录下多一套：

```text
eval/*/bfcl/
├── result/
└── score/
    └── data_overall.csv
eval/*/logs/
├── generation.log
├── evaluation_raw.log
└── evaluation.log
```

- `bfcl/score/data_overall.csv`: BFCL 官方评测表。
- `logs/evaluation.log`: pipeline 兼容格式，里面会写 `test_score` 和 `accuracy`。

### 12.9 最常用的结果读取位置

训练/蒸馏追踪：

```text
manifest.json
commands.jsonl
results.json
logs/*.log
```

蒸馏数据：

```text
distill/filtered_sft.parquet
```

数据指标：

```text
dataobs/data_metrics_summary.csv
dataobs/metrics/split_*_metrics.json
```

SFT 表现：

```text
eval/sft/generated/responses_labeled.metrics.json
```

RL 表现：

```text
eval/grpo/generated/responses_labeled.metrics.json
```

RL 训练过程：

```text
logs/grpo.log
```

## 13. Observation 实验还不确定 / 缺失的部分

下面这些不是已经跑实验前的小问题，而是会影响 observation 结论能否解释清楚的关键缺口。

### 13.1 实验定义还不确定

- Teacher 列表已初步固定在 `DataObs/config/teacher_registry.json`。主实验用 Qwen2.5-Instruct 同系列，跨系列/reasoning/任务专家 teacher 分开报告。
- 自蒸馏默认定义为 `self_base`：用 student base model 作为 teacher。若使用 SFT checkpoint 自己生成，实验 id 必须写 `self_sft`。
- Seed data 的版本还没统一：prompt-only seed、human reasoning seed、teacher reasoning seed、teacher rewrite seed 的来源和列格式要固定，否则 SFT/RL 差异可能来自数据格式而不是 reasoning。
- “更大的 teacher 有没有意义”的因变量建议固定为：`teacher_pass_rate`、`prompt_pass_rate`、`kept_rate`、`SFT test_score`、`RL test_score`、`RL gain`、`elapsed_sec`。其中 distill 相关数值已经写入 `distill/filtered_sft.summary.json`。
- “数据质量”的定义建议分两层：DataObs 的 quality/diversity/entropy/PPL/IFD，加上 distill summary 的 teacher pass rate、score distribution、empty answer rate、answer length distribution。format error rate 和 duplicate rate 还需要继续补。
- Difficulty 的定义需要选择：可以用 base model pass rate、teacher pass rate、reward margin、题目 metadata level；不同定义会得到不同 easy/medium/hard bucket。
- Diversity 和 coverage 的定义需要分开：diversity 是样本间差异，coverage 是知识点/题型覆盖；现在还没有明确用 embedding、n-gram、topic、metadata 还是 reward behavior 来衡量。
- Size 实验需要控制训练 token budget：如果大数据集同时带来更多 optimizer steps，无法判断是 data size 还是训练步数导致增益。
- Temperature / num samples 实验需要记录候选总数和过滤后样本数：否则 `n=16` 的收益可能只是因为保留样本更多，不一定是 generation diversity 更好。
- SFT-RL 相关性需要定义统计口径：比较 final score、best score、RL gain、convergence step、entropy 曲线，还是固定 step 的 validation score。

### 13.2 数据构造和脚本缺失

- teacher model registry 已有初版：`DataObs/config/teacher_registry.json`；后续需要随着实际可用 GPU 和新增 checkpoint 更新。
- 缺少 distill sweep 脚本：现在 pipeline 能单次跑 distill，但还没有批量展开 teacher x temperature x num_samples x seed_data 的 runner。
- distill 单次 summary 已有：`filtered_sft.summary.json` 会记录 pass rate、kept rows、candidate rows、score 分布、平均长度。还缺跨实验汇总表、重复率、格式错误率。
- 缺少 SFT/RL 统一结果表：还没有自动把 DataObs metrics、SFT eval、GRPO eval、GRPO logs 合成一个 CSV/JSONL。
- 缺少 controlled subset builder：还不能自动生成 fixed-size、difficulty bucket、diversity bucket、coverage-controlled 子集。
- 缺少 human reasoning 数据转换器：如果拿原始带 reasoning 的数据，需要统一转成 `question/answer`，并保留 `reasoning_source=human` 元信息。
- 缺少 distilled SFT -> RL parquet 转换器：当前 RL 默认用原始 seed parquet；如果想研究“蒸馏数据本身作为 RL prompt/answer/reference”的影响，需要单独构造 RL 数据格式。
- 缺少 GRPO log parser：目前还不能自动抽取 entropy、reward curve、validation curve、best step、convergence step、训练失败原因。
- 缺少失败样本分析：需要把 SFT/RL eval 的错题按原始 seed index、difficulty、teacher_score、data metrics 回连，才能解释后验意义。
- 缺少去重和泄漏检查：distilled train、SFT val、eval test 之间需要检查 prompt/answer 重复或近重复。

### 13.3 当前流程风险

- 代码类数据集虽然已经能 distill/eval，且 `mbpp`、`mbppplus`、`humaneval`、`humanevalplus` 已接入 GRPO reward router，但 reward 会执行代码测试，训练速度和 timeout 风险明显高于数学/选择题。
- `livecodebench` 目前能 distill/eval，但还没有接入 GRPO reward router；如果要做 RL，需要先补 router 并做小规模 smoke。
- `bfcl` 是单独 function-calling evaluator，不走普通 parquet reward function；暂时不能直接纳入当前 distill/SFT/RL observation pipeline。
- Math 类 reward 依赖 `math_verify` 等环境包；所有实验命令应固定在 `verl-cot` 环境运行，避免 base python 缺依赖。
- `MODEL_NAME`、`experiment_id`、输出路径需要强制包含 dataset/teacher/temp/n_samples/filter/seed，否则后续结果容易无法追踪。
- 大 teacher 蒸馏成本高，正式 sweep 前必须先用 `--smoke-num-rows 8/32` 验证每个 dataset 的格式、reward 和输出行数。
