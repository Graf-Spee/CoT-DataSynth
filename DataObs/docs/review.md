# DataObs Structure Review

本文档记录重构后的 `DataObs/` 结构和后续可继续清理的点。

## 当前结构

```text
DataObs/
  README.md
  config/
    data_pipeline.yaml
    teacher_registry.json
  docs/
    metrics.md
    runbook.md
    review.md
    eval_dataset_validation.md
    eval_dataset_validation.json
  experiment_instruction/
    README.md
    exp*/commands.sh
  lib/
    README.md
    __init__.py
    data_obs.py
    data_metrics.py
    advanced_metrics.py
    analysis_pipeline.py
    training_pipeline.py
    training_pipeline_parallel.py
    dataobs_eval_runner.py
  scripts/
    experiment_pipeline.py
    data_obs_pipeline.py
    split_seed_prompts.py
  tools/
    README.md
    analyze_observation_deepseek.py
    compute_metrics.py
    data_obs_pipeline_math_legacy.py
    run_dataobs_eval_serial.py
    validate_eval_datasets.py
```

## 主入口

优先保持可跑的是：

```text
DataObs/scripts/experiment_pipeline.py
```

它负责编排：

```text
distill -> metrics -> sft -> sft_eval -> grpo -> grpo_eval
```

其中：

- `distill` 调用 repo 根目录的 `DataObs/lib/data_process/cot_distill_teacher_filter.py`。
- `metrics` 调用 `DataObs/scripts/data_obs_pipeline.py`。
- `sft` 调用 repo 根目录的 `DataObs/lib/training/sft_dataobs.sh`。
- `sft_eval` 调用 repo 根目录的 `scripts/eval.sh`。
- `grpo` 调用 repo 根目录的 `DataObs/lib/training/grpo_from_sft.sh`。
- `grpo_eval` 调用 repo 根目录的 `DataObs/lib/evaluation/eval_grpo_dataobs.sh`。

因此重构 DataObs 时，不应该破坏这三个 DataObs 内部脚本：

- `scripts/experiment_pipeline.py`
- `scripts/data_obs_pipeline.py`
- `scripts/split_seed_prompts.py`

## Config

| 文件 | 作用 |
|---|---|
| `config/data_pipeline.yaml` | 原 `config_template.yaml`，现在放到 DataObs 自己的 config 目录下。当前仍主要作为模板，主 pipeline 还是 CLI 参数驱动。 |
| `config/teacher_registry.json` | teacher/student registry，记录模型路径、系列、大小、推荐 TP、用途。后续建议让 `experiment_pipeline.py` 支持 `--teacher-id` 读取它。 |

## Lib

`lib/` 只放 pipeline 会调用的可复用函数和类。

| 文件 | 作用 | 后续建议 |
|---|---|---|
| `data_obs.py` | split、结果收集、GPU 分配、基础数据结构。 | 可拆 `io_utils.py`、`gpu_utils.py`。 |
| `data_metrics.py` | 基础数据指标。 | 与 `advanced_metrics.py` 的 prompt/answer 提取 helper 可统一。 |
| `advanced_metrics.py` | diversity、entropy、PPL、IFD、reward difficulty。 | 文件较大，后续拆成 `similarity.py`、`diversity.py`、`ppl_ifd.py`。 |
| `analysis_pipeline.py` | correlations 和可视化。 | 可与 reporting 工具共享读取/格式化逻辑。 |
| `training_pipeline.py` | 旧 DataObs blocking 训练调度。 | 可和 parallel 版本合并。 |
| `training_pipeline_parallel.py` | 旧 DataObs 显存感知并行训练调度。 | 可抽出 GPU scheduler。 |
| `dataobs_eval_runner.py` | DataObs eval runner 辅助逻辑。 | 可考虑让 `experiment_pipeline.py` 复用。 |

## Scripts

`scripts/` 已收敛到 pipeline 相关入口：

| 文件 | 作用 |
|---|---|
| `experiment_pipeline.py` | 当前主实验 pipeline。 |
| `data_obs_pipeline.py` | 数据指标 pipeline，主要服务 `metrics` stage。 |
| `split_seed_prompts.py` | exp3100 setting A/B 的 prompt split 工具。 |

不要再把一次性分析脚本、验证脚本、legacy 脚本放回 `scripts/`。

## Tools

`tools/` 放辅助工具和 legacy 脚本：

| 文件 | 作用 | 状态 |
|---|---|---|
| `compute_metrics.py` | 对已有 split 补算 metrics。 | legacy，当前偏 jsonl split。 |
| `validate_eval_datasets.py` | 验证 eval dataset schema/reward/prompt。 | 可保留。 |
| `analyze_observation_deepseek.py` | 调 DeepSeek API 生成 observation 结论。 | 可保留。 |
| `data_obs_pipeline_math_legacy.py` | 原 MATH/eval 专用 pipeline 变体。 | 后续应合并或删除。 |
| `run_dataobs_eval_serial.py` | 旧 serial eval runner。 | 依赖 legacy pipeline，后续应合并或删除。 |

## 已删除/移动

- 删除 `DataObs/test/`。
- 删除 `DataObs/Point.md`。
- 删除 `DataObs/**/__pycache__/`。
- `DataObs/pipeline_lib/` 重命名为 `DataObs/lib/`。
- `DataObs/config_template.yaml` 移到 `DataObs/config/data_pipeline.yaml`。
- 非主入口脚本从 `DataObs/scripts/` 移到 `DataObs/tools/`。

## 后续清理顺序

1. 让 `experiment_pipeline.py` 支持 `--teacher-id`，读取 `config/teacher_registry.json`。
2. 把 dataset defaults/reward/eval path 抽成 `config/dataset_registry.json`，避免 `eval.sh`、`experiment_pipeline.py`、`validate_eval_datasets.py` 三处漂移。
3. 合并或删除 `tools/data_obs_pipeline_math_legacy.py` 和 `tools/run_dataobs_eval_serial.py`。
4. 拆 `lib/metrics/advanced_metrics.py`。
5. 合并 `lib/training/training_pipeline.py` 和 `lib/training/training_pipeline_parallel.py`。
