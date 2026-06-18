# DataObs

数据观察、蒸馏、训练、评测和实验管理管线。当前框架把主链路入口、复用库和辅助工具分开维护。

`DataObs` 现在保留两类 pipeline：

- `scripts/experiment_pipeline.py`: 主入口，负责 distill -> metrics -> SFT -> SFT eval -> GRPO -> GRPO eval。当前实验 recipes 都应优先走这个入口。
- `scripts/data_obs_pipeline.py`: 数据指标/分 split/相关性分析入口，主要被 `experiment_pipeline.py` 的 `metrics` stage 调用。
- `lib/evaluation/dataobs_eval_runner.py`: Python-native eval runner，支持 dataset-specific prompt template、custom eval data path 和 smoke-test row limit。

## Dashboard

```bash
python DataObs/tools/experiment_dashboard.py \
  --experiments-root /data/hrh/COT/experiments \
  --host 127.0.0.1 \
  --port 7860
```

## Directory Layout

```text
DataObs/
  config/                  # DataObs 自己的配置和 registry
    data_pipeline.yaml      # 数据指标 pipeline 配置模板
    teacher_registry.json   # teacher/student 模型 registry
  lib/                      # pipeline 调用的函数和类
  scripts/                  # pipeline 入口脚本，保持精简
  tools/                    # 非主链路工具/legacy/验证脚本
  docs/                     # runbook、metrics、结构说明
  experiment_instruction/   # 可直接跑的实验命令
```

## Main Pipeline

```bash
cd /home/hrh/CoT-DataSynth
python DataObs/scripts/experiment_pipeline.py \
  --experiment-id distill_smoke_gsm8k_qwen3_5_9b \
  --dataset gsm8k \
  --base-model /data/pretrain_models/Qwen3.5-0.8B \
  --teacher-model /data/pretrain_models/Qwen3.5-9B \
  --output-dir /data/hrh/COT/experiments \
  --stages distill \
  --gpu-ids 7 \
  --smoke-num-rows 8 \
  --teacher-num-samples 1 \
  --teacher-temperature 0.7 \
  --teacher-batch-size 8 \
    --teacher-max-new-tokens 1024
```

实验命令集合见：

```text
DataObs/experiment_instruction/
```

详细实验设计和待确认项见：

```text
DataObs/docs/runbook.md
```

## Scripts Policy

`scripts/` 只放 pipeline 直接入口：

- `experiment_pipeline.py`
- `data_obs_pipeline.py`
- `split_seed_prompts.py`

辅助分析、验证、历史兼容脚本放在 `tools/`。
