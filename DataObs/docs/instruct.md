# experiment_pipeline.py 参数说明

本文档记录 `DataObs/scripts/experiment_pipeline.py` 当前支持的命令行参数，并按阶段分类。

## 全局参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--experiment-id` | 必填 | 实验名，输出目录为 `{output_dir}/{experiment_id}` |
| `--dataset` | 必填 | 数据集名，如 `gsm8k` |
| `--base-model` | 必填 | 学生/base 模型路径 |
| `--output-dir` | `/data/hrh/COT/experiments` | 实验根目录 |
| `--stages` | `sft,sft_eval,grpo,grpo_eval` | 要跑的阶段，逗号分隔 |
| `--gpu-ids` | `0` | 默认 GPU ids，各阶段未单独指定时使用 |
| `--dry-run` | `False` | 只打印命令，不实际运行 |
| `--data-variant` | 空 | metrics 的 data_name；也写入 manifest |
| `--reasoning-source` | 空 | 只写入 manifest，用于标记 reasoning 来源 |
| `--model-name` | 空 | 覆盖 eval/SFT/GRPO 用的 `MODEL_NAME` |

可用 stage 名：

```text
distill, metrics, sft, sft_eval, grpo, grpo_eval
```

## Distill 阶段

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--distill-dataset` | 空 | 传给 distill 脚本的数据集名；为空则用 `--dataset` |
| `--distill-input` | 空 | 蒸馏输入 parquet；为空则按 dataset 使用内置默认路径 |
| `--distill-output` | 空 | 蒸馏输出；为空则 `{exp_dir}/distill/filtered_sft.parquet` |
| `--teacher-model` | 空 | 教师模型路径；跑 `distill` 时必需 |
| `--distill-gpu-ids` | 空 | distill 专用 GPU；为空则用 `--gpu-ids` |
| `--teacher-num-samples` | `1` | 每条 prompt 采样多少条 CoT |
| `--teacher-temperature` | `0.7` | teacher generation temperature |
| `--teacher-top-p` | `0.95` | teacher generation top_p |
| `--teacher-batch-size` | `128` | teacher generation batch size |
| `--teacher-max-new-tokens` | `8192` | teacher 最大生成 token 数 |
| `--teacher-tensor-parallel-size` | `1` | teacher vLLM tensor parallel size |
| `--teacher-gpu-memory-utilization` | `0.95` | teacher vLLM 显存占比 |
| `--teacher-correct-threshold` | `0.99` | teacher filter 正确阈值 |
| `--teacher-do-sample` | `False` | 显式开启 sampling |
| `--disable-teacher-filter` | `False` | 不做 teacher correctness filter |
| `--smoke-num-rows` | `0` | 只跑前 N 行；0 表示全量 |

如果 `--teacher-num-samples > 1`，代码会自动把 `teacher_do_sample` 设为 true。

## Metrics 阶段

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--metrics-data` | 空 | metrics 输入数据；为空则用 SFT 数据路径 |
| `--metrics-model` | 空 | 额外传给 `data_obs_pipeline.py --model` |
| `--n-splits` | `10` | 切分数量 |
| `--similarity-type` | `jaccard` | diversity/similarity 类型 |
| `--metrics-arg` | `[]` | 额外参数，可重复传；原样追加到 metrics 命令 |

metrics 默认输出到：

```text
{exp_dir}/dataobs
```

## SFT 阶段

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--sft-data` | 空 | SFT train parquet；为空则用 distill output |
| `--sft-val-data` | 空 | SFT val parquet；为空则同 `--sft-data` |
| `--sft-output-dir` | 空 | SFT 输出目录；为空则 `{exp_dir}/sft` |
| `--sft-checkpoint` | 空 | 给后续 eval/GRPO 指定已有 SFT checkpoint |
| `--sft-gpu-ids` | 空 | SFT 专用 GPU；为空则用 `--gpu-ids` |
| `--sft-epochs` | `1` | SFT epoch 数 |
| `--sft-arg` | `[]` | 额外 Hydra/脚本参数，可重复传 |
| `--sft-env` | `[]` | SFT 环境变量，格式 `KEY=VALUE`，可重复传 |

SFT 实际调用：

```text
DataObs/lib/training/sft_dataobs.sh
```

并设置环境变量：

```text
MODEL_NAME={model_name 或 experiment_id-sft}
DATA_NAME={dataset}
```

## SFT Eval 阶段

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--eval-gpu-ids` | 空 | eval 专用 GPU；为空则用 `--gpu-ids` |
| `--sft-eval-output-dir` | 空 | SFT eval 输出；为空则 `{exp_dir}/eval/sft` |
| `--eval-arg` | `[]` | 额外参数，可重复传；追加到 `scripts/eval.sh` |
| `--eval-env` | `[]` | eval 环境变量，格式 `KEY=VALUE`，可重复传 |
| `--sft-checkpoint` | 空 | 指定评测哪个 SFT checkpoint；否则自动找 SFT 输出下最新 checkpoint |

SFT eval 会用到全局的：

```text
--dataset
--base-model
--model-name
```

并设置环境变量：

```text
BASE_MODEL={base_model}
MODEL_NAME={model_name 或 experiment_id-sft}
TARGET_EVAL_DIR={sft_eval_output_dir 或 exp_dir/eval/sft}
```

## GRPO 阶段

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--rl-train-data` | 空 | RL train parquet；为空则按 dataset 使用内置默认路径 |
| `--rl-val-data` | 空 | RL val parquet；为空则按 dataset 使用内置默认路径 |
| `--rl-train-from-distill-kept` | `False` | 用 distill filter 保留下来的 prompt 构造 RL train |
| `--grpo-output-dir` | 空 | GRPO 输出目录；为空则 `{exp_dir}/grpo` |
| `--grpo-gpu-ids` | 空 | GRPO 专用 GPU；为空则用 `--gpu-ids` |
| `--grpo-step` | 空 | 不是 GRPO 训练用；给 `grpo_eval` 指定 eval step |
| `--grpo-arg` | `[]` | 额外参数，可重复传；追加到 GRPO 脚本 |
| `--grpo-env` | `[]` | GRPO 环境变量，格式 `KEY=VALUE`，可重复传 |
| `--sft-checkpoint` | 空 | GRPO 从哪个 SFT checkpoint 开始；为空自动找最新 |

GRPO 实际调用：

```text
DataObs/lib/training/grpo_from_sft.sh
```

并设置环境变量：

```text
MODEL_NAME={model_name 或 experiment_id-grpo}
DATA_NAME={dataset}
```

## GRPO Eval 阶段

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--grpo-eval-output-dir` | 空 | GRPO eval 输出；为空则 `{exp_dir}/eval/grpo` |
| `--eval-gpu-ids` | 空 | eval 专用 GPU；为空则用 `--gpu-ids` |
| `--eval-env` | `[]` | eval 环境变量，格式 `KEY=VALUE`，可重复传 |
| `--grpo-step` | 空 | 如果传了，会设置环境变量 `STEP={grpo_step}` |

GRPO eval 实际调用：

```text
DataObs/lib/evaluation/eval_grpo_dataobs.sh
```

## 隐式默认和注意事项

- `distill_input`、`rl_train_data`、`rl_val_data` 为空时，会按 `--dataset` 查内置默认路径。
- 支持内置默认路径的数据集包括：`gsm8k`、`math-500`、`math`、`aqua_rat`、`arc-challenge`、`strategyQA`、`commonsenseQA`、`mbpp`、`mbppplus`、`humaneval`、`humanevalplus`、`livecodebench`、`numinamath`。
- `--sft-data` 为空时默认吃 distill 输出。如果只跑 `sft` 而不跑 `distill`，要么提前有默认 distill 输出，要么显式传 `--sft-data`。
- `--metrics-arg`、`--sft-arg`、`--eval-arg`、`--grpo-arg` 都是 `action=append`，可以重复写多次。
