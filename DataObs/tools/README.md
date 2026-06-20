# DataObs Tools

`tools/` 放不属于主 pipeline 入口的辅助脚本：

- `compute_metrics.py`: 对已有 split 补算 metrics 的 legacy 工具。
- `experiment_dashboard.py`: 本地实验看板，按 experiment-id 查看 pipeline 阶段状态、日志、eval 指标和 GRPO 曲线。
- `estimate_dataset_difficulty.py`: 用指定模型对每条数据重复采样并按正确次数估计 difficulty。
- `grpo_log_metrics.py`: 从 `grpo.log` 提取 reward、entropy、KL、显存、耗时等 GRPO 指标，导出 CSV 和曲线图。
- `validate_eval_datasets.py`: 验证 eval 数据集 schema/reward/prompt。
- `analyze_observation_deepseek.py`: 用 DeepSeek API 生成 observation 结论。
- `data_obs_pipeline_math_legacy.py`: 旧 MATH/eval 专用 pipeline 变体，后续应合并进 `scripts/data_obs_pipeline.py` 或删除。
- `run_dataobs_eval_serial.py`: 旧 serial eval runner，依赖 `data_obs_pipeline_math_legacy.py`。

主实验不要优先从这里启动，优先使用 `DataObs/scripts/experiment_pipeline.py`。

## Difficulty 估计

用某个模型对每道题采样 10 次，统计做对几次，得到每条数据的难度：

```bash
python DataObs/tools/estimate_dataset_difficulty.py \
  --dataset gsm8k \
  --model-id /data/pretrain_models/Qwen2.5-7B-Instruct \
  --gpu-ids 0 \
  --num-attempts 10 \
  --output-dir /data/hrh/COT/difficulty/gsm8k_qwen2_5_7b \
  --write-bucket-parquets
```

输出：

- `<prefix>.parquet`: 每条原始数据的 `pass_count`、`pass_rate`、`difficulty_bucket`。
- `<prefix>.candidates.parquet`: 每次采样答案和 reward 分数。
- `<prefix>_{easy,medium,hard}.parquet`: 可选写出的原始 seed 数据分桶。
- `<prefix>_sft_{easy,medium,hard}.parquet`: 如果传 `--sft-file distill/filtered_sft.parquet`，会按 `source_index` 切出可直接给 exp5000 使用的 SFT 分桶。

## 实验看板

启动本地 dashboard：

```bash
python DataObs/tools/experiment_dashboard.py \
  --experiments-root /data/hrh/COT/experiments \
  --host 127.0.0.1 \
  --port 7860
```

页面里填入或选择 `experiment-id`，例如 `pipeline_another_gsm8k_qwen7b`。看板会读取：

- `commands.jsonl`：阶段状态、耗时、启动参数。
- `logs/*.log`：各阶段日志尾部。
- `eval/{sft,grpo}/generated/responses_labeled.metrics.json`：SFT/GRPO accuracy、pass@1。
- `grpo_metrics/{reward,entropy,grad_norm,kl}.png`：GRPO 训练曲线。
- `distill/filtered_sft.summary.json`：distill 保留样本数和 teacher pass rate。

点击 `Refresh GRPO` 会从当前 `logs/grpo.log` 重新生成 GRPO 指标和曲线。

## GRPO 训练曲线

训练过程中可以重复运行下面命令查看当前收敛趋势：

```bash
python DataObs/tools/grpo_log_metrics.py \
  --log /data/hrh/COT/experiments/pipeline_check_gsm8k_qwen7b/logs/grpo.log \
  --output-dir /data/hrh/COT/experiments/pipeline_check_gsm8k_qwen7b/grpo_metrics \
  --plot
```

主要输出：

- `grpo_metrics.csv`：每个 step 的原始数值指标。
- `grpo_metrics_derived.csv`：增加 rolling mean、accuracy proxy、grad clip hit 等派生指标。
- `metric_summary.csv`：每个指标的 first/last/delta/mean/std/min/max/best step。
- `summary.json`：核心指标摘要。
- `reward.{png,pdf}`：训练 reward、平滑 reward、验证 reward；对 0/1 reward 数据集可近似看作 accuracy/pass@1。
- `entropy.{png,pdf}`：policy entropy。
- `grad_norm.{png,pdf}`：gradient norm。
- `kl.{png,pdf}`：KL penalty / PPO KL。
