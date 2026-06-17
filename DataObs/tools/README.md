# DataObs Tools

`tools/` 放不属于主 pipeline 入口的辅助脚本：

- `compute_metrics.py`: 对已有 split 补算 metrics 的 legacy 工具。
- `grpo_log_metrics.py`: 从 `grpo.log` 提取 reward、entropy、KL、显存、耗时等 GRPO 指标，导出 CSV 和曲线图。
- `validate_eval_datasets.py`: 验证 eval 数据集 schema/reward/prompt。
- `analyze_observation_deepseek.py`: 用 DeepSeek API 生成 observation 结论。
- `data_obs_pipeline_math_legacy.py`: 旧 MATH/eval 专用 pipeline 变体，后续应合并进 `scripts/data_obs_pipeline.py` 或删除。
- `run_dataobs_eval_serial.py`: 旧 serial eval runner，依赖 `data_obs_pipeline_math_legacy.py`。

主实验不要优先从这里启动，优先使用 `DataObs/scripts/experiment_pipeline.py`。

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
