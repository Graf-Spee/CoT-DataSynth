# DataObs Lib

`lib/` 只放 DataObs pipeline 会调用的实现代码；`DataObs/scripts/` 只保留入口脚本。

当前分类：

- `data_process/`: 数据生成、蒸馏、过滤、schema 转换，以及 split/结果收集/GPU 分配等基础数据流程。
- `metrics/`: 基础数据指标、diversity、entropy、PPL、IFD、reward difficulty。
- `analysis/`: correlations 和可视化。
- `training/`: DataObs 专用 SFT/GRPO 启动脚本，以及旧 DataObs blocking/parallel 训练调度。
- `evaluation/`: DataObs 专用评测、BFCL 评测、结果汇总和 eval runner 辅助逻辑。
- `model_ops/`: 模型操作辅助脚本，例如 LoRA merge。

后续如果继续重构，优先拆 `metrics/advanced_metrics.py`，再合并 `training/training_pipeline.py` 和 `training/training_pipeline_parallel.py`。
