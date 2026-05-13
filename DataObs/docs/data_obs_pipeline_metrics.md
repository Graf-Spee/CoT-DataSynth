# data_obs_pipeline.py 指标说明（按当前代码实现）

本文档基于 `DataObs/scripts/data_obs_pipeline.py` 与 `DataObs/pipeline_lib/*.py` 的当前实现整理，描述每个指标的真实含义与产出位置。

## 1. Pipeline 会计算哪些“数据指标”

在 `Phase 2: Computing Data Metrics` 中，每个 split 会依次计算：

1. `compute_data_statistics`
2. `compute_data_quality_metrics`
3. `compute_dataset_diversity`
4. `compute_dataset_entropy`
5. `compute_ppl_metrics`（仅当传入 `--model`）
6. `compute_ifd_metrics`（仅当传入 `--model`）

结果写入：

- `metrics/split_{i}_metrics.json`（单 split）
- `data_metrics_summary.csv`（所有 split 汇总）

## 2. 统计指标（statistics）

来源：`pipeline_lib/data_metrics.py::compute_data_statistics`

- `num_samples`
  - 含义：该 split 的样本数。

- `avg_prompt_length`
  - 含义：`prompt` 文本长度（字符数）均值。
  - 细节：如果 `prompt` 是 list（如多轮消息），会拼接每个 `content` 的长度后求和。

- `std_prompt_length`
  - 含义：`prompt` 长度标准差。

- `min_prompt_length` / `max_prompt_length`
  - 含义：`prompt` 长度最小值 / 最大值。

- `avg_response_length`
  - 含义：`extra_info.answer` 文本长度（字符数）均值。

- `std_response_length`
  - 含义：`extra_info.answer` 长度标准差。

- `min_response_length` / `max_response_length`
  - 含义：`extra_info.answer` 长度最小值 / 最大值。

- `num_unique_data_sources`
  - 含义：`data_source` 去重后的数量。

- `num_unique_abilities`
  - 含义：`ability` 去重后的数量。

## 3. 质量指标（quality）

来源：`pipeline_lib/data_metrics.py::compute_data_quality_metrics`

- `answer_coverage`
  - 含义：有非空 `extra_info.answer` 的样本占比。
  - 公式：`has_answer / len(data)`。

- `format_validity`
  - 含义：同时满足“有 `prompt` 且有可提取 `answer`”的样本占比。

- `prompt_uniqueness`
  - 含义：`prompt` 字符串去重比例（简单唯一性）。
  - 公式：`unique_prompts / len(data)`。

## 4. 多样性指标（diversity）

来源：`pipeline_lib/advanced_metrics.py::compute_dataset_diversity`

### 4.1 相似度类型

由 `--similarity_type` 指定，支持：

- `jaccard`
- `levenshtein`
- `cosine`
- `jaro_winkler`
- `ngram`
- `bertouch`
- `bleu`
- `rouge`

### 4.2 计算对象

由 `--compute_on` 指定：

- `prompt`：只看 prompt
- `answer`：只看 answer
- `both`：prompt + answer

### 4.3 指标字段

假设 `similarity_type=jaccard`，则输出键会带前缀 `jaccard_`（其他类型同理）：

- `{sim}_diversity_score`
  - 含义：多样性分数。
  - 公式：`1 - avg_similarity`。

- `{sim}_avg_similarity`
  - 含义：样本两两相似度均值（上三角，不含对角线）。

- `{sim}_min_similarity`
  - 含义：样本两两相似度最小值。

- `{sim}_max_similarity`
  - 含义：样本两两相似度最大值。

- `{sim}_std_similarity`
  - 含义：样本两两相似度标准差。

- `diversity_similarity_type`
  - 含义：本次计算实际使用的相似度类型字符串。

说明：默认会在 split 内做采样（`sample_size=100`）后再计算 pairwise 相似度。

## 5. 熵指标（entropy）

来源：`pipeline_lib/advanced_metrics.py::compute_dataset_entropy`

基于 prompt 文本计算 Shannon entropy：

- `avg_char_entropy` / `std_char_entropy` / `min_char_entropy` / `max_char_entropy`
  - 含义：字符级熵的均值/标准差/最小值/最大值。

- `avg_word_entropy` / `std_word_entropy` / `min_word_entropy` / `max_word_entropy`
  - 含义：词级熵的均值/标准差/最小值/最大值。

## 6. PPL 指标（可选）

来源：`pipeline_lib/advanced_metrics.py::compute_ppl_metrics`

触发条件：在 pipeline 中传入 `--model`（HuggingFace 模型名或本地路径）。

- `avg_ppl`
- `std_ppl`
- `min_ppl` 
- `max_ppl`

含义：使用指定语言模型对 answer 文本计算困惑度（Perplexity）后的统计量。

## 7. IFD 指标（可选）

来源：`pipeline_lib/advanced_metrics.py::compute_ifd_metrics`

触发条件：在 pipeline 中传入 `--model`。

IFD 定义（代码实现）：

- `IFD = Loss(with_prompt) / Loss(without_prompt)`

输出字段：

- `avg_ifd`
- `std_ifd`
- `min_ifd`
- `max_ifd`

含义：按样本计算 IFD 后在 split 内聚合的统计量。

## 8. 相关性分析阶段（analysis）会产出哪些“分析指标”

在 `Phase 4: Analysis and Visualization` 中，pipeline 会把“数据指标”与“训练/评测结果”做相关性分析。

### 8.1 训练/评测结果字段来源

从 `training/split_{i}/training_results.json` 读取所有数值字段，常见包括：

- `train_loss`
- `val_loss`
- `val_accuracy`（若训练日志里解析得到）
- `test_accuracy`（若评测阶段写入）

### 8.2 相关性计算方式

来源：`pipeline_lib/analysis_pipeline.py::CorrelationAnalyzer.compute_correlations`

- 默认方法：`pearson`
- 仅对数值列计算
- 每对指标最少要求 3 个共同 split 点
- 常数列（标准差为 0）会跳过

输出：

- `observation/correlations.csv`
  - 列：`pair`, `correlation`
  - `pair` 形如：`avg_char_entropy vs test_accuracy`

- `observation/strong_correlations_0.6.csv`
  - 过滤条件：
    - `|correlation| >= 0.6`
    - 且默认排除任一侧为 `min_*` / `max_*` 的指标对

- `observation/strong_correlations_0.4.csv`
  - 过滤条件：
    - `|correlation| >= 0.4`
    - 且默认排除任一侧为 `min_*` / `max_*` 的指标对

补充说明：

- `correlations.csv` 仍保留全部可计算 pair（不做 `min/max` 过滤）。
- 只有 strong correlations 汇总会默认过滤 `min/max` 指标对。

### 8.3 可视化输出（按当前实现）

来源：`pipeline_lib/analysis_pipeline.py::AnalysisVisualizer.plot_correlation_heatmap`

- 分组热力图会先排除 `min_*` / `max_*` 数值列。
- 会生成的常见文件：
  - `correlation_heatmap_length.png`
  - `correlation_heatmap_diversity.png`
  - `correlation_heatmap_entropy.png`
  - `correlation_heatmap_ppl_ifd.png`（`ppl` 与 `ifd` 已合并）
  - `correlation_heatmap_quality.png`
  - `correlation_heatmap_full.png`
- 不再生成：`correlation_heatmap_training.png`

## 9. 默认是否计算

- 默认会算：statistics + quality + diversity + entropy
- 默认不会算：PPL + IFD（需要显式给 `--model`）
- `--skip_metrics` 会跳过所有数据指标计算
- `--skip_analysis` 会跳过相关性分析和可视化

## 10. 一句话总结

`data_obs_pipeline.py` 的核心是：

1. 对每个 split 计算数据侧指标（长度/质量/多样性/熵/可选 PPL/IFD）
2. 收集每个 split 的训练或测试表现（如 `test_accuracy`）
3. 计算两者之间的 Pearson 相关系数并输出强相关列表

## 11. DeepSeek 自动结论脚本

脚本：`DataObs/scripts/analyze_observation_deepseek.py`

用途：读取 `observation/correlations.csv` + `observation/strong_correlations_*.csv`，结合指标定义，自动生成实验结论 Markdown。

最小示例：

```bash
export DEEPSEEK_API_KEY="你的key"
python /home/hrh/CoT-DataSynth/DataObs/scripts/analyze_observation_deepseek.py \
  --obs_dir /data/hjw/outputs/MATH-CoT-Qwen3B/observation \
  --threshold 0.4
```

`--threshold` 含义：

- 强相关阈值（`|correlation| >= threshold`）。
- 优先读取 `strong_correlations_{threshold}.csv`；
- 若不存在则从 `correlations.csv` 现场过滤。
