#!/usr/bin/env python3
"""
Analyze observation correlations and generate a readable report.

Inputs:
- observation/correlations.csv
- observation/strong_correlations_0.4.csv
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MIN_MAX_METRIC_PATTERN = re.compile(r"(^|_)(min|max)($|_)")


METRIC_MEANINGS: Dict[str, str] = {
    "num_samples": "该 split 的样本数",
    "avg_prompt_length": "prompt 平均长度（字符级）",
    "std_prompt_length": "prompt 长度标准差",
    "avg_response_length": "answer 平均长度（字符级）",
    "std_response_length": "answer 长度标准差",
    "num_unique_data_sources": "去重后的 data_source 数量",
    "num_unique_abilities": "去重后的 ability 数量",
    "answer_coverage": "有 answer 的样本占比",
    "format_validity": "同时包含有效 prompt+answer 的样本占比",
    "prompt_uniqueness": "prompt 去重比例",
    "diversity_similarity_type": "多样性使用的相似度函数类型（类别字段，通常不参与数值相关性）",
    "avg_char_entropy": "字符级 Shannon 熵均值",
    "std_char_entropy": "字符级 Shannon 熵标准差",
    "avg_word_entropy": "词级 Shannon 熵均值",
    "std_word_entropy": "词级 Shannon 熵标准差",
    "avg_ppl": "answer 文本平均困惑度（PPL）",
    "std_ppl": "PPL 标准差",
    "avg_ifd": "IFD 均值，定义为 Loss(with_prompt)/Loss(without_prompt)",
    "std_ifd": "IFD 标准差",
    "train_loss": "训练集 loss",
    "val_loss": "验证集 loss",
    "val_accuracy": "验证集准确率",
}


def _is_min_max_metric(metric_name: str) -> bool:
    return bool(MIN_MAX_METRIC_PATTERN.search(metric_name.lower()))


def _resolve_observation_dir(path: Path) -> Path:
    if (path / "correlations.csv").exists():
        return path
    if (path / "observation" / "correlations.csv").exists():
        return path / "observation"
    raise FileNotFoundError(
        f"Cannot find correlations.csv under: {path} or {path / 'observation'}"
    )


def _split_pair(pair: str) -> Tuple[str, str]:
    parts = [p.strip() for p in str(pair).split(" vs ")]
    if len(parts) == 2:
        return parts[0], parts[1]
    return str(pair), ""


def _metric_family(metric: str) -> str:
    m = metric.lower()
    if "ifd" in m or "ppl" in m:
        return "ppl_ifd"
    if "diversity" in m or "similarity" in m:
        return "diversity"
    if "entropy" in m:
        return "entropy"
    if "length" in m:
        return "length"
    if "coverage" in m or "validity" in m or "uniqueness" in m:
        return "quality"
    if "loss" in m or "accuracy" in m:
        return "training"
    return "other"


def _describe_metric(metric: str) -> str:
    if metric in METRIC_MEANINGS:
        return METRIC_MEANINGS[metric]

    if metric.endswith("_diversity_score"):
        sim = metric.split("_", 1)[0]
        return f"{sim} 相似度下的多样性分数（1 - avg_similarity）"
    if metric.endswith("_avg_similarity"):
        sim = metric.split("_", 1)[0]
        return f"{sim} 两两样本相似度均值"
    if metric.endswith("_std_similarity"):
        sim = metric.split("_", 1)[0]
        return f"{sim} 两两样本相似度标准差"

    return "指标含义未在内置字典中定义，可结合 data_obs_pipeline_metrics.md 补充"


def _format_corr(v: float) -> str:
    return f"{v:+.4f}"


def _load_correlations(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if "pair" not in df.columns or "correlation" not in df.columns:
        raise ValueError(f"{file_path} must contain columns: pair, correlation")
    df = df.copy()
    parsed = df["pair"].apply(_split_pair)
    df["data_metric"] = parsed.apply(lambda x: x[0])
    df["result_metric"] = parsed.apply(lambda x: x[1])
    df["abs_corr"] = df["correlation"].abs()
    df["is_minmax_pair"] = df["data_metric"].apply(_is_min_max_metric) | df["result_metric"].apply(_is_min_max_metric)
    return df


def _top_rows(df: pd.DataFrame, n: int = 8) -> List[Dict[str, str]]:
    if df.empty:
        return []
    rows: List[Dict[str, str]] = []
    for _, r in df.head(n).iterrows():
        rows.append({
            "pair": str(r["pair"]),
            "corr": _format_corr(float(r["correlation"])),
            "data_metric": str(r["data_metric"]),
            "result_metric": str(r["result_metric"]),
        })
    return rows


def build_report(obs_dir: Path, threshold: float) -> str:
    corr_path = obs_dir / "correlations.csv"
    strong_path = obs_dir / f"strong_correlations_{threshold:.1f}.csv"

    corr_df = _load_correlations(corr_path)
    if strong_path.exists():
        strong_df = _load_correlations(strong_path)
    else:
        logger.warning("%s not found, fallback to filtering correlations.csv", strong_path.name)
        strong_df = corr_df[corr_df["abs_corr"] >= threshold].copy()

    corr_no_minmax = corr_df[~corr_df["is_minmax_pair"]].copy()
    strong_no_minmax = strong_df[~strong_df["is_minmax_pair"]].copy()
    if strong_no_minmax.empty:
        strong_no_minmax = corr_no_minmax[corr_no_minmax["abs_corr"] >= threshold].copy()

    strong_no_minmax = strong_no_minmax.sort_values("abs_corr", ascending=False)
    pos_top = _top_rows(strong_no_minmax[strong_no_minmax["correlation"] > 0].sort_values("correlation", ascending=False))
    neg_top = _top_rows(strong_no_minmax[strong_no_minmax["correlation"] < 0].sort_values("correlation", ascending=True))

    by_result_lines: List[str] = []
    if not strong_no_minmax.empty:
        grouped = strong_no_minmax.groupby("result_metric", dropna=False)
        for result_metric, g in grouped:
            if not result_metric:
                continue
            g_sorted = g.sort_values("abs_corr", ascending=False)
            top = g_sorted.iloc[0]
            by_result_lines.append(
                f"- `{result_metric}`: 强相关 {len(g)} 条，最强的是 `{top['data_metric']}` ({_format_corr(float(top['correlation']))})"
            )

    family_lines: List[str] = []
    if not strong_no_minmax.empty:
        fam = strong_no_minmax.copy()
        fam["family"] = fam["data_metric"].apply(_metric_family)
        fam_stats = (
            fam.groupby("family")
            .agg(count=("pair", "count"), mean_abs=("abs_corr", "mean"))
            .sort_values(["count", "mean_abs"], ascending=[False, False])
        )
        for family, row in fam_stats.iterrows():
            family_lines.append(
                f"- `{family}`: {int(row['count'])} 条，平均 |corr| = {float(row['mean_abs']):.4f}"
            )

    key_data_metrics = sorted(set(strong_no_minmax["data_metric"].dropna().tolist()))
    metric_meaning_lines = [
        f"- `{m}`: {_describe_metric(m)}"
        for m in key_data_metrics
    ]

    lines: List[str] = []
    lines.append("# Observation Correlation Analysis")
    lines.append("")
    lines.append("## 1) 数据概览")
    lines.append(f"- observation 目录: `{obs_dir}`")
    lines.append(f"- correlations.csv 总条数: {len(corr_df)}")
    lines.append(f"- 去除 min/max 后条数: {len(corr_no_minmax)}")
    lines.append(f"- strong_correlations_{threshold:.1f}.csv 条数: {len(strong_df)}")
    lines.append(f"- strong 去除 min/max 后条数: {len(strong_no_minmax)}")
    lines.append("")
    lines.append("## 2) 最强正相关 (去除 min/max)")
    if pos_top:
        for r in pos_top:
            lines.append(f"- `{r['pair']}`: {r['corr']}")
    else:
        lines.append("- 无")
    lines.append("")
    lines.append("## 3) 最强负相关 (去除 min/max)")
    if neg_top:
        for r in neg_top:
            lines.append(f"- `{r['pair']}`: {r['corr']}")
    else:
        lines.append("- 无")
    lines.append("")
    lines.append("## 4) 按训练指标汇总")
    if by_result_lines:
        lines.extend(by_result_lines)
    else:
        lines.append("- 无可用强相关条目")
    lines.append("")
    lines.append("## 5) 按数据指标类型汇总")
    if family_lines:
        lines.extend(family_lines)
    else:
        lines.append("- 无可用强相关条目")
    lines.append("")
    lines.append("## 6) 指标含义速查（基于强相关里出现的数据指标）")
    if metric_meaning_lines:
        lines.extend(metric_meaning_lines)
    else:
        lines.append("- 无")
    lines.append("")
    lines.append("## 7) 观察建议")
    lines.append("- 先优先看 `|corr|` 大且在多个训练指标上方向一致的指标，这类通常更稳定。")
    lines.append("- 若某指标只在极少数 split 上有效，建议回看原始 split 数据，确认是否受采样噪声影响。")
    lines.append("- IFD/PPL 建议联合看 `avg_*` 与 `std_*`，均值反映总体难度，标准差反映样本异质性。")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze observation correlations and generate report.")
    parser.add_argument(
        "--obs_dir",
        required=True,
        help="Path to observation directory OR run output directory containing observation/",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Strong correlation threshold used to read strong_correlations_{threshold}.csv (default: 0.4)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown path (default: <obs_dir>/observation_analysis.md)",
    )
    args = parser.parse_args()

    obs_dir = _resolve_observation_dir(Path(args.obs_dir))
    report = build_report(obs_dir=obs_dir, threshold=args.threshold)
    output_path = Path(args.output) if args.output else (obs_dir / "observation_analysis.md")
    try:
        output_path.write_text(report, encoding="utf-8")
        logger.info("Saved analysis report: %s", output_path)
    except OSError as e:
        logger.warning("Could not write report to %s: %s", output_path, e)

    print(report)


if __name__ == "__main__":
    main()
