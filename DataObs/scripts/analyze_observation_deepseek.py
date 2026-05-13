#!/usr/bin/env python3
"""
Use DeepSeek API to analyze observation correlations and export markdown conclusions.

Inputs:
- observation/correlations.csv
- observation/strong_correlations_0.4.csv
- (optional) DataObs/docs/data_obs_pipeline_metrics.md for metric meanings
"""

import argparse
import json
import logging
import os
import re
import textwrap
import urllib.error
import urllib.request
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
    "min_prompt_length": "prompt 长度最小值",
    "max_prompt_length": "prompt 长度最大值",
    "avg_response_length": "answer 平均长度（字符级）",
    "std_response_length": "answer 长度标准差",
    "min_response_length": "answer 长度最小值",
    "max_response_length": "answer 长度最大值",
    "num_unique_data_sources": "去重后的 data_source 数量",
    "num_unique_abilities": "去重后的 ability 数量",
    "answer_coverage": "有 answer 的样本占比",
    "format_validity": "同时包含有效 prompt+answer 的样本占比",
    "prompt_uniqueness": "prompt 去重比例",
    "avg_char_entropy": "字符级 Shannon 熵均值",
    "std_char_entropy": "字符级 Shannon 熵标准差",
    "min_char_entropy": "字符级 Shannon 熵最小值",
    "max_char_entropy": "字符级 Shannon 熵最大值",
    "avg_word_entropy": "词级 Shannon 熵均值",
    "std_word_entropy": "词级 Shannon 熵标准差",
    "min_word_entropy": "词级 Shannon 熵最小值",
    "max_word_entropy": "词级 Shannon 熵最大值",
    "avg_ppl": "answer 文本平均困惑度（PPL）",
    "std_ppl": "PPL 标准差",
    "min_ppl": "PPL 最小值",
    "max_ppl": "PPL 最大值",
    "avg_ifd": "IFD 均值，定义为 Loss(with_prompt)/Loss(without_prompt)",
    "std_ifd": "IFD 标准差",
    "min_ifd": "IFD 最小值",
    "max_ifd": "IFD 最大值",
    "train_loss": "训练集 loss",
    "val_loss": "验证集 loss",
    "val_accuracy": "验证集准确率",
    "test_accuracy": "测试集准确率",
}


def _is_min_max_metric(metric_name: str) -> bool:
    return bool(MIN_MAX_METRIC_PATTERN.search(metric_name.lower()))


def _resolve_observation_dir(path: Path) -> Path:
    if (path / "correlations.csv").exists():
        return path
    candidate = path / "observation"
    if (candidate / "correlations.csv").exists():
        return candidate
    raise FileNotFoundError(
        f"Cannot find correlations.csv under {path} or {candidate}"
    )


def _split_pair(pair: str) -> Tuple[str, str]:
    parts = [p.strip() for p in str(pair).split(" vs ")]
    if len(parts) == 2:
        return parts[0], parts[1]
    return str(pair), ""


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
    if metric.endswith("_min_similarity"):
        sim = metric.split("_", 1)[0]
        return f"{sim} 两两样本相似度最小值"
    if metric.endswith("_max_similarity"):
        sim = metric.split("_", 1)[0]
        return f"{sim} 两两样本相似度最大值"

    return "未在内置字典中定义"


def _load_correlations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "pair" not in df.columns or "correlation" not in df.columns:
        raise ValueError(f"{path} must include columns: pair, correlation")
    df = df.copy()
    parsed = df["pair"].apply(_split_pair)
    df["data_metric"] = parsed.apply(lambda x: x[0])
    df["result_metric"] = parsed.apply(lambda x: x[1])
    df["abs_corr"] = df["correlation"].abs()
    return df


def _select_top(df: pd.DataFrame, n: int) -> List[Dict[str, object]]:
    rows = []
    for _, r in df.head(n).iterrows():
        rows.append(
            {
                "pair": str(r["pair"]),
                "correlation": float(r["correlation"]),
                "abs_corr": float(r["abs_corr"]),
                "data_metric": str(r["data_metric"]),
                "result_metric": str(r["result_metric"]),
            }
        )
    return rows


def _extract_metric_meanings(metrics: List[str]) -> Dict[str, str]:
    return {m: _describe_metric(m) for m in sorted(set(metrics))}


def _load_metric_doc(repo_root: Path) -> str:
    doc_path = repo_root / "DataObs" / "docs" / "data_obs_pipeline_metrics.md"
    if not doc_path.exists():
        return ""
    try:
        text = doc_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    # Keep size small while preserving key semantics.
    return text[:12000]


def _build_payload(
    obs_dir: Path,
    corr_df: pd.DataFrame,
    strong_df: pd.DataFrame,
    threshold: float,
    exclude_min_max: bool,
    top_n: int,
    metric_doc_excerpt: str,
) -> Dict[str, object]:
    full = corr_df.copy()
    strong = strong_df.copy()

    if exclude_min_max:
        full = full[
            ~full["data_metric"].apply(_is_min_max_metric)
            & ~full["result_metric"].apply(_is_min_max_metric)
        ].copy()
        strong = strong[
            ~strong["data_metric"].apply(_is_min_max_metric)
            & ~strong["result_metric"].apply(_is_min_max_metric)
        ].copy()

    full = full.sort_values("abs_corr", ascending=False)
    strong = strong.sort_values("abs_corr", ascending=False)

    top_all = _select_top(full, top_n)
    top_strong = _select_top(strong, top_n)
    top_pos = _select_top(strong[strong["correlation"] > 0].sort_values("correlation", ascending=False), top_n)
    top_neg = _select_top(strong[strong["correlation"] < 0].sort_values("correlation", ascending=True), top_n)

    metrics_for_meaning = full["data_metric"].tolist() + full["result_metric"].tolist()
    metric_meanings = _extract_metric_meanings(metrics_for_meaning)

    by_result = {}
    for result_metric, g in strong.groupby("result_metric", dropna=False):
        if not result_metric:
            continue
        g_sorted = g.sort_values("abs_corr", ascending=False)
        by_result[result_metric] = _select_top(g_sorted, min(5, top_n))

    return {
        "observation_dir": str(obs_dir),
        "threshold": threshold,
        "exclude_min_max": exclude_min_max,
        "counts": {
            "correlations_total": int(len(corr_df)),
            "strong_total": int(len(strong_df)),
            "correlations_used": int(len(full)),
            "strong_used": int(len(strong)),
        },
        "top_correlations_all": top_all,
        "top_correlations_strong": top_strong,
        "top_positive_strong": top_pos,
        "top_negative_strong": top_neg,
        "by_training_metric": by_result,
        "metric_meanings": metric_meanings,
        "metric_doc_excerpt": metric_doc_excerpt,
    }


def _call_deepseek_chat(
    api_key: str,
    api_base: str,
    model: str,
    temperature: float,
    max_tokens: int,
    payload: Dict[str, object],
    goal: str,
    auto_continue: bool = True,
    max_rounds: int = 8,
    timeout_sec: int = 120,
) -> str:
    system_prompt = (
        "你是资深机器学习实验分析师，专注冷启动数据工程。"
        "请根据给定相关性数据和指标定义，输出结构化、可执行、可验证的实验观察结论，"
        "目标是指导冷启动阶段该优先优化哪些数据质量维度。"
        "不要编造不存在的数据。"
    )

    user_prompt = textwrap.dedent(
        """
        请基于下面 JSON 数据，生成一份中文 Markdown 实验结论报告。
        本次分析目标：
        {goal}

        输出要求：
        1. 标题为“# 实验观察结论（DeepSeek）”。
        2. 包含以下部分：
           - 1) 冷启动结论摘要（3-6 条）
           - 2) 关键相关性解读（按训练指标分组：如 test_accuracy/val_loss/train_loss）
           - 3) 冷启动优先级排序（给出 Top 数据指标，并说明为什么优先）
           - 4) 指标语义与因果方向提醒（哪些相关性更可能是 proxy，哪些值得做干预实验）
           - 5) 下一轮实验建议（至少 5 条，必须具体到可执行动作）
             每条建议要包含：改动指标、改动方向（升/降）、预期影响指标（test_accuracy/val_loss/train_loss）、验证方法
           - 6) 风险与有效性威胁（至少 3 条）
           - 7) 新的数据质量评估指标示例（至少 5 个）
             每个示例必须包含：指标名、定义/公式、可计算方式（可给伪代码）、为什么对冷启动有价值、
             预期与 test_accuracy/val_loss/train_loss 的关系方向、以及最小验证实验设计
        3. 每条关键结论尽量引用具体 pair 和相关系数（保留 4 位小数）。
        4. 如果发现指标定义存在实现偏差风险，请明确指出（例如 IFD 的实现细节可能导致解释偏差）。
        5. 禁止输出与输入数据不一致的数字。

        输入数据 JSON：
        """
    ).format(goal=goal).strip()

    def _request(messages: List[Dict[str, str]]) -> Tuple[str, str]:
        request_body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        url = api_base.rstrip("/") + "/chat/completions"
        data = json.dumps(request_body).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                response_text = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"DeepSeek API HTTPError {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"DeepSeek API URLError: {e}") from e

        try:
            obj = json.loads(response_text)
            choice = obj["choices"][0]
            content = choice["message"]["content"]
            finish_reason = str(choice.get("finish_reason", ""))
            return content, finish_reason
        except Exception as e:
            raise RuntimeError(f"Failed to parse DeepSeek response: {e}; raw={response_text[:500]}") from e

    base_user_msg = user_prompt + "\n\n```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": base_user_msg},
    ]

    full_content_parts: List[str] = []
    rounds = 0
    while True:
        rounds += 1
        content, finish_reason = _request(messages)
        full_content_parts.append(content)

        if not auto_continue:
            break
        if finish_reason not in {"length", "max_tokens"}:
            break
        if rounds >= max_rounds:
            logger.warning("Reached max continuation rounds (%d), stop continuing", max_rounds)
            break

        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": (
                "继续上一次输出，严格从中断处续写，不要重复已写内容。"
                "请保持同一份 Markdown 文档结构并完整收尾。"
            ),
        })

    return "".join(full_content_parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze observation with DeepSeek API and export markdown conclusions."
    )
    parser.add_argument(
        "--obs_dir",
        required=True,
        help="Path to observation directory OR run output directory containing observation/",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Read strong_correlations_{threshold}.csv, default: 0.4",
    )
    parser.add_argument(
        "--include_min_max",
        action="store_true",
        help="Include min/max metrics in analysis payload (default: excluded)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Top-N pairs to pass for each ranking list",
    )
    parser.add_argument(
        "--api_base",
        default="https://api.deepseek.com",
        help="DeepSeek API base URL",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="DeepSeek chat model, e.g. deepseek-chat / deepseek-reasoner",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8000,
        help="Max generated tokens per API call (auto-continue may issue multiple calls)",
    )
    parser.add_argument(
        "--no_auto_continue",
        action="store_true",
        help="Disable auto-continue when response is cut by token limit",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=8,
        help="Maximum continuation rounds when response is truncated",
    )
    parser.add_argument(
        "--api_key_env",
        default="DEEPSEEK_API_KEY",
        help="Environment variable name for DeepSeek API key",
    )
    parser.add_argument(
        "--goal",
        default=(
            "我当前在做冷启动数据研究，流程是：先把同一数据集划分成多个 split；"
            "对每个 split 计算数据质量与分布指标（长度、覆盖率、唯一性、多样性、熵、PPL/IFD 等）；"
            "然后分别训练并得到 train_loss、val_loss、test_accuracy；"
            "最后分析“数据指标 -> 训练表现”的相关关系。"
            "请你基于 correlations.csv 和 strong_correlations_0.4.csv，回答："
            "1) 哪些数据性质对冷启动最有用（按优先级排序）；"
            "2) 这些性质更可能提升 test_accuracy 还是降低 val_loss/train_loss；"
            "3) 哪些相关性可能只是 proxy 或偶然相关；"
            "4) 下一轮我该如何改数据（筛选、重采样、去重、长度控制、难度配比、分布约束），"
            "每条建议给出可执行规则和验证方案；"
            "5) 除了现有指标，再提出一批潜在改进指标（可计算、可解释、可用于下一轮数据迭代），"
            "并说明预期作用方向。"
        ),
        help="Analysis goal passed to DeepSeek prompt",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown path (default: <obs_dir>/observation_conclusion_deepseek.md)",
    )
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"Missing API key. Please set env var: {args.api_key_env}")

    obs_dir = _resolve_observation_dir(Path(args.obs_dir))
    corr_path = obs_dir / "correlations.csv"
    strong_path = obs_dir / f"strong_correlations_{args.threshold:.1f}.csv"

    corr_df = _load_correlations(corr_path)
    if strong_path.exists():
        strong_df = _load_correlations(strong_path)
    else:
        logger.warning("%s not found, fallback to threshold filtering from correlations.csv", strong_path.name)
        strong_df = corr_df[corr_df["abs_corr"] >= args.threshold].copy()

    repo_root = Path(__file__).resolve().parents[2]
    metric_doc_excerpt = _load_metric_doc(repo_root)

    payload = _build_payload(
        obs_dir=obs_dir,
        corr_df=corr_df,
        strong_df=strong_df,
        threshold=args.threshold,
        exclude_min_max=not args.include_min_max,
        top_n=args.top_n,
        metric_doc_excerpt=metric_doc_excerpt,
    )

    logger.info(
        "Calling DeepSeek API with model=%s, corr_used=%s, strong_used=%s",
        args.model,
        payload["counts"]["correlations_used"],
        payload["counts"]["strong_used"],
    )
    markdown = _call_deepseek_chat(
        api_key=api_key,
        api_base=args.api_base,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        payload=payload,
        goal=args.goal,
        auto_continue=not args.no_auto_continue,
        max_rounds=args.max_rounds,
    )

    output_path = Path(args.output) if args.output else (obs_dir / "observation_conclusion_deepseek.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown.strip() + "\n", encoding="utf-8")
    logger.info("Saved DeepSeek conclusion markdown to: %s", output_path)
    print(output_path)


if __name__ == "__main__":
    main()
