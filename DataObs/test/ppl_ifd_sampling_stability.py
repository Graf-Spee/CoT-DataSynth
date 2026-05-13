#!/usr/bin/env python3
"""
Estimate whether sampled PPL/IFD can represent full-split values.

Workflow:
1. Compute full-split per-sample PPL/IFD for one split (single pass baseline).
2. For multiple sample ratios, repeatedly subsample from baseline values.
3. Report sampling error vs. full baseline.
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Avoid tokenizer parallelism fork warnings in long offline runs.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_lib.advanced_metrics import compute_ifd_for_sample, compute_ppl_for_text

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _extract_prompt_text(item: Dict) -> str:
    prompt = item.get("prompt", None)
    if prompt is not None:
        if isinstance(prompt, list):
            return " ".join(str(p.get("content", "")) for p in prompt).strip()
        return str(prompt).strip()

    for key in ("question", "query", "instruction", "input"):
        value = item.get(key, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _extract_answer_text(item: Dict) -> str:
    extra_info = item.get("extra_info", {})
    if isinstance(extra_info, dict):
        answer = extra_info.get("answer", None)
        if answer is not None and str(answer).strip():
            return str(answer).strip()

    for key in ("answer", "output", "response", "solution"):
        value = item.get(key, None)
        if value is not None and str(value).strip():
            return str(value).strip()

    reward_model = item.get("reward_model", {})
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth", None)
        if ground_truth is not None and str(ground_truth).strip():
            return str(ground_truth).strip()
    return ""


def _load_split(path: str) -> List[Dict]:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p).to_dict("records")
    if p.suffix == ".jsonl":
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows
    if p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported split format: {p}")


def _parse_ratios(ratio_str: str) -> List[float]:
    ratios = []
    for x in ratio_str.split(","):
        x = x.strip()
        if not x:
            continue
        v = float(x)
        if v <= 0 or v > 1:
            raise ValueError(f"Ratio must be in (0, 1], got {v}")
        ratios.append(v)
    if not ratios:
        raise ValueError("No valid ratios provided")
    return sorted(set(ratios))


def _safe_rel_err(sample: float, baseline: float) -> float:
    if baseline == 0:
        return math.nan
    return abs(sample - baseline) / abs(baseline)


def _sampling_report(
    values: List[float],
    metric: str,
    ratios: List[float],
    repeats: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    arr = np.asarray(values, dtype=np.float64)
    baseline = float(np.mean(arr))
    rng = np.random.default_rng(seed)

    records = []
    for ratio in ratios:
        n = max(1, int(round(len(arr) * ratio)))
        for rep in range(repeats):
            idx = rng.choice(len(arr), size=n, replace=False)
            sample_mean = float(np.mean(arr[idx]))
            abs_err = abs(sample_mean - baseline)
            rel_err = _safe_rel_err(sample_mean, baseline)
            records.append(
                {
                    "metric": metric,
                    "ratio": ratio,
                    "repeat": rep,
                    "sample_size": n,
                    "sample_mean": sample_mean,
                    "baseline_mean": baseline,
                    "abs_error": abs_err,
                    "rel_error": rel_err,
                }
            )

    raw_df = pd.DataFrame(records)
    summary_df = (
        raw_df.groupby(["metric", "ratio", "sample_size"], as_index=False)
        .agg(
            repeats=("repeat", "count"),
            baseline_mean=("baseline_mean", "first"),
            sample_mean_avg=("sample_mean", "mean"),
            sample_mean_std=("sample_mean", "std"),
            abs_error_mean=("abs_error", "mean"),
            abs_error_p95=("abs_error", lambda x: np.percentile(x, 95)),
            rel_error_mean=("rel_error", "mean"),
            rel_error_p95=("rel_error", lambda x: np.percentile(x.dropna(), 95) if len(x.dropna()) else np.nan),
        )
        .sort_values(["metric", "ratio"])
    )

    baseline_info = {
        "metric": metric,
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    return raw_df, summary_df, baseline_info


def main() -> None:
    parser = argparse.ArgumentParser(description="PPL/IFD sampling stability on one split")
    parser.add_argument("--split_path", required=True, help="Path to split parquet/jsonl/json")
    parser.add_argument("--model", required=True, help="HF model path/name for PPL/IFD")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--ratios", default="0.01,0.02,0.05,0.1,0.2,0.3,0.5", help="Comma-separated sample ratios")
    parser.add_argument("--repeats", type=int, default=20, help="Repeats per ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_records", type=int, default=None, help="Optional cap for debugging")
    parser.add_argument("--output_dir", required=True, help="Directory for report outputs")
    args = parser.parse_args()

    ratios = _parse_ratios(args.ratios)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading split data from %s", args.split_path)
    data = _load_split(args.split_path)
    if args.max_records and len(data) > args.max_records:
        random.Random(args.seed).shuffle(data)
        data = data[: args.max_records]
        logger.info("Using max_records=%d after shuffle", args.max_records)

    answers: List[str] = []
    pairs: List[Tuple[str, str]] = []
    for item in data:
        prompt = _extract_prompt_text(item)
        answer = _extract_answer_text(item)
        if answer:
            answers.append(answer)
        if prompt and answer:
            pairs.append((prompt, answer))

    if not answers:
        raise ValueError("No valid answers found for PPL")
    if not pairs:
        raise ValueError("No valid prompt-answer pairs found for IFD")

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model/tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    model.eval()

    start = time.time()

    ppl_values: List[float] = []
    if tqdm is not None:
        answer_iter = tqdm(answers, desc="Full PPL", unit="sample", dynamic_ncols=True)
    else:
        answer_iter = answers

    for i, answer in enumerate(answer_iter):
        try:
            value = compute_ppl_for_text(answer, model, tokenizer, device=args.device)
            if not np.isnan(value) and not np.isinf(value):
                ppl_values.append(float(value))
        except Exception as e:
            logger.warning("PPL failed at sample %d: %s", i, e)

    ifd_values: List[float] = []
    if tqdm is not None:
        pair_iter = tqdm(pairs, desc="Full IFD", unit="sample", dynamic_ncols=True)
    else:
        pair_iter = pairs

    for i, (prompt, answer) in enumerate(pair_iter):
        try:
            value = compute_ifd_for_sample(prompt, answer, model, tokenizer, device=args.device)
            if not np.isnan(value) and not np.isinf(value) and value > 0:
                ifd_values.append(float(value))
        except Exception as e:
            logger.warning("IFD failed at sample %d: %s", i, e)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not ppl_values:
        raise ValueError("No valid full-split PPL values")
    if not ifd_values:
        raise ValueError("No valid full-split IFD values")

    ppl_raw, ppl_summary, ppl_baseline = _sampling_report(
        ppl_values, "ppl", ratios, args.repeats, args.seed
    )
    ifd_raw, ifd_summary, ifd_baseline = _sampling_report(
        ifd_values, "ifd", ratios, args.repeats, args.seed + 1
    )

    raw_df = pd.concat([ppl_raw, ifd_raw], ignore_index=True)
    summary_df = pd.concat([ppl_summary, ifd_summary], ignore_index=True)

    raw_path = out_dir / "sampling_stability_raw.csv"
    summary_path = out_dir / "sampling_stability_summary.csv"
    json_path = out_dir / "sampling_stability_report.json"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    report = {
        "split_path": args.split_path,
        "model": args.model,
        "device": args.device,
        "ratios": ratios,
        "repeats": args.repeats,
        "seed": args.seed,
        "max_records": args.max_records,
        "elapsed_sec": round(time.time() - start, 2),
        "baseline": {
            "ppl": ppl_baseline,
            "ifd": ifd_baseline,
        },
        "outputs": {
            "raw_csv": str(raw_path),
            "summary_csv": str(summary_path),
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("Done. Report: %s", json_path)
    logger.info("PPL baseline mean=%.6f | IFD baseline mean=%.6f", ppl_baseline["mean"], ifd_baseline["mean"])


if __name__ == "__main__":
    main()
