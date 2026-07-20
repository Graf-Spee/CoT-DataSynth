#!/usr/bin/env python3
"""Compute and plot pass@K from a vLLM responses parquet."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _score_response(compute_score, response: str, ground_truth: Any) -> bool:
    try:
        return float(compute_score(response, ground_truth)) > 0.0
    except Exception:
        return False


def _extract_prediction(extract_pred, response: str) -> str:
    if extract_pred is None:
        return response.strip()
    try:
        pred = extract_pred(response)
    except Exception:
        return ""
    return "" if pred is None else str(pred).strip()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.replace(",", "").replace("$", "").strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _answers_match(prediction: Any, ground_truth: Any) -> bool:
    pred_float = _to_float(prediction)
    gt_float = _to_float(ground_truth)
    if pred_float is not None and gt_float is not None:
        return pred_float == gt_float
    pred = "" if prediction is None else str(prediction).strip().lower()
    gt = "" if ground_truth is None else str(ground_truth).strip().lower()
    return pred == gt


def _majority_vote(predictions: list[str]) -> str:
    counts = Counter(predictions)
    if not counts:
        return ""
    max_count = max(counts.values())
    tied = {pred for pred, count in counts.items() if count == max_count}
    for pred in predictions:
        if pred in tied:
            return pred
    return ""


def _pass_at_k_estimate(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k estimator used by Codex/HumanEval style evaluation."""
    if num_correct <= 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


def compute_passk(
    responses_path: Path,
    reward_path: Path,
    output_dir: Path,
    *,
    max_k: int | None = None,
    baseline_pass1: float | None = None,
    title: str = "pass@K",
) -> dict[str, Any]:
    responses_path = responses_path.resolve()
    reward_path = reward_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(responses_path)
    if "responses" not in df.columns:
        raise ValueError(f"Missing 'responses' column in {responses_path}")
    if "reward_model" not in df.columns:
        raise ValueError(f"Missing 'reward_model' column in {responses_path}")

    reward_module = _load_module(reward_path, "passk_reward")
    compute_score = getattr(reward_module, "compute_score")
    extract_pred = getattr(reward_module, "extract_pred", None)

    per_problem: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        responses = row["responses"]
        if hasattr(responses, "tolist"):
            responses = responses.tolist()
        responses = list(responses)
        ground_truth = row["reward_model"]["ground_truth"]
        correct = [_score_response(compute_score, str(response), ground_truth) for response in responses]
        predictions = [_extract_prediction(extract_pred, str(response)) for response in responses]
        per_problem.append(
            {
                "index": int(idx),
                "ground_truth": ground_truth,
                "num_samples": len(correct),
                "num_correct": int(sum(correct)),
                "correct": correct,
                "predictions": predictions,
            }
        )

    if not per_problem:
        raise ValueError(f"No rows found in {responses_path}")

    n_values = {item["num_samples"] for item in per_problem}
    if len(n_values) != 1:
        raise ValueError(f"Expected same number of samples per problem, got: {sorted(n_values)}")
    n = n_values.pop()
    if n <= 0:
        raise ValueError("Each problem has zero responses")

    max_k = n if max_k is None or max_k <= 0 else min(max_k, n)
    k_values = list(range(1, max_k + 1))

    rows = []
    pass_at_1 = sum(_pass_at_k_estimate(n, item["num_correct"], 1) for item in per_problem) / len(per_problem)
    gain_baseline = pass_at_1 if baseline_pass1 is None else baseline_pass1
    for k in k_values:
        prefix_pass = sum(any(item["correct"][:k]) for item in per_problem) / len(per_problem)
        pass_at_k = sum(_pass_at_k_estimate(n, item["num_correct"], k) for item in per_problem) / len(per_problem)
        sc_correct = 0
        for item in per_problem:
            voted_answer = _majority_vote(item["predictions"][:k])
            if _answers_match(voted_answer, item["ground_truth"]):
                sc_correct += 1
        sc_accuracy = sc_correct / len(per_problem)
        rows.append(
            {
                "k": k,
                "pass_at_k": pass_at_k,
                "prefix_pass_at_k": prefix_pass,
                "self_consistency_accuracy": sc_accuracy,
                "self_consistency_gain": sc_accuracy - gain_baseline,
                "percent": pass_at_k * 100.0,
                "prefix_percent": prefix_pass * 100.0,
                "self_consistency_percent": sc_accuracy * 100.0,
                "self_consistency_gain_pp": (sc_accuracy - gain_baseline) * 100.0,
            }
        )

    metrics_df = pd.DataFrame(rows)
    csv_path = output_dir / "passk_metrics.csv"
    json_path = output_dir / "passk_metrics.json"
    per_problem_path = output_dir / "passk_per_problem.json"
    png_path = output_dir / "passk_curve.png"

    metrics_df.to_csv(csv_path, index=False)
    payload = {
        "responses_path": str(responses_path),
        "reward_path": str(reward_path),
        "num_problems": len(per_problem),
        "num_samples_per_problem": n,
        "pass_at_1_baseline": pass_at_1,
        "self_consistency_gain_baseline": gain_baseline,
        "self_consistency_gain_baseline_source": "pass@1_estimate" if baseline_pass1 is None else "external",
        "passk": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    per_problem_path.write_text(json.dumps(per_problem, indent=2, ensure_ascii=False), encoding="utf-8")

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.plot(metrics_df["k"], metrics_df["percent"], marker="o", linewidth=2.2, label="pass@K")
    ax.plot(
        metrics_df["k"],
        metrics_df["prefix_percent"],
        marker="s",
        linewidth=1.6,
        linestyle="--",
        label="first-K empirical",
        alpha=0.75,
    )
    ax.plot(
        metrics_df["k"],
        metrics_df["self_consistency_percent"],
        marker="^",
        linewidth=2.0,
        label="self-consistency",
    )
    ax.set_xlabel("K")
    ax.set_ylabel("Performance (%)")
    ax.set_title(title)
    ax.set_xticks(k_values)
    ymax = max(
        metrics_df["prefix_percent"].max(),
        metrics_df["percent"].max(),
        metrics_df["self_consistency_percent"].max(),
    )
    ax.set_ylim(0, min(100, ymax + 8))
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)

    gain_png_path = output_dir / "self_consistency_gain_curve.png"
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    ax.axhline(0, color="#666666", linewidth=1.0)
    ax.plot(
        metrics_df["k"],
        metrics_df["self_consistency_gain_pp"],
        marker="o",
        linewidth=2.0,
        color="#2ca02c",
    )
    ax.set_xlabel("K")
    ax.set_ylabel("SC Gain (pp)")
    ax.set_title("Self-Consistency Gain")
    ax.set_xticks(k_values)
    fig.tight_layout()
    fig.savefig(gain_png_path)
    plt.close(fig)

    print(f"[INFO] Wrote {csv_path}")
    print(f"[INFO] Wrote {json_path}")
    print(f"[INFO] Wrote {png_path}")
    print(f"[INFO] Wrote {gain_png_path}")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute and plot pass@K from vLLM responses.")
    parser.add_argument("--responses", required=True, help="Path to generated/responses.parquet")
    parser.add_argument(
        "--reward",
        default=str(REPO_ROOT / "verl" / "utils" / "reward_score" / "gsm8k.py"),
        help="Reward Python file with compute_score().",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for pass@K artifacts")
    parser.add_argument("--max-k", type=int, default=0, help="Maximum K to plot (0 = all samples)")
    parser.add_argument(
        "--baseline-pass1",
        type=float,
        default=None,
        help="Optional pass@1 baseline accuracy in [0, 1] for self-consistency gain.",
    )
    parser.add_argument("--title", default="Qwen3-4B on GSM8K pass@K")
    args = parser.parse_args()

    compute_passk(
        responses_path=Path(args.responses),
        reward_path=Path(args.reward),
        output_dir=Path(args.output_dir),
        max_k=args.max_k,
        baseline_pass1=args.baseline_pass1,
        title=args.title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
