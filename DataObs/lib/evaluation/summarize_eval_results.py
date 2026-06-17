#!/usr/bin/env python3
"""Summarize evaluation outputs produced by DataObs evaluation scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _as_list(value):
    if isinstance(value, list):
        return value
    return [value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DataObs/verl evaluation results.")
    parser.add_argument("path", help="Evaluation directory or responses_labeled.json/parquet path")
    parser.add_argument("--show-errors", type=int, default=0, help="Print N incorrect examples")
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_dir():
        json_path = path / "generated" / "responses_labeled.json"
        parquet_path = path / "generated" / "responses_labeled.parquet"
        if json_path.exists():
            path = json_path
        elif parquet_path.exists():
            path = parquet_path
        else:
            raise FileNotFoundError(f"No generated/responses_labeled.json or parquet under {args.path}")

    if path.suffix == ".json":
        df = pd.read_json(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported result file: {path}")

    if "is_correct" not in df.columns or "reward_scores" not in df.columns:
        raise KeyError(f"Expected columns is_correct and reward_scores, got: {list(df.columns)}")

    row_correct = df["is_correct"].apply(lambda xs: any(bool(x) for x in _as_list(xs)))
    response_correct = [bool(x) for xs in df["is_correct"] for x in _as_list(xs)]
    response_scores = [float(x) for xs in df["reward_scores"] for x in _as_list(xs)]

    total_rows = len(df)
    correct_rows = int(row_correct.sum())
    total_responses = len(response_correct)
    correct_responses = sum(response_correct)
    mean_reward = sum(response_scores) / total_responses if total_responses else 0.0

    print(f"file: {path}")
    print(f"num_examples: {total_rows}")
    print(f"num_responses: {total_responses}")
    print(f"pass@1/example_accuracy: {correct_rows / total_rows * 100:.4f}% ({correct_rows}/{total_rows})")
    print(f"response_accuracy: {correct_responses / total_responses * 100:.4f}% ({correct_responses}/{total_responses})")
    print(f"mean_reward: {mean_reward:.6f}")

    if args.show_errors > 0:
        shown = 0
        print("\nIncorrect examples:")
        for idx, row in df.loc[~row_correct].iterrows():
            extra = row.get("extra_info", {}) or {}
            reward_model = row.get("reward_model", {}) or {}
            question = extra.get("question")
            if question is None:
                prompt = row.get("prompt")
                question = prompt[0].get("content") if isinstance(prompt, list) and prompt else prompt
            responses = _as_list(row.get("responses", []))
            preds = _as_list(row.get("preds", [])) if "preds" in row else []
            print("-" * 80)
            print(f"index: {idx}")
            print(f"question: {question}")
            print(f"ground_truth: {reward_model.get('ground_truth')}")
            print(f"pred: {preds[0] if preds else None}")
            print(f"response: {responses[0] if responses else None}")
            shown += 1
            if shown >= args.show_errors:
                break


if __name__ == "__main__":
    main()
