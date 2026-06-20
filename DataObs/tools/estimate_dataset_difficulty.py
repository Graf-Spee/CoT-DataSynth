#!/usr/bin/env python3
"""Estimate per-row dataset difficulty with repeated teacher sampling.

For each input prompt, this script asks a model to solve it multiple times,
scores every response with the dataset reward function, and writes a row-level
difficulty table. A low pass count means the prompt is hard for this model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DataObs.scripts.experiment_pipeline import dataset_default  # noqa: E402


def load_distill_module() -> Any:
    module_path = REPO_ROOT / "DataObs" / "lib" / "data_process" / "cot_distill_teacher_filter.py"
    spec = importlib.util.spec_from_file_location("cot_distill_teacher_filter_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load distill module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DISTILL = load_distill_module()
LocalTeacher = DISTILL.LocalTeacher
SUPPORTED_DATASETS = DISTILL.SUPPORTED_DATASETS
_append_reasoning_suffix = DISTILL._append_reasoning_suffix
_get_ground_truth = DISTILL._get_ground_truth
_load_reward_functions = DISTILL._load_reward_functions
_normalize_dataset_name = DISTILL._normalize_dataset_name


def safe_mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def difficulty_bucket(pass_count: int, *, hard_max_correct: int, easy_min_correct: int) -> str:
    if pass_count <= hard_max_correct:
        return "hard"
    if pass_count >= easy_min_correct:
        return "easy"
    return "medium"


def prepare_rows(df: pd.DataFrame, dataset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_index, row in df.iterrows():
        question, messages = _append_reasoning_suffix(row["prompt"], dataset)
        gold = _get_ground_truth(row["reward_model"])
        if gold is None:
            raise ValueError(f"Missing reward_model.ground_truth at row index {source_index}")
        rows.append(
            {
                "source_index": int(source_index) if isinstance(source_index, int) else str(source_index),
                "question": question,
                "messages": messages,
                "gold_answer": gold,
                "data_source": row.get("data_source", dataset),
            }
        )
    return rows


def write_bucket_parquets(
    *,
    input_df: pd.DataFrame,
    difficulty_df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    source_index_to_label = {str(label): label for label in input_df.index.tolist()}

    for bucket in ("easy", "medium", "hard"):
        source_indices = difficulty_df.loc[difficulty_df["difficulty_bucket"] == bucket, "source_index"].tolist()
        labels: list[Any] = []
        for source_index in source_indices:
            label = source_index_to_label.get(str(source_index))
            if label is not None:
                labels.append(label)
        bucket_df = input_df.loc[labels].copy() if labels else input_df.iloc[0:0].copy()
        out_path = output_dir / f"{prefix}_{bucket}.parquet"
        bucket_df.to_parquet(out_path, index=False)
        paths[bucket] = str(out_path)
    return paths


def write_sft_bucket_parquets(
    *,
    sft_file: Path,
    difficulty_df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
) -> dict[str, str]:
    sft_df = pd.read_parquet(sft_file)
    if "source_index" not in sft_df.columns:
        raise ValueError(f"SFT parquet must contain source_index column: {sft_file}")

    index_to_bucket = {
        str(row["source_index"]): row["difficulty_bucket"]
        for row in difficulty_df[["source_index", "difficulty_bucket"]].to_dict("records")
    }
    work_df = sft_df.copy()
    work_df["_difficulty_bucket"] = work_df["source_index"].map(lambda x: index_to_bucket.get(str(x), "unknown"))

    paths: dict[str, str] = {}
    for bucket in ("easy", "medium", "hard"):
        bucket_df = work_df[work_df["_difficulty_bucket"] == bucket].drop(columns=["_difficulty_bucket"])
        out_path = output_dir / f"{prefix}_sft_{bucket}.parquet"
        bucket_df.to_parquet(out_path, index=False)
        paths[bucket] = str(out_path)
    return paths


def run(args: argparse.Namespace) -> None:
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    dataset = _normalize_dataset_name(args.dataset)
    input_file = args.input_file or dataset_default(dataset, "distill_input")
    if not input_file:
        raise SystemExit(f"[ERROR] No default input for dataset={args.dataset}. Pass --input-file.")

    input_path = Path(input_file)
    if not input_path.is_file():
        raise SystemExit(f"[ERROR] Input parquet not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"{dataset}_difficulty_{Path(args.model_id).name}"
    output_file = Path(args.output_file) if args.output_file else output_dir / f"{prefix}.parquet"
    candidates_file = (
        Path(args.candidates_file)
        if args.candidates_file
        else output_file.with_name(f"{output_file.stem}.candidates.parquet")
    )
    summary_file = output_file.with_name(f"{output_file.stem}.summary.json")

    df = pd.read_parquet(input_path)
    if args.max_rows > 0:
        df = df.head(args.max_rows)

    prepared_rows = prepare_rows(df, dataset)
    compute_score = _load_reward_functions(dataset)

    teacher = LocalTeacher(
        model_id=args.model_id,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    generated_responses: list[list[str]] = []
    for start in tqdm(range(0, len(prepared_rows), args.gen_batch_size), desc="Generation", unit="batch"):
        batch_rows = prepared_rows[start : start + args.gen_batch_size]
        generated_responses.extend(
            teacher.generate_batch(
                messages_batch=[row["messages"] for row in batch_rows],
                n=args.num_attempts,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        )

    row_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    for row_data, answers in tqdm(
        zip(prepared_rows, generated_responses),
        total=len(prepared_rows),
        desc="Scoring",
        unit="row",
    ):
        scores: list[float] = []
        pass_count = 0
        for attempt_index, answer in enumerate(answers):
            score = float(compute_score(answer, row_data["gold_answer"]))
            is_correct = score >= args.correct_threshold
            pass_count += int(is_correct)
            scores.append(score)
            candidate_records.append(
                {
                    "source_index": row_data["source_index"],
                    "attempt_index": int(attempt_index),
                    "answer": answer,
                    "teacher_score": score,
                    "is_correct": bool(is_correct),
                    "answer_char_len": int(len(str(answer))),
                }
            )

        bucket = difficulty_bucket(
            pass_count,
            hard_max_correct=args.hard_max_correct,
            easy_min_correct=args.easy_min_correct,
        )
        row_records.append(
            {
                "source_index": row_data["source_index"],
                "dataset": dataset,
                "data_source": row_data["data_source"],
                "question": row_data["question"],
                "gold_answer": row_data["gold_answer"],
                "model_id": args.model_id,
                "num_attempts": int(args.num_attempts),
                "pass_count": int(pass_count),
                "fail_count": int(args.num_attempts - pass_count),
                "pass_rate": float(pass_count / args.num_attempts) if args.num_attempts else 0.0,
                "difficulty_bucket": bucket,
                "difficulty_score": float(1.0 - pass_count / args.num_attempts) if args.num_attempts else 1.0,
                "score_mean": safe_mean(scores),
                "score_min": min(scores) if scores else None,
                "score_max": max(scores) if scores else None,
            }
        )

    difficulty_df = pd.DataFrame(row_records)
    difficulty_df.to_parquet(output_file, index=False)
    pd.DataFrame(candidate_records).to_parquet(candidates_file, index=False)

    bucket_paths: dict[str, str] = {}
    if args.write_bucket_parquets:
        bucket_paths = write_bucket_parquets(
            input_df=df,
            difficulty_df=difficulty_df,
            output_dir=output_dir,
            prefix=prefix,
        )

    sft_bucket_paths: dict[str, str] = {}
    if args.sft_file:
        sft_path = Path(args.sft_file)
        if not sft_path.is_file():
            raise SystemExit(f"[ERROR] SFT parquet not found: {sft_path}")
        sft_bucket_paths = write_sft_bucket_parquets(
            sft_file=sft_path,
            difficulty_df=difficulty_df,
            output_dir=output_dir,
            prefix=prefix,
        )

    bucket_counts = difficulty_df["difficulty_bucket"].value_counts().to_dict() if not difficulty_df.empty else {}
    summary = {
        "dataset": dataset,
        "input_file": str(input_path),
        "model_id": args.model_id,
        "output_file": str(output_file),
        "candidates_file": str(candidates_file),
        "seed_bucket_parquets": bucket_paths,
        "sft_bucket_parquets": sft_bucket_paths,
        "num_rows": int(len(difficulty_df)),
        "num_attempts": int(args.num_attempts),
        "correct_threshold": float(args.correct_threshold),
        "hard_max_correct": int(args.hard_max_correct),
        "easy_min_correct": int(args.easy_min_correct),
        "bucket_counts": {str(k): int(v) for k, v in bucket_counts.items()},
        "mean_pass_rate": float(difficulty_df["pass_rate"].mean()) if not difficulty_df.empty else None,
        "elapsed_sec": float(time.time() - start_time),
        "generation": {
            "gpu_ids": args.gpu_ids,
            "gen_batch_size": int(args.gen_batch_size),
            "tensor_parallel_size": int(args.tensor_parallel_size),
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "do_sample": bool(args.do_sample),
            "dtype": args.dtype,
        },
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    del teacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate dataset difficulty by sampling a teacher model multiple times per row."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset name. Supported: {', '.join(SUPPORTED_DATASETS)}.",
    )
    parser.add_argument("--input-file", default="", help="Input processed parquet. Defaults to pipeline distill input.")
    parser.add_argument("--model-id", required=True, help="Teacher/base model path or model id.")
    parser.add_argument("--output-dir", default="/data/hrh/COT/difficulty", help="Output directory.")
    parser.add_argument("--output-prefix", default="", help="Output filename prefix.")
    parser.add_argument("--output-file", default="", help="Override row-level output parquet path.")
    parser.add_argument("--candidates-file", default="", help="Override candidate output parquet path.")
    parser.add_argument(
        "--sft-file",
        default="",
        help="Optional filtered_sft.parquet to split into <prefix>_sft_easy/medium/hard.parquet by source_index.",
    )

    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--num-attempts", type=int, default=10, help="Number of sampled answers per row.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gen-batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--correct-threshold", type=float, default=0.99)
    parser.add_argument("--max-rows", type=int, default=0, help="Debug limit. 0 means all rows.")

    parser.add_argument("--hard-max-correct", type=int, default=3, help="pass_count <= this is hard.")
    parser.add_argument("--easy-min-correct", type=int, default=8, help="pass_count >= this is easy.")
    parser.add_argument(
        "--write-bucket-parquets",
        action="store_true",
        help="Also write original seed rows into <prefix>_easy/medium/hard.parquet.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
