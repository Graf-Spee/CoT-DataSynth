#!/usr/bin/env python3
"""
Run BFCL generation + evaluation with DataObs-compatible inputs/outputs.

Usage:
  python scripts/eval_bfcl_dataobs.py \
    <model_path> <base_model> <dataset_name> <output_path> <gpu_id>
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

BFCL_LOCAL_SERVER_PORT = "11053"


def _print_header(title: str) -> None:
    print("")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(title)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


def _run_and_tee(cmd: list[str], cwd: Path, log_path: Path, env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"$ {' '.join(shlex.quote(x) for x in cmd)}\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        return proc.wait()


def _count_gpus(gpu_id: str) -> int:
    ids = [x.strip() for x in gpu_id.split(",") if x.strip()]
    return max(1, len(ids))


def _resolve_test_category(dataset_name: str) -> str:
    ds = dataset_name.strip()
    ds_lower = ds.lower()
    if ds_lower in {"bfcl", "bfcl_v4", "bfcl-v4"}:
        return "all_scoring"
    if ds_lower.startswith("bfcl:"):
        value = ds.split(":", 1)[1].strip()
        return value or "all_scoring"
    if ds_lower.startswith("bfcl_"):
        value = ds.split("_", 1)[1].strip()
        return value or "all_scoring"
    return ds


def _extract_overall_accuracy(score_csv: Path, model_name: str) -> Optional[float]:
    if not score_csv.exists():
        return None

    with score_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    chosen = None
    for row in rows:
        model_col = (row.get("Model") or "").strip()
        if model_col == model_name:
            chosen = row
            break

    if chosen is None:
        chosen = rows[0]

    overall = (chosen.get("Overall Acc") or "").strip()
    # Expected format: "73.45%"
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", overall)
    if not match:
        return None
    return float(match.group(1)) / 100.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run BFCL generation/eval with DataObs-compatible interface."
    )
    parser.add_argument("checkpoint_path")
    parser.add_argument("base_model")
    parser.add_argument("dataset_name")
    parser.add_argument("output_path")
    parser.add_argument("gpu_id", nargs="?", default="0")
    parser.add_argument(
        "--bfcl-model-name",
        default=None,
        help="BFCL registry model name passed to `bfcl generate/evaluate`. "
        "Default: use base_model as model name.",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "sglang"],
        default="vllm",
        help="Local inference backend for BFCL generation.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization passed to BFCL generation.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Optional thread count for BFCL generation.",
    )
    parser.add_argument(
        "--partial-eval",
        action="store_true",
        help="Enable BFCL partial evaluation.",
    )
    args = parser.parse_args()

    model_path = Path(args.checkpoint_path).resolve()
    eval_output_dir = Path(args.output_path).resolve()
    logs_dir = eval_output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"[ERROR] Model path not found: {model_path}")
        return 1

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    bfcl_root = (repo_root / "gorilla" / "berkeley-function-call-leaderboard").resolve()
    if not bfcl_root.exists():
        print(f"[ERROR] BFCL root not found: {bfcl_root}")
        return 1

    bfcl_model_name = args.bfcl_model_name or args.base_model
    test_category = _resolve_test_category(args.dataset_name)
    n_gpus = _count_gpus(args.gpu_id)

    print("==========================================")
    print("BFCL Evaluation")
    print("==========================================")
    print(f"Model Path: {model_path}")
    print(f"Base Model: {args.base_model}")
    print(f"BFCL Model Name: {bfcl_model_name}")
    print(f"Test Category: {test_category}")
    print(f"GPU: {args.gpu_id}")
    print(f"LOCAL_SERVER_PORT: {BFCL_LOCAL_SERVER_PORT}")
    print(f"BFCL Root: {bfcl_root}")
    print("==========================================")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    env["HF_HUB_OFFLINE"] = env.get("HF_HUB_OFFLINE", "1")
    env["TRANSFORMERS_OFFLINE"] = env.get("TRANSFORMERS_OFFLINE", "1")
    env["WANDB_MODE"] = env.get("WANDB_MODE", "offline")
    env["LOCAL_SERVER_PORT"] = BFCL_LOCAL_SERVER_PORT
    env.setdefault("LOCAL_SERVER_ENDPOINT", "localhost")

    bfcl_dir = eval_output_dir / "bfcl"
    env["BFCL_PROJECT_ROOT"] = str(bfcl_dir)
    result_dir_rel = "result"
    score_dir_rel = "score"
    (bfcl_dir / "result").mkdir(parents=True, exist_ok=True)
    (bfcl_dir / "score").mkdir(parents=True, exist_ok=True)

    _print_header("Stage 1: BFCL Generation")
    gen_cmd = [
        sys.executable,
        "-m",
        "bfcl_eval",
        "generate",
        "--model",
        bfcl_model_name,
        "--test-category",
        test_category,
        "--backend",
        args.backend,
        "--num-gpus",
        str(n_gpus),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--local-model-path",
        str(model_path),
        "--result-dir",
        result_dir_rel,
        "--allow-overwrite",
    ]
    if args.num_threads is not None:
        gen_cmd.extend(["--num-threads", str(args.num_threads)])

    ret = _run_and_tee(gen_cmd, cwd=bfcl_root, log_path=logs_dir / "generation.log", env=env)
    if ret != 0:
        print("[ERROR] Generation Failed!")
        return 1
    print("[INFO] Generation Done!")

    _print_header("Stage 2: BFCL Evaluation")
    eval_cmd = [
        sys.executable,
        "-m",
        "bfcl_eval",
        "evaluate",
        "--model",
        bfcl_model_name,
        "--test-category",
        test_category,
        "--result-dir",
        result_dir_rel,
        "--score-dir",
        score_dir_rel,
    ]
    if args.partial_eval:
        eval_cmd.append("--partial-eval")

    raw_eval_log = logs_dir / "evaluation_raw.log"
    ret = _run_and_tee(eval_cmd, cwd=bfcl_root, log_path=raw_eval_log, env=env)
    if ret != 0:
        print("[ERROR] Evaluation Failed!")
        return 1
    print("[INFO] Evaluation Done!")

    score_csv = bfcl_dir / "score" / "data_overall.csv"
    accuracy = _extract_overall_accuracy(score_csv=score_csv, model_name=bfcl_model_name)
    final_eval_log = logs_dir / "evaluation.log"
    with final_eval_log.open("w", encoding="utf-8") as fout:
        if accuracy is not None:
            fout.write(f"test_score: {accuracy:.6f}\n")
            fout.write(f"accuracy: {accuracy:.6f}\n")
        else:
            fout.write("accuracy: N/A\n")
        fout.write("\n")
        fout.write(f"score_csv: {score_csv}\n")
        fout.write(f"raw_eval_log: {raw_eval_log}\n\n")
        if raw_eval_log.exists():
            fout.write(raw_eval_log.read_text(encoding="utf-8"))

    print("")
    print("==========================================")
    print("Evaluation Results:")
    print("==========================================")
    if accuracy is None:
        print("[WARNING] Could not parse overall accuracy from BFCL score CSV.")
        print(f"See {score_csv} and {logs_dir / 'evaluation.log'} for details")
    else:
        print(f"test_score: {accuracy:.6f}")
        print(f"accuracy: {accuracy:.6f}")

    print("")
    print(f"Output directory: {eval_output_dir}")
    print(f"BFCL result directory: {bfcl_dir / 'result'}")
    print(f"BFCL score directory: {bfcl_dir / 'score'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
