#!/usr/bin/env python3
"""Run OpenCompass-style Qwen3 base evals for AIME25, IFEval, and GPQA-Diamond."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_BIN = Path("/home/hrh/anaconda3/envs/verl-torch280/bin/python")
DATASETS = ("aime25", "ifeval", "gpqa-diamond")


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace("-", "_").replace(".", "_")


def run_one(
    *,
    model_path: str,
    dataset: str,
    gpu_ids: str,
    tensor_parallel_size: int,
    output_root: Path,
    gpu_memory_utilization: float,
    max_response_len: int,
) -> dict:
    model_name = Path(model_path.rstrip("/")).name
    out_dir = output_root / model_name / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PYTHON_BIN),
        str(REPO_ROOT / "DataObs" / "lib" / "evaluation" / "eval_vllm.py"),
        "--model-path", model_path,
        "--base-model", model_path,
        "--dataset", dataset,
        "--output-dir", str(out_dir),
        "--gpu-ids", gpu_ids,
        "--num-samples", "1",
        "--temperature", "0.0",
        "--top-p", "1.0",
        "--top-k", "-1",
        "--max-response-len", str(max_response_len),
        "--batch-size", "1",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--prompt-template-method", "zeroshot",
        "--seed", "42",
        "--enable-thinking",
    ]

    start = time.time()
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    elapsed = time.time() - start
    result_path = out_dir / "generated" / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "dataset": dataset,
        "output_dir": str(out_dir),
        "elapsed_sec_total": elapsed,
        **result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--output-root", default="/data/hrh/COT/eval_outputs/qwen3_base_opencompass_aligned")
    parser.add_argument("--max-response-len", type=int, default=32768)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    model_name = Path(args.model_path.rstrip("/")).name
    summary_path = output_root / model_name / "summary.json"
    summary = {
        "model_path": args.model_path,
        "gpu_ids": args.gpu_ids,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_response_len": args.max_response_len,
        "enable_thinking": True,
        "datasets": {},
    }

    for dataset in DATASETS:
        try:
            result = run_one(
                model_path=args.model_path,
                dataset=dataset,
                gpu_ids=args.gpu_ids,
                tensor_parallel_size=args.tensor_parallel_size,
                output_root=output_root,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_response_len=args.max_response_len,
            )
            summary["datasets"][dataset] = {"success": True, **result}
        except Exception as exc:
            summary["datasets"][dataset] = {
                "success": False,
                "error": repr(exc),
            }
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
