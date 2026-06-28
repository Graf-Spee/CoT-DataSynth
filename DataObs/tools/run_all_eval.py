#!/usr/bin/env python3
"""Run base-model vLLM evaluation serially for the standard DataObs dataset suite."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("/data/hrh/COT/evals")
DEFAULT_GPU_MEMORY_UTILIZATION = 0.6
DEFAULT_TENSOR_PARALLEL_SIZE = 1
DEFAULT_PROMPT_TEMPLATE_METHOD = "zeroshot"

DATASETS = [
    "gsm8k",
    "math-500",
    "numinamath",
    "arc-challenge",
    "aqua-rat",
    "commonsenseqa",
    "strategyqa",
    "mbpp",
    "mbppplus",
    "humaneval",
    "humanevalplus",
]


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned.strip("._-") or "model"


def model_dir_name(model_path: str) -> str:
    path = Path(model_path.rstrip("/"))
    return safe_name(path.name or model_path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DataObs vLLM base-model eval for gsm8k/math/code/common-sense datasets."
    )
    parser.add_argument(
        "--gpu-ids",
        required=True,
        help="GPU ids used by evaluation, e.g. 0 or 0,1.",
    )
    parser.add_argument(
        "--base-model",
        "--model-path",
        dest="model_path",
        required=True,
        help="Base model path to evaluate.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        dest="tensor_parallel_size",
        type=int,
        default=DEFAULT_TENSOR_PARALLEL_SIZE,
        help=f"vLLM tensor parallel size. Default: {DEFAULT_TENSOR_PARALLEL_SIZE}.",
    )
    parser.add_argument(
        "--prompt-template-method",
        choices=["plain", "zeroshot", "fewshot"],
        default=DEFAULT_PROMPT_TEMPLATE_METHOD,
        help=f"Prompt template method for eval data preparation. Default: {DEFAULT_PROMPT_TEMPLATE_METHOD}.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from DataObs.lib.evaluation.eval_vllm import _normalize_dataset, run_vllm_eval

    model_root = DEFAULT_OUTPUT_ROOT / model_dir_name(args.model_path)
    model_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "model_path": args.model_path,
        "gpu_ids": args.gpu_ids,
        "output_dir": str(model_root),
        "eval_backend": "vllm",
        "gpu_memory_utilization": DEFAULT_GPU_MEMORY_UTILIZATION,
        "tensor_parallel_size": args.tensor_parallel_size,
        "prompt_template_method": args.prompt_template_method,
        "datasets": {},
    }
    write_json(model_root / "summary.json", summary)

    for idx, dataset in enumerate(DATASETS, start=1):
        print("=" * 80, flush=True)
        print(f"[{idx}/{len(DATASETS)}] Evaluating base model on dataset={dataset}", flush=True)
        print(f"model_path={args.model_path}", flush=True)
        print(f"output_root={model_root}", flush=True)

        normalized_dataset = _normalize_dataset(dataset)
        dataset_dir = model_root / safe_name(dataset)
        method_dir = dataset_dir / safe_name(args.prompt_template_method)
        try:
            result = run_vllm_eval(
                model_path=args.model_path,
                base_model=args.model_path,
                dataset=normalized_dataset,
                output_dir=str(method_dir),
                gpu_ids=args.gpu_ids,
                num_samples=1,
                temperature=0.6,
                top_p=0.95,
                max_response_len=1024,
                batch_size=8,
                gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
                tensor_parallel_size=args.tensor_parallel_size,
                prompt_template_method=args.prompt_template_method,
                max_samples=None,
                seed=42,
            )
        except Exception as exc:
            failure = {
                "success": False,
                "dataset_name": dataset,
                "normalized_dataset_name": normalized_dataset,
                "model_path": args.model_path,
                "eval_output_dir": str(method_dir),
                "error": repr(exc),
            }
            write_json(method_dir / "result.json", failure)
            summary.setdefault("datasets", {}).setdefault(dataset, {})[args.prompt_template_method] = failure
            write_json(model_root / "summary.json", summary)
            print(f"[ERROR] dataset={dataset} failed: {exc}", file=sys.stderr, flush=True)
            return 1

        result_payload = {
            "success": True,
            "dataset_name": dataset,
            "normalized_dataset_name": normalized_dataset,
            "model_path": args.model_path,
            "eval_output_dir": str(method_dir),
            **result,
        }
        write_json(method_dir / "result.json", result_payload)
        summary.setdefault("datasets", {}).setdefault(dataset, {})[args.prompt_template_method] = result_payload
        write_json(model_root / "summary.json", summary)

        print(f"[OK] dataset={dataset} accuracy={result.get('accuracy')}", flush=True)
        print(f"[OK] output={method_dir}", flush=True)

    print("=" * 80, flush=True)
    print(f"[OK] All evaluations completed. Summary: {model_root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
