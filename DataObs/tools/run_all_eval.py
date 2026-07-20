#!/usr/bin/env python3
"""Run base-model vLLM evaluation serially for the standard DataObs dataset suite."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = "/data/hrh/COT/evals"
DEFAULT_TENSOR_PARALLEL_SIZE = 1
DEFAULT_PROMPT_TEMPLATE_METHOD = "zeroshot"
DEFAULT_MAX_RESPONSE_LEN = 16384
DEFAULT_ENABLE_THINKING = False

# Edit this list to control the model batch evaluated by this script.
# gpu_memory_utilization is per model because larger checkpoints such as Qwen3-32B
# often need a different vLLM KV-cache reservation.
MODEL_SPECS: list[dict[str, Any]] = [
    # {
    #     "model_path": "/data/pretrain_models/Qwen3-1.7B",
    #     "gpu_memory_utilization": 0.95,
    # },
    # {
    #     "model_path": "/data/pretrain_models/Qwen3-4B",
    #     "gpu_memory_utilization": 0.95,
    # },
    # {
    #     "model_path": "/data/pretrain_models/Qwen3-8B",
    #     "gpu_memory_utilization": 0.95,
    # },
    # {
    #     "model_path": "/data/pretrain_models/Qwen3-14B",
    #     "gpu_memory_utilization": 0.95,
    # },
    # {
    #     "model_path": "/data/pretrain_models/Qwen3-32B",
    #     "gpu_memory_utilization": 0.95,
    # },
    # {
    #     "model_path": "/data/pretrain_models/Llama-3.1-8B-Instruct",
    #     "gpu_memory_utilization": 0.8,
    # },
    {
        "model_path": "/data/pretrain_models/DeepSeek-R1-Distill-Qwen-32B",
        "gpu_memory_utilization": 0.8,
    },
]

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


SUMMARY_CONFIG_KEYS = (
    "model_path",
    "output_dir",
    "eval_backend",
    "prompt_template_method",
    "max_response_len",
    "enable_thinking",
)


def build_summary_config(
    *,
    model_path: str,
    model_root: Path,
    gpu_ids: str,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    prompt_template_method: str,
) -> dict[str, Any]:
    return {
        "model_path": model_path,
        "gpu_ids": gpu_ids,
        "output_dir": str(model_root),
        "eval_backend": "vllm",
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "prompt_template_method": prompt_template_method,
        "max_response_len": DEFAULT_MAX_RESPONSE_LEN,
        "enable_thinking": DEFAULT_ENABLE_THINKING,
    }


def load_or_create_summary(summary_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    fresh_summary = {**config, "datasets": {}}
    if not summary_path.exists():
        return fresh_summary

    try:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[WARN] Existing summary is invalid JSON, rebuilding: {summary_path} ({exc})", flush=True)
        return fresh_summary

    if not isinstance(existing, dict):
        print(f"[WARN] Existing summary is not an object, rebuilding: {summary_path}", flush=True)
        return fresh_summary

    mismatched_keys = [
        key for key in SUMMARY_CONFIG_KEYS
        if existing.get(key) != config.get(key)
    ]
    if mismatched_keys:
        joined = ", ".join(mismatched_keys)
        print(f"[INFO] Summary config changed ({joined}); rebuilding: {summary_path}", flush=True)
        return fresh_summary

    summary = dict(existing)
    summary.update(config)
    if not isinstance(summary.get("datasets"), dict):
        summary["datasets"] = {}
    return summary


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
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for evaluation outputs. Default: {DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking steps in evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from DataObs.lib.evaluation.eval_vllm import _normalize_dataset

    for model_idx, model_spec in enumerate(MODEL_SPECS, start=1):
        model_path = str(model_spec["model_path"])
        gpu_memory_utilization = float(model_spec["gpu_memory_utilization"])
        model_root = Path(args.output_root) / model_dir_name(model_path)
        model_root.mkdir(parents=True, exist_ok=True)

        summary_path = model_root / "summary.json"
        summary_config = build_summary_config(
            model_path=model_path,
            model_root=model_root,
            gpu_ids=args.gpu_ids,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            prompt_template_method=args.prompt_template_method,
        )
        summary = load_or_create_summary(summary_path, summary_config)
        write_json(summary_path, summary)

        for dataset_idx, dataset in enumerate(DATASETS, start=1):
            print("=" * 80, flush=True)
            print(
                f"[model {model_idx}/{len(MODEL_SPECS)}][dataset {dataset_idx}/{len(DATASETS)}] "
                f"Evaluating dataset={dataset}",
                flush=True,
            )
            print(f"model_path={model_path}", flush=True)
            print(f"gpu_memory_utilization={gpu_memory_utilization}", flush=True)
            print(f"output_root={model_root}", flush=True)

            normalized_dataset = _normalize_dataset(dataset)
            dataset_dir = model_root / safe_name(dataset)
            method_dir = dataset_dir / safe_name(args.prompt_template_method)
            try:
                cmd = [
                    sys.executable,
                    str(REPO_ROOT / "DataObs" / "lib" / "evaluation" / "eval_vllm.py"),
                    "--model-path", model_path,
                    "--base-model", model_path,
                    "--dataset", normalized_dataset,
                    "--output-dir", str(method_dir),
                    "--gpu-ids", args.gpu_ids,
                    "--num-samples", "1",
                    "--temperature", "0.0",
                    "--top-p", "1.0",
                    "--top-k", "-1",
                    "--max-response-len", str(DEFAULT_MAX_RESPONSE_LEN),
                    "--batch-size", "8",
                    "--gpu-memory-utilization", str(gpu_memory_utilization),
                    "--tensor-parallel-size", str(args.tensor_parallel_size),
                    "--prompt-template-method", args.prompt_template_method,
                    "--seed", "42",
                ]
                if args.enable_thinking or DEFAULT_ENABLE_THINKING:
                    cmd.append("--enable-thinking")

                subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
                result_path = method_dir / "generated" / "results.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as exc:
                failure = {
                    "success": False,
                    "dataset_name": dataset,
                    "normalized_dataset_name": normalized_dataset,
                    "model_path": model_path,
                    "eval_output_dir": str(method_dir),
                    "error": repr(exc),
                }
                write_json(method_dir / "result.json", failure)
                summary.setdefault("datasets", {}).setdefault(dataset, {})[args.prompt_template_method] = failure
                write_json(summary_path, summary)
                print(f"[ERROR] model={model_path} dataset={dataset} failed: {exc}", file=sys.stderr, flush=True)
                return 1

            result_payload = {
                "success": True,
                "dataset_name": dataset,
                "normalized_dataset_name": normalized_dataset,
                "model_path": model_path,
                "eval_output_dir": str(method_dir),
                **result,
            }
            write_json(method_dir / "result.json", result_payload)
            summary.setdefault("datasets", {}).setdefault(dataset, {})[args.prompt_template_method] = result_payload
            write_json(summary_path, summary)

            print(f"[OK] model={model_path} dataset={dataset} accuracy={result.get('accuracy')}", flush=True)
            print(f"[OK] output={method_dir}", flush=True)

        print("=" * 80, flush=True)
        print(f"[OK] Model evaluation completed. Summary: {summary_path}", flush=True)

    print("=" * 80, flush=True)
    print("[OK] All model evaluations completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
