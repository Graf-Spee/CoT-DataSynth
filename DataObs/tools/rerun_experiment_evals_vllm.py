#!/usr/bin/env python3
"""Rerun vLLM SFT/GRPO eval for completed DataObs experiments.

The experiment list is intentionally explicit so batch reruns are reviewable.
Each entry should point to an experiment directory under /data/hrh/COT/experiments
that already has both an SFT checkpoint and a GRPO actor checkpoint.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = REPO_ROOT / "DataObs" / "scripts" / "experiment_pipeline_vllm.py"


EXPERIMENTS: list[dict[str, str]] = [
    {
        "experiment_dir": "/data/hrh/COT/experiments/exp1100_gsm8k_qwen3_teacher_qwen3_1_7b_t0.7_n4",
        "dataset": "gsm8k",
        "base_model": "/data/pretrain_models/Qwen3-4B",
    },
    {
        "experiment_dir": "/data/hrh/COT/experiments/exp1100_gsm8k_qwen3_teacher_qwen3_4b_t0.7_n4",
        "dataset": "gsm8k",
        "base_model": "/data/pretrain_models/Qwen3-4B",
    },
    {
        "experiment_dir": "/data/hrh/COT/experiments/exp1100_gsm8k_qwen3_teacher_qwen3_8b_t0.7_n4",
        "dataset": "gsm8k",
        "base_model": "/data/pretrain_models/Qwen3-4B",
    },
    {
        "experiment_dir": "/data/hrh/COT/experiments/exp1200_gsm8k_qwen3_teacher_type_deepseek_r1_qwen32b_t0.7_n4",
        "dataset": "gsm8k",
        "base_model": "/data/pretrain_models/Qwen3-4B",
    },
    {
        "experiment_dir": "/data/hrh/COT/experiments/exp1200_gsm8k_qwen3_teacher_type_qwen3_14b_t0.7_n4",
        "dataset": "gsm8k",
        "base_model": "/data/pretrain_models/Qwen3-4B",
    },
    {
        "experiment_dir": "/data/hrh/COT/experiments/exp1200_math500_qwen3_teacher_type_qwen3_14b_t0.7_n4",
        "dataset": "math-500",
        "base_model": "/data/pretrain_models/Qwen3-4B",
    },
    {
        "experiment_dir": "/data/hrh/COT/experiments/exp3100_gsm8k_qwen3_settingA_sft_eq_rl_prompt",
        "dataset": "gsm8k",
        "base_model": "/data/pretrain_models/Qwen3-4B",
    },
    {
        "experiment_dir": "/data/hrh/COT/experiments/exp3100_gsm8k_qwen3_settingB_sft_ne_rl_prompt",
        "dataset": "gsm8k",
        "base_model": "/data/pretrain_models/Qwen3-4B",
    },
    {
        "experiment_dir": "/data/hrh/COT/experiments/exp5000_gsm8k_qwen3_difficulty_easy_distill",
        "dataset": "gsm8k",
        "base_model": "/data/pretrain_models/Qwen3-4B",
    },
    {
        "experiment_dir": "/data/hrh/COT/experiments/exp5000_gsm8k_qwen3_difficulty_hard_distill",
        "dataset": "gsm8k",
        "base_model": "/data/pretrain_models/Qwen3-4B",
    },
    {
        "experiment_dir": "/data/hrh/COT/experiments/exp5000_gsm8k_qwen3_difficulty_medium_distill",
        "dataset": "gsm8k",
        "base_model": "/data/pretrain_models/Qwen3-4B",
    },
    {
        "experiment_dir": "/data/hrh/COT/experiments/exp7000_gsm8k_qwen3_teacher14b_pool_n8_same_prompt_n1",
        "dataset": "gsm8k",
        "base_model": "/data/pretrain_models/Qwen3-4B",
    },
    {
        "experiment_dir": "/data/hrh/COT/experiments/exp7000_gsm8k_qwen3_teacher14b_pool_n8_same_prompt_n2",
        "dataset": "gsm8k",
        "base_model": "/data/pretrain_models/Qwen3-4B",
    },
    {
        "experiment_dir": "/data/hrh/COT/experiments/exp7000_gsm8k_qwen3_teacher14b_pool_n8_same_prompt_n4",
        "dataset": "gsm8k",
        "base_model": "/data/pretrain_models/Qwen3-4B",
    },
    {
        "experiment_dir": "/data/hrh/COT/experiments/exp7000_gsm8k_qwen3_teacher14b_pool_n8_same_prompt_n8",
        "dataset": "gsm8k",
        "base_model": "/data/pretrain_models/Qwen3-4B",
    },
    {
        "experiment_dir": "/data/hrh/COT/experiments/pipeline_gsm8k_qwen3_4b_0620",
        "dataset": "gsm8k",
        "base_model": "/data/pretrain_models/Qwen3-4B",
    },
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resume_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("experiment_id", "")),
        str(record.get("dataset", "")),
        str(record.get("sft_checkpoint", "")),
        str(record.get("grpo_step", "")),
    )


def is_hf_or_lora_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "adapter_model.safetensors").is_file():
        return True
    if not (path / "config.json").is_file():
        return False
    model_files = [
        path / "model.safetensors",
        path / "pytorch_model.bin",
        path / "model.safetensors.index.json",
    ]
    return any(p.is_file() for p in model_files) or any(path.glob("*.safetensors"))


def latest_global_step(path: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for item in path.glob("global_step_*"):
        if not item.is_dir():
            continue
        step = item.name.removeprefix("global_step_")
        if step.isdigit():
            candidates.append((int(step), item))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def resolve_sft_checkpoint(exp_dir: Path) -> Path | None:
    sft_dir = exp_dir / "sft"
    if is_hf_or_lora_dir(sft_dir):
        return sft_dir

    checkpoint_last = sft_dir / "checkpoint-last"
    if is_hf_or_lora_dir(checkpoint_last):
        return checkpoint_last

    latest = latest_global_step(sft_dir)
    if latest and is_hf_or_lora_dir(latest):
        return latest

    return None


def latest_grpo_step_id(exp_dir: Path) -> str | None:
    grpo_dir = exp_dir / "grpo"
    latest_file = grpo_dir / "latest_checkpointed_iteration.txt"
    if latest_file.is_file():
        step = latest_file.read_text(encoding="utf-8").strip().removeprefix("global_step_")
        if step:
            return step

    latest = latest_global_step(grpo_dir)
    if latest:
        return latest.name.removeprefix("global_step_")

    return None


def has_grpo_actor(exp_dir: Path, step_id: str) -> bool:
    actor_dir = exp_dir / "grpo" / f"global_step_{step_id}" / "actor"
    lora_dir = actor_dir / "lora_adapter"
    hf_dir = actor_dir / "huggingface"
    merged_dir = actor_dir / "merged_model"
    return (
        (lora_dir / "adapter_model.safetensors").is_file()
        or is_hf_or_lora_dir(hf_dir)
        or is_hf_or_lora_dir(merged_dir)
    )


def validate_experiment(spec: dict[str, str]) -> dict[str, Any]:
    exp_dir = Path(spec["experiment_dir"]).resolve()
    sft_checkpoint = resolve_sft_checkpoint(exp_dir)
    grpo_step = latest_grpo_step_id(exp_dir)
    ok = (
        exp_dir.is_dir()
        and bool(spec.get("dataset"))
        and bool(spec.get("base_model"))
        and sft_checkpoint is not None
        and grpo_step is not None
        and has_grpo_actor(exp_dir, grpo_step)
    )
    return {
        "ok": ok,
        "experiment_dir": str(exp_dir),
        "experiment_id": exp_dir.name,
        "output_dir": str(exp_dir.parent),
        "dataset": spec.get("dataset", ""),
        "base_model": spec.get("base_model", ""),
        "sft_checkpoint": str(sft_checkpoint) if sft_checkpoint else "",
        "grpo_step": grpo_step or "",
    }


def build_cmd(item: dict[str, Any], args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--experiment-id", item["experiment_id"],
        "--dataset", item["dataset"],
        "--base-model", item["base_model"],
        "--output-dir", item["output_dir"],
        "--stages", "sft_eval,grpo_eval",
        "--gpu-ids", args.gpu_ids,
        "--sft-checkpoint", item["sft_checkpoint"],
        "--grpo-step", item["grpo_step"],
        "--eval-gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--eval-tensor-parallel-size", str(args.tensor_parallel_size),
    ]
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun DataObs experiment SFT/GRPO evals through experiment_pipeline_vllm.py."
    )
    parser.add_argument("--gpu-ids", required=True, help="GPU ids for evaluation, e.g. 0 or 0,1.")
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        dest="tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor parallel size for evaluation. Default: 1.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="vLLM GPU memory utilization for evaluation. Default: 0.8.",
    )
    parser.add_argument(
        "--summary-path",
        default="/data/hrh/COT/experiments/rerun_experiment_evals_vllm.summary.json",
        help="Path for batch progress summary.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print pipeline commands without running eval.")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore successful records in summary and rerun every listed experiment.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    valid_items: list[dict[str, Any]] = []
    skipped_items: list[dict[str, Any]] = []
    for spec in EXPERIMENTS:
        item = validate_experiment(spec)
        if item["ok"] or args.dry_run:
            valid_items.append(item)
        else:
            skipped_items.append(item)

    summary_path = Path(args.summary_path)
    completed: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    previous_runs: list[dict[str, Any]] = []
    if summary_path.exists() and not args.force_rerun:
        try:
            previous_summary = load_json(summary_path)
            if isinstance(previous_summary, dict) and isinstance(previous_summary.get("runs"), list):
                previous_runs = [run for run in previous_summary["runs"] if isinstance(run, dict)]
                completed = {
                    resume_key(run): run
                    for run in previous_runs
                    if run.get("success") is True and not run.get("dry_run_only")
                }
        except json.JSONDecodeError as exc:
            print(f"[WARN] Existing summary is invalid JSON; starting fresh: {summary_path} ({exc})", flush=True)

    items_to_run: list[dict[str, Any]] = []
    skipped_completed: list[dict[str, Any]] = []
    for item in valid_items:
        if not args.force_rerun and resume_key(item) in completed:
            skipped_completed.append(item)
        else:
            items_to_run.append(item)

    summary: dict[str, Any] = {
        "gpu_ids": args.gpu_ids,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "pipeline_script": str(PIPELINE_SCRIPT),
        "dry_run": bool(args.dry_run),
        "force_rerun": bool(args.force_rerun),
        "experiments_total": len(EXPERIMENTS),
        "experiments_to_run": len(valid_items),
        "experiments_remaining": len(items_to_run),
        "skipped_completed": skipped_completed,
        "skipped": skipped_items,
        "runs": [] if args.force_rerun else previous_runs,
    }
    write_json(summary_path, summary)

    if skipped_completed:
        print(f"[INFO] Resume enabled: skipping {len(skipped_completed)} completed experiments.", flush=True)

    for idx, item in enumerate(items_to_run, start=1):
        cmd = build_cmd(item, args)
        print("=" * 80, flush=True)
        print(
            f"[{idx}/{len(items_to_run)}] Rerunning eval for {item['experiment_id']} "
            f"dataset={item['dataset']} grpo_step={item['grpo_step']}",
            flush=True,
        )
        print(" ".join(cmd), flush=True)

        run_record = {
            "experiment_id": item["experiment_id"],
            "experiment_dir": item["experiment_dir"],
            "dataset": item["dataset"],
            "base_model": item["base_model"],
            "sft_checkpoint": item["sft_checkpoint"],
            "grpo_step": item["grpo_step"],
            "cmd": cmd,
        }

        if args.dry_run:
            run_record["returncode"] = 0
            run_record["success"] = True
            run_record["dry_run_only"] = True
            summary["runs"].append(run_record)
            write_json(summary_path, summary)
            continue

        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True)
        run_record["returncode"] = proc.returncode
        run_record["success"] = proc.returncode == 0
        summary["runs"].append(run_record)
        write_json(summary_path, summary)

        if proc.returncode != 0:
            print(f"[ERROR] Eval rerun failed for {item['experiment_id']}", file=sys.stderr, flush=True)
            print(f"[ERROR] Summary: {summary_path}", file=sys.stderr, flush=True)
            return proc.returncode

    print("=" * 80, flush=True)
    print(f"[OK] Completed {len(items_to_run)} eval reruns. Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
