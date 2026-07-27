#!/usr/bin/env python3
"""Closed-loop Training Dynamics Guided Data Optimization pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DataEvo.lib.controller import DataController, DataGenerationStrategy, initial_strategy, load_teacher_pool
from DataEvo.lib.prompt_agent import run_prompt_agent
from DataEvo.lib.training_dynamics import write_training_dynamics


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def build_round_cmd(
    *,
    args: argparse.Namespace,
    extra_args: list[str],
    round_index: int,
    strategy: DataGenerationStrategy,
    rounds_dir: Path,
) -> tuple[list[str], Path]:
    experiment_id = f"{args.evo_id}_r{round_index:03d}"
    experiment_dir = rounds_dir / experiment_id
    cmd = [
        sys.executable,
        str(REPO_ROOT / args.pipeline_script),
        "--experiment-id",
        experiment_id,
        "--dataset",
        args.dataset,
        "--base-model",
        args.base_model,
        "--teacher-model",
        strategy.teacher_model,
        "--output-dir",
        str(rounds_dir),
        "--stages",
        args.round_stages,
        "--gpu-ids",
        args.gpu_ids,
        "--teacher-temperature",
        str(strategy.teacher_temperature),
        "--teacher-top-p",
        str(strategy.teacher_top_p),
        "--teacher-num-samples",
        str(strategy.teacher_num_samples),
        "--teacher-max-new-tokens",
        str(strategy.teacher_max_new_tokens),
        "--evo-difficulty",
        strategy.difficulty_profile,
        "--evo-reason-length",
        strategy.reason_length_profile,
        "--evo-teacher-type",
        strategy.teacher_type or "base",
    ]
    if strategy.prompt_suffix:
        cmd += ["--evo-prompt-suffix", strategy.prompt_suffix]
    if strategy.teacher_tensor_parallel_size:
        cmd += ["--teacher-tensor-parallel-size", str(strategy.teacher_tensor_parallel_size)]
    else:
        cmd += ["--teacher-tensor-parallel-size", str(args.teacher_tensor_parallel_size)]
    if strategy.teacher_num_samples > 1 or args.teacher_do_sample:
        cmd.append("--teacher-do-sample")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.distill_input:
        cmd += ["--distill-input", args.distill_input]
    if args.smoke_num_rows > 0:
        cmd += ["--smoke-num-rows", str(args.smoke_num_rows)]
    if args.teacher_batch_size:
        cmd += ["--teacher-batch-size", str(args.teacher_batch_size)]
    if args.teacher_gpu_memory_utilization:
        cmd += ["--teacher-gpu-memory-utilization", str(args.teacher_gpu_memory_utilization)]
    cmd.extend(extra_args)
    return cmd, experiment_dir


def run_round(cmd: list[str], *, dry_run: bool) -> None:
    print("\n>>>>>>>> dataevo_round")
    print(" ".join(cmd))
    if dry_run:
        return
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True)
    if proc.returncode != 0:
        raise SystemExit(f"[ERROR] DataEvo round failed with code {proc.returncode}")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run Training Dynamics Guided Data Optimization: "
            "data generation -> model training -> dynamics analysis -> strategy update."
        )
    )
    parser.add_argument("--evo-id", required=True, help="Closed-loop experiment id.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--teacher-model", required=True, help="Initial teacher model path or id.")
    parser.add_argument("--output-dir", default="/data/hrh/COT/dataevo_experiments")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--round-stages", default="distill,metrics,sft,grpo", help="Inner DataEvo stages per round.")
    parser.add_argument("--pipeline-script", default="DataEvo/scripts/experiment_pipeline_vllm.py")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--teacher-registry", default="DataEvo/config/teacher_registry.json")
    parser.add_argument("--teacher-temperature", type=float, default=0.7)
    parser.add_argument("--teacher-top-p", type=float, default=0.95)
    parser.add_argument("--teacher-num-samples", type=int, default=1)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=8192)
    parser.add_argument("--teacher-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--teacher-batch-size", type=int, default=128)
    parser.add_argument("--teacher-gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--teacher-do-sample", action="store_true")

    parser.add_argument("--distill-input", default="")
    parser.add_argument("--smoke-num-rows", type=int, default=0)
    parser.add_argument("--initial-strategy", default="", help="Optional JSON strategy file to start from.")
    parser.add_argument(
        "--prompt-agent-mode",
        choices=["heuristic", "llm", "off"],
        default="heuristic",
        help="How to rewrite the next-round distillation prompt suffix.",
    )
    parser.add_argument(
        "--prompt-agent-cmd",
        default="",
        help=(
            "External LLM agent command. It receives a JSON context on stdin and should "
            "print JSON with prompt_suffix and rationale."
        ),
    )
    args, extra_args = parser.parse_known_args()
    if args.rounds < 1:
        parser.error("--rounds must be >= 1")
    return args, extra_args


def main() -> int:
    args, extra_args = parse_args()
    evo_dir = Path(args.output_dir).resolve() / args.evo_id
    rounds_dir = evo_dir / "rounds"
    feedback_dir = evo_dir / "feedback"
    history_file = evo_dir / "history.jsonl"

    teacher_pool = load_teacher_pool(str(REPO_ROOT / args.teacher_registry) if args.teacher_registry else None)
    if args.initial_strategy:
        strategy = DataGenerationStrategy.from_json(json.loads(Path(args.initial_strategy).read_text(encoding="utf-8")))
    else:
        strategy = initial_strategy(
            teacher_model=args.teacher_model,
            teacher_pool=teacher_pool,
            teacher_temperature=args.teacher_temperature,
            teacher_top_p=args.teacher_top_p,
            teacher_num_samples=args.teacher_num_samples,
            teacher_max_new_tokens=args.teacher_max_new_tokens,
        )
    controller = DataController(teacher_pool)

    write_json(
        evo_dir / "manifest.json",
        {
            "evo_id": args.evo_id,
            "dataset": args.dataset,
            "base_model": args.base_model,
            "rounds": args.rounds,
            "round_stages": args.round_stages,
            "pipeline_script": args.pipeline_script,
            "extra_args": extra_args,
            "prompt_agent_mode": args.prompt_agent_mode,
            "prompt_agent_cmd": args.prompt_agent_cmd,
            "closed_loop": "data_generation -> model_training -> training_dynamics_analysis -> strategy_update",
        },
    )

    for round_index in range(args.rounds):
        round_strategy_path = feedback_dir / f"strategy_round_{round_index:03d}.json"
        write_json(round_strategy_path, strategy.to_json())

        cmd, experiment_dir = build_round_cmd(
            args=args,
            extra_args=extra_args,
            round_index=round_index,
            strategy=strategy,
            rounds_dir=rounds_dir,
        )
        append_jsonl(history_file, {"round": round_index, "strategy": strategy.to_json(), "cmd": cmd, "experiment_dir": str(experiment_dir)})
        run_round(cmd, dry_run=args.dry_run)

        dynamics_path = feedback_dir / f"dynamics_round_{round_index:03d}.json"
        dynamics = write_training_dynamics(experiment_dir, dynamics_path)
        if round_index == args.rounds - 1:
            break
        strategy, update = controller.update(strategy, dynamics)
        strategy, prompt_agent_update = run_prompt_agent(
            mode=args.prompt_agent_mode,
            command=args.prompt_agent_cmd,
            round_index=round_index,
            strategy=strategy,
            dynamics=dynamics,
            controller_update=update,
            output_dir=feedback_dir,
        )
        update["prompt_agent"] = prompt_agent_update
        update["output_strategy_after_prompt_agent"] = strategy.to_json()
        update_path = feedback_dir / f"update_round_{round_index:03d}_to_{round_index + 1:03d}.json"
        write_json(update_path, update)
        append_jsonl(history_file, {"round": round_index, "next_strategy": strategy.to_json(), "update": update})

    write_json(feedback_dir / "final_strategy.json", strategy.to_json())
    print(f"\nDone. DataEvo dir: {evo_dir}")
    print(f"History: {history_file}")
    print(f"Feedback: {feedback_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
