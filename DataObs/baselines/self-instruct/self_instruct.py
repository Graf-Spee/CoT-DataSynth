#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
try:
    from .bootstrap_instructions import bootstrap_instructions
    from .common import load_seed_tasks, write_outputs
    from .generate_instances import generate_instances
    from .identify_clf_or_not import identify_clf_or_not
    from .prepare_for_finetuning import prepare_for_finetuning
except ImportError:
    from bootstrap_instructions import bootstrap_instructions
    from common import load_seed_tasks, write_outputs
    from generate_instances import generate_instances
    from identify_clf_or_not import identify_clf_or_not
    from prepare_for_finetuning import prepare_for_finetuning


def run(args: argparse.Namespace) -> None:
    start_time = time.time()
    seed_tasks = load_seed_tasks(args.input_file, args.smoke_num_rows)
    state_path = Path(args.resume_state) if args.resume_state else Path(f"{args.output_file}.distill_state.json")
    state: dict = {}

    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("dataset") != args.dataset or state.get("input_file") != args.input_file:
            raise SystemExit(f"[ERROR] Distill state belongs to a different dataset or input file: {state_path}")
        print(f"[INFO] Resuming distill state from: {state_path}")

    def save_state() -> None:
        state["dataset"] = args.dataset
        state["input_file"] = args.input_file
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = Path(f"{state_path}.tmp")
        temporary_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        temporary_path.replace(state_path)
        print(f"[INFO] Saved distill state: {state_path}")

    if state.get("completed") and Path(args.output_file).is_file():
        print(f"[INFO] Distill output already complete: {args.output_file}")
        return

    bootstrapped_instructions = state.get("bootstrapped_instructions")
    if bootstrapped_instructions is None:
        bootstrapped_instructions = bootstrap_instructions(seed_tasks, args)
        state["bootstrapped_instructions"] = bootstrapped_instructions
        save_state()

    classified_instructions = state.get("classified_instructions")
    if classified_instructions is None:
        classified_instructions = identify_clf_or_not(bootstrapped_instructions, args)
        state["classified_instructions"] = classified_instructions
        save_state()

    generated_items = state.get("generated_items")
    if generated_items is None:
        generated_items = generate_instances(classified_instructions, args)
        state["generated_items"] = generated_items
        save_state()

    sft_rows = prepare_for_finetuning(generated_items, args.dataset)

    print(f"Loaded seed tasks: {len(seed_tasks)}")
    print(f"Bootstrapped instructions: {len(bootstrapped_instructions)}")
    print(f"Classification-annotated instructions: {len(classified_instructions)}")
    print(f"Generated items: {len(generated_items)}")
    print(f"SFT rows: {len(sft_rows)}")

    if sft_rows:
        print("First SFT row preview:")
        print(sft_rows[0])

    write_outputs(
        args.output_file,
        generated_items,
        sft_rows,
        args=args,
        seed_task_count=len(seed_tasks),
        bootstrapped_instruction_count=len(bootstrapped_instructions),
        classified_instruction_count=len(classified_instructions),
        elapsed_sec=time.time() - start_time,
    )
    state["completed"] = True
    save_state()
    print(f"Wrote output to: {args.output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Instruct baseline.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--resume-state", default="", help="Path to a resumable distill state file.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--gen-batch-size", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--correct-threshold", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-teacher-filter", action="store_true")
    parser.add_argument("--smoke-num-rows", type=int, default=0)
    parser.add_argument("--bootstrap-num-instructions", type=int, default=100)
    parser.add_argument("--bootstrap-num-prompt-instructions", type=int, default=8)
    parser.add_argument("--bootstrap-request-batch-size", type=int, default=5)
    parser.add_argument("--bootstrap-max-new-tokens", type=int, default=1024)
    parser.add_argument("--bootstrap-temperature", type=float, default=0.7)
    parser.add_argument("--bootstrap-top-p", type=float, default=0.5)
    parser.add_argument("--bootstrap-similarity-threshold", type=float, default=0.7)
    parser.add_argument("--bootstrap-use-clf-seed-tasks-only", action="store_true")
    parser.add_argument("--bootstrap-allow-stub-fallback", action="store_true")
    parser.add_argument("--classification-max-new-tokens", type=int, default=3)
    parser.add_argument("--classification-request-batch-size", type=int, default=5)
    parser.add_argument("--classification-allow-heuristic-fallback", action="store_true")
    parser.add_argument("--instance-request-batch-size", type=int, default=5)
    parser.add_argument("--instance-max-instances-to-generate", type=int, default=5)
    parser.add_argument("--instance-max-new-tokens-clf", type=int, default=300)
    parser.add_argument("--instance-max-new-tokens-gen", type=int, default=350)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
