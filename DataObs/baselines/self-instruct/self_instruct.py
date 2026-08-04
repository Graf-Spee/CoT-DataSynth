#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
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
    bootstrapped_instructions = bootstrap_instructions(seed_tasks, args)
    classified_instructions = identify_clf_or_not(bootstrapped_instructions, args)
    generated_items = generate_instances(classified_instructions, args)
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
    print(f"Wrote output to: {args.output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Instruct baseline.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
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
