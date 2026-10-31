#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def _configure_log_file(log_file: str | None) -> None:
    if not log_file:
        return
    path = Path(log_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)

try:
    from .common import load_seed_tasks, write_outputs
    from .filter_code_tests import filter_code_tests
    from .filter_rip import filter_rip
    from .filter_answer_consistency import filter_answer_consistency
    from .generate_code_tasks import generate_code_tasks
    from .generate_cot_tasks import generate_cot_tasks
    from .generate_instruction_tasks import generate_instruction_tasks
    from .prepare_for_finetuning import prepare_for_finetuning
except ImportError:
    from common import load_seed_tasks, write_outputs
    from filter_code_tests import filter_code_tests
    from filter_rip import filter_rip
    from filter_answer_consistency import filter_answer_consistency
    from generate_code_tasks import generate_code_tasks
    from generate_cot_tasks import generate_cot_tasks
    from generate_instruction_tasks import generate_instruction_tasks
    from prepare_for_finetuning import prepare_for_finetuning


REASONING_DATASETS = {
    "gsm8k",
    "math",
    "math500",
    "aquarat",
    "arcchallenge",
    "strategyqa",
    "commonsenseqa",
    "numinamath",
}

CODE_DATASETS = {
    "mbpp",
    "mbppplus",
    "humaneval",
    "humanevalplus",
}


def _normalize_dataset_name(name: str) -> str:
    return str(name or "").strip().lower().replace("_", "").replace("-", "")


def _resolve_task_modes(args: argparse.Namespace) -> list[str]:
    mode = str(args.cot_task_mode or "auto").strip().lower()
    if mode == "reasoning":
        return ["reasoning"]
    if mode == "instruction":
        return ["instruction"]
    if mode == "code":
        return ["code"]
    if mode == "both":
        return ["reasoning", "instruction"]
    if mode == "all":
        return ["reasoning", "instruction", "code"]

    normalized_dataset = _normalize_dataset_name(args.dataset)
    if normalized_dataset in CODE_DATASETS:
        return ["code"]
    if normalized_dataset in REASONING_DATASETS:
        return ["reasoning"]
    return ["instruction"]


def run(args: argparse.Namespace) -> None:
    _configure_log_file(args.log_file)
    start_time = time.time()
    seed_tasks = load_seed_tasks(args.input_file, args.smoke_num_rows)
    task_modes = _resolve_task_modes(args)

    generated_reasoning_items: list[dict] = []
    filtered_reasoning_items: list[dict] = []
    generated_instruction_items: list[dict] = []
    filtered_instruction_items: list[dict] = []
    generated_code_items: list[dict] = []
    filtered_code_items: list[dict] = []

    if "reasoning" in task_modes:
        generated_reasoning_items = generate_cot_tasks(seed_tasks, args)
        filtered_reasoning_items = filter_answer_consistency(generated_reasoning_items, args)

    if "instruction" in task_modes:
        generated_instruction_items = generate_instruction_tasks(seed_tasks, args)
        filtered_instruction_items = filter_rip(generated_instruction_items, args)

    if "code" in task_modes:
        generated_code_items = generate_code_tasks(seed_tasks, args)
        filtered_code_items = filter_code_tests(generated_code_items, args)

    filtered_items = filtered_reasoning_items + filtered_instruction_items + filtered_code_items
    sft_rows = prepare_for_finetuning(filtered_items, args.dataset)
    consistency_passed_count = sum(1 for item in filtered_reasoning_items if item.get("consistency_passed"))
    rip_passed_count = sum(1 for item in filtered_instruction_items if item.get("rip_passed"))
    code_passed_count = sum(1 for item in filtered_code_items if item.get("code_passed"))

    print(f"Loaded seed tasks: {len(seed_tasks)}")
    print(f"Task modes: {', '.join(task_modes)}")
    print(f"Generated reasoning tasks: {len(generated_reasoning_items)}")
    print(f"Answer-consistency passed: {consistency_passed_count}")
    print(f"Generated instruction tasks: {len(generated_instruction_items)}")
    print(f"RIP passed: {rip_passed_count}")
    print(f"Generated code tasks: {len(generated_code_items)}")
    print(f"Code-tests passed: {code_passed_count}")
    print(f"SFT rows: {len(sft_rows)}")
    if sft_rows:
        print("First SFT row preview:")
        print(sft_rows[0])

    write_outputs(
        args.output_file,
        filtered_items,
        sft_rows,
        args=args,
        seed_task_count=len(seed_tasks),
        generated_task_count=len(generated_reasoning_items) + len(generated_instruction_items) + len(generated_code_items),
        consistency_passed_count=consistency_passed_count,
        rip_passed_count=rip_passed_count,
        code_passed_count=code_passed_count,
        elapsed_sec=time.time() - start_time,
    )
    print(f"Wrote output to: {args.output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoT-Self-Instruct baseline.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--gen-batch-size", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-custom-all-reduce", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-file", default="")
    parser.add_argument("--smoke-num-rows", type=int, default=0)
    parser.add_argument("--cot-task-mode", choices=["auto", "reasoning", "instruction", "code", "both", "all"], default="auto")
    parser.add_argument("--cot-num-instructions", type=int, default=100)
    parser.add_argument("--cot-num-fewshot-examples", type=int, default=2)
    parser.add_argument("--cot-num-prompts-per-request", type=int, default=10)
    parser.add_argument("--cot-max-new-tokens", type=int, default=2048)
    parser.add_argument("--cot-temperature", type=float, default=0.7)
    parser.add_argument("--cot-top-p", type=float, default=0.95)
    parser.add_argument("--cot-do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--instruction-num-instructions", type=int, default=100)
    parser.add_argument("--instruction-num-fewshot-examples", type=int, default=2)
    parser.add_argument("--instruction-max-new-tokens", type=int, default=2048)
    parser.add_argument("--instruction-temperature", type=float, default=0.8)
    parser.add_argument("--instruction-top-p", type=float, default=0.9)
    parser.add_argument("--instruction-do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--consistency-k", type=int, default=16)
    parser.add_argument("--consistency-max-new-tokens", type=int, default=1024)
    parser.add_argument("--consistency-temperature", type=float, default=0.6)
    parser.add_argument("--consistency-top-p", type=float, default=0.95)
    parser.add_argument("--consistency-do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--consistency-min-vote-ratio", type=float, default=0.5)
    parser.add_argument("--rip-num-responses", type=int, default=32)
    parser.add_argument("--rip-response-max-new-tokens", type=int, default=1024)
    parser.add_argument("--rip-response-temperature", type=float, default=1.0)
    parser.add_argument("--rip-response-top-p", type=float, default=1.0)
    parser.add_argument("--rip-response-do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rip-score-max-new-tokens", type=int, default=32)
    parser.add_argument("--rip-min-score", type=int, default=4)
    parser.add_argument("--code-num-samples", type=int, default=1)
    parser.add_argument("--code-max-new-tokens", type=int, default=2048)
    parser.add_argument("--code-temperature", type=float, default=0.7)
    parser.add_argument("--code-top-p", type=float, default=0.95)
    parser.add_argument("--code-do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--code-correct-threshold", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
