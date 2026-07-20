#!/usr/bin/env python3
"""Shared utilities for heuristic CoT data augmentation scripts."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DataObs.lib.data_process.cot_distill_teacher_filter import (  # noqa: E402
    DATASET_GROUP,
    REASONING_SUFFIX,
    LocalTeacher,
    _as_messages,
    _get_ground_truth,
    _last_user_text,
    _load_reward_functions,
    _normalize_dataset_name,
    _to_builtin,
)

try:  # noqa: E402
    from .heuristic_prompts import (
        consistency_check_prompt_math,
        consistency_check_prompt_mcq,
        gen_reasoning_prompt,
        icl_samples,
        prompt_for_backward_question,
    )
    from .heuristic_text_utils import get_true_false, remove_backward_answer
except ImportError:  # direct script execution
    from heuristic_prompts import (
        consistency_check_prompt_math,
        consistency_check_prompt_mcq,
        gen_reasoning_prompt,
        icl_samples,
        prompt_for_backward_question,
    )
    from heuristic_text_utils import get_true_false, remove_backward_answer

__all__ = [
    "DATASET_GROUP",
    "REASONING_SUFFIX",
    "LocalTeacher",
    "add_common_args",
    "base_summary",
    "baseline_task",
    "batched_generate",
    "build_answer_aug_prompt",
    "build_reasoning_prompt",
    "build_rephrase_prompt",
    "clean_rephrase",
    "consistency_check_prompt_math",
    "consistency_check_prompt_mcq",
    "consistency_prompt_template",
    "dataset_group",
    "extract_reward_prediction",
    "flatten",
    "get_true_false",
    "icl_samples",
    "load_input_rows",
    "make_teacher",
    "normalize_dataset",
    "prompt_for_backward_question",
    "remove_backward_answer",
    "require_backward_task",
    "score_answer",
    "validate_common_args",
    "write_outputs",
]

SUPPORTED_DATASETS = tuple(DATASET_GROUP.keys())

DATASET_TO_BASELINE_TASK = {
    "arc-challenge": "ARC",
    "commonsenseqa": "CSQA",
    "gsm8k": "GSM8K",
    "math": "MATH",
    "math-500": "MATH",
    "strategyqa": "SQA",
}
REVERSE_THINKING_DATASETS = tuple(DATASET_TO_BASELINE_TASK.keys())


def normalize_dataset(name: str) -> str:
    """Normalize user-provided dataset aliases to DataObs canonical names."""
    return _normalize_dataset_name(name)


def baseline_task(dataset: str) -> str | None:
    """Map DataObs datasets to the original baseline task names when prompts exist."""
    return DATASET_TO_BASELINE_TASK.get(dataset)


def dataset_group(dataset: str) -> str:
    """Return the DataObs reward/prompt group for a canonical dataset."""
    return DATASET_GROUP[dataset]


def build_reasoning_prompt(dataset: str, question: str) -> str:
    """Build the teacher CoT prompt for a question. Prefer original baseline prompts when available."""
    task = baseline_task(dataset)
    if task and task in gen_reasoning_prompt:
        return question.rstrip() + "\n" + gen_reasoning_prompt[task]

    group = dataset_group(dataset)
    suffix = REASONING_SUFFIX[group]
    return question.rstrip() + "\n\n" + suffix


def consistency_prompt_template(dataset: str) -> str:
    """Select the RevThink consistency-check prompt template for math or non-math data."""
    group = dataset_group(dataset)
    if group in {"gsm8k", "math"}:
        return consistency_check_prompt_math
    return consistency_check_prompt_mcq


def require_backward_task(dataset: str) -> str:
    """Validate Reverse Thinking support and return the matching baseline ICL task name."""
    if DATASET_GROUP[dataset] == "code_tests":
        raise ValueError(f"Reverse Thinking is not supported for coding dataset {dataset!r}.")
    task = baseline_task(dataset)
    if not task or task not in icl_samples:
        supported = ", ".join(REVERSE_THINKING_DATASETS)
        raise ValueError(
            f"Reverse Thinking needs original backward-question ICL prompts and is not covered "
            f"for dataset {dataset!r}. Supported datasets: {supported}."
        )
    return task


def clean_rephrase(text: str) -> str:
    """Remove common model-introduced rephrase prefixes and surrounding quotes."""
    prefix = re.compile(
        r"^\s*(?:Rephrase[sd]?|Rephrased)(?:\s+the)?\s+(?:above\s+)?question[:\-\s]*",
        re.IGNORECASE,
    )
    text = prefix.sub("", text).lstrip("-: ").strip()
    if len(text) > 1 and text[0] in "\"'" and text[-1] in "\"'":
        text = text[1:-1]
    return text.strip()


REPHRASE_EXAMPLE = """
Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Rephrase the above question: What amount of money does Olivia have left after buying five bagels at $3 each if she started with $23?
""".strip()


def build_rephrase_prompt(question: str) -> str:
    """Build the MetaMath-style question rephrasing prompt from the seed question."""
    return f"{REPHRASE_EXAMPLE}\n\nQuestion: {question}\nRephrase the above question:"


ANSWER_AUG_HEADER = """
Q: Liam had 4 boxes of apples with 6 apples each. He ate 5 apples. How many apples are left?
A: Let's think step by step. 4 * 6 = 24 apples. 24 - 5 = 19. The answer is: 19

Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there?
A: Let's think step by step. Half of 16 is 8. Half of 8 is 4. The answer is: 4
""".strip()


def build_answer_aug_prompt(dataset: str, question: str, use_original_prompt: bool) -> str:
    """Build the prompt for sampling alternate reasoning paths for the same question."""
    if use_original_prompt:
        return f"{ANSWER_AUG_HEADER}\n\nQ: {question}\nA: Let's think step by step."
    return build_reasoning_prompt(dataset, question)


def score_answer(dataset: str, answer: str, gold_answer: Any) -> float:
    """Score a generated answer with the same verl reward function used by DataObs distillation."""
    return float(_load_reward_functions(dataset)(answer, gold_answer))


def extract_reward_prediction(dataset: str, answer: str) -> str:
    """Extract a final prediction using the dataset's verl reward module, when available."""
    reward_module = _load_reward_functions(dataset).__module__
    module = sys.modules[reward_module]
    extract_pred = getattr(module, "extract_pred", None)
    if extract_pred is None:
        return ""
    pred = extract_pred(answer)
    return "" if pred is None else str(pred)


def load_input_rows(input_file: str, dataset: str, smoke_num_rows: int = 0) -> List[Dict[str, Any]]:
    """Load DataObs-format parquet rows and expose question/gold metadata for augmentation."""
    path = Path(input_file)
    if path.suffix.lower() != ".parquet":
        raise ValueError("Input must be a DataObs-format parquet file.")

    df = pd.read_parquet(path)
    if smoke_num_rows > 0:
        df = df.head(smoke_num_rows)
    rows = []
    for idx, row in df.iterrows():
        messages = _as_messages(row["prompt"])
        question = _last_user_text(messages).strip()
        gold = _get_ground_truth(row["reward_model"])
        if gold is None:
            raise ValueError(f"Missing gold answer at row index {idx}.")

        rows.append(
            {
                "source_index": int(idx) if isinstance(idx, int) else str(idx),
                "dataset": dataset,
                "data_source": row.get("data_source", dataset),
                "question": question,
                "gold_answer": _to_builtin(gold),
            }
        )
    return rows


def batched_generate(
    teacher: LocalTeacher,
    prompts: List[str],
    *,
    n: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    gen_batch_size: int,
    desc: str,
) -> List[List[str]]:
    """Run vLLM teacher generation over plain-text prompts in batches."""
    outputs: List[List[str]] = []
    for start in tqdm(range(0, len(prompts), gen_batch_size), desc=desc, unit="batch"):
        batch = prompts[start : start + gen_batch_size]
        batch_messages = [[{"role": "user", "content": prompt}] for prompt in batch]
        outputs.extend(
            teacher.generate_batch(
                messages_batch=batch_messages,
                n=n,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
        )
    return outputs


def make_teacher(args: argparse.Namespace) -> LocalTeacher:
    """Create the local teacher model with the same backend used by DataObs distillation."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    return LocalTeacher(
        model_id=args.model_id,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


def add_common_args(parser: argparse.ArgumentParser, *, dataset_required: bool = True) -> None:
    """Register CLI arguments shared by all heuristic augmentation scripts."""
    parser.add_argument(
        "--dataset",
        required=dataset_required,
        default=None if dataset_required else "gsm8k",
        help=f"Dataset/task. Supported: {', '.join(SUPPORTED_DATASETS)}",
    )
    parser.add_argument("--input-file", required=True, help="Input DataObs-format parquet path.")
    parser.add_argument("--output-file", required=True, help="Output parquet path.")
    parser.add_argument("--model-id", required=True, help="Local teacher model path or model id.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--gen-batch-size", type=int, default=128)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--correct-threshold", type=float, default=0.99)
    parser.add_argument("--disable-teacher-filter", action="store_true")
    parser.add_argument("--smoke-num-rows", type=int, default=0)


def validate_common_args(parser: argparse.ArgumentParser, args: argparse.Namespace, *, num_samples: int = 1) -> None:
    """Normalize common arguments and enforce generation constraints before running."""
    try:
        args.dataset = normalize_dataset(args.dataset)
    except ValueError as exc:
        parser.error(str(exc))
    if num_samples > 1 and not args.do_sample:
        parser.error("Generating more than one sample per prompt requires --do-sample.")
    if args.gen_batch_size < 1:
        parser.error("--gen-batch-size must be >= 1.")
    if args.smoke_num_rows < 0:
        parser.error("--smoke-num-rows must be >= 0.")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be >= 1.")
    if not (0 < args.gpu_memory_utilization <= 1):
        parser.error("--gpu-memory-utilization must be in (0, 1].")


def write_outputs(
    *,
    output_file: str,
    output_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    """Write kept rows, candidate rows, and a JSON summary beside the requested output file."""
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_parquet(out_path, index=False)

    candidates_path = out_path.with_name(f"{out_path.stem}.candidates.parquet")
    pd.DataFrame(candidate_rows).to_parquet(candidates_path, index=False)

    summary = dict(summary)
    summary["output_file"] = str(out_path)
    summary["candidates_file"] = str(candidates_path)
    summary_path = out_path.with_name(f"{out_path.stem}.summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def base_summary(
    *,
    method: str,
    args: argparse.Namespace,
    input_rows: int,
    candidate_rows: List[Dict[str, Any]],
    output_rows: List[Dict[str, Any]],
    start_time: float,
) -> Dict[str, Any]:
    """Create the common summary payload shared by all augmentation methods."""
    passed = [row for row in candidate_rows if row.get("teacher_filter_passed")]
    return {
        "method": method,
        "dataset": args.dataset,
        "input_file": args.input_file,
        "model_id": args.model_id,
        "num_input_rows": input_rows,
        "num_candidates": len(candidate_rows),
        "num_passed_candidates": len(passed),
        "num_kept": len(output_rows),
        "teacher_pass_rate": len(passed) / len(candidate_rows) if candidate_rows else 0.0,
        "kept_rate": len(output_rows) / len(candidate_rows) if candidate_rows else 0.0,
        "teacher_filter_applied": not args.disable_teacher_filter,
        "correct_threshold": args.correct_threshold,
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": args.do_sample,
            "max_new_tokens": args.max_new_tokens,
            "gen_batch_size": args.gen_batch_size,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_ids": args.gpu_ids,
        },
        "elapsed_sec": time.time() - start_time,
    }


def flatten(items: Iterable[Iterable[Any]]) -> List[Any]:
    """Flatten a two-level iterable into a list."""
    return [x for group in items for x in group]
