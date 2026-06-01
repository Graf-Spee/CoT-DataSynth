#!/usr/bin/env python3
"""CoT data distillation: local teacher generation + teacher-correctness filter.

Supported datasets: commonsenseqa, mbpp

Input parquet requirements:
- `prompt`: question/prompt payload (string or chat-message list)
- `reward_model`: dict-like object containing `ground_truth`

Output parquet columns:
- `question`: original `prompt` value from input
- `answer`: generated CoT text
- `gold_answer`: value from `reward_model["ground_truth"]`
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore

# Avoid CUDA re-init errors in vLLM worker subprocesses.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

REASONING_SUFFIX = {
    # Distillation-Bench style for MCQ tasks
    "commonsenseqa": (
        'Provide your step-by-step reasoning to the question first, and then '
        'print "The answer is (x)" where x is A, B, C, D or E, at the end of your response.'
    ),
    # Distillation-Bench style adapted for code generation output format
    "mbpp": (
        "Provide your step-by-step reasoning first, and then output the final "
        "Python solution wrapped in ```python ... ```."
    ),
}


def _normalize_dataset_name(name: str) -> str:
    key = name.strip().lower().replace("-", "").replace("_", "")
    if key in {"commonsenseqa", "commonsenseq"}:
        return "commonsenseqa"
    if key in {"mbpp"}:
        return "mbpp"
    raise ValueError(f"Unsupported dataset: {name}. Only commonsenseqa and mbpp are supported.")


def _load_reward_functions(dataset_name: str):
    """Load dataset-specific reward/extract functions from verl.utils.reward_score."""
    dataset = _normalize_dataset_name(dataset_name)

    if dataset == "commonsenseqa":
        from verl.utils.reward_score import multiple_choice as reward_mod
        return reward_mod.compute_score
    if dataset == "mbpp":
        from verl.utils.reward_score import mbpp as reward_mod
        return reward_mod.compute_score

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _append_reasoning_suffix(question: str, dataset: str) -> List[Dict[str, str]]:
    suffix = REASONING_SUFFIX.get(_normalize_dataset_name(dataset), "").strip()
    if not suffix or not question:
        raise ValueError(f"Unsupported dataset: {dataset}")

    out = [{"role": "user", "content": question + "\n" + suffix}]
    return out


@dataclass
class LocalTeacher:
    model_id: str
    trust_remote_code: bool
    dtype: str
    tensor_parallel_size: int
    gpu_memory_utilization: float

    def __post_init__(self) -> None:
        # Keep tokenizer formatting logic from current script; generation backend moves to vLLM.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Lazy import to keep module import lightweight when only checking --help.
        from vllm import LLM

        self.llm = LLM(
            model=self.model_id,
            trust_remote_code=self.trust_remote_code,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype=self.dtype,
        )

    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        n: int,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> List[List[str]]:
        if n > 1 and not do_sample:
            raise ValueError("num-cots > 1 requires sampling. Please set --do-sample.")

        prompts: List[str] = []
        for messages in messages_batch:
            prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            n=n,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            max_tokens=max_new_tokens,
        )
        outputs = self.llm.generate(prompts, sampling_params)

        all_responses: List[List[str]] = []
        for out in outputs:
            all_responses.append([o.text.strip() for o in out.outputs])
        return all_responses


def _prepare_generation_rows(df: pd.DataFrame, dataset: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        question = row["prompt"]    # str
        reward_model = row["reward_model"]
        gold = reward_model.get("ground_truth", None) if isinstance(reward_model, dict) else None
        if gold is None:
            raise ValueError(f"Missing gold answer at row index {i}")

        messages = _append_reasoning_suffix(question, dataset)
        rows.append({"question": question, "gold_answer": gold, "messages": messages})
    return rows


def run(args: argparse.Namespace) -> None:
    dataset = _normalize_dataset_name(args.dataset)
    compute_score = _load_reward_functions(dataset)

    df = pd.read_parquet(args.input_file)
    if args.smoke_num_rows > 0:
        df = df.head(args.smoke_num_rows)
    prepared_rows = _prepare_generation_rows(df, dataset)

    teacher = LocalTeacher(
        model_id=args.model_id,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    output_rows: List[Dict[str, Any]] = []
    total_candidates = 0
    total_kept = 0

    # Stage 2: generation (vLLM batched)
    generated_responses: List[List[str]] = []
    for start in tqdm(range(0, len(prepared_rows), args.gen_batch_size), desc="Generation stage", unit="batch"):
        batch_rows = prepared_rows[start : start + args.gen_batch_size]
        batch_messages = [r["messages"] for r in batch_rows]
        batch_outputs = teacher.generate_batch(
            messages_batch=batch_messages,
            n=args.num_cots,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        generated_responses.extend(batch_outputs)

    # Stage 3: teacher correctness filtering
    for row_data, answers in tqdm(
        zip(prepared_rows, generated_responses),
        total=len(prepared_rows),
        desc="Filter stage",
        unit="sample",
    ):
        question = row_data["question"]
        gold = row_data["gold_answer"]
        for answer in answers:
            total_candidates += 1
            score = compute_score(answer, gold)
            passed_teacher_filter = score >= args.correct_threshold
            if args.disable_teacher_filter or passed_teacher_filter:
                total_kept += 1
                row_out = {"question": question, "answer": answer, "gold_answer": gold}
                if args.disable_teacher_filter:
                    row_out["teacher_filter_passed"] = bool(passed_teacher_filter)
                output_rows.append(row_out)

    out_columns = ["question", "answer", "gold_answer"]
    if args.disable_teacher_filter:
        out_columns.append("teacher_filter_passed")
    out_df = pd.DataFrame(output_rows, columns=out_columns)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    print(
        json.dumps(
            {
                "dataset": dataset,
                "input_file": args.input_file,
                "output_file": str(out_path),
                "num_input_rows": int(len(df)),
                "num_prepared_rows": int(len(prepared_rows)),
                "num_cots_per_row": int(args.num_cots),
                "num_candidates": int(total_candidates),
                "num_kept": int(total_kept),
                "teacher_filter_applied": bool(not args.disable_teacher_filter),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoT distillation with teacher correctness filtering.")
    parser.add_argument("--dataset", required=True, choices=["commonsenseqa", "mbpp"], help="Dataset name.")
    parser.add_argument("--input-file", required=True, help="Input processed parquet path.")
    parser.add_argument("--output-file", required=True, help="Output parquet path.")

    parser.add_argument("--model-id", required=True, help="Local teacher model path or model id.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code.")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model load dtype.",
    )

    parser.add_argument("--num-cots", type=int, default=4, help="N: number of CoT samples per row.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max generation tokens per CoT.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling; required when num-cots > 1.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Sampling top-p.")
    parser.add_argument("--gen-batch-size", type=int, default=128, help="Batch size for vLLM generation stage.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory utilization (0,1].",
    )
    parser.add_argument(
        "--correct-threshold",
        type=float,
        default=0.99,
        help="Threshold on reward score to keep a sample.",
    )
    parser.add_argument(
        "--disable-teacher-filter",
        action="store_true",
        help="If set, keep all generated CoTs without applying teacher correctness filtering.",
    )
    parser.add_argument(
        "--smoke-num-rows",
        type=int,
        default=0,
        help="If > 0, only process the first K rows for quick smoke testing.",
    )

    args = parser.parse_args()
    if args.num_cots < 1:
        parser.error("--num-cots must be >= 1.")
    if args.num_cots > 1 and not args.do_sample:
        parser.error("--num-cots > 1 requires --do-sample.")
    if args.gen_batch_size < 1:
        parser.error("--gen-batch-size must be >= 1.")
    if not (0 < args.gpu_memory_utilization <= 1):
        parser.error("--gpu-memory-utilization must be in (0, 1].")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be >= 1.")
    if args.smoke_num_rows < 0:
        parser.error("--smoke-num-rows must be >= 0.")
    return args


if __name__ == "__main__":
    run(parse_args())
