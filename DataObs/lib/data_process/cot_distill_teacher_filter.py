#!/usr/bin/env python3
"""CoT data distillation: local teacher generation + teacher-correctness filter.

Supported datasets:
- Multiple choice: arc-challenge/ai2_arc, aqua_rat, commonsenseqa
- Math: gsm8k, math, math-500, numinamath
- Boolean QA: strategyqa
- Code generation: mbpp, mbppplus, humaneval, humanevalplus

Input parquet requirements:
- `prompt`: question/prompt payload (string or chat-message list)
- `reward_model`: dict-like object containing `ground_truth`

Output parquet columns:
- `question`: original user prompt text from input
- `answer`: generated CoT text
- `gold_answer`: value from `reward_model["ground_truth"]`
- metadata columns for tracking distillation variants
"""

from __future__ import annotations

import argparse
import math
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer  # type: ignore

# Avoid CUDA re-init errors in vLLM worker subprocesses.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

SUPPORTED_DATASETS = (
    "arc-challenge",
    "aqua_rat",
    "commonsenseqa",
    "gsm8k",
    "humaneval",
    "humanevalplus",
    "math",
    "math-500",
    "mbpp",
    "mbppplus",
    "numinamath",
    "strategyqa",
)

DATASET_GROUP = {
    "arc-challenge": "multiple_choice",
    "aqua_rat": "multiple_choice",
    "commonsenseqa": "multiple_choice",
    "gsm8k": "gsm8k",
    "math": "math",
    "math-500": "math",
    "numinamath": "math",
    "strategyqa": "truefalse",
    "mbpp": "code_tests",
    "mbppplus": "code_tests",
    "humaneval": "code_tests",
    "humanevalplus": "code_tests",
}

REASONING_SUFFIX = {
    "multiple_choice": (
        "Additional distillation instruction: provide step-by-step reasoning, "
        'then end with exactly this format: "The answer is (X)" where X is the correct option label.'
    ),
    "gsm8k": (
        'Additional distillation instruction: provide step-by-step reasoning and end with "#### <number>".'
    ),
    "math": (
        r"Additional distillation instruction: provide step-by-step reasoning and put the final answer in \boxed{}."
    ),
    "truefalse": (
        'Additional distillation instruction: provide concise reasoning, '
        'then end with exactly "So the answer is yes" or "So the answer is no".'
    ),
    "code_tests": (
        "Additional distillation instruction: reason briefly, then output the final Python solution "
        "as the last markdown code block in this format: ```python\n...\n```."
    ),
}


def _normalize_dataset_name(name: str) -> str:
    raw = name.strip().lower()
    key = raw.replace("-", "").replace("_", "")
    if key in {"arc", "arcchallenge", "ai2arc", "arcchalleng"} or raw == "arc-":
        return "arc-challenge"
    if key in {"aquarat", "aqua"}:
        return "aqua_rat"
    if key in {"commonsenseqa", "commonsenseq", "csqa"}:
        return "commonsenseqa"
    if key == "gsm8k":
        return "gsm8k"
    if key in {"humaneval", "openaihumaneval"}:
        return "humaneval"
    if key in {"humanevalplus", "humaneval+"}:
        return "humanevalplus"
    if key == "math":
        return "math"
    if key in {"math500"}:
        return "math-500"
    if key == "mbpp":
        return "mbpp"
    if key in {"mbppplus", "mbpp+"}:
        return "mbppplus"
    if key in {"numinamath", "numina"}:
        return "numinamath"
    if key in {"strategyqa", "strategy"}:
        return "strategyqa"
    supported = ", ".join(SUPPORTED_DATASETS)
    raise ValueError(f"Unsupported dataset: {name}. Supported datasets: {supported}. BFCL uses a separate evaluator.")


def _load_reward_functions(dataset_name: str) -> Callable[..., float]:
    """Load dataset-specific reward/extract functions from verl.utils.reward_score."""
    dataset = _normalize_dataset_name(dataset_name)
    group = DATASET_GROUP[dataset]

    if group == "multiple_choice":
        from verl.utils.reward_score import multiple_choice as reward_mod
        return reward_mod.compute_score
    if group == "gsm8k":
        from verl.utils.reward_score import gsm8k as reward_mod
        return reward_mod.compute_score
    if group == "math":
        from verl.utils.reward_score import math_verify as reward_mod
        return reward_mod.compute_score
    if group == "truefalse":
        from verl.utils.reward_score import truefalse as reward_mod
        return reward_mod.compute_score
    if group == "code_tests":
        from verl.utils.reward_score import mbpp as reward_mod
        return reward_mod.compute_score
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict, list)):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def _as_messages(prompt: Any) -> List[Dict[str, str]]:
    prompt = _to_builtin(prompt)
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, dict):
        role = str(prompt.get("role", "user"))
        content = str(prompt.get("content", ""))
        return [{"role": role, "content": content}]
    if isinstance(prompt, list):
        messages: List[Dict[str, str]] = []
        for item in prompt:
            item = _to_builtin(item)
            if isinstance(item, dict):
                role = str(item.get("role", "user"))
                content = str(item.get("content", ""))
            else:
                role = "user"
                content = str(item)
            if content.strip():
                messages.append({"role": role, "content": content})
        if messages:
            return messages
    raise ValueError(f"Unsupported prompt payload type: {type(prompt).__name__}")


def _last_user_text(messages: List[Dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return messages[-1].get("content", "") if messages else ""


def _append_reasoning_suffix(prompt: Any, dataset: str) -> tuple[str, List[Dict[str, str]]]:
    canonical = _normalize_dataset_name(dataset)
    group = DATASET_GROUP[canonical]
    suffix = REASONING_SUFFIX.get(group, "").strip()
    messages = _as_messages(prompt)
    question = _last_user_text(messages).strip()
    if not suffix or not question:
        raise ValueError(f"Unsupported dataset: {dataset}")

    out = [dict(message) for message in messages]
    for idx in range(len(out) - 1, -1, -1):
        if out[idx].get("role") == "user":
            out[idx]["content"] = out[idx].get("content", "").rstrip() + "\n\n" + suffix
            break
    else:
        out.append({"role": "user", "content": suffix})
    return question, out


def _get_ground_truth(reward_model: Any) -> Any:
    reward_model = _to_builtin(reward_model)
    if isinstance(reward_model, dict):
        return _to_builtin(reward_model.get("ground_truth", None))
    return None


def _safe_mean(values: List[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _safe_quantile(values: List[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] * (1 - weight) + ordered[hi] * weight)


def _summarize_distill(
    *,
    args: argparse.Namespace,
    dataset: str,
    input_rows: int,
    prepared_rows: int,
    candidate_rows: List[Dict[str, Any]],
    kept_rows: List[Dict[str, Any]],
    elapsed_sec: float,
    output_path: Path,
    candidates_path: Path,
) -> Dict[str, Any]:
    scores = [float(row["teacher_score"]) for row in candidate_rows]
    answer_lengths = [float(row["answer_char_len"]) for row in candidate_rows]
    kept_answer_lengths = [float(len(str(row["answer"]))) for row in kept_rows]
    passed = [row for row in candidate_rows if row["teacher_filter_passed"]]
    source_indices = {row["source_index"] for row in candidate_rows}
    source_indices_passed = {row["source_index"] for row in passed}

    num_candidates = len(candidate_rows)
    num_kept = len(kept_rows)
    return {
        "dataset": dataset,
        "input_file": args.input_file,
        "output_file": str(output_path),
        "candidates_file": str(candidates_path),
        "model_id": args.model_id,
        "num_input_rows": int(input_rows),
        "num_prepared_rows": int(prepared_rows),
        "num_cots_per_row": int(args.num_cots),
        "num_candidates": int(num_candidates),
        "num_passed_candidates": int(len(passed)),
        "num_kept": int(num_kept),
        "teacher_pass_rate": float(len(passed) / num_candidates) if num_candidates else 0.0,
        "kept_rate": float(num_kept / num_candidates) if num_candidates else 0.0,
        "prompt_pass_rate": float(len(source_indices_passed) / len(source_indices)) if source_indices else 0.0,
        "num_prompts_with_passed_candidate": int(len(source_indices_passed)),
        "teacher_filter_applied": bool(not args.disable_teacher_filter),
        "correct_threshold": float(args.correct_threshold),
        "generation": {
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "do_sample": bool(args.do_sample),
            "max_new_tokens": int(args.max_new_tokens),
            "gen_batch_size": int(args.gen_batch_size),
            "tensor_parallel_size": int(args.tensor_parallel_size),
            "gpu_ids": args.gpu_ids,
        },
        "score": {
            "mean": _safe_mean(scores),
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "p25": _safe_quantile(scores, 0.25),
            "p50": _safe_quantile(scores, 0.50),
            "p75": _safe_quantile(scores, 0.75),
        },
        "answer_length_chars": {
            "mean_all_candidates": _safe_mean(answer_lengths),
            "mean_kept": _safe_mean(kept_answer_lengths),
            "min_all_candidates": min(answer_lengths) if answer_lengths else None,
            "max_all_candidates": max(answer_lengths) if answer_lengths else None,
        },
        "empty_answer_rate": float(sum(1 for row in candidate_rows if not str(row["answer"]).strip()) / num_candidates)
        if num_candidates
        else 0.0,
        "elapsed_sec": float(elapsed_sec),
        "supported_datasets": list(SUPPORTED_DATASETS),
    }


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
    canonical = _normalize_dataset_name(dataset)
    rows: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        question, messages = _append_reasoning_suffix(row["prompt"], canonical)
        gold = _get_ground_truth(row["reward_model"])
        if gold is None:
            raise ValueError(f"Missing gold answer at row index {i}")

        rows.append(
            {
                "source_index": int(i) if isinstance(i, int) else str(i),
                "dataset": canonical,
                "data_source": row.get("data_source", canonical),
                "question": question,
                "gold_answer": gold,
                "messages": messages,
            }
        )
    return rows


def run(args: argparse.Namespace) -> None:
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

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
    candidate_rows: List[Dict[str, Any]] = []
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
        for sample_idx, answer in enumerate(answers):
            total_candidates += 1
            score = compute_score(answer, gold)
            passed_teacher_filter = score >= args.correct_threshold
            candidate_rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "gold_answer": gold,
                    "dataset": row_data["dataset"],
                    "data_source": row_data["data_source"],
                    "source_index": row_data["source_index"],
                    "sample_index": int(sample_idx),
                    "teacher_score": float(score),
                    "teacher_filter_passed": bool(passed_teacher_filter),
                    "answer_char_len": int(len(str(answer))),
                }
            )
            if args.disable_teacher_filter or passed_teacher_filter:
                total_kept += 1
                row_out = {
                    "question": question,
                    "answer": answer,
                    "gold_answer": gold,
                    "dataset": row_data["dataset"],
                    "data_source": row_data["data_source"],
                    "source_index": row_data["source_index"],
                    "teacher_score": float(score),
                    "teacher_filter_passed": bool(passed_teacher_filter),
                }
                output_rows.append(row_out)

    out_columns = [
        "question",
        "answer",
        "gold_answer",
        "dataset",
        "data_source",
        "source_index",
        "teacher_score",
        "teacher_filter_passed",
    ]
    out_df = pd.DataFrame(output_rows, columns=out_columns)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    candidates_path = out_path.with_name(f"{out_path.stem}.candidates.parquet")
    candidates_df = pd.DataFrame(candidate_rows)
    candidates_df.to_parquet(candidates_path, index=False)

    summary = _summarize_distill(
        args=args,
        dataset=dataset,
        input_rows=len(df),
        prepared_rows=len(prepared_rows),
        candidate_rows=candidate_rows,
        kept_rows=output_rows,
        elapsed_sec=time.time() - start_time,
        output_path=out_path,
        candidates_path=candidates_path,
    )
    summary_path = out_path.with_name(f"{out_path.stem}.summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    del teacher
    import gc
    gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoT distillation with teacher correctness filtering.")
    parser.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset name. Supported: {', '.join(SUPPORTED_DATASETS)}. BFCL is not supported here.",
    )
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

    parser.add_argument("--gpu-ids", type=str, default="0", help="The GPU IDs of all usable GPUs.")

    parser.add_argument("--num-cots", type=int, default=4, help="N: number of CoT samples per row.")
    parser.add_argument("--max-new-tokens", type=int, default=8192, help="Max generation tokens per CoT.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling; required when num-cots > 1.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Sampling top-p.")
    parser.add_argument("--gen-batch-size", type=int, default=128, help="Batch size for vLLM generation stage.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
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
    try:
        args.dataset = _normalize_dataset_name(args.dataset)
    except ValueError as exc:
        parser.error(str(exc))
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
