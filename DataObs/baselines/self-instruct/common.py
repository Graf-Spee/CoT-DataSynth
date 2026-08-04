from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DataObs.lib.data_process.cot_distill_teacher_filter import LocalTeacher  # noqa: E402

# Keep the same vLLM multiprocessing behavior as the main distillation path.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def prompt_to_messages(prompt: object) -> list[dict]:
    if isinstance(prompt, list):
        return [msg for msg in prompt if isinstance(msg, dict)]
    if isinstance(prompt, tuple):
        return [msg for msg in prompt if isinstance(msg, dict)]
    tolist = getattr(prompt, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if isinstance(converted, list):
            return [msg for msg in converted if isinstance(msg, dict)]
    return []


def extract_instruction(task: dict) -> str | None:
    instruction = task.get("instruction") or task.get("question")
    if instruction:
        return str(instruction).strip()

    prompt = task.get("prompt")
    if isinstance(prompt, str):
        text = prompt.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(parsed, list):
            for message in reversed(parsed):
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content")
                    if content:
                        return str(content).strip()
        return text

    messages = prompt_to_messages(prompt)
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if content:
                return str(content).strip()

    return None


def extract_answer(task: dict) -> str:
    answer = task.get("answer") or task.get("output") or task.get("response")
    if answer is not None:
        return str(answer).strip()
    gold_answer = task.get("gold_answer")
    if gold_answer is not None:
        return str(gold_answer).strip()
    reward_model = task.get("reward_model")
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
        if ground_truth is not None:
            return str(ground_truth).strip()
    return ""


def normalize_instruction_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("\"'`")
    return text


def dedupe_instructions(instructions: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for instruction in instructions:
        normalized = normalize_instruction_text(instruction)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def load_seed_tasks(input_file: str, smoke_num_rows: int) -> list[dict]:
    path = Path(input_file)

    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    elif path.suffix.lower() == ".parquet":
        rows = pd.read_parquet(path).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}. Expected .jsonl or .parquet.")

    if smoke_num_rows > 0:
        rows = rows[:smoke_num_rows]

    return rows


def make_teacher(args: object) -> LocalTeacher:
    os.environ["CUDA_VISIBLE_DEVICES"] = getattr(args, "gpu_ids", "0")
    # Some local runs fail to auto-detect the vLLM target device in this env.
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")
    return LocalTeacher(
        model_id=getattr(args, "model_id"),
        trust_remote_code=getattr(args, "trust_remote_code", False),
        dtype=getattr(args, "dtype", "auto"),
        tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.95),
    )


def batched_generate(
    teacher: LocalTeacher,
    prompts: list[str],
    *,
    n: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    gen_batch_size: int,
    desc: str,
    enable_thinking: bool = False,
    use_chat_template: bool = True,
    stop_sequences: list[str] | None = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> list[list[str]]:
    return [
        [item["text"] for item in batch]
        for batch in batched_generate_full(
            teacher,
            prompts,
            n=n,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            gen_batch_size=gen_batch_size,
            desc=desc,
            enable_thinking=enable_thinking,
            use_chat_template=use_chat_template,
            stop_sequences=stop_sequences,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
    ]


def batched_generate_full(
    teacher: LocalTeacher,
    prompts: list[str],
    *,
    n: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    gen_batch_size: int,
    desc: str,
    enable_thinking: bool = False,
    use_chat_template: bool = True,
    stop_sequences: list[str] | None = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> list[list[dict[str, object]]]:
    from vllm import SamplingParams

    outputs: list[list[dict[str, object]]] = []
    for start in tqdm(range(0, len(prompts), gen_batch_size), desc=desc, unit="batch"):
        batch = prompts[start : start + gen_batch_size]
        if use_chat_template:
            batch_messages = [[{"role": "user", "content": prompt}] for prompt in batch]
            rendered_prompts: list[str] = []
            for messages in batch_messages:
                kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                    "enable_thinking": enable_thinking,
                }
                try:
                    rendered = teacher.tokenizer.apply_chat_template(messages, **kwargs)
                except TypeError:
                    kwargs.pop("enable_thinking", None)
                    rendered = teacher.tokenizer.apply_chat_template(messages, **kwargs)
                rendered_prompts.append(rendered)
        else:
            rendered_prompts = batch

        sampling_params = SamplingParams(
            n=n,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            max_tokens=max_new_tokens,
            stop=stop_sequences or None,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        batch_outputs = teacher.llm.generate(rendered_prompts, sampling_params)
        outputs.extend(
            [
                [
                    {
                        "text": output.text.strip(),
                        "finish_reason": getattr(output, "finish_reason", None),
                        "stop_reason": getattr(output, "stop_reason", None),
                    }
                    for output in out.outputs
                ]
                for out in batch_outputs
            ]
        )
    return outputs


def _get_arg(args: object | None, name: str, default: object = None) -> object:
    if args is None:
        return default
    return getattr(args, name, default)


def write_outputs(
    output_file: str,
    generated_items: list[dict],
    sft_rows: list[dict],
    *,
    args: object | None = None,
    seed_task_count: int | None = None,
    bootstrapped_instruction_count: int | None = None,
    classified_instruction_count: int | None = None,
    elapsed_sec: float | None = None,
) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sft_rows).to_parquet(path, index=False)

    candidates = []
    item_metadata_by_source = {}
    for item in generated_items:
        item_metadata_by_source[item.get("source_index")] = {
            key: value
            for key, value in item.items()
            if key not in {"instruction", "answer", "raw_seed_task"}
        }

    for row in sft_rows:
        item_metadata = item_metadata_by_source.get(row.get("source_index"), {})
        candidates.append(
            {
                **row,
                **item_metadata,
                "candidate_instruction": row.get("instruction", row.get("original_instruction", row.get("question"))),
                "candidate_answer": row.get("instance_output", row.get("answer", "")),
                "candidate_retained": True,
            }
        )
    pd.DataFrame(candidates).to_parquet(path.with_name(f"{path.stem}.candidates.parquet"), index=False)

    teacher_passed_candidates = sum(1 for row in candidates if bool(row.get("teacher_filter_passed", False)))
    candidate_source_indices = {row.get("source_index") for row in candidates if row.get("source_index") is not None}
    kept_source_indices = {row.get("source_index") for row in sft_rows if row.get("source_index") is not None}
    candidate_retained = sum(1 for row in candidates if bool(row.get("candidate_retained", False)))

    summary = {
        "method": "self_instruct",
        "dataset": _get_arg(args, "dataset"),
        "input_file": _get_arg(args, "input_file"),
        "output_file": str(path),
        "candidates_file": str(path.with_name(f"{path.stem}.candidates.parquet")),
        "model_id": _get_arg(args, "model_id"),
        "num_input_rows": int(seed_task_count if seed_task_count is not None else len(generated_items)),
        "num_bootstrapped_instructions": int(
            bootstrapped_instruction_count if bootstrapped_instruction_count is not None else len(generated_items)
        ),
        "num_classified_instructions": int(
            classified_instruction_count if classified_instruction_count is not None else len(generated_items)
        ),
        "num_candidates": int(len(candidates)),
        "num_passed_candidates": int(teacher_passed_candidates),
        "num_kept": int(len(sft_rows)),
        "candidate_retained_rows": int(candidate_retained),
        "teacher_filter_applied": False,
        "correct_threshold": _get_arg(args, "correct_threshold"),
        "teacher_pass_rate": float(teacher_passed_candidates / len(candidates)) if candidates else 0.0,
        "kept_rate": float(len(sft_rows) / len(candidates)) if candidates else 0.0,
        "prompt_pass_rate": float(len(kept_source_indices) / len(candidate_source_indices)) if candidate_source_indices else 0.0,
        "num_prompts_with_passed_candidate": int(len(kept_source_indices)),
        "generation": {
            "temperature": _get_arg(args, "temperature"),
            "top_p": _get_arg(args, "top_p"),
            "do_sample": bool(_get_arg(args, "do_sample", False)),
            "max_new_tokens": _get_arg(args, "max_new_tokens"),
            "gen_batch_size": _get_arg(args, "gen_batch_size"),
            "tensor_parallel_size": _get_arg(args, "tensor_parallel_size"),
            "gpu_ids": _get_arg(args, "gpu_ids"),
            "gpu_memory_utilization": _get_arg(args, "gpu_memory_utilization"),
        },
        "bootstrap": {
            "num_instructions": _get_arg(args, "bootstrap_num_instructions"),
            "num_prompt_instructions": _get_arg(args, "bootstrap_num_prompt_instructions"),
            "request_batch_size": _get_arg(args, "bootstrap_request_batch_size"),
            "max_new_tokens": _get_arg(args, "bootstrap_max_new_tokens"),
            "temperature": _get_arg(args, "bootstrap_temperature"),
            "top_p": _get_arg(args, "bootstrap_top_p"),
            "similarity_threshold": _get_arg(args, "bootstrap_similarity_threshold"),
            "use_clf_seed_tasks_only": bool(_get_arg(args, "bootstrap_use_clf_seed_tasks_only", False)),
            "allow_stub_fallback": bool(_get_arg(args, "bootstrap_allow_stub_fallback", False)),
        },
        "classification": {
            "max_new_tokens": _get_arg(args, "classification_max_new_tokens"),
            "request_batch_size": _get_arg(args, "classification_request_batch_size"),
            "allow_heuristic_fallback": bool(_get_arg(args, "classification_allow_heuristic_fallback", False)),
        },
        "instance_generation": {
            "request_batch_size": _get_arg(args, "instance_request_batch_size"),
            "max_instances_to_generate": _get_arg(args, "instance_max_instances_to_generate"),
            "max_new_tokens_clf": _get_arg(args, "instance_max_new_tokens_clf"),
            "max_new_tokens_gen": _get_arg(args, "instance_max_new_tokens_gen"),
        },
        "elapsed_sec": float(elapsed_sec) if elapsed_sec is not None else None,
        "input_rows": int(seed_task_count if seed_task_count is not None else len(generated_items)),
        "candidate_rows": int(len(candidates)),
        "output_rows": int(len(sft_rows)),
    }
    path.with_name(f"{path.stem}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
