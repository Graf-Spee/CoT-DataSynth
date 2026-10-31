from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set vLLM target device early so LocalTeacher import does not try to infer the
# backend before our runtime config is in place.
os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")

from DataObs.lib.data_process.cot_distill_teacher_filter import LocalTeacher  # noqa: E402

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

BOXED_ANSWER_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


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
    instruction = task.get("instruction") or task.get("question") or task.get("problem")
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
    answer = (
        task.get("answer")
        or task.get("gold_answer")
        or task.get("output")
        or task.get("response")
        or task.get("canonical_solution")
    )
    if answer is not None:
        return str(answer).strip()

    reward_model = task.get("reward_model")
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
        if ground_truth is not None:
            return str(ground_truth).strip()
    return ""


def extract_reward_ground_truth(task: dict) -> object:
    reward_model = task.get("reward_model")
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth")
    return None


def normalize_text(text: str) -> str:
    text = THINK_BLOCK_RE.sub("", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def strip_reasoning_block(text: str) -> str:
    return THINK_BLOCK_RE.sub("", str(text or "")).strip()


def extract_boxed_answer(text: str) -> str:
    matches = BOXED_ANSWER_RE.findall(str(text or ""))
    if matches:
        return matches[-1].strip()
    return ""


def load_seed_tasks(input_file: str, smoke_num_rows: int) -> list[dict]:
    path = Path(input_file)
    if path.suffix.lower() == ".jsonl":
        rows: list[dict] = []
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
    return LocalTeacher(
        model_id=getattr(args, "model_id"),
        trust_remote_code=getattr(args, "trust_remote_code", False),
        dtype=getattr(args, "dtype", "auto"),
        tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.95),
        max_model_len=getattr(args, "max_model_len", None),
        enforce_eager=getattr(args, "enforce_eager", False),
        disable_custom_all_reduce=getattr(args, "disable_custom_all_reduce", False),
    )


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
            rendered_prompts: list[str] = []
            for prompt in batch:
                kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                    "enable_thinking": enable_thinking,
                }
                try:
                    rendered = teacher.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        **kwargs,
                    )
                except TypeError:
                    kwargs.pop("enable_thinking", None)
                    rendered = teacher.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        **kwargs,
                    )
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
    generated_task_count: int | None = None,
    consistency_passed_count: int | None = None,
    rip_passed_count: int | None = None,
    code_passed_count: int | None = None,
    elapsed_sec: float | None = None,
) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sft_rows).to_parquet(path, index=False)

    candidates = []
    for item in generated_items:
        candidate_row = dict(item)
        candidate_row["candidate_instruction"] = item.get("question", item.get("instruction", ""))
        candidate_row["candidate_answer"] = (
            item.get("target_answer")
            or item.get("rip_best_response", "")
            or item.get("code_best_response", "")
        )
        candidate_row["candidate_retained"] = bool(
            item.get("consistency_passed", False)
            or item.get("rip_passed", False)
            or item.get("code_passed", False)
        )
        candidates.append(candidate_row)
    pd.DataFrame(candidates).to_parquet(path.with_name(f"{path.stem}.candidates.parquet"), index=False)

    kept_source_indices = {row.get("source_index") for row in sft_rows if row.get("source_index") is not None}
    candidate_source_indices = {row.get("source_index") for row in candidates if row.get("source_index") is not None}
    passed_count = sum(
        1
        for row in candidates
        if bool(
            row.get("consistency_passed", False)
            or row.get("rip_passed", False)
            or row.get("code_passed", False)
        )
    )
    consistency_count = sum(1 for row in candidates if bool(row.get("consistency_passed", False)))
    rip_count = sum(1 for row in candidates if bool(row.get("rip_passed", False)))
    code_count = sum(1 for row in candidates if bool(row.get("code_passed", False)))
    reasoning_count = sum(1 for row in candidates if str(row.get("task_type", "")) == "reasoning")
    instruction_count = sum(1 for row in candidates if str(row.get("task_type", "")) == "instruction")
    code_task_count = sum(1 for row in candidates if str(row.get("task_type", "")) == "code")
    summary = {
        "method": "cot_self_instruct",
        "dataset": _get_arg(args, "dataset"),
        "input_file": _get_arg(args, "input_file"),
        "output_file": str(path),
        "candidates_file": str(path.with_name(f"{path.stem}.candidates.parquet")),
        "model_id": _get_arg(args, "model_id"),
        "num_input_rows": int(seed_task_count if seed_task_count is not None else len(generated_items)),
        "num_generated_tasks": int(generated_task_count if generated_task_count is not None else len(generated_items)),
        "num_reasoning_tasks": int(reasoning_count),
        "num_instruction_tasks": int(instruction_count),
        "num_code_tasks": int(code_task_count),
        "num_candidates": int(len(candidates)),
        "num_consistency_passed": int(consistency_passed_count if consistency_passed_count is not None else consistency_count),
        "num_rip_passed": int(rip_passed_count if rip_passed_count is not None else rip_count),
        "num_code_passed": int(code_passed_count if code_passed_count is not None else code_count),
        "num_kept": int(len(sft_rows)),
        "teacher_filter_applied": False,
        "kept_rate": float(len(sft_rows) / len(candidates)) if candidates else 0.0,
        "prompt_pass_rate": float(len(kept_source_indices) / len(candidate_source_indices)) if candidate_source_indices else 0.0,
        "num_prompts_with_passed_candidate": int(len(kept_source_indices)),
        "cot_generation": {
            "num_instructions": _get_arg(args, "cot_num_instructions"),
            "num_fewshot_examples": _get_arg(args, "cot_num_fewshot_examples"),
            "num_prompts_per_request": _get_arg(args, "cot_num_prompts_per_request"),
            "max_new_tokens": _get_arg(args, "cot_max_new_tokens"),
            "temperature": _get_arg(args, "cot_temperature"),
            "top_p": _get_arg(args, "cot_top_p"),
            "do_sample": bool(_get_arg(args, "cot_do_sample", True)),
        },
        "consistency_filter": {
            "k": _get_arg(args, "consistency_k"),
            "max_new_tokens": _get_arg(args, "consistency_max_new_tokens"),
            "temperature": _get_arg(args, "consistency_temperature"),
            "top_p": _get_arg(args, "consistency_top_p"),
            "do_sample": bool(_get_arg(args, "consistency_do_sample", True)),
            "min_vote_ratio": _get_arg(args, "consistency_min_vote_ratio"),
        },
        "instruction_generation": {
            "num_instructions": _get_arg(args, "instruction_num_instructions"),
            "num_fewshot_examples": _get_arg(args, "instruction_num_fewshot_examples"),
            "max_new_tokens": _get_arg(args, "instruction_max_new_tokens"),
            "temperature": _get_arg(args, "instruction_temperature"),
            "top_p": _get_arg(args, "instruction_top_p"),
            "do_sample": bool(_get_arg(args, "instruction_do_sample", True)),
        },
        "rip_filter": {
            "num_responses": _get_arg(args, "rip_num_responses"),
            "response_max_new_tokens": _get_arg(args, "rip_response_max_new_tokens"),
            "response_temperature": _get_arg(args, "rip_response_temperature"),
            "response_top_p": _get_arg(args, "rip_response_top_p"),
            "response_do_sample": bool(_get_arg(args, "rip_response_do_sample", True)),
            "score_max_new_tokens": _get_arg(args, "rip_score_max_new_tokens"),
            "min_score": _get_arg(args, "rip_min_score"),
        },
        "code_generation": {
            "num_samples": _get_arg(args, "code_num_samples"),
            "max_new_tokens": _get_arg(args, "code_max_new_tokens"),
            "temperature": _get_arg(args, "code_temperature"),
            "top_p": _get_arg(args, "code_top_p"),
            "do_sample": bool(_get_arg(args, "code_do_sample", True)),
            "correct_threshold": _get_arg(args, "code_correct_threshold"),
        },
        "runtime": {
            "gen_batch_size": _get_arg(args, "gen_batch_size"),
            "tensor_parallel_size": _get_arg(args, "tensor_parallel_size"),
            "gpu_ids": _get_arg(args, "gpu_ids"),
            "gpu_memory_utilization": _get_arg(args, "gpu_memory_utilization"),
        },
        "elapsed_sec": float(elapsed_sec) if elapsed_sec is not None else None,
    }
    path.with_name(f"{path.stem}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
