from __future__ import annotations

import json
from functools import partial
from multiprocessing import Pool
import random
import re
import string
import sys
from statistics import mean

try:
    from .common import (
        batched_generate_full,
        extract_answer,
        extract_instruction,
        make_teacher,
    )
    from .templates import BOOTSTRAP_PROMPT, BOOTSTRAP_PROMPT_CLF, BOOTSTRAP_QWEN_CHAT_TEMPLATE
except ImportError:
    from common import (
        batched_generate_full,
        extract_answer,
        extract_instruction,
        make_teacher,
    )
    from templates import BOOTSTRAP_PROMPT, BOOTSTRAP_PROMPT_CLF, BOOTSTRAP_QWEN_CHAT_TEMPLATE


THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
BAD_WORDS = ("image", "images", "graph", "graphs", "picture", "pictures", "file", "files", "map", "maps", "draw", "plot", "go to")
OFFICIAL_STOP_SEQUENCES = ["\n\n", "\n16", "16.", "16 ."]

random.seed(42)


try:
    from rouge_score import rouge_scorer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    rouge_scorer = None


def encode_prompt(prompt_instructions: list[str], classification: bool = False) -> str:
    if classification:
        prompt = BOOTSTRAP_PROMPT_CLF + "\n"
    else:
        prompt = BOOTSTRAP_PROMPT + "\n"
    for idx, instruction in enumerate(prompt_instructions):
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"{idx + 1}. {instruction}\n"
    prompt += f"{len(prompt_instructions) + 1}."
    return prompt


def format_qwen_bootstrap_prompt(prompt: str, next_index: int) -> str:
    return BOOTSTRAP_QWEN_CHAT_TEMPLATE.format(
        next_index=next_index,
        prompt=prompt,
    )


def find_word_in_string(word: str, text: str) -> bool:
    return re.compile(rf"\b({re.escape(word)})\b", flags=re.IGNORECASE).search(text) is not None


def sample_machine_instructions(machine_instructions: list[str], similarities: object | None = None, n: int = 2) -> list[str]:
    return random.sample(machine_instructions, min(n, len(machine_instructions)))


def post_process_gpt3_response(response: dict[str, object] | None) -> list[str]:
    if response is None:
        return []
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return []
    if first_choice.get("finish_reason") == "length":
        return []

    text = THINK_BLOCK.sub("", str(first_choice.get("text", "") or "")).strip()
    if not text:
        return []

    raw_instructions = re.split(r"\n\d+\s?\. ", "\n" + text)

    instructions: list[str] = []
    for inst in raw_instructions:
        inst = re.sub(r"\s+", " ", inst).strip()
        inst = inst.strip().capitalize()
        if not inst:
            continue
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        if any(find_word_in_string(word, inst) for word in BAD_WORDS):
            continue
        if inst.startswith("Write a program"):
            continue
        if inst[0] in string.punctuation:
            continue
        if not inst[0].isascii():
            continue
        instructions.append(inst)
    return instructions


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _prepare_seed_instructions(seed_tasks: list[dict], use_clf_seed_tasks_only: bool) -> list[str]:
    extracted_seed_instructions = []
    for task in seed_tasks:
        if use_clf_seed_tasks_only:
            is_classification = task.get("is_classification")
            if is_classification is not None and not _coerce_bool(is_classification):
                continue
        instruction = extract_instruction(task)
        if instruction:
            extracted_seed_instructions.append(instruction)
    return extracted_seed_instructions


def _sample_prompt_instructions(seed_instructions: list[str], machine_instructions: list[str], num_prompt_instructions: int) -> list[str]:
    prompt_instructions = sample_machine_instructions(machine_instructions, similarities=None, n=2)
    remaining = max(0, num_prompt_instructions - len(prompt_instructions))
    if remaining > 0 and seed_instructions:
        prompt_instructions += random.sample(seed_instructions, min(len(seed_instructions), remaining))
    random.shuffle(prompt_instructions)
    return prompt_instructions


def bootstrap_instructions(seed_tasks: list[dict], args: object | None = None) -> list[dict]:
    seed_instructions = _prepare_seed_instructions(seed_tasks, getattr(args, "bootstrap_use_clf_seed_tasks_only", False))
    use_stub_fallback = bool(getattr(args, "bootstrap_allow_stub_fallback", False))
    stub_records = []
    for idx, task in enumerate(seed_tasks):
        instruction = extract_instruction(task)
        if not instruction:
            continue
        stub_records.append(
            {
                "instruction": instruction,
                "answer": extract_answer(task),
                "source_index": idx,
                "raw_seed_task": task,
                "generation_method": "self_instruct_stub",
                "original_instruction": instruction,
            }
        )

    if not seed_instructions:
        return []

    target_count = getattr(args, "bootstrap_num_instructions", 8) if args is not None else 8
    num_prompt_instructions = getattr(args, "bootstrap_num_prompt_instructions", 8) if args is not None else 8
    request_batch_size = getattr(args, "bootstrap_request_batch_size", 5) if args is not None else 5
    max_new_tokens = getattr(args, "bootstrap_max_new_tokens", getattr(args, "max_new_tokens", 1024))
    temperature = getattr(args, "bootstrap_temperature", 0.7)
    top_p = getattr(args, "bootstrap_top_p", 0.5)
    similarity_threshold = getattr(args, "bootstrap_similarity_threshold", 0.7)
    num_cpus = int(getattr(args, "bootstrap_num_cpus", 4))

    machine_instructions: list[str] = []
    generated_records: list[dict] = []
    request_idx = 0
    bootstrap_error = ""

    try:
        if rouge_scorer is None:
            raise RuntimeError("rouge_score is required for official-equivalent bootstrap similarity filtering")

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        teacher = make_teacher(args) if args is not None and getattr(args, "model_id", "") else None
        if teacher is None:
            raise RuntimeError("bootstrap requires a model_id for local generation")
        all_instructions = list(seed_instructions)

        while len(machine_instructions) < target_count:
            batch_prompts: list[str] = []
            batch_prompt_metadata: list[tuple[str, int]] = []
            for _ in range(request_batch_size):
                prompt_instructions = _sample_prompt_instructions(seed_instructions, machine_instructions, num_prompt_instructions)
                base_prompt = encode_prompt(
                    prompt_instructions,
                    classification=getattr(args, "bootstrap_use_clf_seed_tasks_only", False),
                )
                next_index = len(prompt_instructions) + 1
                batch_prompts.append(format_qwen_bootstrap_prompt(base_prompt, next_index))
                batch_prompt_metadata.append((base_prompt, next_index))

            outputs = batched_generate_full(
                teacher,
                batch_prompts,
                n=1,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                gen_batch_size=request_batch_size,
                desc="Bootstrap instructions",
                enable_thinking=False,
                use_chat_template=True,
                stop_sequences=OFFICIAL_STOP_SEQUENCES,
                frequency_penalty=0.0,
                presence_penalty=2.0,
            )

            instructions: list[str] = []
            all_metadata: list[dict[str, object]] = []
            for prompt_idx, ((base_prompt, next_index), batch_outputs) in enumerate(zip(batch_prompt_metadata, outputs)):
                first_output = batch_outputs[0] if batch_outputs else {}
                raw_text = str(first_output.get("text", "") or "")
                finish_reason = first_output.get("finish_reason")
                stop_reason = first_output.get("stop_reason")
                official_response = {
                    "choices": [
                        {
                            "text": raw_text,
                            "finish_reason": finish_reason,
                        }
                    ],
                    "stop_reason": stop_reason,
                }
                candidates = post_process_gpt3_response(official_response)
                instructions.extend(candidates)
                all_metadata.extend(
                    [
                        {
                            "prompt": base_prompt,
                            "response": official_response,
                            "model_prompt": batch_prompts[prompt_idx],
                            "next_index": next_index,
                            "prompt_index": prompt_idx,
                        }
                    ]
                    * len(candidates)
                )

            for inst, metadata in zip(instructions, all_metadata):
                with Pool(num_cpus) as pool:
                    rouge_score_objects = pool.map(
                        partial(scorer.score, inst),
                        all_instructions,
                    )
                rouge_scores = [score["rougeL"].fmeasure for score in rouge_score_objects]

                max_score = max(rouge_scores) if rouge_scores else 0.0
                if max_score > similarity_threshold:
                    continue

                most_similar = {}
                if rouge_scores and all_instructions:
                    top_indices = sorted(range(len(rouge_scores)), key=lambda i: rouge_scores[i], reverse=True)[:10]
                    most_similar = {all_instructions[i]: rouge_scores[i] for i in top_indices}

                generated_records.append(
                    {
                        "instruction": inst,
                        "answer": "",
                        "source_index": f"bootstrap_{request_idx}_{len(machine_instructions)}",
                        "raw_seed_task": None,
                        "generation_method": "self_instruct_bootstrap",
                        "original_instruction": inst,
                        "most_similar": most_similar,
                        "avg_similarity_score": float(mean(rouge_scores)) if rouge_scores else 0.0,
                        "metadata": metadata,
                        "request_idx": request_idx,
                        "bootstrap_prompt": metadata["prompt"],
                        "bootstrap_model_prompt": metadata["model_prompt"],
                        "bootstrap_next_index": metadata["next_index"],
                        "bootstrap_request_idx": request_idx,
                        "bootstrap_prompt_index": metadata["prompt_index"],
                        "bootstrap_seed_count": len(seed_instructions),
                        "bootstrap_requested_count": target_count,
                        "bootstrap_num_prompt_instructions": num_prompt_instructions,
                        "bootstrap_raw_generation": metadata["response"]["choices"][0]["text"],
                        "bootstrap_finish_reason": metadata["response"]["choices"][0].get("finish_reason"),
                        "bootstrap_stop_reason": metadata["response"].get("stop_reason"),
                        "bootstrap_candidate_count": len(candidates),
                        "bootstrap_avg_similarity_score": float(mean(rouge_scores)) if rouge_scores else 0.0,
                        "bootstrap_most_similar": json.dumps(most_similar, ensure_ascii=False),
                        "bootstrap_error": "",
                    }
                )
                machine_instructions.append(inst)
                all_instructions.append(inst)
                if len(machine_instructions) >= target_count:
                    break

            request_idx += 1
            if len(machine_instructions) >= target_count:
                break

    except Exception as exc:
        bootstrap_error = f"{type(exc).__name__}: {exc}"

    if generated_records:
        for record in generated_records:
            if not record.get("bootstrap_error"):
                record["bootstrap_error"] = bootstrap_error
        return generated_records

    if not use_stub_fallback:
        return []

    for record in stub_records:
        record["bootstrap_prompt"] = ""
        record["bootstrap_seed_count"] = len(seed_instructions)
        record["bootstrap_requested_count"] = target_count
        record["bootstrap_num_prompt_instructions"] = num_prompt_instructions if seed_instructions else 0
        record["bootstrap_raw_generation"] = ""
        record["bootstrap_finish_reason"] = None
        record["bootstrap_stop_reason"] = None
        record["bootstrap_candidate_count"] = 0
        record["bootstrap_avg_similarity_score"] = 0.0
        record["bootstrap_most_similar"] = "{}"
        record["bootstrap_error"] = bootstrap_error
        record["generation_method"] = "self_instruct_stub"
    return stub_records
