from __future__ import annotations

import random
import re

try:
    from .common import (
        batched_generate_full,
        extract_answer,
        extract_boxed_answer,
        extract_instruction,
        make_teacher,
        normalize_text,
        strip_reasoning_block,
    )
    from .templates import INSTRUCTION_PROMPT_TEMPLATE, REASONING_PROMPT_TEMPLATE
except ImportError:
    from common import (
        batched_generate_full,
        extract_answer,
        extract_boxed_answer,
        extract_instruction,
        make_teacher,
        normalize_text,
        strip_reasoning_block,
    )
    from templates import INSTRUCTION_PROMPT_TEMPLATE, REASONING_PROMPT_TEMPLATE

QUESTION_BLOCK_RE = re.compile(
    r"\[New Question Begin\]\s*(.*?)\s*\[New Question End\]",
    re.DOTALL | re.IGNORECASE,
)
FINAL_ANSWER_BLOCK_RE = re.compile(
    r"\[Final Answer to New Question Begin\]\s*(.*?)\s*\[Final Answer to New Question End\]",
    re.DOTALL | re.IGNORECASE,
)
PROMPT_BLOCK_RE = re.compile(
    r"\[New Prompt Begin\]\s*(.*?)\s*\[New Prompt End\]",
    re.DOTALL | re.IGNORECASE,
)
FINAL_ANSWER_LINE_RE = re.compile(
    r"(?:final answer|answer)\s*[:：]\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
GSM8K_FINAL_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
TRAILING_NUMERIC_RE = re.compile(
    r"(-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)(?![\s\S]*-?\d)",
)


def _canonicalize_target_answer(text: str) -> str:
    raw = str(text or "").strip()
    boxed = extract_boxed_answer(raw)
    if boxed:
        return normalize_text(boxed)

    gsm8k_matches = GSM8K_FINAL_RE.findall(raw)
    if gsm8k_matches:
        return normalize_text(gsm8k_matches[-1].replace(",", ""))

    final_answer_matches = FINAL_ANSWER_LINE_RE.findall(raw)
    if final_answer_matches:
        return normalize_text(final_answer_matches[-1])

    trailing_numeric = TRAILING_NUMERIC_RE.search(raw)
    if trailing_numeric:
        return normalize_text(trailing_numeric.group(1))

    short_lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if short_lines and len(short_lines[-1]) <= 32:
        return normalize_text(short_lines[-1])
    return normalize_text(raw)


def _format_reasoning_examples(seed_tasks: list[dict], num_examples: int) -> str:
    pool = list(seed_tasks)
    random.shuffle(pool)
    examples = []
    for task in pool:
        question = extract_instruction(task)
        answer = extract_answer(task)
        if not question or not answer:
            continue
        examples.append(
            "\n".join(
                [
                    "[Question Begin]",
                    question.strip(),
                    "[Question End]",
                    "[Answer Begin]",
                    answer.strip(),
                    "[Answer End]",
                ]
            )
        )
        if len(examples) >= num_examples:
            break
    return "\n\n".join(examples)


def _format_instruction_examples(seed_tasks: list[dict], num_examples: int) -> str:
    pool = list(seed_tasks)
    random.shuffle(pool)
    examples = []
    for task in pool:
        instruction = extract_instruction(task)
        if not instruction:
            continue
        examples.append(
            "\n".join(
                [
                    "[Prompt Begin]",
                    instruction.strip(),
                    "[Prompt End]",
                ]
            )
        )
        if len(examples) >= num_examples:
            break
    return "\n\n".join(examples)


def _parse_reasoning_generation(raw_text: str) -> list[dict]:
    text = strip_reasoning_block(raw_text)
    question_matches = list(QUESTION_BLOCK_RE.finditer(text))
    answer_matches = list(FINAL_ANSWER_BLOCK_RE.finditer(text))
    pairs = min(len(question_matches), len(answer_matches))
    items = []
    for idx in range(pairs):
        qmatch = question_matches[idx]
        amatch = answer_matches[idx]
        question = normalize_text(qmatch.group(1))
        raw_target_answer = str(amatch.group(1) or "").strip()
        target_answer = _canonicalize_target_answer(raw_target_answer)
        reasoning_text = text[qmatch.end() : amatch.start()].strip()
        if not question or not target_answer:
            continue
        items.append(
            {
                "question": question,
                "target_answer": target_answer,
                "target_answer_raw": raw_target_answer,
                "reasoning_text": reasoning_text,
            }
        )
    return items


def _parse_instruction_generation(raw_text: str) -> list[str]:
    text = strip_reasoning_block(raw_text)
    prompts = []
    for match in PROMPT_BLOCK_RE.finditer(text):
        prompt = normalize_text(match.group(1))
        if prompt:
            prompts.append(prompt)
    return prompts


def generate_cot_tasks(seed_tasks: list[dict], args: object) -> list[dict]:
    """Generate reasoning-style synthetic tasks using the paper-style prompt."""
    if not seed_tasks:
        return []

    teacher = make_teacher(args)
    target_count = int(getattr(args, "cot_num_instructions", 100))
    num_fewshot = int(getattr(args, "cot_num_fewshot_examples", 8))
    num_prompts_per_request = int(getattr(args, "cot_num_prompts_per_request", 10))
    batch_size = int(getattr(args, "gen_batch_size", 1))
    generated_items: list[dict] = []

    while len(generated_items) < target_count:
        prompts: list[str] = []
        prompt_examples: list[str] = []
        for _ in range(batch_size):
            reasoning_examples = _format_reasoning_examples(seed_tasks, num_fewshot)
            prompts.append(REASONING_PROMPT_TEMPLATE.format(examples=reasoning_examples))
            prompt_examples.append(reasoning_examples)

        batch_outputs = batched_generate_full(
            teacher,
            prompts,
            n=1,
            max_new_tokens=int(getattr(args, "cot_max_new_tokens", 2048)),
            do_sample=bool(getattr(args, "cot_do_sample", True)),
            temperature=float(getattr(args, "cot_temperature", 0.7)),
            top_p=float(getattr(args, "cot_top_p", 0.95)),
            gen_batch_size=batch_size,
            desc="Generate CoT synthetic tasks",
            enable_thinking=False,
            use_chat_template=True,
        )

        for prompt, seed_examples, outputs in zip(prompts, prompt_examples, batch_outputs):
            first_output = outputs[0] if outputs else {}
            raw_text = str(first_output.get("text", "") or "")
            parsed_items = _parse_reasoning_generation(raw_text)
            for parsed in parsed_items:
                generated_items.append(
                    {
                        "source_index": len(generated_items),
                        "question": parsed["question"],
                        "target_answer": parsed["target_answer"],
                        "reasoning_text": parsed["reasoning_text"],
                        "cot_prompt": prompt,
                        "cot_seed_examples": seed_examples,
                        "cot_raw_generation": raw_text,
                        "cot_generation_metadata": first_output,
                        "generation_method": "cot_self_instruct_reasoning",
                        "augmentation_method": "cot_self_instruct",
                        "task_type": "reasoning",
                        "instruction_prompt_template": INSTRUCTION_PROMPT_TEMPLATE,
                    }
                )
                if len(generated_items) >= target_count:
                    break
            if len(generated_items) >= target_count:
                break

        if not batch_outputs or all(not outputs or not outputs[0].get("text") for outputs in batch_outputs):
            break

        if num_prompts_per_request <= 0:
            break

    return generated_items[:target_count]
