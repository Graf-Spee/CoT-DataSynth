from __future__ import annotations

import random
import re

try:
    from .common import batched_generate_full, extract_instruction, make_teacher, normalize_text, strip_reasoning_block
    from .templates import INSTRUCTION_PROMPT_TEMPLATE
except ImportError:
    from common import batched_generate_full, extract_instruction, make_teacher, normalize_text, strip_reasoning_block
    from templates import INSTRUCTION_PROMPT_TEMPLATE


PROMPT_BLOCK_RE = re.compile(
    r"\[New Prompt Begin\]\s*(.*?)\s*\[New Prompt End\]",
    re.DOTALL | re.IGNORECASE,
)
QUESTION_STYLE_RE = re.compile(
    r"^(what is|what's|how many|how much|calculate|compute|solve|find the|if a|if an|determine)\b",
    re.IGNORECASE,
)
MATHISH_RE = re.compile(
    r"(\b\d+\b.*\b\d+\b)|(\bmeters?\b)|(\bkilometers?\b)|(\bkm/?h\b)|(\bperimeter\b)|(\bdistance\b)",
    re.IGNORECASE,
)
INSTRUCTION_VERB_RE = re.compile(
    r"^(rewrite|summarize|extract|classify|translate|format|list|organize|convert|paraphrase|generate|write|turn|return|respond|create|rephrase)\b",
    re.IGNORECASE,
)


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


def _parse_instruction_generation(raw_text: str) -> list[str]:
    text = strip_reasoning_block(raw_text)
    prompts: list[str] = []
    for match in PROMPT_BLOCK_RE.finditer(text):
        prompt = normalize_text(match.group(1))
        if prompt:
            prompts.append(prompt)
    return prompts


def _is_instruction_like(prompt: str) -> bool:
    text = normalize_text(prompt)
    if not text:
        return False
    lowered = text.lower()
    if QUESTION_STYLE_RE.search(text):
        return False
    if text.endswith("?") and not INSTRUCTION_VERB_RE.search(text):
        return False
    if MATHISH_RE.search(text) and not INSTRUCTION_VERB_RE.search(text):
        return False
    if any(
        phrase in lowered
        for phrase in (
            "step by step",
            "show your work",
            "what is the answer",
            "find the answer",
            "solve the problem",
            "calculate the",
        )
    ):
        return False
    return True


def generate_instruction_tasks(seed_tasks: list[dict], args: object) -> list[dict]:
    if not seed_tasks:
        return []

    teacher = make_teacher(args)
    target_count = int(getattr(args, "instruction_num_instructions", getattr(args, "cot_num_instructions", 100)))
    num_fewshot = int(getattr(args, "instruction_num_fewshot_examples", 8))
    batch_size = int(getattr(args, "gen_batch_size", 1))
    generated_items: list[dict] = []

    while len(generated_items) < target_count:
        prompts: list[str] = []
        prompt_examples: list[str] = []
        for _ in range(batch_size):
            instruction_examples = _format_instruction_examples(seed_tasks, num_fewshot)
            prompts.append(INSTRUCTION_PROMPT_TEMPLATE.format(examples=instruction_examples))
            prompt_examples.append(instruction_examples)

        batch_outputs = batched_generate_full(
            teacher,
            prompts,
            n=1,
            max_new_tokens=int(getattr(args, "instruction_max_new_tokens", 1024)),
            do_sample=bool(getattr(args, "instruction_do_sample", True)),
            temperature=float(getattr(args, "instruction_temperature", 0.7)),
            top_p=float(getattr(args, "instruction_top_p", 0.95)),
            gen_batch_size=batch_size,
            desc="Generate instruction synthetic tasks",
            enable_thinking=False,
            use_chat_template=True,
        )

        for prompt, seed_examples, outputs in zip(prompts, prompt_examples, batch_outputs):
            first_output = outputs[0] if outputs else {}
            raw_text = str(first_output.get("text", "") or "")
            parsed_prompts = _parse_instruction_generation(raw_text)
            for parsed_prompt in parsed_prompts:
                if not _is_instruction_like(parsed_prompt):
                    continue
                generated_items.append(
                    {
                        "source_index": len(generated_items),
                        "question": parsed_prompt,
                        "instruction": parsed_prompt,
                        "target_answer": "",
                        "reasoning_text": "",
                        "cot_prompt": prompt,
                        "cot_seed_examples": seed_examples,
                        "cot_raw_generation": raw_text,
                        "cot_generation_metadata": first_output,
                        "generation_method": "cot_self_instruct_instruction",
                        "augmentation_method": "cot_self_instruct",
                        "task_type": "instruction",
                    }
                )
                if len(generated_items) >= target_count:
                    break
            if len(generated_items) >= target_count:
                break

        if not batch_outputs or all(not outputs or not outputs[0].get("text") for outputs in batch_outputs):
            break

    return generated_items[:target_count]
