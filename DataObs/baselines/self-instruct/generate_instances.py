from __future__ import annotations

import re

try:
    from .common import batched_generate_full, make_teacher
    from .templates import (
        INPUT_FIRST_TEMPLATE_FOR_GEN,
        INSTANCE_CLF_QWEN_CHAT_TEMPLATE,
        INSTANCE_GEN_QWEN_CHAT_TEMPLATE,
        OUTPUT_FIRST_TEMPLATE_FOR_CLF,
    )
except ImportError:
    from common import batched_generate_full, make_teacher
    from templates import (
        INPUT_FIRST_TEMPLATE_FOR_GEN,
        INSTANCE_CLF_QWEN_CHAT_TEMPLATE,
        INSTANCE_GEN_QWEN_CHAT_TEMPLATE,
        OUTPUT_FIRST_TEMPLATE_FOR_CLF,
    )


QUOTED_LABEL_PATTERN = re.compile(r'"([^"]+)"')
OR_LABEL_PATTERN = re.compile(
    r"\b(?:as|into|with|answer|output)\s+([A-Za-z][A-Za-z0-9 _-]{0,40}?)\s+or\s+([A-Za-z][A-Za-z0-9 _-]{0,40}?)(?:[.,;\n]|$)",
    re.IGNORECASE,
)
SELECT_LABEL_PATTERN = re.compile(
    r"\bselect\s+((?:[A-Za-z0-9][A-Za-z0-9 _-]*,\s*)+[A-Za-z0-9][A-Za-z0-9 _-]*\s*,?\s*or\s+[A-Za-z0-9][A-Za-z0-9 _-]*)(?:[.,;\n]|$)",
    re.IGNORECASE,
)


def normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", str(label).strip().strip("\"'`"))


def dedupe_labels(labels: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for label in labels:
        normalized = normalize_label(label)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def extract_allowed_labels(instruction: str) -> list[str]:
    text = instruction.strip()
    quoted = dedupe_labels(QUOTED_LABEL_PATTERN.findall(text))
    if len(quoted) >= 2:
        return quoted

    match = OR_LABEL_PATTERN.search(text)
    if match:
        return dedupe_labels([match.group(1), match.group(2)])

    match = SELECT_LABEL_PATTERN.search(text)
    if match:
        chunk = match.group(1).replace(" or ", ", ")
        return dedupe_labels([part.strip() for part in chunk.split(",")])

    lowered = text.lower()
    if "yes or no" in lowered:
        return ["Yes", "No"]
    if "true or false" in lowered:
        return ["True", "False"]
    return []


def build_instance_prompt(instruction: str, is_classification: bool) -> str:
    if is_classification:
        allowed_labels = extract_allowed_labels(instruction)
        allowed_labels_block = ""
        if allowed_labels:
            allowed_labels_block = (
                "Allowed labels for the final task: "
                + ", ".join(allowed_labels)
                + "\nFor the final task, every `Class label:` must be exactly one of the allowed labels above."
            )
        return INSTANCE_CLF_QWEN_CHAT_TEMPLATE.format(
            allowed_labels_block=allowed_labels_block,
            template=OUTPUT_FIRST_TEMPLATE_FOR_CLF,
            instruction=instruction.strip(),
        )
    return INSTANCE_GEN_QWEN_CHAT_TEMPLATE.format(
        template=INPUT_FIRST_TEMPLATE_FOR_GEN,
        instruction=instruction.strip(),
    )

def generate_instances(instruction_items: list[dict], args: object | None = None) -> list[dict]:
    """Generate raw instances following the official Self-Instruct step 3 structure."""
    if not instruction_items:
        return []

    prompts = [
        build_instance_prompt(
            str(item.get("instruction", "")).strip(),
            bool(item.get("is_classification", False)),
        )
        for item in instruction_items
    ]

    raw_generations = [""] * len(instruction_items)
    raw_metadata: list[dict[str, object]] = [{} for _ in instruction_items]
    generation_error = ""
    if args is not None and getattr(args, "model_id", ""):
        try:
            teacher = make_teacher(args)
            stop_sequences = [
                f"Example {getattr(args, 'instance_max_instances_to_generate', 5) + 1}",
                "Task:",
            ]
            batch_size = getattr(args, "instance_request_batch_size", getattr(args, "gen_batch_size", 5))
            for start in range(0, len(instruction_items), batch_size):
                batch_items = instruction_items[start : start + batch_size]
                batch_prompts = prompts[start : start + batch_size]
                batch_has_classification = any(bool(item.get("is_classification", False)) for item in batch_items)
                batch_outputs = batched_generate_full(
                    teacher,
                    batch_prompts,
                    n=1,
                    # The official script uses 300 if the request batch contains any
                    # classification task, otherwise 350.
                    max_new_tokens=(
                        getattr(args, "instance_max_new_tokens_clf", 300)
                        if batch_has_classification
                        else getattr(args, "instance_max_new_tokens_gen", 350)
                    ),
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    gen_batch_size=batch_size,
                    desc="Generate raw instances",
                    enable_thinking=False,
                    use_chat_template=True,
                    stop_sequences=stop_sequences,
                    frequency_penalty=0.0,
                    presence_penalty=1.5,
                )
                for absolute_idx, output_batch in enumerate(batch_outputs, start=start):
                    first_output = output_batch[0] if output_batch else {}
                    raw_generations[absolute_idx] = str(first_output.get("text", "") or "")
                    raw_metadata[absolute_idx] = first_output
        except Exception as exc:
            generation_error = f"{type(exc).__name__}: {exc}"

    generated = []
    for item, prompt, raw_text, metadata in zip(instruction_items, prompts, raw_generations, raw_metadata):
        instruction_metadata = {
            key: value
            for key, value in item.items()
            if key not in {"instruction", "answer", "raw_seed_task"}
        }
        generation_method = "self_instruct_instance_teacher" if not generation_error else "self_instruct_instance_error"
        generated.append(
            {
                **item,
                "instance_prompt": prompt,
                "instance_raw_generation": raw_text,
                "raw_instances": raw_text,
                "instance_metadata": metadata,
                "instruction_metadata": instruction_metadata,
                "most_similar": item.get("bootstrap_most_similar", "{}"),
                "avg_similarity_score": item.get("bootstrap_avg_similarity_score", 0.0),
                "request_idx": item.get("bootstrap_request_idx"),
                "instance_generation_method": generation_method,
                "instance_template_type": (
                    "official_output_first_clf"
                    if bool(item.get("is_classification", False))
                    else "official_input_first_gen"
                ),
                "instance_prompt_style": "official_template_qwen_chat_wrapper",
                "instance_finish_reason": metadata.get("finish_reason"),
                "instance_stop_reason": metadata.get("stop_reason"),
                "instance_error": generation_error,
            }
        )
    return generated
