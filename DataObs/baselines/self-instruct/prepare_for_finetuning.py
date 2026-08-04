from __future__ import annotations

import random
import re

random.seed(123)

INPUT_OUTPUT_SPLIT = re.compile(r"Output\s*\d*\s*:", re.IGNORECASE)
INPUT_PREFIX = re.compile(r"^Input\s*\d*\s*:", re.IGNORECASE)
INPUT_MARKER = re.compile(r"Input\s*\d*\s*:", re.IGNORECASE)
EXAMPLE_SPLIT = re.compile(r"Example\s?\d*\.?", re.IGNORECASE)
CLASS_LABEL_SPLIT = re.compile(r"Class label:", re.IGNORECASE)


def encode_instance(instruction: str, inst_input: str, inst_output: str) -> tuple[str, str]:
    instruction = instruction.strip()
    inst_input = inst_input.strip()
    inst_output = inst_output.strip()
    encoding_templates_w_input = [
        ("{instruction}\nInput: {input}\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n\nInput: {input}\n\nOutput:", " {output}<|endoftext|>"),
        ("Task: {instruction}\nInput: {input}\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n\n{input}\n\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n\n{input}\n\n", "{output}<|endoftext|>"),
        ("{instruction}\n{input}\n\n", "{output}<|endoftext|>"),
        ("Task: {instruction}\n\n{input}\n\n", "{output}<|endoftext|>"),
    ]
    encoding_templates_wo_input = [
        ("{instruction} Output:", " {output}<|endoftext|>"),
        ("{instruction}\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n", "{output}<|endoftext|>"),
        ("{instruction}\n\n", "{output}<|endoftext|>"),
        ("Task: {instruction}\n\n", "{output}<|endoftext|>"),
    ]
    if inst_input:
        prompt_template, completion_template = random.choice(encoding_templates_w_input)
        prompt = prompt_template.format(instruction=instruction, input=inst_input)
        completion = completion_template.format(output=inst_output)
    else:
        prompt_template, completion_template = random.choice(encoding_templates_wo_input)
        prompt = prompt_template.format(instruction=instruction)
        completion = completion_template.format(output=inst_output)
    return prompt, completion


def parse_input_output(response_text: str) -> tuple[str, str]:
    if INPUT_OUTPUT_SPLIT.findall(response_text):
        parts = INPUT_OUTPUT_SPLIT.split(response_text, maxsplit=1)
        inst_input = parts[0].strip()
        inst_output = parts[1].strip() if len(parts) > 1 else ""
    else:
        inst_input = ""
        inst_output = response_text.strip()

    if INPUT_MARKER.findall(inst_output):
        inst_output = INPUT_MARKER.split(inst_output, maxsplit=1)[0].strip()

    inst_input = INPUT_PREFIX.sub("", inst_input).strip()
    return inst_input, inst_output


def filter_duplicate_instances(instances: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    same_input_diff_output = False
    for i in range(1, len(instances)):
        for j in range(i):
            if not instances[i][1]:
                continue
            if instances[i][1] == instances[j][1] and instances[i][2] != instances[j][2]:
                same_input_diff_output = True
                break
        if same_input_diff_output:
            break
    if same_input_diff_output:
        return []
    return list(set(instances))


def filter_invalid_instances(instances: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    filtered = []
    for instruction, inst_input, inst_output in instances:
        if inst_input == inst_output:
            continue
        if not inst_output:
            continue
        if inst_input.strip().endswith(":") or inst_output.strip().endswith(":"):
            continue
        filtered.append((instruction, inst_input, inst_output))
    return filtered


def get_finish_reason(response_metadata: object) -> str | None:
    if isinstance(response_metadata, dict):
        finish_reason = response_metadata.get("finish_reason")
        if finish_reason is not None:
            return str(finish_reason)
        response = response_metadata.get("response")
        if isinstance(response, dict):
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                choice = choices[0]
                if isinstance(choice, dict) and choice.get("finish_reason") is not None:
                    return str(choice["finish_reason"])
    return None


def normalize_local_generation_for_official_parser(raw_text: str, *, is_classification: bool) -> str:
    text = str(raw_text or "").strip()
    if not text or is_classification:
        return text
    lowered = text.lower()
    if lowered.startswith("output:") or lowered.startswith("example 1"):
        return text
    # Local chat models may answer directly instead of continuing the completion.
    return f"Output: {text}"


def parse_instances_for_generation_task(
    raw_text: str,
    instruction: str,
    response_metadata: object,
) -> list[tuple[str, str, str]]:
    instances: list[tuple[str, str, str]] = []
    text = str(raw_text or "").strip()
    if not text:
        return []

    if EXAMPLE_SPLIT.findall(text):
        parts = [part.strip() for part in EXAMPLE_SPLIT.split(text) if part.strip()]
        for part in parts:
            inst_input, inst_output = parse_input_output(part)
            instances.append((instruction.strip(), inst_input.strip(), inst_output.strip()))
    elif INPUT_OUTPUT_SPLIT.findall(text):
        inst_input, inst_output = parse_input_output(text)
        instances.append((instruction.strip(), inst_input.strip(), inst_output.strip()))
    else:
        return []

    if get_finish_reason(response_metadata) == "length":
        instances = instances[:-1]

    instances = filter_invalid_instances(instances)
    instances = filter_duplicate_instances(instances)
    return instances


def parse_instances_for_classification_task(
    raw_text: str,
    instruction: str,
    response_metadata: object,
) -> list[tuple[str, str, str]]:
    text = str(raw_text or "").strip()
    if not text:
        return []

    instances: list[tuple[str, str, str]] = []
    if "Class label:" in text:
        parts = CLASS_LABEL_SPLIT.split(text)[1:]
        for part in parts:
            fields = part.strip().split("\n", 1)
            if len(fields) == 2:
                class_label = fields[0].strip()
                inst_input = fields[1].strip()
            else:
                class_label = fields[0].strip()
                inst_input = ""
            instances.append((instruction.strip(), inst_input, class_label))
    else:
        return []

    if get_finish_reason(response_metadata) == "length":
        instances = instances[:-1]

    instances = filter_invalid_instances(instances)
    instances = filter_duplicate_instances(instances)
    return instances


def preprocess_instance_input(inst_input: str) -> str:
    processed_input = inst_input
    if random.random() < 0.5:
        colon_words = re.findall(r"(\w+):", processed_input)
        if len(set(colon_words)) == 1:
            processed_input = processed_input.split(":", 1)[1].strip()
        else:
            processed_input = processed_input.strip()
        processed_input = processed_input.replace("\n\n", "\n")
    return processed_input.strip()


def build_training_question(instruction: str, inst_input: str) -> str:
    instruction = instruction.strip()
    inst_input = inst_input.strip()
    if not inst_input:
        return instruction
    return f"{instruction}\n\nInput: {inst_input}"


def prepare_for_finetuning(generated_items: list[dict], dataset: str) -> list[dict]:
    """Convert generated instruction instances into cleaned DataObs SFT rows.

    This stage now follows the official Self-Instruct prepare step more closely:
    parse raw generated instances, filter invalid/duplicate pairs, and then
    reformat retained examples into training rows.
    """
    parsed_training_instances: list[dict] = []
    seen_prompt_completion: set[tuple[str, str]] = set()

    for item in generated_items:
        instruction = str(item.get("instruction", "")).strip()
        if not instruction:
            continue

        raw_text = str(item.get("instance_raw_generation", "") or "").strip()
        raw_instances = str(item.get("raw_instances", raw_text) or "").strip()
        is_classification = bool(item.get("is_classification", False))
        instance_metadata = item.get("instance_metadata", {})
        parser_raw_instances = normalize_local_generation_for_official_parser(
            raw_instances,
            is_classification=is_classification,
        )

        if is_classification:
            parsed_instances = parse_instances_for_classification_task(
                parser_raw_instances,
                instruction,
                instance_metadata,
            )
            parser_name = "classification"
        else:
            parsed_instances = parse_instances_for_generation_task(
                parser_raw_instances,
                instruction,
                instance_metadata,
            )
            parser_name = "generation"

        if not parsed_instances:
            continue

        sampled_instances = random.sample(parsed_instances, min(len(parsed_instances), 5))
        parsed_count = len(sampled_instances)

        for instance_index, (_, inst_input, inst_output) in enumerate(sampled_instances):
            parsed_training_instances.append(
                {
                    "instruction": instruction,
                    "instance_input": inst_input.strip(),
                    "instance_output": inst_output.strip(),
                    "is_classification": is_classification,
                    "prepare_parser": parser_name,
                    "parsed_instance_index": instance_index,
                    "parsed_instance_count": parsed_count,
                    "item": item,
                }
            )

    rows = []
    for parsed in parsed_training_instances:
        item = parsed["item"]
        instruction = str(parsed["instruction"]).strip()
        processed_input = preprocess_instance_input(str(parsed["instance_input"]))
        inst_output = str(parsed["instance_output"]).strip()
        question = build_training_question(instruction, processed_input)
        prompt, completion = encode_instance(instruction, processed_input, inst_output)
        prompt_completion = (prompt, completion)
        if prompt_completion in seen_prompt_completion:
            continue
        seen_prompt_completion.add(prompt_completion)

        rows.append(
            {
                "question": question,
                "answer": inst_output,
                "gold_answer": inst_output,
                "prompt": prompt,
                "completion": completion,
                "instruction": instruction,
                "instance_input": processed_input,
                "instance_output": inst_output,
                "dataset": dataset,
                "data_source": "self_instruct",
                "source_index": item.get("source_index"),
                "augmentation_method": "self_instruct",
                "generation_method": item.get("generation_method", "self_instruct_stub"),
                "instance_generation_method": item.get("instance_generation_method"),
                "original_instruction": item.get("original_instruction", instruction),
                "teacher_filter_passed": True,
                "is_classification": bool(parsed["is_classification"]),
                "prepare_parser": parsed["prepare_parser"],
                "parsed_instance_index": parsed["parsed_instance_index"],
                "parsed_instance_count": parsed["parsed_instance_count"],
                "instance_prompt": item.get("instance_prompt"),
                "instance_raw_generation": item.get("instance_raw_generation"),
                "raw_instances": item.get("raw_instances"),
                "instance_prompt_style": item.get("instance_prompt_style"),
                "instance_template_type": item.get("instance_template_type"),
                "instance_error": item.get("instance_error"),
                "classification_detection_method": item.get("classification_detection_method"),
                "classification_detection_reason": item.get("classification_detection_reason"),
                "bootstrap_prompt": item.get("bootstrap_prompt"),
                "bootstrap_seed_count": item.get("bootstrap_seed_count"),
                "bootstrap_requested_count": item.get("bootstrap_requested_count"),
                "bootstrap_raw_generation": item.get("bootstrap_raw_generation"),
                "bootstrap_candidate_count": item.get("bootstrap_candidate_count"),
                "bootstrap_error": item.get("bootstrap_error"),
            }
        )

    random.shuffle(rows)
    return rows
