from __future__ import annotations

import re

try:
    from .common import batched_generate_full, make_teacher
    from .templates import CLASSIFICATION_QWEN_CHAT_TEMPLATE, CLASSIFICATION_TEMPLATE_1
except ImportError:
    from common import batched_generate_full, make_teacher
    from templates import CLASSIFICATION_QWEN_CHAT_TEMPLATE, CLASSIFICATION_TEMPLATE_1


THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
MCQ_OPTION_LINE = re.compile(r"^\s*(?:[A-Da-d][\).]|[1-4][\).])\s+.+$", re.MULTILINE)
YES_NO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
OFFICIAL_CLASSIFICATION_STOP_SEQUENCES = ["\n"]

CLASSIFICATION_HINTS = (
    "which of the following",
    "select the correct option",
    "choose the correct option",
    "choose one of the following",
    "multiple choice",
    "yes or no",
    "true or false",
    "is this statement",
    "classify the",
    "determine whether",
    "label the",
    "sentiment of",
    "topic of",
    "category of",
)


def build_classification_prompt(instruction: str) -> str:
    return CLASSIFICATION_QWEN_CHAT_TEMPLATE.format(
        template=CLASSIFICATION_TEMPLATE_1,
        instruction=instruction.strip(),
    )


def detect_classification(instruction: str) -> tuple[bool, str]:
    text = instruction.strip()
    lowered = text.lower()

    option_lines = MCQ_OPTION_LINE.findall(text)
    if len(option_lines) >= 2:
        return True, "multiple_choice_options"

    for hint in CLASSIFICATION_HINTS:
        if hint in lowered:
            return True, f"hint:{hint}"

    return False, "default_non_classification"


def parse_classification_response(text: str) -> tuple[bool | None, str]:
    cleaned = THINK_BLOCK.sub("", str(text or "")).strip()
    match = YES_NO_PATTERN.match(cleaned)
    if not match:
        match = YES_NO_PATTERN.search(cleaned)
    if not match:
        return None, "unparseable_model_response"
    label = match.group(1).strip().lower()
    return label == "yes", f"model:{label}"


def identify_clf_or_not(instruction_items: list[dict], args: object | None = None) -> list[dict]:
    """Classify tasks using the official Self-Instruct template when possible."""
    if not instruction_items:
        return []

    allow_heuristic_fallback = bool(getattr(args, "classification_allow_heuristic_fallback", False))
    prompts = [build_classification_prompt(str(item.get("instruction", "")).strip()) for item in instruction_items]
    raw_generations = [""] * len(instruction_items)
    generation_error = ""
    if args is not None and getattr(args, "model_id", ""):
        try:
            teacher = make_teacher(args)
            outputs = batched_generate_full(
                teacher,
                prompts,
                n=1,
                max_new_tokens=getattr(args, "classification_max_new_tokens", 3),
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                gen_batch_size=getattr(args, "classification_request_batch_size", getattr(args, "gen_batch_size", 5)),
                desc="Identify classification tasks",
                enable_thinking=False,
                stop_sequences=OFFICIAL_CLASSIFICATION_STOP_SEQUENCES,
            )
            raw_generations = [str(batch[0].get("text", "") or "") if batch else "" for batch in outputs]
            raw_metadata = [batch[0] if batch else {} for batch in outputs]
        except Exception as exc:
            generation_error = f"{type(exc).__name__}: {exc}"
            raw_metadata = [{} for _ in instruction_items]
    else:
        raw_metadata = [{} for _ in instruction_items]

    annotated = []
    for item, prompt, raw_text, metadata in zip(instruction_items, prompts, raw_generations, raw_metadata):
        instruction = str(item.get("instruction", "")).strip()
        model_decision, reason = parse_classification_response(raw_text)
        if model_decision is None:
            if allow_heuristic_fallback:
                heuristic_decision, heuristic_reason = detect_classification(instruction)
                is_classification = heuristic_decision
                detection_method = "heuristic_fallback" if raw_text or generation_error else "heuristic"
                detection_reason = generation_error or heuristic_reason if generation_error else f"{reason}|{heuristic_reason}"
            else:
                is_classification = False
                detection_method = "official_template_unparsed"
                detection_reason = generation_error or reason
        else:
            is_classification = model_decision
            detection_method = "official_template_vllm"
            detection_reason = reason
        annotated.append(
            {
                **item,
                "is_classification": is_classification,
                "classification_prompt": prompt,
                "classification_prompt_style": "official_template_qwen_chat_wrapper",
                "classification_raw_generation": raw_text,
                "classification_finish_reason": metadata.get("finish_reason"),
                "classification_stop_reason": metadata.get("stop_reason"),
                "classification_error": generation_error,
                "classification_detection_method": detection_method,
                "classification_detection_reason": detection_reason,
            }
        )
    return annotated
