from __future__ import annotations

from collections import Counter
import re

try:
    from .common import batched_generate_full, extract_boxed_answer, make_teacher, normalize_text
    from .templates import ANSWER_CONSISTENCY_SOLVE_TEMPLATE
except ImportError:
    from common import batched_generate_full, extract_boxed_answer, make_teacher, normalize_text
    from templates import ANSWER_CONSISTENCY_SOLVE_TEMPLATE


FINAL_ANSWER_LINE_RE = re.compile(
    r"(?:final answer|answer)\s*[:：]\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
GSM8K_FINAL_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
TRAILING_NUMERIC_RE = re.compile(
    r"(-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)(?![\s\S]*-?\d)",
)


def _fallback_answer(text: str) -> str:
    raw = str(text or "")
    normalized = normalize_text(raw)
    if not normalized:
        return ""
    gsm8k_matches = GSM8K_FINAL_RE.findall(raw)
    if gsm8k_matches:
        return normalize_text(gsm8k_matches[-1].replace(",", ""))
    final_answer_matches = FINAL_ANSWER_LINE_RE.findall(raw)
    if final_answer_matches:
        return normalize_text(final_answer_matches[-1])
    trailing_numeric = TRAILING_NUMERIC_RE.search(raw)
    if trailing_numeric:
        return normalize_text(trailing_numeric.group(1))
    if len(normalized) <= 32:
        return normalized
    return ""


def _extract_final_answer(text: str) -> str:
    boxed = extract_boxed_answer(text)
    if boxed:
        return normalize_text(boxed)
    return _fallback_answer(text)


def filter_answer_consistency(generated_items: list[dict], args: object) -> list[dict]:
    """Keep generated reasoning tasks whose resampled answers agree by majority."""
    if not generated_items:
        return []

    teacher = make_teacher(args)
    k = int(getattr(args, "consistency_k", 5))
    batch_size = int(getattr(args, "gen_batch_size", 1))
    min_vote_ratio = float(getattr(args, "consistency_min_vote_ratio", 0.5))
    prompts = [
        ANSWER_CONSISTENCY_SOLVE_TEMPLATE.format(question=str(item.get("question", "")).strip())
        for item in generated_items
    ]

    all_outputs: list[list[list[dict[str, object]]]] = []
    for sample_idx in range(k):
        sample_outputs = batched_generate_full(
            teacher,
            prompts,
            n=1,
            max_new_tokens=int(getattr(args, "consistency_max_new_tokens", 1024)),
            do_sample=bool(getattr(args, "consistency_do_sample", True)),
            temperature=float(getattr(args, "consistency_temperature", 0.7)),
            top_p=float(getattr(args, "consistency_top_p", 0.95)),
            gen_batch_size=batch_size,
            desc=f"Answer-consistency sample {sample_idx + 1}/{k}",
            enable_thinking=False,
            use_chat_template=True,
        )
        all_outputs.append(sample_outputs)

    filtered_items: list[dict] = []
    for item_idx, item in enumerate(generated_items):
        raw_responses: list[str] = []
        parsed_answers: list[str] = []
        for sample_outputs in all_outputs:
            output = sample_outputs[item_idx][0] if sample_outputs[item_idx] else {}
            text = str(output.get("text", "") or "")
            raw_responses.append(text)
            parsed_answers.append(_extract_final_answer(text))

        nonempty_answers = [ans for ans in parsed_answers if ans]
        if nonempty_answers:
            counts = Counter(nonempty_answers)
            majority_answer, majority_count = counts.most_common(1)[0]
            vote_ratio = majority_count / len(parsed_answers)
        else:
            majority_answer, majority_count, vote_ratio = "", 0, 0.0

        target_answer = normalize_text(str(item.get("target_answer", "")))
        consistency_passed = bool(
            majority_answer
            and target_answer
            and majority_answer == target_answer
            and vote_ratio >= min_vote_ratio
        )

        enriched = {
            **item,
            "consistency_prompt": prompts[item_idx],
            "consistency_raw_responses": raw_responses,
            "consistency_answers": parsed_answers,
            "consistency_majority_answer": majority_answer,
            "consistency_majority_count": majority_count,
            "consistency_vote_ratio": vote_ratio,
            "consistency_passed": consistency_passed,
            "teacher_filter_passed": consistency_passed,
        }
        if consistency_passed:
            filtered_items.append(enriched)
        else:
            filtered_items.append(enriched)

    return filtered_items
