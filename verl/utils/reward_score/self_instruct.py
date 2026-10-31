"""Reference-similarity reward for generated Self-Instruct examples."""

from __future__ import annotations

import re
from collections import Counter


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s]")


def _answer_text(value: object) -> str:
    if isinstance(value, dict):
        value = value.get("ground_truth", "")
    return "" if value is None else str(value)


def _comparable_text(value: object) -> str:
    text = _answer_text(value).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    code_blocks = _CODE_BLOCK_RE.findall(text)
    if code_blocks:
        text = code_blocks[-1]
    return " ".join(text.split())


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _TOKEN_RE.findall(prediction.casefold())
    ref_tokens = _TOKEN_RE.findall(reference.casefold())
    if not pred_tokens or not ref_tokens:
        return 0.0

    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def compute_score(solution_str: str, ground_truth: object, **_: object) -> float:
    """Score a rollout against the retained Self-Instruct teacher response.

    This is an imitation reward, not benchmark correctness. Thinking text and
    Markdown code fences are ignored before multiset token F1 is calculated.
    """

    prediction = _comparable_text(solution_str)
    reference = _comparable_text(ground_truth)
    if not prediction or not reference:
        return 0.0
    if prediction == reference:
        return 1.0
    return _token_f1(prediction, reference)
