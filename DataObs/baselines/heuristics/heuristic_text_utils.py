# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small RevThink text helpers used by the heuristic augmentation scripts."""

from __future__ import annotations

import re


_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_ANY_THINK_TAG = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)


def get_true_false(text: str) -> str:
    """Return the last true/false token in generated text, or N/A if absent."""
    if not text:
        return "N/A"
    match = re.findall(r"(true|false)", text, re.IGNORECASE)
    return match[-1].lower() if match else "N/A"


def strip_thinking_blocks(text: str) -> str:
    """Drop complete Qwen-style thinking blocks and any remaining think tags."""
    if not text:
        return ""
    text = _THINK_BLOCK.sub("", str(text))
    return _ANY_THINK_TAG.sub("", text).strip()


def strip_output_prefix(text: str) -> str:
    """Extract the last labeled model output field without retaining later sections."""
    text = strip_thinking_blocks(text)
    if not text:
        return ""

    labels = (
        "OUTPUT",
        "FINAL CREATED QUESTION",
        "FINAL QUESTION",
        "REPHRASED QUESTION",
        "INVERSE QUESTION",
        "BACKWARD QUESTION",
    )
    label_pattern = "|".join(re.escape(label) for label in labels)
    label_re = re.compile(r"(?im)^\s*(?:\*\*)?(?:" + label_pattern + r")\s*:\s*(?:\*\*)?\s*")
    matches = list(label_re.finditer(text))
    if matches:
        text = text[matches[-1].end() :]

    stop_re = re.compile(
        r"(?im)^\s*(?:INPUT|EXPLANATION|RATIONALE|SOLUTION|VERIFICATION(?: AND MODIFICATION)?|FINAL ANSWER)\s*:\s*"
    )
    text = stop_re.split(text, maxsplit=1)[0]
    return text.strip()


def _strip_wrapping_quotes(text: str) -> str:
    text = text.strip().strip("-: ").strip()
    while len(text) > 1 and text[0] in "\"'`“‘" and text[-1] in "\"'`”’":
        text = text[1:-1].strip()
    return text


def _select_last_question_block(text: str) -> str:
    """Prefer the last paragraph/line that looks like the requested question."""
    text = text.strip()
    if not text:
        return ""

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    question_blocks = [(idx, p) for idx, p in enumerate(paragraphs) if "?" in p]
    if question_blocks:
        idx, block = question_blocks[-1]
        option_blocks = []
        for paragraph in paragraphs[idx + 1 :]:
            if re.match(r"^\s*(?:\([A-Z]\)|[A-Z][\.\)])\s+", paragraph):
                option_blocks.append(paragraph)
                continue
            break
        return "\n".join([block, *option_blocks]).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    question_lines = [line for line in lines if "?" in line]
    if question_lines:
        return question_lines[-1].strip()
    return lines[-1].strip() if lines else text


def _drop_answer_leakage(text: str) -> str:
    """Remove trailing answer annotations from generated questions."""
    text = re.split(
        r"(?i)\s+(?:the\s+correct\s+answer\s+is|the\s+answer\s+is|final\s+answer)\s*:?\s+|(?:^|\s)answer\s*:\s*",
        text,
        maxsplit=1,
    )[0]
    return text.strip()


def remove_backward_answer(text: str) -> str:
    """Remove trailing MCQ, math, and yes/no answer annotations from a backward question."""
    text = strip_thinking_blocks(text)
    if not text:
        return ""
    return remove_backward_answer_annotation(text)


def remove_backward_answer_annotation(text: str) -> str:
    """Remove trailing backward-question answer annotations while preserving reasoning text."""
    text = "" if text is None else str(text).strip()
    if not text:
        return ""
    answer_value = (
        r"(?:"
        r"\([A-Z]\)|[A-Z]|"
        r"yes|no|true|false|"
        r"\\boxed\{[^{}]*\}|"
        r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?"
        r")"
    )
    trailing_patterns = [
        rf"\s*(?:The\s+correct\s+answer\s+is|The\s+answer\s+is|Answer)\s*:?\s*{answer_value}\s*[\.\)]*\s*$",
        rf"\s*\(?\s*(?:correct\s+answer|answer)\s*[:=]\s*{answer_value}\s*\)?\s*[\.\)]*\s*$",
    ]
    for pattern in trailing_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    return _strip_wrapping_quotes(text)


def clean_generated_question(text: str) -> str:
    """Return a reusable generated question without thinking, labels, or answer leakage."""
    raw_text = "" if text is None else str(text)
    has_unclosed_think = bool(re.search(r"<think\b", raw_text, re.IGNORECASE)) and not bool(
        re.search(r"</think>", raw_text, re.IGNORECASE)
    )
    if has_unclosed_think:
        return ""
    text = strip_output_prefix(text)
    text = _drop_answer_leakage(text)
    text = remove_backward_answer(text)
    text = _select_last_question_block(text)
    text = _drop_answer_leakage(text)
    text = remove_backward_answer(text)
    return _strip_wrapping_quotes(text)
