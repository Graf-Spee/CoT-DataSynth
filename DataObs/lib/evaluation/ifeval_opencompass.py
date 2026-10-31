"""OpenCompass-aligned IFEval scoring helpers.

OpenCompass reports four percentage metrics for IFEval:
Prompt-level-strict-accuracy, Inst-level-strict-accuracy,
Prompt-level-loose-accuracy, and Inst-level-loose-accuracy.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    from .ifeval_vendor import instructions_registry
except ImportError:  # Support direct script-style imports from this directory.
    from ifeval_vendor import instructions_registry


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int, list[str]]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def load_ifeval_references(path: str | Path) -> list[dict[str, Any]]:
    """Load OpenCompass/Google IFEval jsonl references."""
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"IFEval data not found: {source}")
    if source.suffix not in {".jsonl", ".json"}:
        raise ValueError(
            "IFEval data must be the OpenCompass-compatible JSONL file "
            f"(for example input_data.jsonl), got: {source}"
        )

    references: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            missing = {"key", "instruction_id_list", "prompt", "kwargs"} - set(row)
            if missing:
                raise ValueError(f"IFEval row {line_no} missing fields: {sorted(missing)}")
            references.append(row)
    return references


def _input_from_reference(reference: dict[str, Any]) -> InputExample:
    kwargs = [dict(item) for item in reference["kwargs"]]
    for kwarg in kwargs:
        for key in list(kwarg.keys()):
            if kwarg[key] is None:
                kwarg.pop(key, None)
    return InputExample(
        key=reference["key"],
        instruction_id_list=list(reference["instruction_id_list"]),
        prompt=reference["prompt"],
        kwargs=kwargs,
    )


def _get_instruction_cls(instruction_id: str):
    """Resolve IFEval instruction ids with or without the English `en:` prefix."""
    instruction_cls = instructions_registry.INSTRUCTION_DICT.get(instruction_id)
    if instruction_cls is not None:
        return instruction_cls
    if not instruction_id.startswith("en:"):
        instruction_cls = instructions_registry.INSTRUCTION_DICT.get(f"en:{instruction_id}")
        if instruction_cls is not None:
            return instruction_cls
    raise KeyError(instruction_id)


def _test_instruction_following_strict(inp: InputExample, response: str) -> OutputExample:
    is_following_list = []
    for index, instruction_id in enumerate(inp.instruction_id_list):
        instruction_cls = _get_instruction_cls(instruction_id)
        instruction = instruction_cls(instruction_id)
        instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)
        is_following_list.append(
            bool(isinstance(response, str) and response.strip() and instruction.check_following(response))
        )
    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def _test_instruction_following_loose(inp: InputExample, response: str) -> OutputExample:
    if isinstance(response, str):
        lines = response.split("\n")
        response_remove_first = "\n".join(lines[1:]).strip()
        response_remove_last = "\n".join(lines[:-1]).strip()
        response_remove_both = "\n".join(lines[1:-1]).strip()
        revised_response = response.replace("*", "")
        revised_response_remove_first = response_remove_first.replace("*", "")
        revised_response_remove_last = response_remove_last.replace("*", "")
        revised_response_remove_both = response_remove_both.replace("*", "")
        all_responses = [
            response,
            revised_response,
            response_remove_first,
            response_remove_last,
            response_remove_both,
            revised_response_remove_first,
            revised_response_remove_last,
            revised_response_remove_both,
        ]
    else:
        all_responses = []

    is_following_list = []
    for index, instruction_id in enumerate(inp.instruction_id_list):
        instruction_cls = _get_instruction_cls(instruction_id)
        instruction = instruction_cls(instruction_id)
        instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for candidate in all_responses:
            if candidate.strip() and instruction.check_following(candidate):
                is_following = True
                break
        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def score_ifeval_opencompass(
    predictions: list[str],
    references: list[dict[str, Any]],
    origin_prompt: list[str] | None = None,
) -> dict[str, Any]:
    """Score predictions using the same metric contract as OpenCompass."""
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions/references length mismatch: {len(predictions)} vs {len(references)}"
        )
    if origin_prompt is None:
        origin_prompt = [ref["prompt"] for ref in references]

    prompt_strict_correct = prompt_strict_total = 0
    inst_strict_correct = inst_strict_total = 0
    prompt_loose_correct = prompt_loose_total = 0
    inst_loose_correct = inst_loose_total = 0
    details: dict[str, Any] = {}

    for index, (pred, refer) in enumerate(zip(predictions, references)):
        inp = _input_from_reference(refer)

        strict_example = _test_instruction_following_strict(inp, pred)
        strict_follow_list = strict_example.follow_instruction_list
        prompt_strict_total += 1
        is_strict_correct = all(strict_follow_list)
        prompt_strict_correct += int(is_strict_correct)
        inst_strict_total += len(strict_example.instruction_id_list)
        inst_strict_correct += sum(strict_follow_list)

        loose_example = _test_instruction_following_loose(inp, pred)
        loose_follow_list = loose_example.follow_instruction_list
        prompt_loose_total += 1
        is_loose_correct = all(loose_follow_list)
        prompt_loose_correct += int(is_loose_correct)
        inst_loose_total += len(loose_example.instruction_id_list)
        inst_loose_correct += sum(loose_follow_list)

        if is_strict_correct:
            grade = "strict"
        elif is_loose_correct:
            grade = "loose"
        else:
            grade = "none"

        details[str(index)] = {
            "prompt": origin_prompt[index],
            "pred": pred,
            "refer": refer,
            "strict_follow_instruction_list": strict_follow_list,
            "loose_follow_instruction_list": loose_follow_list,
            "is_strict_correct": is_strict_correct,
            "is_loose_correct": is_loose_correct,
            "is_correct": is_strict_correct,
            "grade": grade,
        }

    return {
        "Prompt-level-strict-accuracy": prompt_strict_correct / prompt_strict_total * 100,
        "Inst-level-strict-accuracy": inst_strict_correct / inst_strict_total * 100,
        "Prompt-level-loose-accuracy": prompt_loose_correct / prompt_loose_total * 100,
        "Inst-level-loose-accuracy": inst_loose_correct / inst_loose_total * 100,
        "details": details,
    }
