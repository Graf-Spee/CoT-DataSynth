#!/usr/bin/env python3
"""Reverse Thinking augmentation adapted from the RevThink baseline code."""

from __future__ import annotations

import argparse
import gc
import re
import time
from typing import Any, Dict, List

try:
    from ._common import (
        add_common_args,
        base_summary,
        batched_generate,
        build_reasoning_prompt,
        consistency_prompt_template,
        extract_reward_prediction,
        get_true_false,
        icl_samples,
        load_input_rows,
        make_teacher,
        prompt_for_backward_question,
        remove_backward_answer,
        require_backward_task,
        score_answer,
        validate_common_args,
        write_outputs,
    )
except ImportError:  # direct script execution
    from _common import (
        add_common_args,
        base_summary,
        batched_generate,
        build_reasoning_prompt,
        consistency_prompt_template,
        extract_reward_prediction,
        get_true_false,
        icl_samples,
        load_input_rows,
        make_teacher,
        prompt_for_backward_question,
        remove_backward_answer,
        require_backward_task,
        score_answer,
        validate_common_args,
        write_outputs,
    )


def _strip_output_prefix(text: str) -> str:
    """Remove an optional OUTPUT: prefix from generated backward questions."""
    match = re.search(r"OUTPUT:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def _bq_training_question(question: str, gold_answer: Any) -> str:
    """Format the backward-question generation task as a training row prompt."""
    return (
        "Generate the inverse question based on the following seed question and its answer:\n"
        f"### Seed Question: {question} The correct answer is ({gold_answer})."
    )


def _consistency_gold_answer(dataset: str, gold_answer: Any) -> Any:
    """Format StrategyQA boolean labels as yes/no for the baseline consistency prompt."""
    if dataset != "strategyqa":
        return gold_answer
    if isinstance(gold_answer, bool):
        return "yes" if gold_answer else "no"
    gold_text = str(gold_answer).strip().lower()
    if gold_text in {"true", "1"}:
        return "yes"
    if gold_text in {"false", "0"}:
        return "no"
    return gold_answer


def run(args: argparse.Namespace) -> None:
    """Run RevThink-style backward question, reasoning, and consistency filtering."""
    start_time = time.time()
    rows = load_input_rows(args.input_file, args.dataset, args.smoke_num_rows)
    task = require_backward_task(args.dataset)
    teacher = make_teacher(args)

    # Stage 1: generate inverse/backward questions from the original question and answer.
    backward_question_prompts = [
        prompt_for_backward_question.format(
            icl_samples=icl_samples[task],
            input_question=f"INPUT: {row['question']} The correct answer is {row['gold_answer']}.",
        )
        for row in rows
    ]
    raw_backward_questions = batched_generate(
        teacher,
        backward_question_prompts,
        n=1,
        max_new_tokens=args.backward_question_max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        gen_batch_size=args.gen_batch_size,
        desc="Backward question stage",
    )

    for row, outputs in zip(rows, raw_backward_questions):
        raw = outputs[0] if outputs else ""
        row["raw_backward_question"] = raw
        row["backward_question"] = remove_backward_answer(_strip_output_prefix(raw))

    # Stage 2: generate forward reasoning for the original question and score it.
    forward_prompts = [build_reasoning_prompt(args.dataset, row["question"]) for row in rows]
    forward_outputs = batched_generate(
        teacher,
        forward_prompts,
        n=1,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        gen_batch_size=args.gen_batch_size,
        desc="Forward reasoning stage",
    )

    for row, outputs in zip(rows, forward_outputs):
        forward_reasoning = outputs[0] if outputs else ""
        row["forward_reasoning"] = forward_reasoning
        row["forward_score"] = score_answer(args.dataset, forward_reasoning, row["gold_answer"])
        row["forward_filter_passed"] = row["forward_score"] >= args.correct_threshold

    # Stage 3: generate reasoning for each backward question.
    backward_reasoning_prompts = [build_reasoning_prompt(args.dataset, row["backward_question"]) for row in rows]
    backward_outputs = batched_generate(
        teacher,
        backward_reasoning_prompts,
        n=1,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        gen_batch_size=args.gen_batch_size,
        desc="Backward reasoning stage",
    )

    for row, outputs in zip(rows, backward_outputs):
        backward_reasoning = outputs[0] if outputs else ""
        row["backward_reasoning"] = backward_reasoning
        row["backward_pred"] = extract_reward_prediction(args.dataset, backward_reasoning)

    # Stage 4: ask the teacher to judge original/backward consistency.
    consistency_template = consistency_prompt_template(args.dataset)
    consistency_prompts = [
        consistency_template.format(
            question=row["question"],
            gold_answer=_consistency_gold_answer(args.dataset, row["gold_answer"]),
            backward_question=row["backward_question"],
            backward_pred=row["backward_pred"],
        )
        for row in rows
    ]
    consistency_outputs = batched_generate(
        teacher,
        consistency_prompts,
        n=1,
        max_new_tokens=args.consistency_max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        gen_batch_size=args.gen_batch_size,
        desc="Consistency stage",
    )

    candidate_rows: List[Dict[str, Any]] = []
    output_rows: List[Dict[str, Any]] = []
    # Stage 5: keep consistent quadruplets with correct forward reasoning.
    for row, outputs in zip(rows, consistency_outputs):
        consistency_reasoning = outputs[0] if outputs else ""
        is_consistent = get_true_false(consistency_reasoning) == "true"
        keep = (args.disable_teacher_filter or row["forward_filter_passed"]) and is_consistent
        candidate = {
            "question": row["question"],
            "answer": row["forward_reasoning"],
            "gold_answer": row["gold_answer"],
            "dataset": args.dataset,
            "data_source": row["data_source"],
            "source_index": row["source_index"],
            "augmentation_method": "reverse_thinking",
            "backward_question": row["backward_question"],
            "raw_backward_question": row["raw_backward_question"],
            "backward_reasoning": row["backward_reasoning"],
            "backward_pred": row["backward_pred"],
            "consistency_reasoning": consistency_reasoning,
            "is_consistent": bool(is_consistent),
            "teacher_score": float(row["forward_score"]),
            "teacher_filter_passed": bool(keep),
            "forward_filter_passed": bool(row["forward_filter_passed"]),
            "answer_char_len": len(str(row["forward_reasoning"])),
        }
        candidate_rows.append(candidate)
        if keep:
            common = {
                "gold_answer": row["gold_answer"],
                "dataset": args.dataset,
                "data_source": row["data_source"],
                "source_index": row["source_index"],
                "augmentation_method": "reverse_thinking",
                "teacher_score": float(row["forward_score"]),
                "teacher_filter_passed": True,
                "original_question": row["question"],
                "backward_question": row["backward_question"],
            }
            output_rows.extend(
                [
                    {
                        **common,
                        "question": row["question"],
                        "answer": row["forward_reasoning"],
                        "reverse_component": "forward_reasoning",
                    },
                    {
                        **common,
                        "question": _bq_training_question(row["question"], row["gold_answer"]),
                        "answer": row["backward_question"],
                        "reverse_component": "backward_question",
                    },
                    {
                        **common,
                        "question": row["backward_question"],
                        "answer": row["backward_reasoning"],
                        "reverse_component": "backward_reasoning",
                        "backward_pred": row["backward_pred"],
                    },
                ]
            )

    summary = base_summary(
        method="reverse_thinking",
        args=args,
        input_rows=len(rows),
        candidate_rows=candidate_rows,
        output_rows=output_rows,
        start_time=start_time,
    )
    summary["num_consistent_quadruplets"] = sum(1 for row in candidate_rows if row["is_consistent"])
    summary["num_forward_passed"] = sum(1 for row in candidate_rows if row["forward_filter_passed"])
    summary["output_rows_per_kept_quadruplet"] = 3
    write_outputs(output_file=args.output_file, output_rows=output_rows, candidate_rows=candidate_rows, summary=summary)

    del teacher
    gc.collect()


def parse_args() -> argparse.Namespace:
    """Parse and validate CLI options for Reverse Thinking augmentation."""
    parser = argparse.ArgumentParser(description="Reverse Thinking data augmentation.")
    add_common_args(parser)
    parser.add_argument("--backward-question-max-new-tokens", type=int, default=1024)
    parser.add_argument("--consistency-max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    validate_common_args(parser, args, num_samples=1)
    return args


if __name__ == "__main__":
    run(parse_args())
