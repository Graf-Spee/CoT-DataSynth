#!/usr/bin/env python3
"""Question Augmentation implemented as backward-question generation."""

from __future__ import annotations

import argparse
import gc
import time
from typing import Any, Dict, List

try:
    from ._common import (
        add_common_args,
        base_summary,
        batched_generate,
        backward_prompt_family,
        build_backward_question_prompt,
        clean_backward_question,
        format_gold_answer_for_prompt,
        load_input_rows,
        make_teacher,
        remove_backward_answer_annotation,
        require_backward_task,
        validate_common_args,
        write_outputs,
    )
except ImportError:  # direct script execution
    from _common import (
        add_common_args,
        base_summary,
        batched_generate,
        backward_prompt_family,
        build_backward_question_prompt,
        clean_backward_question,
        format_gold_answer_for_prompt,
        load_input_rows,
        make_teacher,
        remove_backward_answer_annotation,
        require_backward_task,
        validate_common_args,
        write_outputs,
    )


def _bq_training_question(dataset: str, question: str, gold_answer: Any) -> str:
    """Format the backward-question generation task as a training row prompt."""
    formatted_answer = format_gold_answer_for_prompt(dataset, gold_answer)
    return (
        "Generate the inverse question based on the following seed question and its answer:\n"
        f"### Seed Question: {question}\n"
        f"### Seed Answer: {formatted_answer}"
    )


def run(args: argparse.Namespace) -> None:
    """Generate inverse/backward questions and emit BQ training rows."""
    start_time = time.time()
    require_backward_task(args.dataset)
    prompt_family = backward_prompt_family(args.dataset)
    rows = load_input_rows(args.input_file, args.dataset, args.smoke_num_rows)
    teacher = make_teacher(args)

    prompts = [
        build_backward_question_prompt(args.dataset, row["question"], row["gold_answer"])
        for row in rows
    ]
    raw_outputs = batched_generate(
        teacher,
        prompts,
        n=args.num_backward_questions,
        max_new_tokens=args.backward_question_max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        gen_batch_size=args.gen_batch_size,
        desc="Backward question stage",
    )

    output_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    for row, generations in zip(rows, raw_outputs):
        for question_index, raw_generation in enumerate(generations):
            backward_question = clean_backward_question(raw_generation)
            sft_answer = remove_backward_answer_annotation(raw_generation)
            passed = bool(sft_answer.strip())
            candidate = {
                "question": row["question"],
                "answer": sft_answer,
                "gold_answer": row["gold_answer"],
                "dataset": args.dataset,
                "data_source": row["data_source"],
                "source_index": row["source_index"],
                "augmentation_method": "question_augmentation",
                "question_index": question_index,
                "raw_backward_question": raw_generation,
                "backward_question": backward_question,
                "clean_backward_question": backward_question,
                "backward_prompt_family": prompt_family,
                "teacher_score": 1.0 if passed else 0.0,
                "teacher_filter_passed": bool(passed),
                "answer_char_len": len(str(sft_answer)),
            }
            candidate_rows.append(candidate)
            if passed:
                output_rows.append(
                    {
                        "question": _bq_training_question(args.dataset, row["question"], row["gold_answer"]),
                        "answer": sft_answer,
                        "gold_answer": row["gold_answer"],
                        "dataset": args.dataset,
                        "data_source": row["data_source"],
                        "source_index": row["source_index"],
                        "augmentation_method": "question_augmentation",
                        "teacher_score": 1.0 if passed else 0.0,
                        "teacher_filter_passed": bool(passed),
                        "original_question": row["question"],
                        "raw_original_question": row.get("raw_question", row["question"]),
                        "question_index": question_index,
                        "reverse_component": "backward_question",
                        "raw_backward_question": raw_generation,
                        "backward_question": backward_question,
                        "clean_backward_question": backward_question,
                        "backward_prompt_family": prompt_family,
                    }
                )

    summary = base_summary(
        method="question_augmentation",
        args=args,
        input_rows=len(rows),
        candidate_rows=candidate_rows,
        output_rows=output_rows,
        start_time=start_time,
    )
    summary["num_backward_questions_per_row"] = args.num_backward_questions
    summary["num_nonempty_backward_questions"] = sum(1 for row in candidate_rows if row["teacher_filter_passed"])
    summary["num_clean_backward_questions"] = sum(1 for row in candidate_rows if row["clean_backward_question"])
    summary["teacher_filter_applied"] = False
    summary["filter_type"] = "nonempty_backward_question_generation"
    summary["prompt_policy"] = {
        "backward_question_prompt_family": prompt_family,
        "student_input": "backward_question_generation_task",
        "sft_answer": "raw_backward_question_generation_without_answer_annotation",
        "raw_generation_retained": True,
        "clean_backward_question_retained_for_reuse": True,
        "clean_backward_question_filters_sft_rows": False,
        "sft_answer_cleaning": "remove_backward_answer_annotation_only",
    }
    write_outputs(output_file=args.output_file, output_rows=output_rows, candidate_rows=candidate_rows, summary=summary)

    del teacher
    gc.collect()


def parse_args() -> argparse.Namespace:
    """Parse and validate CLI options for backward-question augmentation."""
    parser = argparse.ArgumentParser(
        description="Question Augmentation data generation via inverse/backward questions."
    )
    add_common_args(parser)
    parser.add_argument(
        "--num-backward-questions",
        "--num-new-questions",
        dest="num_backward_questions",
        type=int,
        default=1,
        help="Backward questions generated per seed row.",
    )
    parser.add_argument("--backward-question-max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.num_backward_questions < 1:
        parser.error("--num-backward-questions must be >= 1.")
    validate_common_args(parser, args, num_samples=args.num_backward_questions)
    return args


if __name__ == "__main__":
    run(parse_args())
