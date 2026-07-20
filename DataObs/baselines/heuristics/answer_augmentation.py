#!/usr/bin/env python3
"""Answer Augmentation by sampling multiple teacher reasoning paths."""

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
        build_answer_aug_prompt,
        load_input_rows,
        make_teacher,
        score_answer,
        validate_common_args,
        write_outputs,
    )
except ImportError:  # direct script execution
    from _common import (
        add_common_args,
        base_summary,
        batched_generate,
        build_answer_aug_prompt,
        load_input_rows,
        make_teacher,
        score_answer,
        validate_common_args,
        write_outputs,
    )


def run(args: argparse.Namespace) -> None:
    """Run answer augmentation by sampling multiple teacher CoTs per original question."""
    start_time = time.time()
    rows = load_input_rows(args.input_file, args.dataset, args.smoke_num_rows)
    teacher = make_teacher(args)

    # Stage 1: build answer-augmentation prompts from the original questions.
    prompts = [
        build_answer_aug_prompt(args.dataset, row["question"], args.use_original_metamath_prompt)
        for row in rows
    ]
    # Stage 2: sample multiple reasoning paths from the teacher.
    answer_outputs = batched_generate(
        teacher,
        prompts,
        n=args.num_augmented_answers,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        gen_batch_size=args.gen_batch_size,
        desc="Answer augmentation stage",
    )

    output_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    # Stage 3: filter generated paths by the DataObs reward function.
    for row, answers in zip(rows, answer_outputs):
        for sample_index, answer in enumerate(answers):
            score = score_answer(args.dataset, answer, row["gold_answer"])
            passed = score >= args.correct_threshold
            candidate = {
                "question": row["question"],
                "answer": answer,
                "gold_answer": row["gold_answer"],
                "dataset": args.dataset,
                "data_source": row["data_source"],
                "source_index": row["source_index"],
                "augmentation_method": "answer_augmentation",
                "sample_index": sample_index,
                "teacher_score": float(score),
                "teacher_filter_passed": bool(passed),
                "answer_char_len": len(str(answer)),
                "used_original_metamath_prompt": bool(args.use_original_metamath_prompt),
            }
            candidate_rows.append(candidate)
            if args.disable_teacher_filter or passed:
                output_rows.append(
                    {
                        "question": row["question"],
                        "answer": answer,
                        "gold_answer": row["gold_answer"],
                        "dataset": args.dataset,
                        "data_source": row["data_source"],
                        "source_index": row["source_index"],
                        "augmentation_method": "answer_augmentation",
                        "teacher_score": float(score),
                        "teacher_filter_passed": bool(passed),
                        "sample_index": sample_index,
                    }
                )

    summary = base_summary(
        method="answer_augmentation",
        args=args,
        input_rows=len(rows),
        candidate_rows=candidate_rows,
        output_rows=output_rows,
        start_time=start_time,
    )
    summary["num_augmented_answers_per_row"] = args.num_augmented_answers
    summary["used_original_metamath_prompt"] = bool(args.use_original_metamath_prompt)
    write_outputs(output_file=args.output_file, output_rows=output_rows, candidate_rows=candidate_rows, summary=summary)

    del teacher
    gc.collect()


def parse_args() -> argparse.Namespace:
    """Parse and validate CLI options for answer augmentation."""
    parser = argparse.ArgumentParser(description="Answer Augmentation data augmentation.")
    add_common_args(parser)
    parser.add_argument("--num-augmented-answers", type=int, default=4)
    parser.add_argument(
        "--use-original-metamath-prompt",
        action="store_true",
        help="Use the short two-shot math prompt from rephrase_questions.py instead of dataset-specific prompts.",
    )
    args = parser.parse_args()
    if args.num_augmented_answers < 1:
        parser.error("--num-augmented-answers must be >= 1.")
    validate_common_args(parser, args, num_samples=args.num_augmented_answers)
    return args


if __name__ == "__main__":
    run(parse_args())
