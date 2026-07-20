#!/usr/bin/env python3
"""Question Rephrasing augmentation with teacher answer filtering."""

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
        build_reasoning_prompt,
        build_rephrase_prompt,
        clean_rephrase,
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
        build_reasoning_prompt,
        build_rephrase_prompt,
        clean_rephrase,
        load_input_rows,
        make_teacher,
        score_answer,
        validate_common_args,
        write_outputs,
    )


def run(args: argparse.Namespace) -> None:
    """Run question rephrasing, teacher CoT generation, and answer filtering."""
    start_time = time.time()
    rows = load_input_rows(args.input_file, args.dataset, args.smoke_num_rows)
    teacher = make_teacher(args)

    # Stage 1: generate semantically equivalent rephrased questions.
    rephrase_prompts = [build_rephrase_prompt(row["question"]) for row in rows]
    rephrase_outputs = batched_generate(
        teacher,
        rephrase_prompts,
        n=args.num_rephrases,
        max_new_tokens=args.rephrase_max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        gen_batch_size=args.gen_batch_size,
        desc="Rephrase stage",
    )

    rephrased_items: List[Dict[str, Any]] = []
    for row, generations in zip(rows, rephrase_outputs):
        for rephrase_index, raw_rephrase in enumerate(generations):
            question = clean_rephrase(raw_rephrase)
            if not question:
                continue
            rephrased_items.append(
                {
                    **row,
                    "augmented_question": question,
                    "rephrase_index": rephrase_index,
                    "raw_rephrase": raw_rephrase,
                }
            )

    # Stage 2: generate teacher CoTs for each rephrased question.
    answer_prompts = [build_reasoning_prompt(args.dataset, item["augmented_question"]) for item in rephrased_items]
    answer_outputs = batched_generate(
        teacher,
        answer_prompts,
        n=args.num_cots,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        gen_batch_size=args.gen_batch_size,
        desc="Answer stage",
    )

    output_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    # Stage 3: keep only candidates whose generated answer matches the gold label.
    for item, answers in zip(rephrased_items, answer_outputs):
        for sample_index, answer in enumerate(answers):
            score = score_answer(args.dataset, answer, item["gold_answer"])
            passed = score >= args.correct_threshold
            candidate = {
                "question": item["augmented_question"],
                "answer": answer,
                "gold_answer": item["gold_answer"],
                "dataset": args.dataset,
                "data_source": item["data_source"],
                "source_index": item["source_index"],
                "augmentation_method": "question_rephrasing",
                "rephrase_index": item["rephrase_index"],
                "sample_index": sample_index,
                "original_question": item["question"],
                "raw_rephrase": item["raw_rephrase"],
                "teacher_score": float(score),
                "teacher_filter_passed": bool(passed),
                "answer_char_len": len(str(answer)),
            }
            candidate_rows.append(candidate)
            if args.disable_teacher_filter or passed:
                output_rows.append(
                    {
                        "question": item["augmented_question"],
                        "answer": answer,
                        "gold_answer": item["gold_answer"],
                        "dataset": args.dataset,
                        "data_source": item["data_source"],
                        "source_index": item["source_index"],
                        "augmentation_method": "question_rephrasing",
                        "teacher_score": float(score),
                        "teacher_filter_passed": bool(passed),
                        "original_question": item["question"],
                        "rephrase_index": item["rephrase_index"],
                        "sample_index": sample_index,
                    }
                )

    summary = base_summary(
        method="question_rephrasing",
        args=args,
        input_rows=len(rows),
        candidate_rows=candidate_rows,
        output_rows=output_rows,
        start_time=start_time,
    )
    summary["num_rephrases_per_row"] = args.num_rephrases
    summary["num_cots_per_rephrase"] = args.num_cots
    summary["num_nonempty_rephrases"] = len(rephrased_items)
    write_outputs(output_file=args.output_file, output_rows=output_rows, candidate_rows=candidate_rows, summary=summary)

    del teacher
    gc.collect()


def parse_args() -> argparse.Namespace:
    """Parse and validate CLI options for question rephrasing."""
    parser = argparse.ArgumentParser(description="Question Rephrasing data augmentation.")
    add_common_args(parser)
    parser.add_argument("--num-rephrases", type=int, default=1, help="Rephrased questions generated per input row.")
    parser.add_argument("--num-cots", type=int, default=1, help="Teacher CoTs generated per rephrased question.")
    parser.add_argument("--rephrase-max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.num_rephrases < 1:
        parser.error("--num-rephrases must be >= 1.")
    if args.num_cots < 1:
        parser.error("--num-cots must be >= 1.")
    validate_common_args(parser, args, num_samples=max(args.num_rephrases, args.num_cots))
    return args


if __name__ == "__main__":
    run(parse_args())
