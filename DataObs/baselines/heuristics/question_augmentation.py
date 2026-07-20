#!/usr/bin/env python3
"""GSM8K Question Augmentation using the Xwin-Math synthetic question prompts."""

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
        load_input_rows,
        make_teacher,
        score_answer,
        validate_common_args,
        write_outputs,
    )


QUESTION_GENERATION_PROMPT = """Please act as a professional math teacher.
Your goal is to create high quality math word problems to help students learn math.
You will be given a math question. Please create a new question based on the Given Question and following instructions.
To achieve the goal, you have three jobs.
# Please generate a similar but new question according to the Given Question.
# Check the question by solving it step-by-step to find out if it adheres to all principles.
# Modify the created question according to your checking comment to ensure it is of high quality.
You have five principles to do this.
# Ensure the new question only asks for one thing, be reasonable, be based on the Given Question, and can be answered with only a number (float or integer). For example, DO NOT ask, 'what is the amount of A, B and C?'.
# Ensure the new question is in line with common sense of life. For example, the amount someone has or pays must be a positive number, and the number of people must be an integer.
# Ensure your student can answer the new question without the given question. If you want to use some numbers, conditions or background in the given question, please restate them to ensure no information is omitted in your new question.
# Please DO NOT include solution in your question.
# If the created question already follows these principles upon your verification. Just keep it without any modification.
Given Question: {question}
Your output should be in the following format:
CREATED QUESTION: <your created question>
VERIFICATION AND MODIFICATION: <solve the question step-by-step and modify it to follow all principles>
FINAL CREATED QUESTION: <your final created question>"""


ANSWER_GENERATION_PROMPT = """Please act as a professional math teacher.
Your goal is to accurately solve a math word problem.
To achieve the goal, you have two jobs.
# Write detailed solution to a Given Question.
# Write the final answer to this question.
You have two principles to do this.
# Ensure the solution is step-by-step.
# Ensure the final answer is just a number (float or integer).
Given Question: {question}
Your output should be in the following format:
SOLUTION: <your detailed solution to the given question>
FINAL ANSWER: <your final answer to the question with only an integer or float number>"""


def build_question_generation_prompt(question: str) -> str:
    """Build the paper's GSM8K synthetic question generation prompt for one seed question."""
    return QUESTION_GENERATION_PROMPT.format(question=question)


def build_answer_generation_prompt(question: str) -> str:
    """Build the paper's GSM8K solution and final-answer generation prompt for one synthetic question."""
    return ANSWER_GENERATION_PROMPT.format(question=question)


def _strip_trailing_section(text: str) -> str:
    """Drop any following all-caps section header that the model may append after a field."""
    return re.split(r"\n\s*[A-Z][A-Z ]{2,}:\s*", text.strip(), maxsplit=1)[0].strip()


def extract_final_created_question(text: str) -> str:
    """Extract FINAL CREATED QUESTION from the question-generation response."""
    match = re.search(r"FINAL CREATED QUESTION:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return _strip_trailing_section(match.group(1))
    match = re.search(r"CREATED QUESTION:\s*(.*?)(?:\n\s*VERIFICATION AND MODIFICATION:|\Z)", text, re.IGNORECASE | re.DOTALL)
    return _strip_trailing_section(match.group(1)) if match else text.strip()


def extract_final_answer(text: str) -> str:
    """Extract the numeric FINAL ANSWER field from the answer-generation response."""
    match = re.search(r"FINAL ANSWER:\s*([^\n]+)", text, re.IGNORECASE)
    if not match:
        return ""
    number = re.search(r"-?\d[\d,]*(?:\.\d+)?", match.group(1))
    return number.group(0).replace(",", "") if number else match.group(1).strip()


def run(args: argparse.Namespace) -> None:
    """Run GSM8K-only synthetic question generation, answer generation, and reward filtering."""
    start_time = time.time()
    rows = load_input_rows(args.input_file, args.dataset, args.smoke_num_rows)
    teacher = make_teacher(args)

    # Stage 1: generate and self-verify new GSM8K-style questions from seed questions.
    question_prompts = [build_question_generation_prompt(row["question"]) for row in rows]
    question_outputs = batched_generate(
        teacher,
        question_prompts,
        n=args.num_new_questions,
        max_new_tokens=args.question_max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        gen_batch_size=args.gen_batch_size,
        desc="Question generation stage",
    )

    synthetic_items: List[Dict[str, Any]] = []
    for row, generations in zip(rows, question_outputs):
        for question_index, raw_generation in enumerate(generations):
            synthetic_question = extract_final_created_question(raw_generation)
            if not synthetic_question:
                continue
            synthetic_items.append(
                {
                    **row,
                    "synthetic_question": synthetic_question,
                    "question_index": question_index,
                    "raw_question_generation": raw_generation,
                }
            )

    # Stage 2: generate a step-by-step solution and final numeric answer for each new question.
    answer_prompts = [build_answer_generation_prompt(item["synthetic_question"]) for item in synthetic_items]
    answer_outputs = batched_generate(
        teacher,
        answer_prompts,
        n=args.num_answers_per_question,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        gen_batch_size=args.gen_batch_size,
        desc="Answer generation stage",
    )

    output_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    # Stage 3: treat the generated final answer as pseudo-gold and keep self-consistent CoTs.
    for item, answers in zip(synthetic_items, answer_outputs):
        for answer_index, answer in enumerate(answers):
            generated_answer = extract_final_answer(answer)
            score = score_answer(args.dataset, answer, generated_answer)
            passed = bool(generated_answer) and score >= args.correct_threshold
            candidate = {
                "question": item["synthetic_question"],
                "answer": answer,
                "gold_answer": generated_answer,
                "dataset": args.dataset,
                "data_source": item["data_source"],
                "source_index": item["source_index"],
                "augmentation_method": "question_augmentation",
                "question_index": item["question_index"],
                "answer_index": answer_index,
                "original_question": item["question"],
                "original_gold_answer": item["gold_answer"],
                "raw_question_generation": item["raw_question_generation"],
                "teacher_score": float(score),
                "teacher_filter_passed": bool(passed),
                "answer_char_len": len(str(answer)),
            }
            candidate_rows.append(candidate)
            if args.disable_teacher_filter or passed:
                output_rows.append(
                    {
                        "question": item["synthetic_question"],
                        "answer": answer,
                        "gold_answer": generated_answer,
                        "dataset": args.dataset,
                        "data_source": item["data_source"],
                        "source_index": item["source_index"],
                        "augmentation_method": "question_augmentation",
                        "teacher_score": float(score),
                        "teacher_filter_passed": bool(passed),
                        "original_question": item["question"],
                        "original_gold_answer": item["gold_answer"],
                        "question_index": item["question_index"],
                        "answer_index": answer_index,
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
    summary["num_new_questions_per_row"] = args.num_new_questions
    summary["num_answers_per_question"] = args.num_answers_per_question
    summary["num_nonempty_synthetic_questions"] = len(synthetic_items)
    summary["paper_prompt_source"] = "Common 7B Language Models Already Possess Strong Math Capabilities, Appendix A"
    write_outputs(output_file=args.output_file, output_rows=output_rows, candidate_rows=candidate_rows, summary=summary)

    del teacher
    gc.collect()


def parse_args() -> argparse.Namespace:
    """Parse and validate CLI options for GSM8K question augmentation."""
    parser = argparse.ArgumentParser(description="GSM8K Question Augmentation data generation.")
    add_common_args(parser, dataset_required=False)
    parser.set_defaults(temperature=1.0, do_sample=True)
    parser.add_argument("--num-new-questions", type=int, default=1, help="Synthetic questions generated per seed row.")
    parser.add_argument("--num-answers-per-question", type=int, default=1, help="Teacher answers generated per synthetic question.")
    parser.add_argument("--question-max-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    validate_common_args(parser, args, num_samples=max(args.num_new_questions, args.num_answers_per_question))
    if args.dataset != "gsm8k":
        parser.error("question_augmentation.py only supports --dataset gsm8k because the paper prompt only covers GSM8K.")
    if args.num_new_questions < 1:
        parser.error("--num-new-questions must be >= 1.")
    if args.num_answers_per_question < 1:
        parser.error("--num-answers-per-question must be >= 1.")
    return args


if __name__ == "__main__":
    run(parse_args())
