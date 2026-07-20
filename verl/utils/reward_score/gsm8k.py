# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.replace(",", "").replace("$", "").strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_solution(solution_str, method="strict"):
    """
    OpenCompass: 
    @misc{2023opencompass,
        title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
        author={OpenCompass Contributors},
        howpublished = {\\url{https://github.com/open-compass/opencompass}},
        year={2023}
    }

    OC-boxed: adds \\boxed to the extraction method, in accordance with current GSM8K prompts. Improvised: 7.2
    """

    if method == "strict":
        # this also tests the formatting of the model
        # Any number of spaces behind "####" is fine
        solutions = re.findall(r"####\s*(\-?[0-9\.\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "").strip()
    elif method == 'opencompass':
        solution_str = solution_str.replace(",", "")
        solution_str = solution_str.split('Question:')[0]           # 截断后续生成内容
        numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', solution_str)
        if not numbers or len(numbers) == 0:
            final_answer = None
        else:
            final_answer =  numbers[-1]
    elif method == 'OC-boxed':
        idx = solution_str.rfind("\\boxed")
        if "\\boxed " in solution_str:
            final_answer = solution_str.split("\\boxed ")[-1].split("$")[0].strip()
        elif idx > 0:
            i = idx
            right_brace_idx = None
            num_left_braces_open = 0
            while i < len(solution_str):
                if solution_str[i] == "{":
                    num_left_braces_open += 1
                if solution_str[i] == "}":
                    num_left_braces_open -= 1
                    if num_left_braces_open == 0:
                        right_brace_idx = i
                        break
                i += 1

            final_answer = None if right_brace_idx is None else solution_str[solution_str.find("{", idx) + 1 : right_brace_idx].strip()
        else:
            final_answer = None

        if final_answer is None:
            solution_str = solution_str.replace(",", "")
            solution_str = solution_str.split('Question:')[0]           # 截断后续生成内容
            numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', solution_str)
            if not numbers or len(numbers) == 0:
                final_answer = None
            else:
                final_answer =  numbers[-1]

    return final_answer


def extract_pred(solution_str: str) -> str:
    """Extract prediction from solution string for evaluation.

    This function is required by the evaluation pipeline.
    """
    answer = extract_solution(solution_str, method="OC-boxed")
    return answer if answer is not None else ""


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method='OC-boxed')
    answer_value = _to_float(answer)
    ground_truth_value = _to_float(ground_truth)
    if answer_value is None or ground_truth_value is None:
        return 0
    if answer_value == ground_truth_value:
        return score
    return format_score
