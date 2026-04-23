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


def extract_solution(solution_str, method="strict"):
    """
    OpenCompass: 
    @misc{2023opencompass,
        title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
        author={OpenCompass Contributors},
        howpublished = {\url{https://github.com/open-compass/opencompass}},
        year={2023}
    }
    """
    
    # assert method in ["strict", "flexible"]

    if method == "strict":
        # this also tests the formatting of the model
        # Any number of spaces behind "####" is fine
        solutions = re.findall(r"####\s*(\-?[0-9\.\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "").strip()
    # elif method == "flexible":  # 41.24%    42.38%
    #     solution_str = solution_str.replace(",", "")
    #     answer = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
    #     final_answer = None
    #     if len(answer) == 0:
    #         # no reward is there is no answer
    #         pass
    #     else:
    #         invalid_str = ["", "."]
    #         # find the last number that is not '.'
    #         for final_answer in reversed(answer):
    #             if final_answer not in invalid_str:
    #                 break

    # elif method == 'vllm':  # weird crash, related to ast?
    #     import ast
    #     solution_str = solution_str.replace(",", "")
    #     numbers = re.findall(r"\d+", solution_str)  # 取所有数字
    #     if len(numbers) < 1:
    #         final_answer = None
    #     final_answer = ast.literal_eval(numbers[-1])
    # elif method == 'vllm-naive':    # 43.67%
    #     solution_str = solution_str.replace(",", "")
    #     numbers = re.findall(r"\d+", solution_str)  # 取所有数字
    #     if len(numbers) < 1:
    #         final_answer = None
    #     final_answer = numbers[-1]
    # elif method == 'eleuther':      # 0%? bugged
    #     pattern = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
    #     answer = re.findall(pattern, solution_str)
    #     final_answer = None
    #     if len(answer) == 0:
    #         # no reward is there is no answer
    #         pass
    #     else:
    #         invalid_str = ["", "."]
    #         # find the last number that is not '.'
    #         for final_answer in reversed(answer):
    #             if final_answer not in invalid_str:
    #                 break
    elif method == 'opencompass':   # 42.53%    43.67%
        solution_str = solution_str.replace(",", "")
        solution_str = solution_str.split('Question:')[0]           # 截断后续生成内容
        numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', solution_str)
        if not numbers:
            final_answer = None
        final_answer =  numbers[-1]
    # elif method == 'modelscope':    # 0% bugged
    #     pattern = r'(-?[0-9.,]{2,})|(-?[0-9]+)'
    #     answer = re.findall(pattern, solution_str)
    #     final_answer = None
    #     if len(answer) == 0:
    #         # no reward is there is no answer
    #         pass
    #     else:
    #         invalid_str = ["", "."]
    #         # find the last number that is not '.'
    #         for final_answer in reversed(answer):
    #             if final_answer not in invalid_str:
    #                 break
    # elif method == 'eelo':      # 42.53%    43.67%
    #     solution_str = solution_str.replace(",", "")
    #     pattern = r"-?\d*\.?\d+"
    #     answer = re.findall(pattern, solution_str)
    #     final_answer = None
    #     if len(answer) == 0:
    #         # no reward is there is no answer
    #         pass
    #     else:
    #         invalid_str = ["", "."]
    #         # find the last number that is not '.'
    #         for final_answer in reversed(answer):
    #             if final_answer not in invalid_str:
    #                 break
    # elif method == 'internal':  # 41.32%    42.38%
    #     solution_str = solution_str.replace(",", "")
    #     pattern = r"[+-]?\d[\d,]*\.?\d*"
    #     answer = re.findall(pattern, solution_str)
    #     final_answer = None
    #     if len(answer) == 0:
    #         # no reward is there is no answer
    #         pass
    #     else:
    #         invalid_str = ["", "."]
    #         # find the last number that is not '.'
    #         for final_answer in reversed(answer):
    #             if final_answer not in invalid_str:
    #                 break

    return final_answer


def extract_pred(solution_str: str) -> str:
    """Extract prediction from solution string for evaluation.

    This function is required by the evaluation pipeline.
    """
    answer = extract_solution(solution_str, method="opencompass")
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
    answer = extract_solution(solution_str=solution_str, method='opencompass')
    if answer is None:
        return 0
    else:
        if float(answer) == float(ground_truth):
            return score
        else:
            return format_score