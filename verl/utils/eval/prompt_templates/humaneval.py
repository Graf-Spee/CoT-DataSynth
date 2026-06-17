"""
Plain is adapted from OpenCompass. Added format specifications to ensure the answer can be correctly processed. Zero-shot is improvised by adding "... step by step reasoning and ..." to Plain.

Reference:
https://github.com/open-compass/opencompass/blob/0524d4992ecda339148b8f59c4cc41991deafc37/opencompass/configs/datasets/humaneval/humaneval_openai_sample_evals_gen_dcae0e.py
"""

humaneval_plain = [
    {"role": "user", "content": "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n{prompt}\nYou should submit your final solution in the following format: ```python\n\n```"},
]

humaneval_zeroshot = [
    {"role": "user", "content": "Read the following function signature and docstring, and fully implement the function described. Your response should only contain step by step reasoing and the code for this function.\n{prompt}\nYou should submit your final solution in the following format: ```python\n\n```"},
]