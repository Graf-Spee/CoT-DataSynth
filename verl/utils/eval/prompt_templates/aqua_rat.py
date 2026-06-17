"""
Plain is adapted from OpenCompass. Zero-shot is improvised by adding "Please ..." to Plain.

Reference:
https://github.com/open-compass/opencompass/blob/0524d4992ecda339148b8f59c4cc41991deafc37/opencompass/configs/datasets/agieval/agieval_gen_617738.py
"""

aqua_rat_plain = [
    {"role": "user", "content": "The following is a AQUA-RAT question. Please select the correct answer.\n{question}\n{options}\nThe answer is "},
]

aqua_rat_zeroshot = [
    {"role": "user", "content": "The following is a AQUA-RAT question. Please select the correct answer.\n{question}\n{options}\nPlease reason step by step. Answer: "},
]
