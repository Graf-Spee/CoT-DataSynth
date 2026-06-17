"""
All adapted from OpenCompass. Few-shot prompting is not implemented, since the shots are not hard-coded in OpenCompass, but selected with fixed indices.

Reference:
https://github.com/open-compass/opencompass/blob/0524d4992ecda339148b8f59c4cc41991deafc37/opencompass/configs/datasets/ARC_c/ARC_c_gen_1e0de5.py
https://github.com/open-compass/opencompass/blob/0524d4992ecda339148b8f59c4cc41991deafc37/opencompass/configs/datasets/ARC_c/ARC_c_cot_gen_926652.py
"""

ZERO_SHOT_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

{choices}""".strip()

CHOICE_3 = "A. {textA}\nB. {textB}\nC. {textC}\n"
CHOICE_4 = "A. {textA}\nB. {textB}\nC. {textC}\nD. {textD}\n"
CHOICE_5 = "A. {textA}\nB. {textB}\nC. {textC}\nD. {textD}\nE. {textE}\n"

arc_challenge_plain = [
    {"role": "user", "content": "Question: {question}\n{choices}Answer:"},
]

arc_challenge_zeroshot = [
    {"role": "user", "content": ZERO_SHOT_TEMPLATE},
]
