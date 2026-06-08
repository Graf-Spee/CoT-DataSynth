"""
Plain and 7 shot templates are adapted from OpenCompass. Zero-shot is improvised by adding "Let's ..." to Plain.

Reference:
https://github.com/open-compass/opencompass/blob/0524d4992ecda339148b8f59c4cc41991deafc37/opencompass/configs/datasets/commonsenseqa/commonsenseqa_gen_1da2d0.py
https://github.com/open-compass/opencompass/blob/0524d4992ecda339148b8f59c4cc41991deafc37/opencompass/configs/datasets/commonsenseqa/commonsenseqa_7shot_cot_gen_734a22.py
"""

commonsenseqa_plain = [
    {"role": "user", "content": "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nE. {E}\nAnswer:"},
]

commonsenseqa_zeroshot = [
    {"role": "user", "content": "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nE. {E}\nAnswer: Let's think step by step."},
]

commonsenseqa_7_shot = [
    {'role': 'user', 'content': 'Q: What do people use to absorb extra ink from a fountain pen? Answer Choices: A.shirt pocket B.calligrapher’s hand C.inkwell D.desk drawer E.blotter'},
    {'role': 'assistant', 'content': 'A: The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. So the answer is E.'},
    
    {'role': 'user', 'content': 'Q: What home entertainment equipment requires cable?Answer Choices: A.radio shack B.substation C.television D.cabinet'},
    {'role': 'assistant', 'content': 'A: The answer must require cable. Of the above choices, only television requires cable. So the answer is C.'},
    
    {'role': 'user', 'content': 'Q: The fox walked from the city into the forest, what was it looking for? Answer Choices: A.pretty flowers B.hen house C.natural habitat D.storybook'},
    {'role': 'assistant', 'content': 'A: The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. So the answer is B.'},
    
    {'role': 'user', 'content': 'Q: Sammy wanted to go to where the people were. Where might he go? Answer Choices: A.populated areas B.race track C.desert D.apartment E.roadblock'},
    {'role': 'assistant', 'content': 'A: The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. So the answer is A.'},
    
    {'role': 'user', 'content': 'Q: Where do you put your grapes just before checking out? Answer Choices: A.mouth B.grocery cart Csuper market D.fruit basket E.fruit market'},
    {'role': 'assistant', 'content': 'A: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. So the answer is B.'},
    
    {'role': 'user', 'content': 'Q: Google Maps and other highway and street GPS services have replaced what? Answer Choices: A.united states B.mexico C.countryside D.atlas'},
    {'role': 'assistant', 'content': 'A: The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. So the answer is D.'},
    
    {'role': 'user', 'content': 'Q: Before getting a divorce, what did the wife feel who was doing all the work? Answer Choices: A.harder B.anguish C.bitterness D.tears E.sadness'},
    {'role': 'assistant', 'content': 'A: The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness. So the answer is C.'},
    
    {'role': 'user', 'content': 'Q:{question}  Answer Choices: A. {A}\nB. {B}\nC. {C}\nD. {D}\nE. {E}\nA:'},
]