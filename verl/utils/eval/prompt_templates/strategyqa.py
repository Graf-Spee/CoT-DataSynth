"""
6 shot template is adapted from OpenCompass. Plain is improvised by taking the last message and adding "Yes or no ...". Zero-shot is improvised by adding "Let's ..." to Plain.

Reference:
https://github.com/open-compass/opencompass/blob/0524d4992ecda339148b8f59c4cc41991deafc37/opencompass/configs/datasets/strategyqa/strategyqa_gen_1180a7.py
"""

strategyqa_plain = [
    {"role": "user", "content": "Question: Yes or no: {question}\nAnswer:"},
]

strategyqa_zeroshot = [
    {"role": "user", "content": "Question: Yes or no: {question}\nAnswer: Let's think step by step."},
]

strategyqa_6_shot = [
    {'role': 'user', 'content': 'Question: Do hamsters provide food for any animals?\nAnswer:'},
    {'role': 'assistant', 'content': 'Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals.\nSo the answer is yes\n'},
    
    {'role': 'user', 'content': 'Question: Could Brooke Shields succeed at University of Pennsylvania?\nAnswer:'},
    {'role': 'assistant', 'content': 'Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.\nSo the answer is yes\n'},
    
    {'role': 'user', 'content': 'Question: Hydrogen\'s atomic number squared exceeds number of Spice Girls?\nAnswer:'},
    {'role': 'assistant', 'content': 'Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen\'s atomic number squared is less than 5.\nSo the answer is no\n'},
    
    {'role': 'user', 'content': 'Question: Is it common to see frost during some college commencements?\nAnswer:'},
    {'role': 'assistant', 'content': 'College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements.\nSo the answer is yes\n'},
    
    {'role': 'user', 'content': 'Question: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?\nAnswer:'},
    {'role': 'assistant', 'content': 'The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam.\nSo the answer is no\n'},
    
    {'role': 'user', 'content': 'Question: Would a pear sink in water?\nAnswer:'},
    {'role': 'assistant', 'content': 'The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float.\nSo the answer is no\n'},
    
    {'role': 'user', 'content': 'Question: {question}\nAnswer:'},
]
