"""
Plain & 4-shot are adapted from OpenCompass. Zero-shot is improvised by adding "Please ..." to Plain.

Reference:
https://github.com/open-compass/opencompass/blob/0524d4992ecda339148b8f59c4cc41991deafc37/opencompass/configs/datasets/mbpp/sanitized_mbpp_mdblock_0shot_nocot_gen_a2e416.py
https://github.com/open-compass/opencompass/blob/0524d4992ecda339148b8f59c4cc41991deafc37/opencompass/configs/datasets/mbpp/sanitized_mbpp_mdblock_gen_a447ff.py
"""

mbpp_plain = [
    {"role": "user", "content": "You are an expert Python programmer, and here is your task:\n{text}\nYour code should pass these tests:\n\n{test_list}\nYou should submit your final solution in the following format: ```python\n\n```"},
]

mbpp_zeroshot = [
    {"role": "user", "content": "You are an expert Python programmer, and here is your task:\n{text}\nYour code should pass these tests:\n\n{test_list}\nPlease reason step by step. You should submit your final solution in the following format: ```python\n\n```"},
]

mbpp_4_shot = [
    {'role': 'user', 'content': "You are an expert Python programmer, and here is your task:\nWrite a function to find the similar elements from the given two tuple lists.\nYour code should pass these tests:\n\nassert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)\nassert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)\n"},
    {'role': 'assistant', 'content': '```python\ndef similar_elements(test_tup1, test_tup2):\n    res = tuple(set(test_tup1) & set(test_tup2))\n    return (res)```'},
    
    {'role': 'user', 'content': "You are an expert Python programmer, and here is your task:\nWrite a python function to identify non-prime numbers.\nYour code should pass these tests:\n\nassert is_not_prime(2) == False\nassert is_not_prime(10) == True\nassert is_not_prime(35) == True\n"},
    {'role': 'assistant', 'content': "```python\nimport math\ndef is_not_prime(n):\n    result = False\n    for i in range(2,int(math.sqrt(n)) + 1):\n        if n %% i == 0:\n            result = True\n    return result```"},
    
    {'role': 'user', 'content': "You are an expert Python programmer, and here is your task:\nWrite a function to find the largest integers from a given list of numbers using heap queue algorithm.\nYour code should pass these tests:\n\nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]\nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]\nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]\n"},
    {'role': 'assistant', 'content': "```python\nimport heapq as hq\ndef heap_queue_largest(nums,n):\n    largest_nums = hq.nlargest(n, nums)\n    return largest_nums```"},
       
    {'role': 'user', 'content': "You are an expert Python programmer, and here is your task:\n{text}\nYour code should pass these tests:\n\n{test_list}\n"},
]
