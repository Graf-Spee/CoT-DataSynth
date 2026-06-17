"""
All adapted from OpenCompass.

Reference:
https://github.com/open-compass/opencompass/blob/0524d4992ecda339148b8f59c4cc41991deafc37/opencompass/configs/datasets/math/math_prm800k_500_0shot_nocot_gen_b27274.py
https://github.com/open-compass/opencompass/blob/0524d4992ecda339148b8f59c4cc41991deafc37/opencompass/configs/datasets/math/math_0shot_gen_11c4b5.py
https://github.com/open-compass/opencompass/blob/0524d4992ecda339148b8f59c4cc41991deafc37/opencompass/configs/datasets/math/math_gen_a58d9d.py
"""

math_plain = [
    {"role": "user", "content": "{problem}\nRemember to put your final answer within \\boxed{{}}."},
]

math_zeroshot = [
    {"role": "user", "content": "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."},
]

math_4_shot = [
    {'role': 'user', 'content': "Problem:\nFind the domain of the expression $\\frac{{\\sqrt{{x-2}}}}{{\\sqrt{{5-x}}}}$.}}\nSolution:"},
    {'role': 'assistant', 'content': 'The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{{[2,5)}}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.\n'},
    
    {'role': 'user', 'content': "Problem:\nIf $\\det \\mathbf{{A}} = 2$ and $\\det \\mathbf{{B}} = 12,$ then find $\\det (\\mathbf{{A}} \\mathbf{{B}}).$\nSolution:"},
    {'role': 'assistant', 'content': "We have that $\\det (\\mathbf{{A}} \\mathbf{{B}}) = (\\det \\mathbf{{A}})(\\det \\mathbf{{B}}) = (2)(12) = \\boxed{{24}}.$\nFinal Answer: The final answer is $24$. I hope it is correct.\n"},
    
    {'role': 'user', 'content': "Problem:\nTerrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?\nSolution:"},
    {'role': 'assistant', 'content': "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$: \\begin{{align*}} 30n&=480\\\\ \\Rightarrow\\qquad n&=480/30=\\boxed{{16}} \\end{{align*}}\nFinal Answer: The final answer is $16$. I hope it is correct.\n"},
    
    {'role': 'user', 'content': "Problem:\nIf the system of equations: \\begin{{align*}} 6x-4y&=a,\\\\ 6y-9x &=b. \\end{{align*}}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{{a}}{{b}},$ assuming $b$ is nonzero.\nSolution:"},
    {'role': 'assistant', 'content': 'If we multiply the first equation by $-\\frac{{3}}{{2}}$, we obtain $$6y-9x=-\\frac{{3}}{{2}}a.$$Since we also know that $6y-9x=b$, we have $$-\\frac{{3}}{{2}}a=b\\Rightarrow\\frac{{a}}{{b}}=\\boxed{{-\\frac{{2}}{{3}}}}.$$\nFinal Answer: The final answer is $-\\frac{{2}}{{3}}$. I hope it is correct.\n'},
    
    {'role': 'user', 'content': "Problem:\n{problem}\nSolution:\n"},
]