from __future__ import annotations

# These two prompt templates are kept as close as possible to the wording used
# by the public third-party CoT-Self-Instruct implementations that mirror the
# paper. Local changes should stay limited to model invocation / output parsing.

REASONING_PROMPT_TEMPLATE = """You are an expert in task creation for synthetic data generation, focusing on producing prompts intended to challenge or improve the reasoning abilities of language models.

Here are a few examples:
{examples}

Your task is to generate 10 diverse new prompts, each designed to require significant chain-of-thought reasoning. These should range from simple multi-step calculations to more complex abstract, causal, or strategic reasoning tasks.

Some guidelines:
- Each prompt should be clear, self-contained, and require deliberate reasoning, not mere fact lookup.
- Make them diverse: include numerical reasoning, logic puzzles, common sense scenarios, symbolic reasoning, or planning.
- Avoid repeating the themes or wording of the seed examples too closely.
- Ensure each prompt is realistic and useful for evaluating or training reasoning.
- Use the required output format exactly.

Required output format:
[New Question Begin]
<new reasoning question>
[New Question End]
Then reason step by step, solve the new question and format output following:
[Final Answer to New Question Begin]
<final answer>
[Final Answer to New Question End]

Repeat that format for each generated prompt.
"""

INSTRUCTION_PROMPT_TEMPLATE = """You are an expert in task creation for synthetic data generation, focusing on producing prompts intended to challenge or improve the instruction-following abilities of language models.

Here are a few examples:
{examples}

Analyze the examples above and identify high-level instruction-following patterns such as rewriting, formatting, extracting information, transforming style, organizing content, or following output constraints.

Your task is to generate 10 new prompts. These should:
- Be diverse and realistic instruction-following tasks
- Encourage the model to follow format, style, transformation, extraction, or organization rules
- Be self-contained and clearly specified
- Not require external tools, web access, or hidden context
- Not ask for step-by-step problem solving, arithmetic, factual QA, or a single numerical answer
- Not be ordinary exam questions or word problems
- Prefer imperatives or direct task requests such as rewrite, summarize, extract, classify, transform, translate, format, list, organize, or generate under constraints
- Use the required output format exactly

Required output format:
[New Prompt Begin]
<new instruction-following prompt>
[New Prompt End]

Repeat for each generated prompt.
"""

# The paper emphasizes Answer-Consistency filtering for verifiable reasoning
# tasks.
ANSWER_CONSISTENCY_SOLVE_TEMPLATE = """Solve the following reasoning problem. Reason step by step, and put your final answer within \\boxed{{}}.

Question:
{question}
"""

# For non-verifiable instruction-following prompts, CoT-Self-Instruct uses RIP
# (Reward-Inducing Prompt) style filtering. We adapt the reward step to a local
# judge prompt so the same teacher can score instruction-following quality.
RIP_RESPONSE_TEMPLATE = """Follow the instruction below carefully and produce the best possible answer.

Instruction:
{instruction}
"""

RIP_SCORE_TEMPLATE = """You are evaluating how well an assistant response follows a user instruction.

Instruction:
{instruction}

Assistant Response:
{response}

Score the response from 1 to 5 using this rubric:
- 1: irrelevant or fails the instruction
- 2: partially follows the instruction but poor quality
- 3: mostly follows the instruction with noticeable issues
- 4: follows the instruction well with minor issues
- 5: fully follows the instruction and is high quality

Reply with exactly one line in this format:
Score: <integer 1-5>
"""
