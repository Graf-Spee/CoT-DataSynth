# Eval Dataset Validation Report

验证范围：`scripts/eval.sh` 支持的数据集。此报告验证数据路径、schema、prompt 样例、ground truth 和 reward function 可用性。

## Summary

| Dataset | Status | Rows | Task | Reward | Can Run |
|---|---:|---:|---|---|---:|
| `arc-challenge` | ok | 1172 | multiple_choice | ok | yes |
| `aqua_rat` | ok | 254 | multiple_choice_math | ok | yes |
| `commonsenseQA` | ok | 1221 | multiple_choice | ok | yes |
| `gsm8k` | ok | 1319 | math_word_problem | ok | yes |
| `livecodebench` | ok | 400 | code_generation | ok | yes |
| `humaneval` | ok | 164 | code_generation | ok | yes |
| `humanevalplus` | ok | 164 | code_generation | ok | yes |
| `math` | ok | 12500 | competition_math | ok | yes |
| `math-500` | ok | 500 | competition_math | ok | yes |
| `mbpp` | ok | 257 | code_generation | ok | yes |
| `mbppplus` | ok | 378 | code_generation | ok | yes |
| `numinamath` | ok | 100 | competition_math | ok | yes |
| `strategyQA` | ok | 687 | true_false | ok | yes |
| `bfcl` | bfcl_separate_path |  | function_calling | separate | yes |

## Eval Flow

普通 parquet 数据集流程：

1. `scripts/eval.sh <dataset> <gpu_ids> [model_path]` 根据 dataset 选择 eval parquet 和 reward function。
2. `verl.trainer.main_generation` 读取 `prompt` 列；prompt 是 chat message list，会用 tokenizer chat template 生成回答。
3. 生成结果写入 `generated/responses.parquet`，新增 `responses` 列。
4. `verl.trainer.main_eval` 读取 `responses` 和 `reward_model.ground_truth`，调用对应 `verl/utils/reward_score/*.py` 打分。
5. 指标写入 `generated/responses_labeled.metrics.json`，逐样本标签写入 `generated/responses_labeled.json`。

BFCL 不走 parquet reward function，`eval.sh bfcl ...` 会调用 `scripts/eval_bfcl_dataobs.py` 和 BFCL 自己的 generate/evaluate。

## Dataset Details

### arc-challenge

- status: `ok`
- task_type: `multiple_choice`
- expected_answer_format: Final answer should be an option label like (A), (B), (C), or (D).
- path: `/data/open_datasets/ai2_arc/ARC-Challenge/test-processed.parquet`
- rows: `1172`
- columns: `prompt, question, choices, answerKey, id, data_source, reward_model`
- sample data_source: `ai2_arc`
- sample ground_truth: `C`
- reward file: `/home/hrh/CoT-DataSynth/verl/utils/reward_score/multiple_choice.py`
- reward smoke score: `1.0`

Prompt sample:

```text
system: You are a helpful assistant.
user: Answer the following multiple choice question step by step.
Your final answer should be exactly one of the option labels: (A), (B), (C), (D).

Question: An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?

Options:
(A) Planetary density will decrease.
(B) Planetary years will become longer.
(C) Planetary days will become shorter.
(D) Planetary gravity will become stronger.

Please explain your reasoning, then clearly state your final answer using the option label ((A), (B), (C), (D)).
```

Example command:

```bash
BASE_MODEL=/data/pretrain_models/Qwen2.5-0.5B-Instruct MODEL_NAME=Qwen2.5-0.5B-arc-challenge bash scripts/eval.sh arc-challenge 0 /data/pretrain_models/Qwen2.5-0.5B-Instruct
```

### aqua_rat

- status: `ok`
- task_type: `multiple_choice_math`
- expected_answer_format: Final answer should be an option label like (A), (B), (C), (D), or (E).
- path: `/data/open_datasets/aqua_rat/processed/test-processed.parquet`
- rows: `254`
- columns: `prompt, question, options, rationale, correct, data_source, reward_model`
- sample data_source: `aqua_rat`
- sample ground_truth: `A`
- reward file: `/home/hrh/CoT-DataSynth/verl/utils/reward_score/multiple_choice.py`
- reward smoke score: `1.0`

Prompt sample:

```text
system: You are a helpful assistant.
user: Solve the following multiple choice question step by step.
Your final answer should be one of the options (A), (B), (C), (D), or (E).

Question: A car is being driven, in a straight line and at a uniform speed, towards the base of a vertical tower. The top of the tower is observed from the car and, in the process, it takes 10 minutes for the angle of elevation to change from 45° to 60°. After how much more time will this car reach the base of the tower?

Options:
(A)5(√3 + 1)
(B)6(√3 + √2)
(C)7(√3 – 1)
(D)8(√3 – 2)
(E)None of these

Please explain your reasoning, then clearly state your final answer ((A), (B), (C), (D), or (E)).
```

Example command:

```bash
BASE_MODEL=/data/pretrain_models/Qwen2.5-0.5B-Instruct MODEL_NAME=Qwen2.5-0.5B-aqua_rat bash scripts/eval.sh aqua_rat 0 /data/pretrain_models/Qwen2.5-0.5B-Instruct
```

### commonsenseQA

- status: `ok`
- task_type: `multiple_choice`
- expected_answer_format: Final answer should be an option label like (A), (B), (C), (D), or (E).
- path: `/data/open_datasets/CommonsenseQA/data/validation-processed.parquet`
- rows: `1221`
- columns: `prompt, question, choices, answerKey, id, data_source, reward_model`
- sample data_source: `commonsenseQA`
- sample ground_truth: `A`
- reward file: `/home/hrh/CoT-DataSynth/verl/utils/reward_score/multiple_choice.py`
- reward smoke score: `1.0`

Prompt sample:

```text
system: You are a helpful assistant.
user: Answer the following multiple choice question step by step.
Your final answer should be exactly one of the option labels: (A), (B), (C), (D), (E).

Question: A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?

Options:
(A) bank
(B) library
(C) department store
(D) mall
(E) new york

Please explain your reasoning, then clearly state your final answer using the option label ((A), (B), (C), (D), (E)).
```

Example command:

```bash
BASE_MODEL=/data/pretrain_models/Qwen2.5-0.5B-Instruct MODEL_NAME=Qwen2.5-0.5B-commonsenseQA bash scripts/eval.sh commonsenseQA 0 /data/pretrain_models/Qwen2.5-0.5B-Instruct
```

### gsm8k

- status: `ok`
- task_type: `math_word_problem`
- expected_answer_format: Final answer should include the numeric answer; prompts ask to output after "####".
- path: `/data/open_datasets/GSM8K/test.parquet`
- rows: `1319`
- columns: `data_source, prompt, ability, reward_model, extra_info`
- sample data_source: `openai/gsm8k`
- sample ground_truth: `18`
- reward file: `/home/hrh/CoT-DataSynth/verl/utils/reward_score/gsm8k.py`
- reward smoke score: `1.0`

Prompt sample:

```text
user: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Let's think step by step and output the final answer after "####".
```

Example command:

```bash
BASE_MODEL=/data/pretrain_models/Qwen2.5-0.5B-Instruct MODEL_NAME=Qwen2.5-0.5B-gsm8k bash scripts/eval.sh gsm8k 0 /data/pretrain_models/Qwen2.5-0.5B-Instruct
```

### livecodebench

- status: `ok`
- task_type: `code_generation`
- expected_answer_format: Final answer should be Python code in a markdown code block.
- path: `/data/open_datasets/livecodebench_code_gen_lite/processed/test_v1.parquet`
- rows: `400`
- columns: `prompt, reward_model, data_source, question_title, question_content, question_id, contest_id, contest_date, starter_code, difficulty, platform, metadata`
- sample data_source: `livecodebench`
- sample ground_truth: `[{"input": "6\nabc\nacb\nbac\nbca\ncab\ncba\n", "output": "YES\nYES\nYES\nNO\nNO\nYES\n", "testtype": "stdin"}]`
- reward file: `/home/hrh/CoT-DataSynth/verl/utils/reward_score/livecodebench.py`
- reward smoke score: `0.0`

Prompt sample:

```text
user: Solve the following programming problem.

Title: A. Short Sort

Description:
There are three cards with letters $\texttt{a}$, $\texttt{b}$, $\texttt{c}$ placed in a row in some order. You can do the following operation at most once: 

 
-  Pick two cards, and swap them.  Is it possible that the row becomes $\texttt{abc}$ after the operation? Output "YES" if it is possible, and "NO" otherwise.

Input

The first line contains a single integer $t$ ($1 \leq t \leq 6$) — the number of test cases.

The only line of each test case contains a single string consisting of each of the three characters $\texttt{a}$, $\texttt{b}$, and $\texttt{c}$ exactly once, representing the cards.

Output

For each test case, output "YES" if you can make the row $\texttt{abc}$ with at most one operation, or "NO" otherwise.

You can output the answer in any case (for example, the strings "yEs", "yes", "Yes" and "YES" will be recognized as a positive answer).Sample Input 1:
6

abc

acb

bac

bca

cab

cba



Sample Output 1:

YES
YES
YES
NO
NO
YES


Note

In the first test case, we don't need to do any operations, since the row is already $\texttt{abc}$.

In the second test case, we can swap $\texttt{c}$ and $\texttt{b}$: $\texttt{acb} \to \texttt{abc}$.

In the third test case, we can swap $\texttt{b}$ and $\texttt{a}$: $\texttt{bac} \to \texttt{abc}$.

In the fourth test case, it is impossible to make $\texttt{abc}$ using at most one operation.

Please write complete, executable Python code to solve this problem. Think step by step, and wrap your final answer in '```python ```'.
```

Example command:

```bash
BASE_MODEL=/data/pretrain_models/Qwen2.5-0.5B-Instruct MODEL_NAME=Qwen2.5-0.5B-livecodebench bash scripts/eval.sh livecodebench 0 /data/pretrain_models/Qwen2.5-0.5B-Instruct
```

### humaneval

- status: `ok`
- task_type: `code_generation`
- expected_answer_format: Final answer should be Python code in a markdown code block.
- path: `/data/open_datasets/humaneval/openai_humaneval/processed/test.parquet`
- rows: `164`
- columns: `prompt, reward_model, data_source, task_id, prompt_original, canonical_solution, test, entry_point`
- sample data_source: `humaneval`
- sample ground_truth: `["\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    ass...`
- reward file: `/home/hrh/CoT-DataSynth/verl/utils/reward_score/mbpp.py`
- reward smoke score: `0.0`

Prompt sample:

```text
user: Complete the following Python function.

Function to complete: `has_close_elements`

```python
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
```

Please write Python code to solve this problem. Think step by step, and wrap your final answer in '```python ```'.
```

Example command:

```bash
BASE_MODEL=/data/pretrain_models/Qwen2.5-0.5B-Instruct MODEL_NAME=Qwen2.5-0.5B-humaneval bash scripts/eval.sh humaneval 0 /data/pretrain_models/Qwen2.5-0.5B-Instruct
```

### humanevalplus

- status: `ok`
- task_type: `code_generation`
- expected_answer_format: Final answer should be Python code in a markdown code block.
- path: `/data/open_datasets/humanevalplus/processed/test.parquet`
- rows: `164`
- columns: `prompt, reward_model, data_source, task_id, prompt_original, canonical_solution, test, entry_point`
- sample data_source: `humanevalplus`
- sample ground_truth: `["\n\nimport numpy as np\n\ndef is_floats(x) -> bool:\n    # check if it is float; List[float]; Tuple[float]\n    if isinstance(x, float):\n        return True\n    if isinstance(x, (list, tuple)):\n        return all(isinstance(i, float) for i in x)\n    if isinstance(x, np.ndarray):\n        retur...`
- reward file: `/home/hrh/CoT-DataSynth/verl/utils/reward_score/mbpp.py`
- reward smoke score: `0.0`

Prompt sample:

```text
user: Complete the following Python function.

Function to complete: `has_close_elements`

```python
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
```

Please write Python code to solve this problem. Think step by step, and wrap your final answer in '```python ```'.
```

Example command:

```bash
BASE_MODEL=/data/pretrain_models/Qwen2.5-0.5B-Instruct MODEL_NAME=Qwen2.5-0.5B-humanevalplus bash scripts/eval.sh humanevalplus 0 /data/pretrain_models/Qwen2.5-0.5B-Instruct
```

### math

- status: `ok`
- task_type: `competition_math`
- expected_answer_format: Final answer should be wrapped in \boxed{}.
- path: `/data/open_datasets/MATH/train_processed.parquet`
- rows: `12500`
- columns: `prompt, problem, solution, data_source, reward_model, level, type`
- sample data_source: `math`
- sample ground_truth: `0`
- reward file: `/home/hrh/CoT-DataSynth/verl/utils/reward_score/math_verify.py`
- warnings: Reward function smoke call failed: ModuleNotFoundError: No module named 'tensordict'

Prompt sample:

```text
system: You are a helpful assistant.
user: Solve the following math problem step by step. Put your final answer in \boxed{}.

Problem: Let \[f(x) = \left\{
\begin{array}{cl} ax+3, &\text{ if }x>2, \\
x-5 &\text{ if } -2 \le x \le 2, \\
2x-b &\text{ if } x <-2.
\end{array}
\right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).
```

Example command:

```bash
BASE_MODEL=/data/pretrain_models/Qwen2.5-0.5B-Instruct MODEL_NAME=Qwen2.5-0.5B-math bash scripts/eval.sh math 0 /data/pretrain_models/Qwen2.5-0.5B-Instruct
```

### math-500

- status: `ok`
- task_type: `competition_math`
- expected_answer_format: Final answer should be wrapped in \boxed{}.
- path: `/data/open_datasets/MATH-500/test-processed.parquet`
- rows: `500`
- columns: `prompt, problem, solution, data_source, reward_model`
- sample data_source: `math-500`
- sample ground_truth: `\left( 3, \frac{\pi}{2} \right)`
- reward file: `/home/hrh/CoT-DataSynth/verl/utils/reward_score/math_verify.py`
- warnings: Reward function smoke call failed: ModuleNotFoundError: No module named 'tensordict'

Prompt sample:

```text
system: You are a helpful assistant.
user: Solve the following math problem step by step. Put your final answer in \boxed{}.

Problem: Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$

Please explain your reasoning, then clearly state your final answer wrapped in in \boxed{}.
```

Example command:

```bash
BASE_MODEL=/data/pretrain_models/Qwen2.5-0.5B-Instruct MODEL_NAME=Qwen2.5-0.5B-math-500 bash scripts/eval.sh math-500 0 /data/pretrain_models/Qwen2.5-0.5B-Instruct
```

### mbpp

- status: `ok`
- task_type: `code_generation`
- expected_answer_format: Final answer should be Python code in a markdown code block.
- path: `/data/open_datasets/mbpp/sanitized/processed/test.parquet`
- rows: `257`
- columns: `prompt, reward_model, data_source, source_file, task_id, prompt_original, code, test_imports, test_list`
- sample data_source: `mbpp`
- sample ground_truth: `["assert remove_Occ(\"hello\",\"l\") == \"heo\"", "assert remove_Occ(\"abcda\",\"a\") == \"bcd\"", "assert remove_Occ(\"PHP\",\"P\") == \"H\""]`
- reward file: `/home/hrh/CoT-DataSynth/verl/utils/reward_score/mbpp.py`
- reward smoke score: `0.0`

Prompt sample:

```text
user: Write a python function to remove first and last occurrence of a given character from the string. Your code should satisfy these tests:

assert remove_Occ("hello","l") == "heo"
assert remove_Occ("abcda","a") == "bcd"
assert remove_Occ("PHP","P") == "H"

Please write Python code to solve this problem. Think step by step, and wrap your final answer in '```python ```'.
```

Example command:

```bash
BASE_MODEL=/data/pretrain_models/Qwen2.5-0.5B-Instruct MODEL_NAME=Qwen2.5-0.5B-mbpp bash scripts/eval.sh mbpp 0 /data/pretrain_models/Qwen2.5-0.5B-Instruct
```

### mbppplus

- status: `ok`
- task_type: `code_generation`
- expected_answer_format: Final answer should be Python code in a markdown code block.
- path: `/data/open_datasets/mbppplus/processed/test.parquet`
- rows: `378`
- columns: `prompt, reward_model, data_source, task_id, prompt_original, code, source_file, test_imports, test_list, test`
- sample data_source: `mbppplus`
- sample ground_truth: `["import numpy as np\nfrom math import inf\n\ndef is_floats(x) -> bool:\n    # check if it is float; List[float]; Tuple[float]\n    if isinstance(x, float):\n        return True\n    if isinstance(x, (list, tuple)):\n        return all(isinstance(i, float) for i in x)\n    if isinstance(x, np.ndarra...`
- reward file: `/home/hrh/CoT-DataSynth/verl/utils/reward_score/mbpp.py`
- reward smoke score: `0.0`

Prompt sample:

```text
user: Write a function to find the shared elements from the given two lists. Your code should satisfy these tests:

assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))
assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))
assert set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))

Please write Python code to solve this problem. Think step by step, and wrap your final answer in '```python ```'.
```

Example command:

```bash
BASE_MODEL=/data/pretrain_models/Qwen2.5-0.5B-Instruct MODEL_NAME=Qwen2.5-0.5B-mbppplus bash scripts/eval.sh mbppplus 0 /data/pretrain_models/Qwen2.5-0.5B-Instruct
```

### numinamath

- status: `ok`
- task_type: `competition_math`
- expected_answer_format: Final answer should be wrapped in \boxed{}.
- path: `/data/open_datasets/NuminaMath-CoT/test-processed.parquet`
- rows: `100`
- columns: `prompt, problem, solution, data_source, reward_model`
- sample data_source: `numinamath`
- sample ground_truth: `\frac{2}{9} \le x^4 + y^4 \le 8`
- reward file: `/home/hrh/CoT-DataSynth/verl/utils/reward_score/math_verify.py`
- warnings: Reward function smoke call failed: ModuleNotFoundError: No module named 'tensordict'

Prompt sample:

```text
system: You are a helpful assistant.
user: Solve the following math problem step by step. Put your final answer in \boxed{}.

Problem: Let  $x, y$  be real numbers such that  $1\le x^2-xy+y^2\le2$ . Show that:
a)  $\dfrac{2}{9}\le x^4+y^4\le 8$ ;
b)  $x^{2n}+y^{2n}\ge\dfrac{2}{3^n}$ , for all  $n\ge3$ .

*Laurențiu Panaitopol* and *Ioan Tomescu*
```

Example command:

```bash
BASE_MODEL=/data/pretrain_models/Qwen2.5-0.5B-Instruct MODEL_NAME=Qwen2.5-0.5B-numinamath bash scripts/eval.sh numinamath 0 /data/pretrain_models/Qwen2.5-0.5B-Instruct
```

### strategyQA

- status: `ok`
- task_type: `true_false`
- expected_answer_format: Final answer should contain "true" or "false".
- path: `/data/open_datasets/StrategyQA/data/test-processed.parquet`
- rows: `687`
- columns: `prompt, qid, term, description, question, answer, facts, data_source, reward_model`
- sample data_source: `strategyQA`
- sample ground_truth: `True`
- reward file: `/home/hrh/CoT-DataSynth/verl/utils/reward_score/truefalse.py`
- reward smoke score: `1.0`

Prompt sample:

```text
system: You are a helpful assistant.
user: Answer the following true of false question step by step, using the given facts.
Your should answer the question only in "true" or "false".

Term & Description: Apollo 13 / A failed crewed mission to land on the Moon

Facts: Apollo 13 was recovered by the USS Iwo Jima. Iwo Jima was captured from the Imperial Japanese Army during World War II by the US in a conflict called the Battle of Iwo Jima.

Question: Was ship that recovered Apollo 13 named after a World War II battle?

Please explain your reasoning, then clearly state your final answer (true or false).
```

Example command:

```bash
BASE_MODEL=/data/pretrain_models/Qwen2.5-0.5B-Instruct MODEL_NAME=Qwen2.5-0.5B-strategyQA bash scripts/eval.sh strategyQA 0 /data/pretrain_models/Qwen2.5-0.5B-Instruct
```

### bfcl

- status: `bfcl_separate_path`
- task_type: `function_calling`
- expected_answer_format: BFCL uses its own bfcl_eval generate/evaluate path, not parquet + reward_model.
- path: `None`

Example command:

```bash
BASE_MODEL=/path/to/model MODEL_NAME=name bash scripts/eval.sh bfcl 0 /path/to/model
```
