# Eval prompt 与 OpenCompass 的差异

本文比较 `scripts/eval.sh` 当前使用的评测 prompt 和本地
`dev/opencompass-main/opencompass/configs/datasets` 里最接近的 OpenCompass 配置。

当前仓库的评测流程是：`scripts/eval.sh` 选择 raw parquet，`verl.trainer.main_generation` 直接读取其中的 `prompt` 列。多数 `prompt` 已经是 chat message list，会再经过 tokenizer chat template。OpenCompass 通常从原始字段用 `PromptTemplate` 或 `RawPromptTemplate` 现场拼 prompt。

## 总览

| 数据集 | 当前仓库 prompt | 最接近的 OpenCompass 配置 | 主要差异 |
| --- | --- | --- | --- |
| `arc-challenge` | 0-shot CoT，答案格式 `(A)` | `ARC_c_gen.py`，`ARC_c_cot_gen_926652.py` | OC 默认是直接 `Answer:`；OC CoT 用 `ANSWER: A`，不是 `(A)` |
| `aqua_rat` | 0-shot CoT，答案格式 `(A)`-`(E)` | AGIEval `aqua-rat` | OC 直接要求选择，提示是 `The answer is`，不要求 CoT |
| `commonsenseQA` | 0-shot CoT，答案格式 `(A)`-`(E)` | `commonsenseqa_gen.py`，`commonsenseqa_7shot_cot_gen_734a22.py` | OC 默认是检索 few-shot；OC CoT 是 7-shot |
| `gsm8k` | 0-shot CoT，答案在 `####` 后 | `gsm8k_gen.py`，0-shot boxed 变体 | OC 默认是 4-shot CoT；OC 0-shot 用 `\boxed{}` |
| `humaneval` | 0-shot，要求 markdown Python block 和 step-by-step | `humaneval_gen.py` | OC 要求只输出代码，不要 markdown，不要 CoT |
| `humanevalplus` | 同 HumanEval | `humaneval_plus_gen.py` | OC prompt 更短，只要求补全代码 |
| `math` | 0-shot CoT，有 system role 和 `Problem:` 前缀，答案 boxed | `math_gen.py` | 目标接近，但 wrapper/system/prefix 不同 |
| `math-500` | 0-shot CoT，有 system role，答案 boxed | `math_500_gen.py` | 目标接近；OC 更短且没有 system role |
| `mbpp` | 0-shot，给 tests，要求 markdown code block 和 step-by-step | `mbpp_gen.py` | OC 是 3-shot，并用 `[BEGIN]...[DONE]` |
| `mbppplus` | 同 MBPP | `mbpp_plus_gen.py` | OC 是 3-shot，并用 `[BEGIN]...[DONE]` |
| `numinamath` | 0-shot CoT，答案 boxed | 未找到直接配置 | 最接近的是通用 `math_gen.py` |
| `strategyQA` | 0-shot，带 term/description/facts，答案 true/false | `strategyqa_gen.py` | OC 是 6-shot CoT，不显式提供 facts wrapper |
| `bfcl` | 单独 BFCL 路径，不走 parquet prompt | 未找到直接配置 | 无法直接对齐 prompt template |

## 逐数据集差异

### arc-challenge

当前仓库：

- 数据：`/data/open_datasets/ai2_arc/ARC-Challenge/test-00000-of-00001.parquet`
- 0-shot chat message。
- 有 `system: You are a helpful assistant.`
- 要求 step by step。
- 最终答案必须是 `(A)`、`(B)`、`(C)`、`(D)` 之一。

OpenCompass：

- 默认入口：`dev/opencompass-main/opencompass/configs/datasets/ARC_c/ARC_c_gen.py`
- 实际导入：`ARC_c_gen_1e0de5.py`
- 默认 prompt：

```text
Question: {question}
A. {textA}
B. {textB}
C. {textC}
D. {textD}
Answer:
```

- CoT 版本：`ARC_c_cot_gen_926652.py`
- CoT 版本要求最后一行是 `ANSWER: $LETTER`。

差异：

- 当前仓库显式要求 CoT 和 `(A)` 格式。
- OC 默认不要求 CoT。
- OC CoT 的答案格式是 `ANSWER: A`，不是 `(A)`。

### aqua_rat

当前仓库：

- 数据：`/data/open_datasets/aqua_rat/raw/test-00000-of-00001.parquet`
- 0-shot chat message，有 system role。
- 要求 step by step。
- 最终答案是 `(A)` 到 `(E)`。

OpenCompass：

- 未找到独立 `aqua_rat` 配置。
- 最接近的是 AGIEval 里的 `aqua-rat`：
  `dev/opencompass-main/opencompass/configs/datasets/agieval/agieval_gen_617738.py`
- prompt 形态：

```text
The following is a AQUA-RAT question. Please select the correct answer.
{question}
{options}
The answer is
```

差异：

- 当前仓库是 CoT prompt，并要求括号选项。
- OC AGIEval 是直接选择题 prompt，通常期望裸字母。

### commonsenseQA

当前仓库：

- 数据：`/data/open_datasets/CommonsenseQA/data/validation-00000-of-00001.parquet`
- 0-shot chat message，有 system role。
- 要求 step-by-step。
- 最终答案是 `(A)` 到 `(E)`。

OpenCompass：

- 默认入口：`commonsenseqa_gen.py`
- 实际导入：`commonsenseqa_gen_c946f2.py`
- 默认使用 MDL 检索 few-shot。
- 单个样例模板：

```text
{question}
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}
Answer:
```

- CoT 版本：`commonsenseqa_7shot_cot_gen_734a22.py`
- CoT 版本是 7-shot，答案形态是 `So the answer is X`。

差异：

- 当前仓库是 0-shot CoT。
- OC 默认是检索 few-shot 直接回答。
- OC CoT 是 few-shot，答案是句子中的裸字母。

### gsm8k

当前仓库：

- 数据：`/data/open_datasets/GSM8K/main/test-00000-of-00001.parquet`
- user-only chat message。
- 结尾：

```text
Let's think step by step and output the final answer after "####".
```

OpenCompass：

- 默认入口：`gsm8k_gen.py`
- 实际导入：`gsm8k_gen_1d7fe4.py`
- 默认是 4-shot CoT：

```text
Question: {question}
Let's think step by step
Answer:
```

- 0-shot 变体如 `gsm8k_0shot_v2_gen_a58960.py`：

```text
{question}
Please reason step by step, and put your final answer within \boxed{}.
```

差异：

- 当前仓库是 0-shot，并使用 `####`。
- OC 默认是 4-shot，示例答案结尾是 `The answer is X`。
- OC 0-shot 变体使用 `\boxed{}`，不是 `####`。

### humaneval

当前仓库：

- 数据：`/data/open_datasets/humaneval/openai_humaneval/test-00000-of-00001.parquet`
- 0-shot。
- 用 markdown code block 展示函数签名和 docstring。
- 要求：

```text
Please write Python code to solve this problem.
Think step by step, and wrap your final answer in '```python ```'.
```

OpenCompass：

- 默认入口：`humaneval_gen.py`
- 实际导入：`humaneval_openai_sample_evals_gen_dcae0e.py`
- prompt：

```text
Read the following function signature and docstring, and fully implement the function described.
Your response should only contain the code for this function.
{prompt}
```

差异：

- 当前仓库要求 markdown Python block 和 step-by-step。
- OC 要求只输出代码，不要 markdown。
- OC 不要求显式 CoT。

### humanevalplus

当前仓库：

- 数据：`/data/open_datasets/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet`
- 和 HumanEval 一样的 wrapper。
- 要求 step-by-step 和 markdown Python code block。

OpenCompass：

- 默认入口：`humaneval_plus_gen.py`
- 实际导入：`humaneval_plus_gen_8e312c.py`
- prompt：

```text
Complete the following python code:
{prompt}
```

差异：

- 当前仓库 instruction 更重。
- OC 更像纯代码补全。
- OC 不要求 markdown，也不要求 CoT。

### math

当前仓库：

- 数据：`/data/open_datasets/MATH/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet`
- chat message，有 system role。
- user prompt：

```text
Solve the following math problem step by step. Put your final answer in \boxed{}.

Problem: {problem}
```

OpenCompass：

- 默认入口：`math_gen.py`
- 实际导入：`math_gen_a58d9d.py`
- prompt：

```text
{problem}
Please reason step by step, and put your final answer within \boxed{}.
```

差异：

- 两者都是 0-shot CoT + boxed。
- 当前仓库多了 system role 和 `Problem:` 前缀。
- OC prompt 更短。

### math-500

当前仓库：

- 数据：`/data/open_datasets/MATH-500/test.parquet`
- chat message，有 system role。
- 要求 step-by-step 和 boxed answer。
- 部分样本还会加：

```text
Please explain your reasoning, then clearly state your final answer wrapped in \boxed{}.
```

OpenCompass：

- 配置：`math_500_gen.py`
- prompt：

```text
{problem}
Please reason step by step, and put your final answer within \boxed{}.
```

差异：

- 大方向一致，都是 boxed。
- 当前仓库有 system role，且有时额外强调解释和最终答案。
- OC 更短，没有 system role。

### mbpp

当前仓库：

- 数据：`/data/open_datasets/mbpp/sanitized/test-00000-of-00001.parquet`
- 0-shot。
- 包含 task text 和 tests。
- 结尾：

```text
Please write Python code to solve this problem.
Think step by step, and wrap your final answer in '```python ```'.
```

OpenCompass：

- 默认入口：`mbpp_gen.py`
- 实际导入：`mbpp_gen_830460.py`
- 3-shot。
- expert-programmer wrapper：

```text
You are an expert Python programmer, and here is your task: {text}
Your code should pass these tests:

{test_list}
```

- 答案格式：

```text
[BEGIN]
...
[DONE]
```

差异：

- 当前仓库是 0-shot。
- OC 是 3-shot。
- 当前仓库期望 markdown Python block；OC 期望 `[BEGIN]...[DONE]`。

### mbppplus

当前仓库：

- 数据：`/data/open_datasets/mbppplus/data/test-00000-of-00001-d5781c9c51e02795.parquet`
- 和 MBPP 一样：0-shot、tests、markdown Python block、step-by-step。

OpenCompass：

- 默认入口：`mbpp_plus_gen.py`
- 实际导入：`mbpp_plus_gen_0b836a.py`
- 3-shot。
- 使用 expert-programmer wrapper 和 `[BEGIN]...[DONE]`。

差异：

- 同 MBPP：当前仓库是 0-shot markdown-code；OC 是 few-shot `[BEGIN]/[DONE]`。

### numinamath

当前仓库：

- 数据：`/data/open_datasets/NuminaMath-CoT/data/test-00000-of-00001.parquet`
- chat message，有 system role。
- 和 MATH 类似：

```text
Solve the following math problem step by step. Put your final answer in \boxed{}.

Problem: {problem}
```

OpenCompass：

- 本地 OpenCompass tree 没找到直接 NuminaMath 配置。
- 最接近的是通用 `math_gen.py`。

差异：

- 没有一一对应的 OpenCompass prompt。
- 最接近的 OC math prompt 也是 boxed，但没有 system role 和 `Problem:` 前缀。

### strategyQA

当前仓库：

- 数据：`/data/open_datasets/StrategyQA/data/test-00000-of-00001-bae602f3ee37f4ca.parquet`
- 0-shot chat message，有 system role。
- 包含：
  - term and description
  - facts
  - question
- 最终答案要求 `true` 或 `false`。

OpenCompass：

- 默认入口：`strategyqa_gen.py`
- 实际导入：`strategyqa_gen_1180a7.py`
- 6-shot CoT。
- 样例形态：

```text
Question: {question}
Answer:
```

- few-shot 答案结尾：

```text
So the answer is yes
```

差异：

- 当前仓库是 0-shot，并显式提供 supporting facts。
- OC 是 6-shot CoT，使用 yes/no 表述。
- 当前仓库目标是 true/false；OC 后处理一般按 yes/no。

### bfcl

当前仓库：

- `scripts/eval.sh bfcl ...` 不走普通 parquet `prompt` + reward function 流程。
- 会调用 BFCL 专用评测脚本。

OpenCompass：

- 本地 OpenCompass tree 没找到直接 BFCL/function-calling 数据集配置。

差异：

- 没有可直接比较的 prompt template。
