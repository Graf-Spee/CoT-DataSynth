import pandas as pd
import json

def format_livecodebench_prompt(row):
    """将 LiveCodeBench 题目格式化为对话形式的 prompt"""
    question_title = row['question_title']
    question_content = row['question_content']
    starter_code = row['starter_code'] if pd.notna(row['starter_code']) else ""
    
    # 构造提示词内容
    prompt_content = f"""Solve the following programming problem.

Title: {question_title}

Description:
{question_content}"""

    # 如果有 starter_code，则追加到提示词中
    if starter_code and starter_code.strip():
        prompt_content += f"""

Starter Code:
```python
{starter_code}
```"""

    prompt_content += """

Please write complete, executable Python code to solve this problem. Think step by step, and wrap your final answer in '```python ```'."""

    return [{"role": "user", "content": prompt_content}]


def process_ground_truth(x):
    """处理 public_test_cases，确保格式正确"""
    if isinstance(x, str):
        try:
            # 尝试解析 JSON 字符串
            return json.loads(x)
        except (json.JSONDecodeError, TypeError):
            # 如果解析失败，保留原字符串
            return x
    return x


# ==================== 主流程 ====================

# 1. 读取 LiveCodeBench 数据集（请替换为实际路径）
input_path = '/data/open_datasets/livecodebench_code_gen_lite/test1.jsonl'
df = pd.read_json(input_path, orient='records', lines=True)

# 2. 创建 reward_model 列，ground_truth 存放 public_test_cases
df['reward_model'] = df['public_test_cases'].apply(
    lambda x: {'ground_truth': process_ground_truth(x)}
)

# 3. 创建 prompt 列
df['prompt'] = df.apply(format_livecodebench_prompt, axis=1)

# 4. 标记数据来源
df['data_source'] = 'livecodebench'

# 5. 选择需要保留的列并保存
# 必须保留的列：prompt, reward_model, data_source
# 可选保留原数据集中的其他列以供后续分析或调试
output_columns = [
    'prompt',
    'reward_model',
    'data_source',
    'question_title',
    'question_content',
    'question_id',
    'contest_id',
    'contest_date',
    'starter_code',
    'difficulty',
    'platform',
    'metadata',
    # 如需保留 private_test_cases 可取消下面注释
    # 'private_test_cases',
]

output_path = '/data/open_datasets/livecodebench_code_gen_lite/processed/test1.parquet'
df[output_columns].to_parquet(output_path, index=False)

# 6. 验证输出
print(f"处理了 {len(df)} 条数据")
print("示例 reward_model:", df.iloc[0]['reward_model'])
print("示例 prompt:", df.iloc[0]['prompt'])