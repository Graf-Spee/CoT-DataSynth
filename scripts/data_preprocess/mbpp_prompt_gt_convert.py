import pandas as pd
import json

def format_mbpp_prompt(row):
    """将 LiveCodeBench 题目格式化为对话形式的 prompt"""
    question_content = row['prompt']
    test_list = "\n".join(row['test_list'])
    test_imports = "\n".join(row['test_imports'])
    
    # 构造提示词内容
    prompt_content = f"""{question_content} Your code should satisfy these tests:

{test_list}"""

    # 如果有 test imports，则追加到提示词中
    if test_imports and test_imports.strip():
        prompt_content += f"""

The following imports are needed to solve the problem:
```python
{test_imports}
```"""

    prompt_content += """

Please write Python code to solve this problem. Think step by step, and wrap your final answer in '```python ```'."""

    return [{"role": "user", "content": prompt_content}]


def process_ground_truth(x):
    """处理 test_list 确保格式正确"""
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
input_path = '/data/open_datasets/mbpp/sanitized/test-00000-of-00001.parquet'
df = pd.read_parquet(input_path)

# 2. 创建 reward_model 列，ground_truth 存放 public_test_cases
df['reward_model'] = df['test_list'].apply(
    lambda x: {'ground_truth': process_ground_truth(x)}
)

# 3. 创建 prompt 列
df['prompt_original'] = df['prompt']
df['prompt'] = df.apply(format_mbpp_prompt, axis=1)

# 4. 标记数据来源
df['data_source'] = 'mbpp'

# 5. 选择需要保留的列并保存
# 必须保留的列：prompt, reward_model, data_source
# 可选保留原数据集中的其他列以供后续分析或调试
output_columns = [
    'prompt',
    'reward_model',
    'data_source',
    'source_file',
    'task_id',
    'prompt_original',
    'code',
    'test_imports',
    'test_list',
]

output_path = '/data/open_datasets/mbpp/sanitized/processed/test.parquet'
df[output_columns].to_parquet(output_path, index=False)

# 6. 验证输出
print(f"处理了 {len(df)} 条数据")
print("示例 reward_model:", df.iloc[0]['reward_model'])
print("示例 prompt:", df.iloc[0]['prompt'])