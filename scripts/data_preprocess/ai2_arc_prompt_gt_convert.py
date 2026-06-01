import pandas as pd
import numpy as np

"""
This preprocessing script can be used for ARC-Challenge and ARC-Easy.
Change the paths and the data source to use it for different datasets.
"""

def format_arc_prompt(row):
    """将ARC-Challenge问题和选项格式化为对话形式"""
    question = row['question']
    choices = row['choices']  # 字典格式: {'text': array([...]), 'label': array([...])}
    
    # 提取选项文本和标签（处理numpy array或普通列表）
    if isinstance(choices, dict):
        choice_texts = choices['text']
        choice_labels = choices['label']
        # 如果是numpy array，转换为list
        if isinstance(choice_texts, np.ndarray):
            choice_texts = choice_texts.tolist()
        if isinstance(choice_labels, np.ndarray):
            choice_labels = choice_labels.tolist()
    else:
        raise TypeError("Column \"choices\" is not of type \"dict\". Please check the parquet file.")
    
    options_lines = {}
    for label, text in zip(choice_labels, choice_texts):
        options_lines[label] = text
        
    prompt_content = f"""{question}\nA. {options_lines['A']}\nB. {options_lines['B']}\nC. {options_lines['C']}\nD. {options_lines['D']}\nE. {options_lines['E']}\nAnswer:"""

    return prompt_content

# Read Original Data
input_path = '/data/open_datasets/CommonsenseQA/data/train-00000-of-00001.parquet'
df = pd.read_parquet(input_path)

df['reward_model'] = df['answerKey'].apply(lambda x: {
    'ground_truth': str(x).strip().upper(),
})

# 创建 prompt 列
df['prompt'] = df.apply(format_arc_prompt, axis=1)

# 创建 data_source 列用于标识数据集
df['data_source'] = 'commonsenseQA'

# 保存处理后的数据（保留原始列便于调试）
output_columns = ['prompt', 'question', 'choices', 'answerKey', 'id', 'data_source', 'reward_model']
df[output_columns].to_parquet(
    '/data/open_datasets/CommonsenseQA/data/train-processed.parquet',
    index=False
)

print(f"成功处理了 {len(df)} 条 ARC-Challenge 数据")
print("\n示例数据:")
print(f"Prompt: {df.iloc[0]['prompt']}")
print(f"Reward Model: {df.iloc[0]['reward_model']}")
print(f"Answer Key: {df.iloc[0]['answerKey']}")