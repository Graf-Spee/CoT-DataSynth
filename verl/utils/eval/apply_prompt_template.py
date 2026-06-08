from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd
import numpy as np
from copy import deepcopy

from verl.utils.eval.prompt_templates.commonsenseqa import commonsenseqa_plain, commonsenseqa_zeroshot, commonsenseqa_7_shot


PreprocessFn = Callable[[pd.DataFrame, str], pd.DataFrame]


def _normalize_dataset_name(dataset_name: str) -> str:
    """
    Normalize dataset name for registry lookup.
    Supports plain names (e.g. "CommonsenseQA").
    """
    raw_name = dataset_name.strip().lower()
    if not raw_name:
        raise ValueError("dataset_name must not be empty.")
    return raw_name


def _commonsenseqa(df: pd.DataFrame, method: str) -> pd.DataFrame:
    normed_method = method.strip().lower()
    assert normed_method in ['plain', 'zeroshot', 'fewshot']

    template = None
    if normed_method == 'plain':
        template = commonsenseqa_plain
    elif normed_method == 'zeroshot':
        template = commonsenseqa_zeroshot
    else:
        template = commonsenseqa_7_shot
    
    def _fill_template(row: pd.Series) -> List[Dict]:
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
            
        prompt = template[0 : len(template)-1]
        last_user = deepcopy(template[-1])
        last_user['content'] = last_user['content'].format(question=question, A=options_lines['A'], B=options_lines['B'], C=options_lines['C'], D=options_lines['D'], E=options_lines['E'])
        prompt.append(last_user)

        return prompt

    df['reward_model'] = df['answerKey'].apply(lambda x: {
        'ground_truth': str(x).strip().upper(),
    })
    df['prompt'] = df.apply(_fill_template, axis=1)
    df['data_source'] = 'CommonsenseQA'

    return df[['prompt', 'reward_model', 'data_source']]


PROMPT_TEMPLATE: Dict[str, PreprocessFn] = {
    "commonsenseqa": _commonsenseqa,
    # "dataset_name": _preprocess_xxx,
}


def apply_prompt_template(dataset_name: str, method: str, input_parquet_path: str, output_parquet_path: str) -> None:
    """
    Read parquet -> apply dataset-specific prompt template -> write parquet.
    Returns the processed dataframe.
    """
    dataset_key = _normalize_dataset_name(dataset_name)
    preprocess_fn = PROMPT_TEMPLATE.get(dataset_key)
    if preprocess_fn is None:
        supported = ", ".join(sorted(PROMPT_TEMPLATE.keys()))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}")

    input_path = Path(input_parquet_path).expanduser()
    if input_path.suffix != ".parquet":
        raise ValueError(f"Input path must be a parquet file, got: {input_parquet_path}")

    output_path = Path(output_parquet_path).expanduser()
    if output_path.suffix != ".parquet":
        output_path = output_path.with_suffix(".parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_parquet(input_path)
    processed_dataframe = preprocess_fn(dataframe, method)
    processed_dataframe.to_parquet(output_path, index=False)
    
    return
