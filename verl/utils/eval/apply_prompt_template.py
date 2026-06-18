from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Mapping, Optional

import pandas as pd
import numpy as np
from copy import deepcopy

try:
    from verl.utils.eval.prompt_templates.aqua_rat import aqua_rat_plain, aqua_rat_zeroshot
    from verl.utils.eval.prompt_templates.arc_challenge import arc_challenge_plain, arc_challenge_zeroshot, CHOICE_3, CHOICE_4, CHOICE_5
    from verl.utils.eval.prompt_templates.commonsenseqa import commonsenseqa_plain, commonsenseqa_zeroshot, commonsenseqa_7_shot
    from verl.utils.eval.prompt_templates.gsm8k import gsm8k_plain, gsm8k_zeroshot, gsm8k_4_shot
    from verl.utils.eval.prompt_templates.humaneval import humaneval_plain, humaneval_zeroshot
    from verl.utils.eval.prompt_templates.math import math_plain, math_zeroshot, math_4_shot
    from verl.utils.eval.prompt_templates.mbpp import mbpp_plain, mbpp_zeroshot, mbpp_4_shot
    from verl.utils.eval.prompt_templates.strategyqa import strategyqa_plain, strategyqa_zeroshot, strategyqa_6_shot
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prompt_templates.aqua_rat import aqua_rat_plain, aqua_rat_zeroshot
    from prompt_templates.arc_challenge import arc_challenge_plain, arc_challenge_zeroshot, CHOICE_3, CHOICE_4, CHOICE_5
    from prompt_templates.commonsenseqa import commonsenseqa_plain, commonsenseqa_zeroshot, commonsenseqa_7_shot
    from prompt_templates.gsm8k import gsm8k_plain, gsm8k_zeroshot, gsm8k_4_shot
    from prompt_templates.humaneval import humaneval_plain, humaneval_zeroshot
    from prompt_templates.math import math_plain, math_zeroshot, math_4_shot
    from prompt_templates.mbpp import mbpp_plain, mbpp_zeroshot, mbpp_4_shot
    from prompt_templates.strategyqa import strategyqa_plain, strategyqa_zeroshot, strategyqa_6_shot


PreprocessFn = Callable[[pd.DataFrame, str], pd.DataFrame]


_METHODS = ['plain', 'zeroshot', 'fewshot']
_LETTER_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _normalize_method(method: str) -> str:
    normed_method = method.strip().lower()
    assert normed_method in _METHODS
    return normed_method


def _normalize_dataset_name(dataset_name: str) -> str:
    """
    Normalize dataset name for registry lookup.
    Supports plain names (e.g. "CommonsenseQA").
    """
    raw_name = dataset_name.strip().lower()
    if not raw_name:
        raise ValueError("dataset_name must not be empty.")
    return raw_name


def _is_missing_scalar(value: Any) -> bool:
    return isinstance(value, float) and pd.isna(value)


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _deepcopy_template_with_last_message(template: List[Dict], **format_kwargs: Any) -> List[Dict]:
    prompt = deepcopy(template[0 : len(template)-1])
    last_user = deepcopy(template[-1])
    last_user['content'] = last_user['content'].format(**format_kwargs)
    prompt.append(last_user)
    return prompt


def _get_column(row: pd.Series, key: str) -> Any:
    if key in row and not _is_missing_scalar(row[key]):
        return row[key]
    raise KeyError(f"Column \"{key}\" is missing from the dataframe.")


def _select_template(method: str, plain: List[Dict], zeroshot: List[Dict], fewshot: Optional[List[Dict]] = None) -> List[Dict]:
    normed_method = _normalize_method(method)
    if normed_method == 'plain':
        return plain
    if normed_method == 'zeroshot':
        return zeroshot
    if fewshot is None:
        raise ValueError("fewshot method is not supported for this dataset.")
    return fewshot


def _format_test_list(test_list: Any) -> str:
    return "\n".join(str(test) for test in _as_list(test_list))


def _normalize_choice_label(label: Any) -> str:
    label = str(label).strip().upper()
    if label.isdigit():
        index = int(label) - 1
        if 0 <= index < len(_LETTER_LABELS):
            return _LETTER_LABELS[index]
    return label


def _extract_choices(choices: Any) -> Dict[str, str]:
    if not isinstance(choices, Mapping):
        raise TypeError("Column \"choices\" is not of type \"dict\". Please check the parquet file.")

    choice_texts = _as_list(choices['text'])
    choice_labels = [_normalize_choice_label(label) for label in _as_list(choices['label'])]
    return {
        label: str(text).strip()
        for label, text in zip(choice_labels, choice_texts)
    }


def _extract_boxed_answer(solution: Any) -> str:
    solution = str(solution)
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution)
    if matches:
        return matches[-1].strip()
    return solution.strip()


def _extract_gsm8k_answer(answer: Any) -> str:
    answer = str(answer)
    if "####" in answer:
        return answer.split("####")[-1].replace(",", "").strip()
    return answer.replace(",", "").strip()


def _aqua_rat(df: pd.DataFrame, method: str) -> pd.DataFrame:
    template = _select_template(method, aqua_rat_plain, aqua_rat_zeroshot)

    def _fill_template(row: pd.Series) -> List[Dict]:
        question = _get_column(row, 'question')
        options = _as_list(_get_column(row, 'options'))
        options_str = "\n".join(options)
        return _deepcopy_template_with_last_message(
            template,
            question=question,
            options=options_str,
        )

    df = df.copy()
    df['reward_model'] = df['correct'].apply(lambda x: {
        'ground_truth': str(x).strip().upper(),
    })
    df['prompt'] = df.apply(_fill_template, axis=1)
    df['data_source'] = 'aqua_rat'
    return df[['prompt', 'reward_model', 'data_source']]


def _arc_challenge(df: pd.DataFrame, method: str) -> pd.DataFrame:
    template = _select_template(method, arc_challenge_plain, arc_challenge_zeroshot)
    df = df.copy()

    def _validate_four_choices(row: pd.Series) -> bool:
        options = _extract_choices(_get_column(row, 'choices'))
        return (all(label in options for label in ['A', 'B', 'C']) and len(options) == 3) or (all(label in options for label in ['A', 'B', 'C', 'D']) and len(options) == 4) or (all(label in options for label in ['A', 'B', 'C', 'D', 'E']) and len(options) == 5)

    invalid_mask = ~df.apply(_validate_four_choices, axis=1)
    if invalid_mask.any():
        raise ValueError(
            "arc_challenge templates support 3-5 choices (A-E). "
            f"Found {int(invalid_mask.sum())} row(s) with a different choice set."
        )
    
    def _deepcopy_choices_template(template: str, **format_kwargs: Any):
        choices = deepcopy(template)
        choices = choices.format(**format_kwargs)
        return choices

    def _fill_template(row: pd.Series) -> List[Dict]:
        options = _extract_choices(_get_column(row, 'choices'))

        if all(label in options for label in ['A', 'B', 'C']) and len(options) == 3:
            choices = _deepcopy_choices_template(CHOICE_3, textA=options['A'], textB=options['B'], textC=options['C'])
        elif all(label in options for label in ['A', 'B', 'C', 'D']) and len(options) == 4:
            choices = _deepcopy_choices_template(CHOICE_4, textA=options['A'], textB=options['B'], textC=options['C'], textD=options['D'])
        elif all(label in options for label in ['A', 'B', 'C', 'D', 'E']) and len(options) == 5:
            choices = _deepcopy_choices_template(CHOICE_5, textA=options['A'], textB=options['B'], textC=options['C'], textD=options['D'], textE=options['E'])

        return _deepcopy_template_with_last_message(
            template,
            question=_get_column(row, 'question'),
            choices=choices
        )

    df['reward_model'] = df['answerKey'].apply(lambda x: {
        'ground_truth': _normalize_choice_label(x),
    })
    df['prompt'] = df.apply(_fill_template, axis=1)
    df['data_source'] = 'arc_challenge'
    return df[['prompt', 'reward_model', 'data_source']]


def _gsm8k(df: pd.DataFrame, method: str) -> pd.DataFrame:
    template = _select_template(method, gsm8k_plain, gsm8k_zeroshot, gsm8k_4_shot)
    df = df.copy()

    def _fill_template(row: pd.Series) -> List[Dict]:
        question = _get_column(row, 'question')
        return _deepcopy_template_with_last_message(template, question=question)

    def _reward_model(row: pd.Series) -> Dict[str, str]:
        ground_truth = _extract_gsm8k_answer(_get_column(row, 'answer'))
        return {'ground_truth': str(ground_truth).strip()}

    df['reward_model'] = df.apply(_reward_model, axis=1)
    df['prompt'] = df.apply(_fill_template, axis=1)
    df['data_source'] = 'gsm8k'
    return df[['prompt', 'reward_model', 'data_source']]


def _math_with_source(df: pd.DataFrame, method: str, data_source: str) -> pd.DataFrame:
    template = _select_template(method, math_plain, math_zeroshot, math_4_shot)
    df = df.copy()

    def _fill_template(row: pd.Series) -> List[Dict]:
        problem = _get_column(row, 'problem')
        return _deepcopy_template_with_last_message(template, problem=problem)

    def _reward_model(row: pd.Series) -> Dict[str, str]:
        if 'answer' in row and not _is_missing_scalar(row['answer']):
            ground_truth = row['answer']
        else:
            ground_truth = _extract_boxed_answer(_get_column(row, 'solution'))
        return {'ground_truth': str(ground_truth).strip()}

    df['reward_model'] = df.apply(_reward_model, axis=1)
    df['prompt'] = df.apply(_fill_template, axis=1)
    df['data_source'] = data_source
    return df[['prompt', 'reward_model', 'data_source']]


def _math(df: pd.DataFrame, method: str) -> pd.DataFrame:
    return _math_with_source(df, method, 'math')


def _math_500(df: pd.DataFrame, method: str) -> pd.DataFrame:
    return _math_with_source(df, method, 'math-500')


def _numinamath(df: pd.DataFrame, method: str) -> pd.DataFrame:
    return _math_with_source(df, method, 'numinamath')


def _humaneval_with_source(df: pd.DataFrame, method: str, data_source: str) -> pd.DataFrame:
    template = _select_template(method, humaneval_plain, humaneval_zeroshot)
    df = df.copy()

    def _fill_template(row: pd.Series) -> List[Dict]:
        prompt = _get_column(row, 'prompt')
        return _deepcopy_template_with_last_message(template, prompt=prompt)

    def _reward_model(row: pd.Series) -> Dict[str, Any]:
        test = _get_column(row, 'test')
        entry_point = _get_column(row, 'entry_point')
        ground_truth = [f"{test}\n\ncheck({entry_point})"]
        return {'ground_truth': ground_truth}

    df['reward_model'] = df.apply(_reward_model, axis=1)
    df['prompt'] = df.apply(_fill_template, axis=1)
    df['data_source'] = data_source
    return df[['prompt', 'reward_model', 'data_source']]


def _humaneval(df: pd.DataFrame, method: str) -> pd.DataFrame:
    return _humaneval_with_source(df, method, 'humaneval')


def _humanevalplus(df: pd.DataFrame, method: str) -> pd.DataFrame:
    return _humaneval_with_source(df, method, 'humanevalplus')


def _mbpp_with_source(df: pd.DataFrame, method: str, data_source: str) -> pd.DataFrame:
    template = _select_template(method, mbpp_plain, mbpp_zeroshot, mbpp_4_shot)
    df = df.copy()

    def _fill_template(row: pd.Series) -> List[Dict]:
        text = _get_column(row, 'text') if 'text' in row else _get_column(row, 'prompt')
        test_list = _format_test_list(_get_column(row, 'test_list'))
        return _deepcopy_template_with_last_message(template, text=text, test_list=test_list)

    def _reward_model(row: pd.Series) -> Dict[str, Any]:
        if 'test' in row and not _is_missing_scalar(row['test']):
            ground_truth = [row['test']]
        else:
            ground_truth = _as_list(_get_column(row, 'test_list'))
        return {'ground_truth': ground_truth}

    df['reward_model'] = df.apply(_reward_model, axis=1)
    df['prompt'] = df.apply(_fill_template, axis=1)
    df['data_source'] = data_source
    return df[['prompt', 'reward_model', 'data_source']]


def _mbpp(df: pd.DataFrame, method: str) -> pd.DataFrame:
    return _mbpp_with_source(df, method, 'mbpp')


def _mbppplus(df: pd.DataFrame, method: str) -> pd.DataFrame:
    return _mbpp_with_source(df, method, 'mbppplus')


def _strategyqa(df: pd.DataFrame, method: str) -> pd.DataFrame:
    template = _select_template(method, strategyqa_plain, strategyqa_zeroshot, strategyqa_6_shot)
    df = df.copy()

    def _fill_template(row: pd.Series) -> List[Dict]:
        return _deepcopy_template_with_last_message(template, question=_get_column(row, 'question'))

    df['reward_model'] = df['answer'].apply(lambda x: {
        'ground_truth': bool(x),
    })
    df['prompt'] = df.apply(_fill_template, axis=1)
    df['data_source'] = 'strategyqa'
    return df[['prompt', 'reward_model', 'data_source']]


def _commonsenseqa(df: pd.DataFrame, method: str) -> pd.DataFrame:
    template = _select_template(method, commonsenseqa_plain, commonsenseqa_zeroshot, commonsenseqa_7_shot)
    df = df.copy()
    
    def _fill_template(row: pd.Series) -> List[Dict]:
        question = _get_column(row, 'question')
        options = _extract_choices(_get_column(row, 'choices'))
        return _deepcopy_template_with_last_message(
            template,
            question=question,
            A=options['A'],
            B=options['B'],
            C=options['C'],
            D=options['D'],
            E=options['E'],
        )

    df['reward_model'] = df['answerKey'].apply(lambda x: {
        'ground_truth': _normalize_choice_label(x),
    })
    df['prompt'] = df.apply(_fill_template, axis=1)
    df['data_source'] = 'commonsenseqa'

    return df[['prompt', 'reward_model', 'data_source']]


PROMPT_TEMPLATE: Dict[str, PreprocessFn] = {
    "aqua_rat": _aqua_rat,
    "aqua-rat": _aqua_rat,
    "arc-challenge": _arc_challenge,
    "arc_challenge": _arc_challenge,
    "ai2_arc": _arc_challenge,
    "commonsenseqa": _commonsenseqa,
    "gsm8k": _gsm8k,
    "humaneval": _humaneval,
    "humanevalplus": _humanevalplus,
    "human-eval-plus": _humanevalplus,
    "math": _math,
    "math-500": _math_500,
    "mbpp": _mbpp,
    "mbppplus": _mbppplus,
    "mbpp-plus": _mbppplus,
    "numinamath": _numinamath,
    "numinamath-cot": _numinamath,
    "strategyqa": _strategyqa,
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
