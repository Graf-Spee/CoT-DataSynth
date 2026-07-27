#!/usr/bin/env python3
"""Prepare distill parquet with clean raw prompts for DataObs distillation."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd

from DataObs.lib.data_process.cot_distill_teacher_filter import (
    _format_student_question,
    _normalize_dataset_name,
    _to_builtin,
)

REQUIRED_COLUMNS = {"prompt", "reward_model", "data_source"}
_LETTER_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _is_missing_scalar(value: Any) -> bool:
    return isinstance(value, float) and pd.isna(value)


def _get_column(row: pd.Series, key: str) -> Any:
    if key in row and not _is_missing_scalar(row[key]):
        return row[key]
    raise KeyError(f'Column "{key}" is missing from the dataframe.')


def _optional_column(row: pd.Series, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and not _is_missing_scalar(row[key]):
            return row[key]
    raise KeyError(f'None of the columns "{", ".join(keys)}" exist in the dataframe.')


def _as_list(value: Any) -> List[Any]:
    value = _to_builtin(value)
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, pd.Series):
        return value.tolist()
    return [value]


def _normalize_choice_label(label: Any) -> str:
    label = str(label).strip().upper()
    if label.isdigit():
        index = int(label) - 1
        if 0 <= index < len(_LETTER_LABELS):
            return _LETTER_LABELS[index]
    return label


def _extract_choices(choices: Any) -> Dict[str, str]:
    choices = _to_builtin(choices)
    if not isinstance(choices, Mapping):
        raise TypeError('Column "choices" is not of type "dict". Please check the parquet file.')

    choice_texts = _as_list(choices["text"])
    choice_labels = [_normalize_choice_label(label) for label in _as_list(choices["label"])]
    return {label: str(text).strip() for label, text in zip(choice_labels, choice_texts)}


def _format_choices(choices: Dict[str, str]) -> str:
    return "\n".join(f"{label}. {text}" for label, text in choices.items())


def _format_option_list(options: Any) -> str:
    return "\n".join(str(option).strip() for option in _as_list(options) if str(option).strip())


def _extract_boxed_answer(solution: Any) -> str:
    text = str(solution)
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def _extract_gsm8k_answer(answer: Any) -> str:
    text = str(answer)
    if "####" in text:
        return text.split("####")[-1].replace(",", "").strip()
    return text.replace(",", "").strip()


def _coerce_bool(answer: Any) -> bool:
    value = _to_builtin(answer)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return bool(value)


def _prompt(question: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": question.strip()}]


def _raw_question(row: pd.Series, keys: tuple[str, ...]) -> str:
    return str(_optional_column(row, keys)).strip()


def _distill_aqua_rat(row: pd.Series) -> tuple[List[Dict[str, str]], Dict[str, Any], str, str]:
    question = _raw_question(row, ("question",))
    options = _format_option_list(_get_column(row, "options"))
    raw_question = question if not options else f"{question}\n\n{options}"
    reward_model = {"ground_truth": str(_get_column(row, "correct")).strip().upper()}
    return _prompt(raw_question), reward_model, "aqua_rat", raw_question


def _distill_arc_challenge(row: pd.Series) -> tuple[List[Dict[str, str]], Dict[str, Any], str, str]:
    question = _raw_question(row, ("question",))
    options = _format_choices(_extract_choices(_get_column(row, "choices")))
    raw_question = f"{question}\n\n{options}"
    reward_model = {"ground_truth": _normalize_choice_label(_get_column(row, "answerKey"))}
    return _prompt(raw_question), reward_model, "arc_challenge", raw_question


def _distill_commonsenseqa(row: pd.Series) -> tuple[List[Dict[str, str]], Dict[str, Any], str, str]:
    question = _raw_question(row, ("question",))
    options = _format_choices(_extract_choices(_get_column(row, "choices")))
    raw_question = f"{question}\n\n{options}"
    reward_model = {"ground_truth": _normalize_choice_label(_get_column(row, "answerKey"))}
    return _prompt(raw_question), reward_model, "commonsenseqa", raw_question


def _distill_gsm8k(row: pd.Series) -> tuple[List[Dict[str, str]], Dict[str, Any], str, str]:
    question = _raw_question(row, ("question",))
    reward_model = {"ground_truth": _extract_gsm8k_answer(_get_column(row, "answer"))}
    return _prompt(question), reward_model, "gsm8k", question


def _distill_math_with_source(row: pd.Series, data_source: str) -> tuple[List[Dict[str, str]], Dict[str, Any], str, str]:
    question = _raw_question(row, ("problem",))
    if "answer" in row and not _is_missing_scalar(row["answer"]):
        ground_truth = row["answer"]
    else:
        ground_truth = _extract_boxed_answer(_get_column(row, "solution"))
    reward_model = {"ground_truth": str(ground_truth).strip()}
    return _prompt(question), reward_model, data_source, question


def _distill_strategyqa(row: pd.Series) -> tuple[List[Dict[str, str]], Dict[str, Any], str, str]:
    question = _raw_question(row, ("question",))
    reward_model = {"ground_truth": _coerce_bool(_get_column(row, "answer"))}
    return _prompt(_format_student_question("strategyqa", question)), reward_model, "strategyqa", question


def _distill_humaneval_with_source(row: pd.Series, data_source: str) -> tuple[List[Dict[str, str]], Dict[str, Any], str, str]:
    prompt = _raw_question(row, ("prompt",))
    test = _get_column(row, "test")
    entry_point = _get_column(row, "entry_point")
    reward_model = {"ground_truth": [f"{test}\n\ncheck({entry_point})"]}
    return _prompt(prompt), reward_model, data_source, prompt


def _distill_mbpp_with_source(row: pd.Series, data_source: str) -> tuple[List[Dict[str, str]], Dict[str, Any], str, str]:
    text = _raw_question(row, ("text", "prompt"))
    test_list = _format_option_list(_get_column(row, "test_list"))
    raw_question = text if not test_list else f"{text}\n\n{test_list}"
    if "test" in row and not _is_missing_scalar(row["test"]):
        ground_truth = [row["test"]]
    else:
        ground_truth = _as_list(_get_column(row, "test_list"))
    reward_model = {"ground_truth": ground_truth}
    return _prompt(raw_question), reward_model, data_source, raw_question


def _build_prepared_rows(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    dataset_key = _normalize_dataset_name(dataset)
    builder_map = {
        "aqua_rat": _distill_aqua_rat,
        "arc-challenge": _distill_arc_challenge,
        "commonsenseqa": _distill_commonsenseqa,
        "gsm8k": _distill_gsm8k,
        "humaneval": lambda row: _distill_humaneval_with_source(row, "humaneval"),
        "humanevalplus": lambda row: _distill_humaneval_with_source(row, "humanevalplus"),
        "math": lambda row: _distill_math_with_source(row, "math"),
        "math-500": lambda row: _distill_math_with_source(row, "math-500"),
        "mbpp": lambda row: _distill_mbpp_with_source(row, "mbpp"),
        "mbppplus": lambda row: _distill_mbpp_with_source(row, "mbppplus"),
        "numinamath": lambda row: _distill_math_with_source(row, "numinamath"),
        "strategyqa": _distill_strategyqa,
    }
    builder = builder_map.get(dataset_key)
    if builder is None:
        supported = ", ".join(sorted(builder_map))
        raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets: {supported}.")

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        prompt, reward_model, data_source, raw_question = builder(row)
        rows.append(
            {
                "prompt": prompt,
                "reward_model": reward_model,
                "data_source": data_source,
                "raw_question": raw_question,
            }
        )
    return pd.DataFrame(rows)


def prepare_distill_data(dataset_name: str, input_path: str, output_path: str) -> Path:
    source = Path(input_path).expanduser()
    output = Path(output_path).expanduser()
    if source.suffix != ".parquet":
        raise ValueError(f"Distill data must be a parquet file, got: {source}")
    if not source.exists():
        raise FileNotFoundError(f"Distill data not found: {source}")
    if output.suffix != ".parquet":
        output = output.with_suffix(".parquet")
    output.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_parquet(source)
    if REQUIRED_COLUMNS.issubset(dataframe.columns):
        shutil.copy2(source, output)
        return output

    prepared = _build_prepared_rows(dataframe, dataset_name)
    prepared.to_parquet(output, index=False)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare distill parquet with raw prompts.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = prepare_distill_data(args.dataset, args.input, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
