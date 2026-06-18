#!/usr/bin/env python3
"""Prepare eval parquet for verl generation/evaluation.

If the source parquet already has prompt/reward_model/data_source columns, it is
copied as-is. Otherwise dataset-specific prompt templates are applied.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"prompt", "reward_model", "data_source"}


def _load_apply_prompt_template():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "verl" / "utils" / "eval" / "apply_prompt_template.py"
    spec = importlib.util.spec_from_file_location("apply_prompt_template_module", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load prompt template module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply_prompt_template


def prepare_eval_data(dataset_name: str, input_path: str, output_path: str, method: str = "zeroshot") -> Path:
    source = Path(input_path).expanduser()
    output = Path(output_path).expanduser()
    if source.suffix != ".parquet":
        raise ValueError(f"Eval data must be a parquet file, got: {source}")
    if not source.exists():
        raise FileNotFoundError(f"Eval data not found: {source}")
    if output.suffix != ".parquet":
        output = output.with_suffix(".parquet")
    output.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_parquet(source)
    if REQUIRED_COLUMNS.issubset(dataframe.columns):
        shutil.copy2(source, output)
    else:
        apply_prompt_template = _load_apply_prompt_template()
        apply_prompt_template(
            dataset_name=dataset_name,
            method=method,
            input_parquet_path=str(source),
            output_parquet_path=str(output),
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare eval parquet with prompt/reward columns.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="zeroshot", choices=["plain", "zeroshot", "fewshot"])
    args = parser.parse_args()

    output = prepare_eval_data(args.dataset, args.input, args.output, args.method)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
