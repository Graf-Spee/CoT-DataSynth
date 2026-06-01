import argparse
import os
import re
from typing import Any

import pandas as pd


DEFAULT_PATHS = {
    "humaneval": {
        "input": "/data/open_datasets/humaneval/openai_humaneval/test-00000-of-00001.parquet",
        "output": "/data/open_datasets/humaneval/openai_humaneval/processed/test.parquet",
        "data_source": "humaneval",
    },
    "humanevalplus": {
        "input": "/data/open_datasets/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet",
        "output": "/data/open_datasets/humanevalplus/processed/test.parquet",
        "data_source": "humanevalplus",
    },
    "mbppplus": {
        "input": "/data/open_datasets/mbppplus/data/test-00000-of-00001-d5781c9c51e02795.parquet",
        "output": "/data/open_datasets/mbppplus/processed/test.parquet",
        "data_source": "mbppplus",
    },
}


def _to_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist") and not isinstance(value, (str, dict, list)):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def format_humaneval_prompt(row: pd.Series) -> list[dict[str, str]]:
    code_prompt = str(row["prompt"]).rstrip()
    entry_point = str(row["entry_point"]).strip()

    prompt_content = f"""Complete the following Python function.

Function to complete: `{entry_point}`

```python
{code_prompt}
```

Please write Python code to solve this problem. Think step by step, and wrap your final answer in '```python ```'."""

    return [{"role": "user", "content": prompt_content}]


def build_humaneval_ground_truth(row: pd.Series) -> dict[str, list[str]]:
    test_code = str(row["test"])
    entry_point = str(row["entry_point"]).strip()

    if entry_point:
        pattern = re.compile(rf"check\\s*\\(\\s*{re.escape(entry_point)}\\s*\\)")
        if pattern.search(test_code) is None:
            test_code = f"{test_code}\n\ncheck({entry_point})"

    return {"ground_truth": [test_code]}


def format_mbppplus_prompt(row: pd.Series) -> list[dict[str, str]]:
    question_content = str(row["prompt"]).strip()

    test_list = [str(x) for x in _to_list(row.get("test_list")) if str(x).strip()]
    test_imports = [str(x) for x in _to_list(row.get("test_imports")) if str(x).strip()]
    test_list_text = "\n".join(test_list)
    test_imports_text = "\n".join(test_imports)

    prompt_content = f"""{question_content} Your code should satisfy these tests:

{test_list_text}"""

    if test_imports:
        prompt_content += f"""

The following imports may be useful:
```python
{test_imports_text}
```"""

    prompt_content += """

Please write Python code to solve this problem. Think step by step, and wrap your final answer in '```python ```'."""

    return [{"role": "user", "content": prompt_content}]


def build_mbppplus_ground_truth(row: pd.Series) -> dict[str, list[str]]:
    full_test_script = str(row.get("test", "")).strip()
    if full_test_script:
        return {"ground_truth": [full_test_script]}

    test_list = [str(x) for x in _to_list(row.get("test_list")) if str(x).strip()]
    return {"ground_truth": test_list}


def process_humaneval(input_path: str, output_path: str, data_source: str) -> None:
    df = pd.read_parquet(input_path)

    df["reward_model"] = df.apply(build_humaneval_ground_truth, axis=1)
    df["prompt_original"] = df["prompt"]
    df["prompt"] = df.apply(format_humaneval_prompt, axis=1)
    df["data_source"] = data_source

    output_columns = [
        "prompt",
        "reward_model",
        "data_source",
        "task_id",
        "prompt_original",
        "canonical_solution",
        "test",
        "entry_point",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[output_columns].to_parquet(output_path, index=False)

    print(f"[{data_source}] processed rows: {len(df)}")
    print(f"[{data_source}] output: {output_path}")
    print(f"[{data_source}] sample reward_model: {df.iloc[0]['reward_model']}")


def process_mbppplus(input_path: str, output_path: str, data_source: str) -> None:
    df = pd.read_parquet(input_path)

    df["reward_model"] = df.apply(build_mbppplus_ground_truth, axis=1)
    df["prompt_original"] = df["prompt"]
    df["prompt"] = df.apply(format_mbppplus_prompt, axis=1)
    df["data_source"] = data_source

    output_columns = [
        "prompt",
        "reward_model",
        "data_source",
        "task_id",
        "prompt_original",
        "code",
        "source_file",
        "test_imports",
        "test_list",
        "test",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[output_columns].to_parquet(output_path, index=False)

    print(f"[{data_source}] processed rows: {len(df)}")
    print(f"[{data_source}] output: {output_path}")
    print(f"[{data_source}] sample reward_model: {df.iloc[0]['reward_model']}")


def process_dataset(dataset_name: str, input_path: str | None, output_path: str | None) -> None:
    cfg = DEFAULT_PATHS[dataset_name]
    src = input_path or cfg["input"]
    dst = output_path or cfg["output"]
    data_source = cfg["data_source"]

    if dataset_name in {"humaneval", "humanevalplus"}:
        process_humaneval(src, dst, data_source)
    elif dataset_name == "mbppplus":
        process_mbppplus(src, dst, data_source)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HumanEval/HumanEvalPlus/MBPPPlus to eval-ready parquet")
    parser.add_argument(
        "--dataset",
        choices=["humaneval", "humanevalplus", "mbppplus", "all"],
        default="all",
        help="Dataset to process",
    )
    parser.add_argument("--input", default=None, help="Optional custom input path (single dataset mode only)")
    parser.add_argument("--output", default=None, help="Optional custom output path (single dataset mode only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset == "all":
        if args.input or args.output:
            raise ValueError("--input/--output are only supported when --dataset is a single dataset")
        for name in ["humaneval", "humanevalplus", "mbppplus"]:
            process_dataset(name, None, None)
    else:
        process_dataset(args.dataset, args.input, args.output)


if __name__ == "__main__":
    main()
