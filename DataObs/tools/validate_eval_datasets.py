#!/usr/bin/env python3
"""Validate eval datasets used by scripts/eval.sh.

The script checks path/schema/reward-function compatibility and records prompt
examples so we can see what each dataset asks the model to do.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class EvalSpec:
    name: str
    aliases: tuple[str, ...]
    path: str | None
    reward_file: str | None
    reward_name: str
    calc_maj: bool
    task_type: str
    expected_answer_format: str
    prompt_key: str = "prompt"
    data_source_key: str = "data_source"
    reward_model_key: str = "reward_model"


SPECS: list[EvalSpec] = [
    EvalSpec(
        "arc-challenge",
        ("arc-challenge", "ai2_arc", "ai2-arc"),
        "/data/open_datasets/ai2_arc/ARC-Challenge/test-processed.parquet",
        "verl/utils/reward_score/multiple_choice.py",
        "compute_score",
        True,
        "multiple_choice",
        "Final answer should be an option label like (A), (B), (C), or (D).",
    ),
    EvalSpec(
        "aqua_rat",
        ("aqua_rat", "aqua-rat"),
        "/data/open_datasets/aqua_rat/processed/test-processed.parquet",
        "verl/utils/reward_score/multiple_choice.py",
        "compute_score",
        True,
        "multiple_choice_math",
        "Final answer should be an option label like (A), (B), (C), (D), or (E).",
    ),
    EvalSpec(
        "commonsenseQA",
        ("commonsenseQA",),
        "/data/open_datasets/CommonsenseQA/data/validation-processed.parquet",
        "verl/utils/reward_score/multiple_choice.py",
        "compute_score",
        True,
        "multiple_choice",
        "Final answer should be an option label like (A), (B), (C), (D), or (E).",
    ),
    EvalSpec(
        "gsm8k",
        ("gsm8k",),
        "/data/open_datasets/GSM8K/test.parquet",
        "verl/utils/reward_score/gsm8k.py",
        "compute_score",
        True,
        "math_word_problem",
        'Final answer should include the numeric answer; prompts ask to output after "####".',
    ),
    EvalSpec(
        "livecodebench",
        ("livecodebench",),
        "/data/open_datasets/livecodebench_code_gen_lite/processed/test_v1.parquet",
        "verl/utils/reward_score/livecodebench.py",
        "compute_score",
        False,
        "code_generation",
        "Final answer should be Python code in a markdown code block.",
    ),
    EvalSpec(
        "humaneval",
        ("humaneval",),
        "/data/open_datasets/humaneval/openai_humaneval/processed/test.parquet",
        "verl/utils/reward_score/mbpp.py",
        "compute_score",
        False,
        "code_generation",
        "Final answer should be Python code in a markdown code block.",
    ),
    EvalSpec(
        "humanevalplus",
        ("humanevalplus", "human-eval-plus"),
        "/data/open_datasets/humanevalplus/processed/test.parquet",
        "verl/utils/reward_score/mbpp.py",
        "compute_score",
        False,
        "code_generation",
        "Final answer should be Python code in a markdown code block.",
    ),
    EvalSpec(
        "math",
        ("math",),
        "/data/open_datasets/MATH/train_processed.parquet",
        "verl/utils/reward_score/math_verify.py",
        "compute_score",
        True,
        "competition_math",
        r"Final answer should be wrapped in \boxed{}.",
    ),
    EvalSpec(
        "math-500",
        ("math-500",),
        "/data/open_datasets/MATH-500/test-processed.parquet",
        "verl/utils/reward_score/math_verify.py",
        "compute_score",
        True,
        "competition_math",
        r"Final answer should be wrapped in \boxed{}.",
    ),
    EvalSpec(
        "mbpp",
        ("mbpp",),
        "/data/open_datasets/mbpp/sanitized/processed/test.parquet",
        "verl/utils/reward_score/mbpp.py",
        "compute_score",
        False,
        "code_generation",
        "Final answer should be Python code in a markdown code block.",
    ),
    EvalSpec(
        "mbppplus",
        ("mbppplus", "mbpp-plus"),
        "/data/open_datasets/mbppplus/processed/test.parquet",
        "verl/utils/reward_score/mbpp.py",
        "compute_score",
        False,
        "code_generation",
        "Final answer should be Python code in a markdown code block.",
    ),
    EvalSpec(
        "numinamath",
        ("numinamath", "numinamath-CoT"),
        "/data/open_datasets/NuminaMath-CoT/test-processed.parquet",
        "verl/utils/reward_score/math_verify.py",
        "compute_score",
        True,
        "competition_math",
        r"Final answer should be wrapped in \boxed{}.",
    ),
    EvalSpec(
        "strategyQA",
        ("strategyQA",),
        "/data/open_datasets/StrategyQA/data/test-processed.parquet",
        "verl/utils/reward_score/truefalse.py",
        "compute_score",
        True,
        "true_false",
        'Final answer should contain "true" or "false".',
    ),
    EvalSpec(
        "bfcl",
        ("bfcl",),
        None,
        None,
        "",
        False,
        "function_calling",
        "BFCL uses its own bfcl_eval generate/evaluate path, not parquet + reward_model.",
    ),
]


def to_builtin(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict, list)):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    return value


def compact(value: Any, limit: int = 700) -> str:
    value = to_builtin(value)
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = text.replace("\n", "\\n")
    return text[:limit] + ("..." if len(text) > limit else "")


def prompt_text(prompt: Any) -> str:
    prompt = to_builtin(prompt)
    if isinstance(prompt, list):
        parts = []
        for msg in prompt:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(msg))
        return "\n".join(parts)
    return str(prompt)


def load_reward(path: str, name: str):
    full = REPO_ROOT / path
    spec = importlib.util.spec_from_file_location(f"validate_reward_{full.stem}", full)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load reward module: {full}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name), full


def candidate_for(spec: EvalSpec, ground_truth: Any) -> str:
    gt = to_builtin(ground_truth)
    if spec.task_type.startswith("multiple_choice"):
        return f"The answer is ({gt})."
    if spec.task_type == "true_false":
        return str(gt).lower()
    if spec.task_type in {"competition_math", "math_word_problem"}:
        return f"The final answer is \\boxed{{{gt}}}. #### {gt}"
    if spec.task_type == "code_generation":
        return "```python\npass\n```"
    return str(gt)


def validate_spec(spec: EvalSpec, sample_idx: int, test_reward: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset": spec.name,
        "aliases": list(spec.aliases),
        "path": spec.path,
        "task_type": spec.task_type,
        "expected_answer_format": spec.expected_answer_format,
        "calc_maj": spec.calc_maj,
        "status": "unknown",
        "errors": [],
        "warnings": [],
    }

    if spec.name == "bfcl":
        result["status"] = "bfcl_separate_path"
        result["can_run_eval_sh"] = True
        result["eval_command"] = "BASE_MODEL=/path/to/model MODEL_NAME=name bash scripts/eval.sh bfcl 0 /path/to/model"
        return result

    assert spec.path is not None
    path = Path(spec.path)
    if not path.exists():
        result["status"] = "missing_data"
        result["errors"].append(f"Missing data file: {path}")
        return result

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        result["status"] = "read_failed"
        result["errors"].append(f"read_parquet failed: {exc}")
        return result

    result["num_rows"] = int(len(df))
    result["columns"] = list(map(str, df.columns))
    required = [spec.prompt_key, spec.data_source_key, spec.reward_model_key]
    missing = [x for x in required if x not in df.columns]
    if missing:
        result["errors"].append(f"Missing columns: {missing}")

    if len(df) == 0:
        result["errors"].append("Dataset is empty")
        result["status"] = "invalid"
        return result

    idx = min(sample_idx, len(df) - 1)
    row = df.iloc[idx].to_dict()
    reward_model = to_builtin(row.get(spec.reward_model_key, {}))
    ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
    prompt = row.get(spec.prompt_key)

    result["sample_index"] = idx
    result["data_source_sample"] = to_builtin(row.get(spec.data_source_key))
    result["reward_model_sample"] = reward_model
    result["ground_truth_sample"] = ground_truth
    result["prompt_sample"] = prompt_text(prompt)
    result["prompt_sample_compact"] = compact(prompt_text(prompt), 900)
    result["chat_message_prompt"] = isinstance(to_builtin(prompt), list)

    if ground_truth is None:
        result["errors"].append("reward_model.ground_truth is missing in sample")
    if not result["chat_message_prompt"]:
        result["warnings"].append("Prompt is not chat-message list; generation may still work for no_chat models only.")

    if spec.reward_file is None:
        result["errors"].append("Missing reward function spec")
    else:
        reward_path = REPO_ROOT / spec.reward_file
        result["reward_file"] = str(reward_path)
        result["reward_file_exists"] = reward_path.exists()
        if not reward_path.exists():
            result["errors"].append(f"Missing reward file: {reward_path}")
        elif test_reward:
            try:
                reward_fn, _ = load_reward(spec.reward_file, spec.reward_name)
                candidate = candidate_for(spec, ground_truth)
                result["reward_test_candidate"] = candidate
                score = reward_fn(candidate, ground_truth, method="strict")
                result["reward_test_score"] = float(score) if isinstance(score, (int, float, bool)) else score
                result["reward_test_ok"] = True
            except Exception as exc:
                result["reward_test_ok"] = False
                result["warnings"].append(f"Reward function smoke call failed: {type(exc).__name__}: {exc}")

    result["eval_command"] = (
        f"BASE_MODEL=/data/pretrain_models/Qwen2.5-0.5B-Instruct "
        f"MODEL_NAME=Qwen2.5-0.5B-{spec.name} "
        f"bash scripts/eval.sh {spec.name} 0 /data/pretrain_models/Qwen2.5-0.5B-Instruct"
    )
    result["can_run_eval_sh"] = not result["errors"]
    result["status"] = "ok" if not result["errors"] else "invalid"
    return result


def write_markdown(results: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Eval Dataset Validation Report")
    lines.append("")
    lines.append("验证范围：`scripts/eval.sh` 支持的数据集。此报告验证数据路径、schema、prompt 样例、ground truth 和 reward function 可用性。")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Dataset | Status | Rows | Task | Reward | Can Run |")
    lines.append("|---|---:|---:|---|---|---:|")
    for r in results:
        rows = r.get("num_rows", "")
        reward = "separate" if r["dataset"] == "bfcl" else ("ok" if r.get("reward_file_exists") else "missing")
        can_run = "yes" if r.get("can_run_eval_sh") else "no"
        lines.append(f"| `{r['dataset']}` | {r.get('status')} | {rows} | {r.get('task_type')} | {reward} | {can_run} |")
    lines.append("")

    lines.append("## Eval Flow")
    lines.append("")
    lines.append("普通 parquet 数据集流程：")
    lines.append("")
    lines.append("1. `scripts/eval.sh <dataset> <gpu_ids> [model_path]` 根据 dataset 选择 eval parquet 和 reward function。")
    lines.append("2. `verl.trainer.main_generation` 读取 `prompt` 列；prompt 是 chat message list，会用 tokenizer chat template 生成回答。")
    lines.append("3. 生成结果写入 `generated/responses.parquet`，新增 `responses` 列。")
    lines.append("4. `verl.trainer.main_eval` 读取 `responses` 和 `reward_model.ground_truth`，调用对应 `verl/utils/reward_score/*.py` 打分。")
    lines.append("5. 指标写入 `generated/responses_labeled.metrics.json`，逐样本标签写入 `generated/responses_labeled.json`。")
    lines.append("")
    lines.append("BFCL 不走 parquet reward function，`eval.sh bfcl ...` 会调用 `scripts/eval_bfcl_dataobs.py` 和 BFCL 自己的 generate/evaluate。")
    lines.append("")

    lines.append("## Dataset Details")
    lines.append("")
    for r in results:
        lines.append(f"### {r['dataset']}")
        lines.append("")
        lines.append(f"- status: `{r.get('status')}`")
        lines.append(f"- task_type: `{r.get('task_type')}`")
        lines.append(f"- expected_answer_format: {r.get('expected_answer_format')}")
        lines.append(f"- path: `{r.get('path')}`")
        if "num_rows" in r:
            lines.append(f"- rows: `{r.get('num_rows')}`")
            lines.append(f"- columns: `{', '.join(r.get('columns', []))}`")
            lines.append(f"- sample data_source: `{r.get('data_source_sample')}`")
            lines.append(f"- sample ground_truth: `{compact(r.get('ground_truth_sample'), 300)}`")
            lines.append(f"- reward file: `{r.get('reward_file')}`")
            if "reward_test_score" in r:
                lines.append(f"- reward smoke score: `{r.get('reward_test_score')}`")
            if r.get("warnings"):
                lines.append(f"- warnings: {'; '.join(r['warnings'])}")
            if r.get("errors"):
                lines.append(f"- errors: {'; '.join(r['errors'])}")
            lines.append("")
            lines.append("Prompt sample:")
            lines.append("")
            lines.append("```text")
            lines.append(str(r.get("prompt_sample", ""))[:2500])
            lines.append("```")
        else:
            if r.get("errors"):
                lines.append(f"- errors: {'; '.join(r['errors'])}")
        lines.append("")
        lines.append("Example command:")
        lines.append("")
        lines.append("```bash")
        lines.append(str(r.get("eval_command", "")))
        lines.append("```")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate eval datasets used by scripts/eval.sh.")
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset names or 'all'.")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--output-dir", default="DataObs/reports/eval_dataset_validation")
    parser.add_argument("--skip-reward-test", action="store_true", help="Do not call reward functions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = {x.strip() for x in args.datasets.split(",") if x.strip()} if args.datasets != "all" else None
    specs = SPECS
    if selected is not None:
        specs = [s for s in SPECS if s.name in selected or any(a in selected for a in s.aliases)]
    out_dir = Path(args.output_dir)
    results = [validate_spec(s, args.sample_idx, not args.skip_reward_test) for s in specs]
    write_json = out_dir / "eval_dataset_validation.json"
    write_json.parent.mkdir(parents=True, exist_ok=True)
    write_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(results, out_dir / "eval_dataset_validation.md")

    ok = sum(1 for r in results if r.get("can_run_eval_sh"))
    print(f"Validated {len(results)} datasets; can_run_eval_sh={ok}/{len(results)}")
    print(f"JSON: {write_json}")
    print(f"Markdown: {out_dir / 'eval_dataset_validation.md'}")
    for r in results:
        print(f"{r['dataset']}: {r.get('status')}")


if __name__ == "__main__":
    main()
