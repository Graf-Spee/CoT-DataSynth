#!/usr/bin/env python3
"""Experiment pipeline for distill -> SFT -> eval -> GRPO -> eval.

This is a lightweight orchestrator around the existing bash/python entrypoints.
It records manifests and commands so experiment variants can be reproduced.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

DISTILL_METHODS = (
    "teacher_correctness_filter",
    "question_rephrasing",
    "answer_augmentation",
    "question_augmentation",
    "reverse_thinking",
)

DISTILL_METHOD_SCRIPTS = {
    "teacher_correctness_filter": REPO_ROOT / "DataObs" / "lib" / "data_process" / "cot_distill_teacher_filter.py",
    "question_rephrasing": REPO_ROOT / "DataObs" / "baselines" / "heuristics" / "question_rephrasing.py",
    "answer_augmentation": REPO_ROOT / "DataObs" / "baselines" / "heuristics" / "answer_augmentation.py",
    "question_augmentation": REPO_ROOT / "DataObs" / "baselines" / "heuristics" / "question_augmentation.py",
    "reverse_thinking": REPO_ROOT / "DataObs" / "baselines" / "heuristics" / "reverse_thinking_augmentation.py",
}

DATASET_DEFAULTS: dict[str, dict[str, str]] = {
    "gsm8k": {
        "distill_input": "/data/open_datasets/GSM8K/main/train-00000-of-00001.parquet",
        "rl_train": "/data/open_datasets/GSM8K/main/train-00000-of-00001.parquet",
        "rl_val": "/data/open_datasets/GSM8K/main/test-00000-of-00001.parquet",
    },
    "math-500": {
        "distill_input": "/data/open_datasets/MATH/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet",
        "rl_train": "/data/open_datasets/MATH/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet",
        "rl_val": "/data/open_datasets/MATH-500/test.parquet",
    },
    "math": {
        "distill_input": "/data/open_datasets/MATH/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet",
        "rl_train": "/data/open_datasets/MATH/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet",
        "rl_val": "/data/open_datasets/MATH/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet",
    },
    "aqua_rat": {
        "distill_input": "/data/open_datasets/aqua_rat/raw/train-00000-of-00001.parquet",
        "rl_train": "/data/open_datasets/aqua_rat/raw/train-00000-of-00001.parquet",
        "rl_val": "/data/open_datasets/aqua_rat/raw/test-00000-of-00001.parquet",
    },
    "arc-challenge": {
        "distill_input": "/data/open_datasets/ai2_arc/ARC-Challenge/train-00000-of-00001.parquet",
        "rl_train": "/data/open_datasets/ai2_arc/ARC-Challenge/train-00000-of-00001.parquet",
        "rl_val": "/data/open_datasets/ai2_arc/ARC-Challenge/test-00000-of-00001.parquet",
    },
    "strategyQA": {
        "distill_input": "/data/open_datasets/StrategyQA/data/train-00000-of-00001-506370352f622815.parquet",
        "rl_train": "/data/open_datasets/StrategyQA/data/train-00000-of-00001-506370352f622815.parquet",
        "rl_val": "/data/open_datasets/StrategyQA/data/test-00000-of-00001-bae602f3ee37f4ca.parquet",
    },
    "commonsenseQA": {
        "distill_input": "/data/open_datasets/CommonsenseQA/data/train-00000-of-00001.parquet",
        "rl_train": "/data/open_datasets/CommonsenseQA/data/train-00000-of-00001.parquet",
        "rl_val": "/data/open_datasets/CommonsenseQA/data/validation-00000-of-00001.parquet",
    },
    "mbpp": {
        "distill_input": "/data/open_datasets/mbpp/sanitized/train-00000-of-00001.parquet",
        "rl_train": "/data/open_datasets/mbpp/sanitized/train-00000-of-00001.parquet",
        "rl_val": "/data/open_datasets/mbpp/sanitized/test-00000-of-00001.parquet",
    },
    "mbppplus": {
        "distill_input": "/data/open_datasets/mbppplus/data/test-00000-of-00001-d5781c9c51e02795.parquet",
        "rl_train": "/data/open_datasets/mbppplus/data/test-00000-of-00001-d5781c9c51e02795.parquet",
        "rl_val": "/data/open_datasets/mbppplus/data/test-00000-of-00001-d5781c9c51e02795.parquet",
    },
    "humaneval": {
        "distill_input": "/data/open_datasets/humaneval/openai_humaneval/test-00000-of-00001.parquet",
        "rl_train": "/data/open_datasets/humaneval/openai_humaneval/test-00000-of-00001.parquet",
        "rl_val": "/data/open_datasets/humaneval/openai_humaneval/test-00000-of-00001.parquet",
    },
    "humanevalplus": {
        "distill_input": "/data/open_datasets/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet",
        "rl_train": "/data/open_datasets/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet",
        "rl_val": "/data/open_datasets/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet",
    },
    "numinamath": {
        "distill_input": "/data/open_datasets/NuminaMath-CoT/data/train-00000-of-00005.parquet",
        "rl_train": "/data/open_datasets/NuminaMath-CoT/data/train-00000-of-00005.parquet",
        "rl_val": "/data/open_datasets/NuminaMath-CoT/data/test-00000-of-00001.parquet",
    },
}

MATH_STYLE_DATASETS = {"gsm8k", "math", "math-500", "numinamath"}
ORIGINAL_METAMATH_NATIVE_DATASETS = {"gsm8k"}
ORIGINAL_METAMATH_BORROWED_DATASETS = {"math", "math-500", "numinamath"}
BACKWARD_PROMPT_FAMILIES = {
    "arc-challenge": "mcq",
    "commonsenseqa": "mcq",
    "gsm8k": "math",
    "math": "math",
    "math-500": "math",
    "strategyqa": "yesno",
}


def normalize_dataset_key(name: str) -> str:
    raw = name.strip()
    key = raw.lower().replace("_", "").replace("-", "")
    if key in {"arc", "arcchallenge", "ai2arc"} or raw.lower() == "arc-":
        return "arc-challenge"
    if key == "aquarat":
        return "aqua_rat"
    if key in {"commonsenseqa", "csqa"}:
        return "commonsenseQA"
    if key == "gsm8k":
        return "gsm8k"
    if key == "humaneval":
        return "humaneval"
    if key in {"humanevalplus", "humaneval+"}:
        return "humanevalplus"
    if key == "math":
        return "math"
    if key == "math500":
        return "math-500"
    if key == "mbpp":
        return "mbpp"
    if key in {"mbppplus", "mbpp+"}:
        return "mbppplus"
    if key in {"numinamath", "numina"}:
        return "numinamath"
    if key == "strategyqa":
        return "strategyQA"
    return raw


def dataset_default(name: str, field: str) -> str:
    return DATASET_DEFAULTS.get(normalize_dataset_key(name), {}).get(field, "")


def _distill_dataset_key(name: str) -> str:
    return normalize_dataset_key(name).lower().replace("_", "-")


def answer_aug_original_prompt_status(dataset: str) -> str:
    key = _distill_dataset_key(dataset)
    if key in ORIGINAL_METAMATH_NATIVE_DATASETS:
        return "native"
    if key in ORIGINAL_METAMATH_BORROWED_DATASETS:
        return "borrowed"
    return "unsupported"


def prompt_support_for_method(method: str, dataset: str, *, use_original_metamath_prompt: bool = False) -> dict[str, Any]:
    key = _distill_dataset_key(dataset)
    group = "unknown"
    if key in {"arc-challenge", "aqua-rat", "commonsenseqa"}:
        group = "multiple_choice"
    elif key in MATH_STYLE_DATASETS:
        group = "math"
    elif key == "strategyqa":
        group = "yesno"
    elif key in {"mbpp", "mbppplus", "humaneval", "humanevalplus"}:
        group = "code"

    if method == "teacher_correctness_filter":
        source = "DataObs fallback prompt"
        if key == "strategyqa":
            source = "DataObs fallback prompt + Yes or no student cue"
        return {"support": "supported", "teacher_prompt": source, "dataset_group": group}

    if method == "answer_augmentation":
        if use_original_metamath_prompt:
            status = answer_aug_original_prompt_status(key)
            return {
                "support": "supported" if status != "unsupported" else "unsupported",
                "teacher_prompt": f"original MetaMath prompt ({status})",
                "dataset_group": group,
            }
        return {"support": "supported", "teacher_prompt": "DataObs dataset-specific prompt", "dataset_group": group}

    if method == "question_rephrasing":
        return {
            "support": "supported",
            "rephrase_prompt": "MetaMath rephrase prompt",
            "answer_prompt": "DataObs dataset-specific prompt",
            "dataset_group": group,
        }

    if method in {"question_augmentation", "reverse_thinking"}:
        family = BACKWARD_PROMPT_FAMILIES.get(key)
        if family is None:
            return {
                "support": "unsupported",
                "teacher_prompt": "no backward-question prompt",
                "dataset_group": group,
            }
        source = "original baseline prompt" if family == "mcq" else f"{family} backward-question prompt"
        return {
            "support": "supported",
            "teacher_prompt": source,
            "backward_prompt_family": family,
            "dataset_group": group,
        }

    return {"support": "unknown", "dataset_group": group}


def now() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def as_list(values: list[str] | None) -> list[str]:
    return list(values or [])


def hydra_int_override(values: list[str] | None, key: str, default: int) -> int:
    prefix = f"{key}="
    result = default
    for item in values or []:
        if not item.startswith(prefix):
            continue
        raw = item.split("=", 1)[1].strip()
        try:
            result = int(raw)
        except ValueError as exc:
            raise SystemExit(f"[ERROR] {key} must be an integer, got: {raw}") from exc
    return result


def parquet_num_rows(path: str) -> int | None:
    if not path.endswith(".parquet"):
        return None
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        pass
    try:
        import pandas as pd

        return len(pd.read_parquet(path))
    except Exception as exc:  # pragma: no cover - best-effort preflight
        print(f"[WARN] Could not inspect parquet row count for {path}: {exc}")
        return None


def parse_env_pairs(values: list[str] | None) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"[ERROR] Environment override must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        if not key:
            raise SystemExit(f"[ERROR] Empty environment variable name in: {item}")
        env[key] = value
    return env


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def _run_prepare_script(script: Path, dataset: str, input_path: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            dataset,
            "--input",
            input_path,
            "--output",
            str(output_path),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    return output_path


def prepare_parquet(dataset: str, input_path: str, output_path: Path) -> Path:
    script = REPO_ROOT / "DataObs" / "lib" / "evaluation" / "prepare_eval_data.py"
    return _run_prepare_script(script, dataset, input_path, output_path)


def prepare_distill_parquet(dataset: str, input_path: str, output_path: Path) -> Path:
    script = REPO_ROOT / "DataObs" / "lib" / "data_process" / "prepare_distill_data.py"
    return _run_prepare_script(script, dataset, input_path, output_path)


def latest_global_step(path: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for item in path.glob("global_step_*"):
        if not item.is_dir():
            continue
        suffix = item.name.removeprefix("global_step_")
        if suffix.isdigit():
            candidates.append((int(suffix), item))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def is_hf_or_lora_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "adapter_model.safetensors").is_file():
        return True
    if not (path / "config.json").is_file():
        return False
    model_files = [
        path / "model.safetensors",
        path / "pytorch_model.bin",
        path / "model.safetensors.index.json",
    ]
    return any(p.is_file() for p in model_files) or any(path.glob("*.safetensors"))


def resolve_sft_checkpoint(sft_dir: Path, fallback: Path | None = None) -> Path:
    if fallback is not None:
        return fallback
    if is_hf_or_lora_dir(sft_dir):
        return sft_dir
    checkpoint_last = sft_dir / "checkpoint-last"
    if is_hf_or_lora_dir(checkpoint_last):
        return checkpoint_last
    latest = latest_global_step(sft_dir)
    if latest and is_hf_or_lora_dir(latest):
        return latest
    return checkpoint_last


def check_input(path: str | None, label: str, required: bool, dry_run: bool) -> None:
    if not path:
        if required:
            raise SystemExit(f"[ERROR] Missing required {label}")
        return
    if not dry_run and not Path(path).exists():
        raise SystemExit(f"[ERROR] {label} not found: {path}")
    if dry_run and path.startswith("/") and not Path(path).exists():
        print(f"[WARN] {label} does not exist yet: {path}")


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.exp_dir = Path(args.output_dir).resolve() / args.experiment_id
        self.logs_dir = self.exp_dir / "logs"
        self.commands_file = self.exp_dir / "commands.jsonl"
        self.results_file = self.exp_dir / "results.json"
        self.manifest_file = self.exp_dir / "manifest.json"
        self.results: dict[str, Any] = {"distill_config": self.build_distill_config()}

    def command_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("CONFIG_DIR", str(REPO_ROOT / "config"))
        if extra:
            env.update(extra)
        return env

    def run_cmd(self, stage: str, cmd: list[str], env_updates: dict[str, str] | None = None) -> None:
        log_path = self.logs_dir / f"{stage}.log"
        entry = {
            "stage": stage,
            "time": now(),
            "cmd": cmd,
            "cmd_str": shell_join(cmd),
            "env": env_updates or {},
            "log_path": str(log_path),
            "dry_run": bool(self.args.dry_run),
        }
        append_jsonl(self.commands_file, entry)

        print(f"\n>>>>>>>> {stage}")
        if env_updates:
            print("env:", " ".join(f"{k}={v}" for k, v in env_updates.items()))
        print(shell_join(cmd))

        if self.args.dry_run:
            return

        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = self.command_env(env_updates)
        start = time.time()
        with log_path.open("w", encoding="utf-8") as fout:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=fout,
                stderr=subprocess.STDOUT,
                text=True,
            )
        elapsed = time.time() - start
        status_entry = {
            "stage": stage,
            "returncode": proc.returncode,
            "elapsed_sec": elapsed,
            "log_path": str(log_path),
        }
        append_jsonl(self.commands_file, status_entry)
        if proc.returncode != 0:
            raise SystemExit(f"[ERROR] Stage {stage} failed. See {log_path}")

    def write_manifest(self, stages: list[str]) -> None:
        manifest = {
            "experiment_id": self.args.experiment_id,
            "created_at": now(),
            "repo_root": str(REPO_ROOT),
            "stages": stages,
            "dataset": self.args.dataset,
            "base_model": self.args.base_model,
            "teacher_model": self.args.teacher_model,
            "distill_method": self.args.distill_method,
            "data_variant": self.args.data_variant,
            "reasoning_source": self.args.reasoning_source,
            "distill_config": self.build_distill_config(),
            "paths": {
                "experiment_dir": str(self.exp_dir),
                "sft_dir": str(self.sft_dir),
                "grpo_dir": str(self.grpo_dir),
                "sft_eval_dir": str(self.sft_eval_dir),
                "grpo_eval_dir": str(self.grpo_eval_dir),
            },
            "args": vars(self.args),
        }
        write_json(self.manifest_file, manifest)

    @property
    def distill_output(self) -> Path:
        return Path(self.args.distill_output) if self.args.distill_output else self.exp_dir / "distill" / "filtered_sft.parquet"

    @property
    def sft_dir(self) -> Path:
        return Path(self.args.sft_output_dir) if self.args.sft_output_dir else self.exp_dir / "sft"

    @property
    def grpo_dir(self) -> Path:
        return Path(self.args.grpo_output_dir) if self.args.grpo_output_dir else self.exp_dir / "grpo"

    @property
    def sft_eval_dir(self) -> Path:
        return Path(self.args.sft_eval_output_dir) if self.args.sft_eval_output_dir else self.exp_dir / "eval" / "sft"

    @property
    def grpo_eval_dir(self) -> Path:
        return Path(self.args.grpo_eval_output_dir) if self.args.grpo_eval_output_dir else self.exp_dir / "eval" / "grpo"

    def sft_data_path(self) -> str:
        if self.args.sft_data:
            return self.args.sft_data
        return str(self.distill_output)

    def sft_val_data_path(self) -> str:
        return self.args.sft_val_data or self.sft_data_path()

    def distill_input_path(self) -> str:
        return self.args.distill_input or dataset_default(self.args.dataset, "distill_input")

    def distill_dataset_name(self) -> str:
        return self.args.distill_dataset or self.args.dataset

    def distill_dataset_key(self) -> str:
        return _distill_dataset_key(self.distill_dataset_name())

    def distill_question_source(self) -> str:
        dataset = self.distill_dataset_key()
        if dataset == "gsm8k":
            return "raw_question"
        if dataset in {"math", "math-500", "numinamath"}:
            return "raw_problem"
        if dataset in {"arc-challenge", "commonsenseqa", "aqua-rat"}:
            return "raw_question_plus_choices"
        if dataset == "strategyqa":
            return "raw_question_plus_yesno"
        if dataset in {"humaneval", "humanevalplus"}:
            return "raw_prompt"
        if dataset in {"mbpp", "mbppplus"}:
            return "raw_text_plus_tests"
        return "raw_question"

    def answer_aug_original_prompt_supported(self) -> bool:
        return self.answer_aug_original_prompt_status() != "unsupported"

    def answer_aug_original_prompt_status(self) -> str:
        return answer_aug_original_prompt_status(self.distill_dataset_key())

    def build_distill_config(self) -> dict[str, Any]:
        method = self.args.distill_method
        method_specific: dict[str, Any] = {
            "teacher_num_samples": self.args.teacher_num_samples,
            "teacher_do_sample": bool(self.args.teacher_do_sample),
            "teacher_dtype": self.args.teacher_dtype,
        }
        if method == "teacher_correctness_filter":
            method_specific["num_cots"] = 1
        elif method == "answer_augmentation":
            method_specific["num_augmented_answers"] = self.args.teacher_num_samples
            method_specific["use_original_metamath_prompt"] = bool(self.args.answer_aug_use_original_metamath_prompt)
            method_specific["original_metamath_prompt_supported"] = self.answer_aug_original_prompt_supported()
            method_specific["original_metamath_prompt_status"] = self.answer_aug_original_prompt_status()
        elif method == "question_rephrasing":
            method_specific["num_rephrases"] = self.args.teacher_num_samples
            method_specific["num_cots_per_rephrase"] = self.args.rephrase_num_cots
        elif method == "question_augmentation":
            method_specific["num_backward_questions"] = self.args.teacher_num_samples
        elif method == "reverse_thinking":
            method_specific["num_backward_questions"] = 1
            method_specific["num_forward_cots"] = 1
            method_specific["num_backward_cots"] = 1
            method_specific["num_consistency_checks"] = 1
        return {
            "method": method,
            "dataset": self.distill_dataset_key(),
            "input_prepare_mode": "distill_raw",
            "prompt_source": "raw_dataset",
            "question_source": self.distill_question_source(),
            "prompt_support": prompt_support_for_method(
                method,
                self.distill_dataset_key(),
                use_original_metamath_prompt=bool(self.args.answer_aug_use_original_metamath_prompt),
            ),
            "eval_default": {
                "enable_thinking": False,
                "prompt_template_method": getattr(self.args, "eval_prompt_template_method", None),
            },
            "method_specific": method_specific,
        }

    def rl_train_data_path(self) -> str:
        return self.args.rl_train_data or dataset_default(self.args.dataset, "rl_train")

    def rl_val_data_path(self) -> str:
        return self.args.rl_val_data or dataset_default(self.args.dataset, "rl_val")

    def prepare_dataset_data(self, data_path: str, split_name: str, *, mode: str = "eval") -> Path:
        prepared = self.exp_dir / "prepared_data" / mode / f"{split_name}.parquet"
        if prepared.exists():
            return prepared
        if mode == "eval":
            return prepare_parquet(self.args.dataset, data_path, prepared)
        if mode == "distill":
            return prepare_distill_parquet(self.distill_dataset_key(), data_path, prepared)
        raise ValueError(f"Unknown prepare mode: {mode}")

    def rl_train_from_distill_kept_path(self) -> Path:
        return self.exp_dir / "rl_data" / "train_from_distill_kept.parquet"

    def build_rl_train_from_distill_kept(self) -> Path:
        output_path = self.rl_train_from_distill_kept_path()
        if output_path.exists():
            return output_path
        distill_path = self.distill_output
        seed_path = Path(self.distill_input_path())
        if not distill_path.exists():
            raise SystemExit(f"[ERROR] Cannot build RL train from distill kept prompts; missing distill output: {distill_path}")
        if not seed_path.exists():
            raise SystemExit(f"[ERROR] Cannot build RL train from distill kept prompts; missing distill input: {seed_path}")

        import pandas as pd

        distill_df = pd.read_parquet(distill_path)
        if "source_index" not in distill_df.columns:
            raise SystemExit(f"[ERROR] Distill output missing source_index column: {distill_path}")
        source_indices = sorted({int(x) for x in distill_df["source_index"].dropna().tolist()})
        if not source_indices:
            raise SystemExit(f"[ERROR] No kept source_index found in distill output: {distill_path}")

        prepared_seed_path = self.prepare_dataset_data(str(seed_path), "distill_seed_for_rl", mode="distill")
        seed_df = pd.read_parquet(prepared_seed_path)
        max_index = len(seed_df) - 1
        bad = [idx for idx in source_indices if idx < 0 or idx > max_index]
        if bad:
            raise SystemExit(f"[ERROR] Distill source_index out of range for {seed_path}: {bad[:5]}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        seed_df.iloc[source_indices].copy().to_parquet(output_path, index=False)
        self.results["rl_train_from_distill_kept"] = str(output_path)
        return output_path

    def prepare_rl_data(self, data_path: str, split_name: str) -> Path:
        prepared = self.exp_dir / "rl_data" / f"{split_name}_prepared.parquet"
        if prepared.exists():
            return prepared
        return prepare_parquet(self.args.dataset, data_path, prepared)

    def eval_model_name(self, suffix: str) -> str:
        return self.args.model_name or f"{self.args.experiment_id}-{suffix}"

    def stage_distill(self) -> None:
        distill_method = self.args.distill_method
        if distill_method == "teacher_correctness_filter" and self.args.teacher_num_samples != 1:
            raise SystemExit(
                "[ERROR] --distill-method teacher_correctness_filter requires "
                "--teacher-num-samples 1. Use --distill-method answer_augmentation "
                "for multi-sample teacher answer baselines."
            )
        if distill_method == "reverse_thinking" and self.args.teacher_num_samples != 1:
            raise SystemExit(
                "[ERROR] --distill-method reverse_thinking does not support "
                "--teacher-num-samples > 1 yet. Use 1 to keep the current single-path setting."
            )
        support_info = prompt_support_for_method(
            distill_method,
            self.distill_dataset_key(),
            use_original_metamath_prompt=bool(self.args.answer_aug_use_original_metamath_prompt),
        )
        if support_info.get("support") == "unsupported":
            raise SystemExit(
                f"[ERROR] distill method {distill_method!r} is not supported for dataset "
                f"{self.distill_dataset_name()!r}: {support_info.get('teacher_prompt', 'no prompt support')}"
            )

        distill_input_source = self.distill_input_path()
        check_input(self.args.teacher_model, "teacher model", True, self.args.dry_run)
        check_input(distill_input_source, "distill input", True, self.args.dry_run)
        if self.args.dry_run and not Path(distill_input_source).exists():
            distill_input = distill_input_source
        else:
            distill_input = str(self.prepare_dataset_data(distill_input_source, "distill_input", mode="distill"))

        cmd = [
            sys.executable,
            str(DISTILL_METHOD_SCRIPTS[distill_method]),
            "--dataset",
            self.distill_dataset_key(),
            "--input-file",
            distill_input,
            "--output-file",
            str(self.distill_output),
            "--model-id",
            self.args.teacher_model,
            "--gpu-ids",
            self.args.distill_gpu_ids or self.args.gpu_ids,
            "--dtype",
            self.args.teacher_dtype,
            "--temperature",
            str(self.args.teacher_temperature),
            "--top-p",
            str(self.args.teacher_top_p),
            "--gen-batch-size",
            str(self.args.teacher_batch_size),
            "--max-new-tokens",
            str(self.args.teacher_max_new_tokens),
            "--tensor-parallel-size",
            str(self.args.teacher_tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.args.teacher_gpu_memory_utilization),
            "--correct-threshold",
            str(self.args.teacher_correct_threshold),
        ]
        if distill_method == "teacher_correctness_filter":
            cmd += ["--num-cots", "1"]
        elif distill_method == "answer_augmentation":
            prompt_status = self.answer_aug_original_prompt_status()
            if self.args.answer_aug_use_original_metamath_prompt and not self.answer_aug_original_prompt_supported():
                raise SystemExit(
                    "[ERROR] --answer-aug-use-original-metamath-prompt is only native for GSM8K "
                    "and borrowed for math/math-500/numinamath. "
                    f"Dataset {self.distill_dataset_name()} should use DataObs dataset-specific prompts."
                )
            if self.args.answer_aug_use_original_metamath_prompt and prompt_status == "borrowed":
                print(
                    "[WARN] --answer-aug-use-original-metamath-prompt is borrowed from the GSM8K/MetaMath "
                    f"answer prompt for {self.distill_dataset_name()}, not a native dataset prompt."
                )
            if not self.args.answer_aug_use_original_metamath_prompt:
                print(
                    "[WARN] answer_augmentation defaults to the DataObs dataset-specific prompt. "
                    "Use --answer-aug-use-original-metamath-prompt only for original MetaMath baseline reproduction."
                )
            cmd += ["--num-augmented-answers", str(self.args.teacher_num_samples)]
            if self.args.answer_aug_use_original_metamath_prompt:
                cmd.append("--use-original-metamath-prompt")
        elif distill_method == "question_rephrasing":
            cmd += [
                "--num-rephrases",
                str(self.args.teacher_num_samples),
                "--num-cots",
                str(self.args.rephrase_num_cots),
                "--rephrase-max-new-tokens",
                str(self.args.rephrase_max_new_tokens),
            ]
        elif distill_method == "question_augmentation":
            cmd += [
                "--num-backward-questions",
                str(self.args.teacher_num_samples),
                "--backward-question-max-new-tokens",
                str(self.args.backward_question_max_new_tokens),
            ]
        elif distill_method == "reverse_thinking":
            cmd += [
                "--backward-question-max-new-tokens",
                str(self.args.backward_question_max_new_tokens),
                "--consistency-max-new-tokens",
                str(self.args.consistency_max_new_tokens),
            ]
        needs_do_sample = bool(self.args.teacher_do_sample)
        if distill_method == "answer_augmentation" and self.args.teacher_num_samples > 1:
            needs_do_sample = True
        elif distill_method == "question_rephrasing" and (
            self.args.teacher_num_samples > 1 or self.args.rephrase_num_cots > 1
        ):
            needs_do_sample = True
        elif distill_method == "question_augmentation" and self.args.teacher_num_samples > 1:
            needs_do_sample = True
        if needs_do_sample:
            cmd.append("--do-sample")
        if self.args.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.args.disable_teacher_filter:
            cmd.append("--disable-teacher-filter")
        if self.args.smoke_num_rows > 0:
            cmd += ["--smoke-num-rows", str(self.args.smoke_num_rows)]
        self.run_cmd("distill", cmd)
        self.results["distill_method"] = distill_method
        self.results["distill_output"] = str(self.distill_output)
        self.results["distill_candidates"] = str(self.distill_output.with_name(f"{self.distill_output.stem}.candidates.parquet"))
        self.results["distill_summary"] = str(self.distill_output.with_name(f"{self.distill_output.stem}.summary.json"))
        self.results["distill_config"] = self.build_distill_config()

    def stage_metrics(self) -> None:
        data_path = self.args.metrics_data or self.sft_data_path()
        check_input(data_path, "metrics data", True, self.args.dry_run)
        cmd = [
            sys.executable,
            str(REPO_ROOT / "DataObs" / "scripts" / "data_obs_pipeline.py"),
            "--data_path",
            data_path,
            "--data_name",
            self.args.data_variant or self.args.experiment_id,
            "--model_id",
            self.args.base_model,
            "--output_dir",
            str(self.exp_dir / "dataobs"),
            "--n_splits",
            str(self.args.n_splits),
            "--skip_training",
            "--skip_analysis",
        ]
        if self.args.metrics_model:
            cmd += ["--model", self.args.metrics_model]
        if self.args.similarity_type:
            cmd += ["--similarity_type", self.args.similarity_type]
        cmd += as_list(self.args.metrics_arg)
        self.run_cmd("metrics", cmd)
        self.results["metrics_dir"] = str(self.exp_dir / "dataobs")

    def stage_sft(self) -> None:
        sft_data = self.sft_data_path()
        sft_val_data = self.sft_val_data_path()
        check_input(self.args.base_model, "base model", True, self.args.dry_run)
        check_input(sft_data, "SFT train data", True, self.args.dry_run)
        check_input(sft_val_data, "SFT val data", True, self.args.dry_run)
        if not self.args.dry_run:
            train_rows = parquet_num_rows(sft_data)
            train_batch_size = hydra_int_override(self.args.sft_arg, "data.train_batch_size", 8)
            if train_rows is not None and train_rows < train_batch_size:
                raise SystemExit(
                    "[ERROR] SFT train data has "
                    f"{train_rows} rows, but data.train_batch_size={train_batch_size}. "
                    "verl SFT uses drop_last=True, so this would run 0 training steps "
                    "and produce no checkpoint. Increase the smoke/distill rows or set "
                    "--sft-arg data.train_batch_size=1 --sft-arg data.micro_batch_size_per_gpu=1."
                )
        cmd = [
            "bash",
            str(REPO_ROOT / "DataObs" / "lib" / "training" / "sft_dataobs.sh"),
            self.args.base_model,
            sft_data,
            sft_val_data,
            str(self.sft_dir),
            self.args.sft_gpu_ids or self.args.gpu_ids,
            str(self.args.sft_epochs),
        ] + as_list(self.args.sft_arg)
        env = {"MODEL_NAME": self.eval_model_name("sft"), "DATA_NAME": self.args.dataset}
        env.update(parse_env_pairs(self.args.sft_env))
        self.run_cmd("sft", cmd, env)
        self.results["sft_dir"] = str(self.sft_dir)

    def stage_sft_eval(self) -> None:
        checkpoint = resolve_sft_checkpoint(self.sft_dir, Path(self.args.sft_checkpoint) if self.args.sft_checkpoint else None)
        if not self.args.dry_run:
            check_input(str(checkpoint), "SFT checkpoint", True, self.args.dry_run)
        cmd = [
            "bash",
            str(REPO_ROOT / "scripts" / "eval.sh"),
            self.args.dataset,
            self.args.eval_gpu_ids or self.args.gpu_ids,
            str(checkpoint),
        ] + as_list(self.args.eval_arg)
        env = {
            "BASE_MODEL": self.args.base_model,
            "MODEL_NAME": self.eval_model_name("sft"),
            "TARGET_EVAL_DIR": str(self.sft_eval_dir),
        }
        env.update(parse_env_pairs(self.args.eval_env))
        self.run_cmd("sft_eval", cmd, env)
        self.results["sft_eval_dir"] = str(self.sft_eval_dir)

    def stage_grpo(self) -> None:
        if self.args.rl_train_from_distill_kept:
            rl_train = str(self.build_rl_train_from_distill_kept())
        else:
            rl_train = str(self.prepare_rl_data(self.rl_train_data_path(), "train"))
        rl_val = str(self.prepare_rl_data(self.rl_val_data_path(), "val"))
        if not rl_train or not rl_val:
            raise SystemExit(f"[ERROR] No RL train/val default for dataset={self.args.dataset}. Pass --rl-train-data and --rl-val-data.")
        checkpoint = resolve_sft_checkpoint(self.sft_dir, Path(self.args.sft_checkpoint) if self.args.sft_checkpoint else None)
        if not self.args.dry_run:
            check_input(str(checkpoint), "SFT checkpoint", True, self.args.dry_run)
        check_input(rl_train, "RL train data", True, self.args.dry_run)
        check_input(rl_val, "RL val data", True, self.args.dry_run)
        cmd = [
            "bash",
            str(REPO_ROOT / "DataObs" / "lib" / "training" / "grpo_from_sft.sh"),
            str(checkpoint),
            rl_train,
            rl_val,
            self.args.grpo_gpu_ids or self.args.gpu_ids,
            str(self.grpo_dir),
            self.args.base_model,
        ] + as_list(self.args.grpo_arg)
        env = {
            "MODEL_NAME": self.eval_model_name("grpo"),
            "DATA_NAME": self.args.dataset,
        }
        env.update(parse_env_pairs(self.args.grpo_env))
        self.run_cmd("grpo", cmd, env)
        self.results["grpo_dir"] = str(self.grpo_dir)

    def stage_grpo_eval(self) -> None:
        check_input(str(self.grpo_dir), "GRPO dir", True, self.args.dry_run)
        cmd = [
            "bash",
            str(REPO_ROOT / "DataObs" / "lib" / "evaluation" / "eval_grpo_dataobs.sh"),
            str(self.grpo_dir),
            self.args.base_model,
            self.args.dataset,
            str(self.grpo_eval_dir),
            self.args.eval_gpu_ids or self.args.gpu_ids,
        ]
        env: dict[str, str] = {}
        if self.args.grpo_step:
            env["STEP"] = str(self.args.grpo_step)
        env.update(parse_env_pairs(self.args.eval_env))
        self.run_cmd("grpo_eval", cmd, env)
        self.results["grpo_eval_dir"] = str(self.grpo_eval_dir)

    def run(self) -> None:
        stages = [x.strip() for x in self.args.stages.split(",") if x.strip()]
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.write_manifest(stages)
        stage_map = {
            "distill": self.stage_distill,
            "metrics": self.stage_metrics,
            "sft": self.stage_sft,
            "sft_eval": self.stage_sft_eval,
            "grpo": self.stage_grpo,
            "grpo_eval": self.stage_grpo_eval,
        }
        for stage in stages:
            if stage not in stage_map:
                raise SystemExit(f"[ERROR] Unknown stage: {stage}. Choices: {', '.join(stage_map)}")
            stage_map[stage]()
            write_json(self.results_file, self.results)
        print(f"\nDone. Experiment dir: {self.exp_dir}")
        print(f"Manifest: {self.manifest_file}")
        print(f"Commands: {self.commands_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment stages from distillation to GRPO evaluation.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--dataset", required=True, help="Dataset name accepted by scripts/eval.sh, e.g. gsm8k.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", default="/data/hrh/COT/experiments")
    parser.add_argument("--stages", default="sft,sft_eval,grpo,grpo_eval",
                        help="Comma-separated stages: distill,metrics,sft,sft_eval,grpo,grpo_eval")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--data-variant", default="")
    parser.add_argument("--reasoning-source", default="")
    parser.add_argument("--model-name", default="", help="Override MODEL_NAME used by eval scripts.")

    # Distillation.
    parser.add_argument("--distill-method", default="teacher_correctness_filter", choices=DISTILL_METHODS)
    parser.add_argument("--distill-dataset", default="")
    parser.add_argument("--distill-input", default="")
    parser.add_argument("--distill-output", default="")
    parser.add_argument("--teacher-model", default="")
    parser.add_argument("--distill-gpu-ids", default="")
    parser.add_argument("--teacher-num-samples", type=int, default=1)
    parser.add_argument("--teacher-temperature", type=float, default=0.7)
    parser.add_argument("--teacher-top-p", type=float, default=0.95)
    parser.add_argument("--teacher-batch-size", type=int, default=128)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=8192)
    parser.add_argument("--teacher-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--teacher-gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument(
        "--teacher-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--teacher-correct-threshold", type=float, default=0.99)
    parser.add_argument("--teacher-do-sample", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-teacher-filter", action="store_true")
    parser.add_argument("--answer-aug-use-original-metamath-prompt", action="store_true")
    parser.add_argument("--rephrase-num-cots", type=int, default=1)
    parser.add_argument("--rephrase-max-new-tokens", type=int, default=512)
    parser.add_argument("--backward-question-max-new-tokens", type=int, default=1024)
    parser.add_argument("--consistency-max-new-tokens", type=int, default=1024)
    parser.add_argument("--smoke-num-rows", type=int, default=0)

    # Metrics.
    parser.add_argument("--metrics-data", default="")
    parser.add_argument("--metrics-model", default="")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--similarity-type", default="jaccard")
    parser.add_argument("--metrics-arg", action="append", default=[])

    # SFT.
    parser.add_argument("--sft-data", default="", help="Parquet with question/answer columns. Defaults to distill output.")
    parser.add_argument("--sft-val-data", default="")
    parser.add_argument("--sft-output-dir", default="")
    parser.add_argument("--sft-checkpoint", default="", help="Use an existing SFT checkpoint for eval/GRPO stages.")
    parser.add_argument("--sft-gpu-ids", default="")
    parser.add_argument("--sft-epochs", type=int, default=1)
    parser.add_argument("--sft-arg", action="append", default=[], help="Extra Hydra arg for sft_dataobs.sh; repeatable.")
    parser.add_argument("--sft-env", action="append", default=[], help="Environment override for SFT, KEY=VALUE; repeatable.")

    # Eval.
    parser.add_argument("--eval-gpu-ids", default="")
    parser.add_argument("--sft-eval-output-dir", default="")
    parser.add_argument("--grpo-eval-output-dir", default="")
    parser.add_argument("--eval-arg", action="append", default=[], help="Extra Hydra arg for eval.sh; repeatable.")
    parser.add_argument("--eval-env", action="append", default=[], help="Environment override for eval scripts, KEY=VALUE; repeatable.")

    # GRPO.
    parser.add_argument("--rl-train-data", default="")
    parser.add_argument("--rl-val-data", default="")
    parser.add_argument(
        "--rl-train-from-distill-kept",
        action="store_true",
        help="Build RL train parquet from prompts that survived distill filtering, so SFT train prompts equal RL train prompts.",
    )
    parser.add_argument("--grpo-output-dir", default="")
    parser.add_argument("--grpo-gpu-ids", default="")
    parser.add_argument("--grpo-step", default="")
    parser.add_argument("--grpo-arg", action="append", default=[], help="Extra Hydra arg for grpo_from_sft.sh; repeatable.")
    parser.add_argument("--grpo-env", action="append", default=[], help="Environment override for GRPO, KEY=VALUE; repeatable.")

    args = parser.parse_args()
    return normalize_distill_args(args)


def normalize_distill_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.distill_method == "reverse_thinking" and args.teacher_num_samples > 1:
        raise SystemExit(
            "[ERROR] --distill-method reverse_thinking does not support "
            "--teacher-num-samples > 1 yet. Use 1 to keep the current single-path setting."
        )

    needs_do_sample = bool(args.teacher_do_sample)
    if args.distill_method == "answer_augmentation" and args.teacher_num_samples > 1:
        needs_do_sample = True
    elif args.distill_method == "question_rephrasing" and (
        args.teacher_num_samples > 1 or getattr(args, "rephrase_num_cots", 1) > 1
    ):
        needs_do_sample = True
    elif args.distill_method == "question_augmentation" and args.teacher_num_samples > 1:
        needs_do_sample = True

    if needs_do_sample:
        args.teacher_do_sample = True
    return args


if __name__ == "__main__":
    print(
        "[DEPRECATED] DataObs/scripts/experiment_pipeline.py uses legacy shell eval paths. "
        "Use DataObs/scripts/experiment_pipeline_vllm.py for current experiments.",
        file=sys.stderr,
    )
    Pipeline(parse_args()).run()
