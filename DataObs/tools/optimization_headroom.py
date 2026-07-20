#!/usr/bin/env python3
"""Measure SFT policy entropy collapse and correlate it with SFT/GRPO results.

The script has two entry points:

1. ``compute``: load the base model and each experiment's SFT checkpoint, then
   estimate policy entropy on a shared prompt/sample set.
2. ``plot``: join the entropy CSV with existing SFT/GRPO metrics and create
   scatter plots plus correlation statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_BASE_MODEL = "/data/pretrain_models/Qwen3-4B"
DEFAULT_EXPERIMENTS_ROOT = "/data/hrh/COT/experiments"
DEFAULT_OUT_DIR = "DataObs/experiment_observations/optimization_headroom_qwen3_gsm8k"
GRPO_VAL_COL = "val-core/gsm8k/reward/mean@1"


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_dir: Path
    dataset: str = "gsm8k"
    base_model: str = DEFAULT_BASE_MODEL

    @property
    def experiment_id(self) -> str:
        return self.experiment_dir.name


DEFAULT_EXPERIMENTS = [
    "exp1100_gsm8k_qwen3_teacher_qwen3_1_7b_t0.7_n4",
    "exp1100_gsm8k_qwen3_teacher_qwen3_4b_t0.7_n4",
    "exp1100_gsm8k_qwen3_teacher_qwen3_8b_t0.7_n4",
    "exp1200_gsm8k_qwen3_teacher_type_deepseek_r1_qwen32b_t0.7_n4",
    "exp1200_gsm8k_qwen3_teacher_type_qwen3_14b_t0.7_n4",
    "exp3100_gsm8k_qwen3_settingA_sft_eq_rl_prompt",
    "exp3100_gsm8k_qwen3_settingB_sft_ne_rl_prompt",
    "exp5000_gsm8k_qwen3_difficulty_easy_distill",
    "exp5000_gsm8k_qwen3_difficulty_hard_distill",
    "exp5000_gsm8k_qwen3_difficulty_medium_distill",
    "exp7000_gsm8k_qwen3_teacher14b_pool_n8_same_prompt_n1",
    "exp7000_gsm8k_qwen3_teacher14b_pool_n8_same_prompt_n2",
    "exp7000_gsm8k_qwen3_teacher14b_pool_n8_same_prompt_n4",
    "exp7000_gsm8k_qwen3_teacher14b_pool_n8_same_prompt_n8",
    "pipeline_gsm8k_qwen3_4b_0620",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_hf_or_lora_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "adapter_model.safetensors").is_file():
        return True
    if not (path / "config.json").is_file():
        return False
    model_markers = [
        path / "model.safetensors",
        path / "pytorch_model.bin",
        path / "model.safetensors.index.json",
    ]
    return any(p.is_file() for p in model_markers) or any(path.glob("*.safetensors"))


def latest_global_step(path: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for item in path.glob("global_step_*"):
        if not item.is_dir():
            continue
        step = item.name.removeprefix("global_step_")
        if step.isdigit():
            candidates.append((int(step), item))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def resolve_sft_checkpoint(exp_dir: Path) -> Path | None:
    sft_dir = exp_dir / "sft"
    if is_hf_or_lora_dir(sft_dir):
        return sft_dir
    checkpoint_last = sft_dir / "checkpoint-last"
    if is_hf_or_lora_dir(checkpoint_last):
        return checkpoint_last
    latest = latest_global_step(sft_dir)
    if latest and is_hf_or_lora_dir(latest):
        return latest
    return None


def short_label(exp_id: str) -> str:
    label = exp_id
    replacements = [
        ("exp1100_gsm8k_qwen3_teacher_qwen3_", "ts:"),
        ("exp1200_gsm8k_qwen3_teacher_type_", "tt:"),
        ("exp3100_gsm8k_qwen3_", "p:"),
        ("exp5000_gsm8k_qwen3_difficulty_", "diff:"),
        ("exp7000_gsm8k_qwen3_teacher14b_pool_n8_same_prompt_", "size:"),
        ("pipeline_gsm8k_qwen3_4b_0620", "pipeline"),
        ("_distill", ""),
        ("_t0.7_n4", ""),
        ("qwen3_", "q"),
    ]
    for old, new in replacements:
        label = label.replace(old, new)
    return label.replace("deepseek_r1_qwen32b", "DeepSeek-R1")


def group_name(exp_id: str) -> str:
    if exp_id.startswith("exp1100_"):
        return "teacher size"
    if exp_id.startswith("exp1200_"):
        return "teacher type"
    if exp_id.startswith("exp3100_"):
        return "prompt setting"
    if exp_id.startswith("exp5000_"):
        return "difficulty"
    if exp_id.startswith("exp7000_"):
        return "seed size"
    if exp_id.startswith("pipeline_"):
        return "pipeline"
    return "other"


def normalize_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        messages = []
        for item in value:
            if isinstance(item, dict):
                messages.append({"role": str(item.get("role", "user")), "content": str(item.get("content", ""))})
            else:
                messages.append({"role": "user", "content": str(item)})
        return messages
    if isinstance(value, dict):
        return [{"role": str(value.get("role", "user")), "content": str(value.get("content", ""))}]
    return [{"role": "user", "content": str(value)}]


def row_to_prompt_response(row: pd.Series) -> tuple[list[dict[str, str]], str | None]:
    if "prompt" in row and row["prompt"] is not None:
        return normalize_messages(row["prompt"]), None
    if "question" in row and pd.notna(row["question"]):
        response = str(row["answer"]) if "answer" in row and pd.notna(row["answer"]) else None
        return [{"role": "user", "content": str(row["question"])}], response
    if "problem" in row and pd.notna(row["problem"]):
        response = str(row["solution"]) if "solution" in row and pd.notna(row["solution"]) else None
        return [{"role": "user", "content": str(row["problem"])}], response
    raise ValueError(f"Cannot infer prompt columns from row columns: {list(row.index)}")


def resolve_sample_path(exp_dir: Path, sample_source: str) -> Path | None:
    candidates = {
        "val": [exp_dir / "rl_data" / "val_prepared.parquet"],
        "rl_train": [
            exp_dir / "rl_data" / "train_prepared.parquet",
            exp_dir / "rl_data" / "train_from_distill_kept.parquet",
        ],
        "sft": [
            exp_dir / "distill" / "filtered_sft.parquet",
            exp7000_same_prompt_subset_path(exp_dir),
        ],
    }
    if sample_source not in candidates:
        path = Path(sample_source)
        return path if path.exists() else None
    for path in candidates[sample_source]:
        if path is not None and path.exists():
            return path
    return None


def exp7000_same_prompt_subset_path(exp_dir: Path) -> Path | None:
    match = re.search(r"exp7000_gsm8k_qwen3_teacher14b_pool_n8_same_prompt_n(\d+)$", exp_dir.name)
    if not match:
        return None
    path = (
        exp_dir.parent
        / "exp7000_size_same_prompt_subsets"
        / f"gsm8k_qwen3_teacher14b_pool_n8_same_prompt_n{match.group(1)}.parquet"
    )
    return path if path.exists() else None


def load_samples(path: Path, max_samples: int, seed: int) -> list[tuple[list[dict[str, str]], str | None]]:
    df = pd.read_parquet(path)
    if max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).sort_index()
    samples = []
    for _, row in df.iterrows():
        samples.append(row_to_prompt_response(row))
    return samples


def import_transformer_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing model dependencies. Run this in the training/eval environment, "
            "or install requirements.txt dependencies: transformers, peft, accelerate."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, PeftModel


def load_tokenizer(tokenizer_path: str, trust_remote_code: bool) -> Any:
    _, _, auto_tokenizer, _ = import_transformer_stack()
    tokenizer = auto_tokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_path: Path | str, base_model: str, torch: Any, auto_model: Any, peft_model: Any, args: argparse.Namespace) -> Any:
    model_path = Path(model_path)
    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    base_kwargs = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if (model_path / "adapter_model.safetensors").is_file():
        model = auto_model.from_pretrained(base_model, **base_kwargs)
        model = peft_model.from_pretrained(model, str(model_path))
    else:
        model = auto_model.from_pretrained(str(model_path), **base_kwargs)
    model.eval()
    return model


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking:
        kwargs["enable_thinking"] = True
    else:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def token_entropy(logits: Any, torch: Any) -> Any:
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def response_token_segments(tokenizer: Any, response_ids: Any) -> dict[str, list[bool]]:
    """Classify response tokens into content/reasoning/final/format segments.

    ``trajectory`` intentionally keeps every response token for backward
    compatibility. The other segments try to remove template artifacts:
    special tokens, whitespace-only tokens, and explicit think tags.
    """

    ids = [int(x) for x in response_ids.tolist()]
    special_ids = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) or [])
    think_id = tokenizer.convert_tokens_to_ids("<think>")
    end_think_id = tokenizer.convert_tokens_to_ids("</think>")
    if not isinstance(think_id, int) or think_id < 0:
        think_id = None
    if not isinstance(end_think_id, int) or end_think_id < 0:
        end_think_id = None

    has_think = think_id in ids if think_id is not None else False
    inside_think = False
    seen_end_think = False
    out = {
        "content": [],
        "reasoning": [],
        "final": [],
        "format": [],
    }
    for token_id in ids:
        is_think_tag = token_id == think_id or token_id == end_think_id
        if token_id == think_id:
            inside_think = True
        elif token_id == end_think_id:
            inside_think = False
            seen_end_think = True

        text = token_text(tokenizer, token_id)
        is_special = token_id in special_ids or is_think_tag
        is_whitespace = text.strip() == ""
        is_content = not is_special and not is_whitespace
        is_reasoning = is_content and inside_think
        if has_think:
            is_final = is_content and seen_end_think and not inside_think
        else:
            is_final = is_content
        is_format = not is_content

        out["content"].append(is_content)
        out["reasoning"].append(is_reasoning)
        out["final"].append(is_final)
        out["format"].append(is_format)

    return out


def entropy_for_samples(
    model: Any,
    tokenizer: Any,
    samples: list[tuple[list[dict[str, str]], str | None]],
    torch: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt_entropies: list[float] = []
    segment_entropies: dict[str, list[float]] = {
        "trajectory": [],
        "content": [],
        "reasoning": [],
        "final": [],
        "format": [],
    }
    segment_token_counts: dict[str, list[int]] = {name: [] for name in segment_entropies}
    device = next(model.parameters()).device
    encoded = []
    for messages, response in samples:
        prompt_text = apply_chat_template(tokenizer, messages, args.enable_thinking)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if len(prompt_ids) == 0:
            continue
        prompt_ids = prompt_ids[-min(args.max_prompt_tokens, args.max_sequence_tokens) :]
        response_ids = None
        if response is not None and args.metric_scope == "trajectory":
            raw_response_ids = tokenizer(str(response), return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            if len(raw_response_ids) > 0:
                response_ids = raw_response_ids[: args.max_response_tokens]

        if response_ids is not None and len(response_ids) > 0:
            max_prompt_for_sequence = max(1, args.max_sequence_tokens - len(response_ids))
            prompt_ids = prompt_ids[-max_prompt_for_sequence:]
            ids = torch.cat([prompt_ids, response_ids], dim=0)
            prompt_logit_pos = len(prompt_ids) - 1
            response_start_logit_pos = len(prompt_ids) - 1
            response_token_count = int(len(response_ids))
            segments = response_token_segments(tokenizer, response_ids)
        else:
            ids = prompt_ids[-args.max_sequence_tokens :]
            prompt_logit_pos = len(ids) - 1
            response_start_logit_pos = -1
            response_token_count = 0
            segments = {}

        encoded.append(
            {
                "input_ids": ids,
                "prompt_logit_pos": int(prompt_logit_pos),
                "response_start_logit_pos": int(response_start_logit_pos),
                "response_token_count": int(response_token_count),
                "segments": segments,
            }
        )

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batch_size = max(1, args.batch_size)
    for start in range(0, len(encoded), batch_size):
        if start and start % args.log_every == 0:
            print(f"[entropy] processed {start}/{len(encoded)} samples", flush=True)
        batch = encoded[start : start + batch_size]
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full((len(batch), max_len), int(pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, item in enumerate(batch):
            ids = item["input_ids"]
            input_ids[i, : len(ids)] = ids
            attention_mask[i, : len(ids)] = 1
        with torch.inference_mode():
            logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
            ent = token_entropy(logits, torch).detach().cpu()

        for i, item in enumerate(batch):
            prompt_entropies.append(float(ent[i, item["prompt_logit_pos"]].item()))
            response_token_count = item["response_token_count"]
            if response_token_count <= 0:
                continue
            start_pos = item["response_start_logit_pos"]
            end_pos = start_pos + response_token_count
            response_ent = ent[i, start_pos:end_pos]
            masks = {
                "trajectory": [True] * response_token_count,
                **item["segments"],
            }
            for segment_name, mask_values in masks.items():
                if not mask_values:
                    continue
                mask = torch.tensor(mask_values, dtype=torch.bool)
                if not bool(mask.any()):
                    continue
                segment_response_ent = response_ent[mask]
                segment_entropies[segment_name].append(float(segment_response_ent.mean().item()))
                segment_token_counts[segment_name].append(int(mask.sum().item()))

    return summarize_entropies(prompt_entropies, segment_entropies, segment_token_counts)


def summarize_entropies(
    prompt: list[float],
    segment_entropies: dict[str, list[float]],
    segment_token_counts: dict[str, list[int]],
) -> dict[str, Any]:
    def stat(values: list[float], name: str) -> dict[str, float | int]:
        if not values:
            return {
                f"{name}_n": 0,
                f"{name}_mean": math.nan,
                f"{name}_std": math.nan,
                f"{name}_median": math.nan,
            }
        arr = np.array(values, dtype=float)
        return {
            f"{name}_n": int(len(values)),
            f"{name}_mean": float(arr.mean()),
            f"{name}_std": float(arr.std(ddof=1)) if len(values) > 1 else 0.0,
            f"{name}_median": float(np.median(arr)),
        }

    out: dict[str, Any] = {}
    out.update(stat(prompt, "prompt_entropy"))
    for segment_name, values in segment_entropies.items():
        out.update(stat(values, f"{segment_name}_entropy"))
        out[f"{segment_name}_tokens_total"] = int(sum(segment_token_counts[segment_name]))
    return out


def read_experiment_list(path: str | None, root: Path, base_model: str, dataset: str) -> list[ExperimentSpec]:
    if path:
        payload = load_json(Path(path))
        specs = []
        for item in payload:
            exp_dir = Path(item.get("experiment_dir", item.get("path", "")))
            if not exp_dir.is_absolute():
                exp_dir = root / exp_dir
            specs.append(
                ExperimentSpec(
                    experiment_dir=exp_dir,
                    dataset=str(item.get("dataset", dataset)),
                    base_model=str(item.get("base_model", base_model)),
                )
            )
        return specs
    return [ExperimentSpec(root / name, dataset=dataset, base_model=base_model) for name in DEFAULT_EXPERIMENTS]


def compute_command(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    torch, auto_model, auto_tokenizer, peft_model = import_transformer_stack()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.experiments_root)
    specs = read_experiment_list(args.experiments_json, root, args.base_model, args.dataset)

    tokenizer = auto_tokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    base_cache: dict[str, dict[str, Any]] = {}

    print(f"[entropy] loading base model: {args.base_model}", flush=True)
    base_model = load_model(args.base_model, args.base_model, torch, auto_model, peft_model, args)

    for spec in specs:
        exp_dir = spec.experiment_dir
        sft_checkpoint = resolve_sft_checkpoint(exp_dir)
        sample_path = resolve_sample_path(exp_dir, args.sample_source)
        if not exp_dir.is_dir() or not sft_checkpoint or not sample_path:
            excluded.append(
                {
                    "experiment_id": exp_dir.name,
                    "experiment_dir": str(exp_dir),
                    "has_exp_dir": exp_dir.is_dir(),
                    "sft_checkpoint": str(sft_checkpoint) if sft_checkpoint else "",
                    "sample_path": str(sample_path) if sample_path else "",
                }
            )
            continue

        print(f"[entropy] {exp_dir.name}: loading samples from {sample_path}", flush=True)
        samples = load_samples(sample_path, args.max_samples, args.seed)
        cache_key = f"{sample_path}:{args.max_samples}:{args.seed}:{args.metric_scope}:{args.max_prompt_tokens}:{args.max_response_tokens}"
        if cache_key not in base_cache:
            base_cache[cache_key] = entropy_for_samples(base_model, tokenizer, samples, torch, args)
        base_stats = base_cache[cache_key]

        print(f"[entropy] {exp_dir.name}: loading SFT checkpoint {sft_checkpoint}", flush=True)
        sft_model = load_model(sft_checkpoint, spec.base_model, torch, auto_model, peft_model, args)
        sft_stats = entropy_for_samples(sft_model, tokenizer, samples, torch, args)
        del sft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        row = {
            "experiment_id": exp_dir.name,
            "label": short_label(exp_dir.name),
            "group": group_name(exp_dir.name),
            "dataset": spec.dataset,
            "base_model": spec.base_model,
            "sft_checkpoint": str(sft_checkpoint),
            "sample_source": args.sample_source,
            "sample_path": str(sample_path),
            "max_samples": args.max_samples,
            "entropy_min": args.entropy_min,
        }
        for key, value in base_stats.items():
            row[f"base_{key}"] = value
        for key, value in sft_stats.items():
            row[f"sft_{key}"] = value
        entropy_scopes = ("prompt", "trajectory", "content", "reasoning", "final", "format")
        for scope in entropy_scopes:
            base_col = f"base_{scope}_entropy_mean"
            sft_col = f"sft_{scope}_entropy_mean"
            if base_col in row and sft_col in row:
                row[f"{scope}_entropy_collapse"] = row[base_col] - row[sft_col]
                row[f"{scope}_optimization_headroom"] = row[sft_col] - args.entropy_min
                row[f"{scope}_collapse_fraction"] = (
                    row[f"{scope}_entropy_collapse"] / row[base_col] if row[base_col] and not math.isnan(row[base_col]) else math.nan
                )
        rows.append(row)

        pd.DataFrame(rows).to_csv(out_dir / "optimization_headroom_entropy.csv", index=False)
        write_json(out_dir / "optimization_headroom_excluded.json", excluded)

    del base_model
    pd.DataFrame(rows).to_csv(out_dir / "optimization_headroom_entropy.csv", index=False)
    write_json(out_dir / "optimization_headroom_excluded.json", excluded)
    write_json(
        out_dir / "optimization_headroom_compute_config.json",
        {
            "base_model": args.base_model,
            "experiments_root": args.experiments_root,
            "sample_source": args.sample_source,
            "max_samples": args.max_samples,
            "metric_scope": args.metric_scope,
            "entropy_min": args.entropy_min,
            "excluded_count": len(excluded),
            "rows": len(rows),
        },
    )
    print(f"[entropy] wrote {out_dir / 'optimization_headroom_entropy.csv'}", flush=True)


def metric_record(path: Path) -> tuple[str, float, int, int]:
    data = load_json(path)
    key = "gsm8k" if "gsm8k" in data else next(iter(data.keys()))
    metrics = data[key]
    return key, float(metrics["mean@1"]), int(metrics.get("correct", -1)), int(metrics.get("total", -1))


def read_grpo_final_val_pct(exp_dir: Path) -> tuple[float, int, float, int]:
    path = exp_dir / "grpo_metrics" / "grpo_metrics.csv"
    df = pd.read_csv(path)
    vals = df[["step", GRPO_VAL_COL]].dropna().copy()
    if vals.empty:
        raise ValueError(f"No non-empty {GRPO_VAL_COL} in {path}")
    vals["step"] = vals["step"].astype(int)
    final = vals.iloc[-1]
    best = vals.loc[vals[GRPO_VAL_COL].idxmax()]
    return float(final[GRPO_VAL_COL]) * 100.0, int(final["step"]), float(best[GRPO_VAL_COL]) * 100.0, int(best["step"])


def collect_performance(root: Path) -> pd.DataFrame:
    rows = []
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        exp_id = exp_dir.name
        sft_path = exp_dir / "eval" / "sft" / "generated" / "responses_labeled.metrics.json"
        grpo_eval_path = exp_dir / "eval" / "grpo" / "generated" / "responses_labeled.metrics.json"
        grpo_metrics_path = exp_dir / "grpo_metrics" / "grpo_metrics.csv"
        if not sft_path.exists():
            continue
        sft_dataset, sft, sft_correct, sft_total = metric_record(sft_path)
        row: dict[str, Any] = {
            "experiment_id": exp_id,
            "sft_eval_mean@1": sft,
            "sft_eval_pct": sft * 100.0,
            "sft_correct": sft_correct,
            "sft_total": sft_total,
            "eval_dataset": sft_dataset,
        }
        if grpo_eval_path.exists():
            grpo_dataset, grpo, grpo_correct, grpo_total = metric_record(grpo_eval_path)
            row.update(
                {
                    "grpo_eval_mean@1": grpo,
                    "grpo_eval_pct": grpo * 100.0,
                    "grpo_correct": grpo_correct,
                    "grpo_total": grpo_total,
                    "grpo_eval_dataset": grpo_dataset,
                    "grpo_eval_minus_sft_pp": (grpo - sft) * 100.0,
                }
            )
        if grpo_metrics_path.exists():
            try:
                final_pct, final_step, best_pct, best_step = read_grpo_final_val_pct(exp_dir)
                row.update(
                    {
                        "grpo_final_val_pct": final_pct,
                        "grpo_final_step": final_step,
                        "grpo_best_val_pct": best_pct,
                        "grpo_best_step": best_step,
                        "grpo_final_val_minus_sft_pp": final_pct - sft * 100.0,
                        "grpo_best_val_minus_sft_pp": best_pct - sft * 100.0,
                    }
                )
            except Exception as exc:
                row["grpo_final_val_error"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def corr_pair(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, Any]:
    sub = df[[x_col, y_col]].dropna()
    if len(sub) < 2:
        return {"x": x_col, "y": y_col, "n": int(len(sub)), "pearson": math.nan, "spearman": math.nan}
    return {
        "x": x_col,
        "y": y_col,
        "n": int(len(sub)),
        "pearson": float(sub[x_col].corr(sub[y_col], method="pearson")),
        "spearman": float(sub[x_col].corr(sub[y_col], method="spearman")),
    }


def style_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#E6E8EB", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    sub = df.dropna(subset=[x_col, y_col]).copy()
    if sub.empty:
        return
    palette = {
        "teacher size": "#4E79A7",
        "teacher type": "#F28E2B",
        "prompt setting": "#E15759",
        "difficulty": "#59A14F",
        "seed size": "#B07AA1",
        "pipeline": "#9C755F",
        "other": "#BAB0AC",
    }
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    for group, part in sub.groupby("group"):
        ax.scatter(
            part[x_col],
            part[y_col],
            s=82,
            color=palette.get(group, "#BAB0AC"),
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
            label=group,
        )
        for _, row in part.iterrows():
            ax.annotate(row["label"], (row[x_col], row[y_col]), xytext=(5, 5), textcoords="offset points", fontsize=8)

    stats = corr_pair(sub, x_col, y_col)
    if len(sub) >= 2 and sub[x_col].nunique() > 1:
        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
        ax.plot(xs, m * xs + b, color="#222222", linestyle="--", linewidth=1.3, label=f"fit: y={m:.2f}x+{b:.1f}")
    ax.text(
        0.02,
        0.98,
        f"n={stats['n']}\nPearson={stats['pearson']:.3f}\nSpearman={stats['spearman']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D0D5DD", "alpha": 0.95},
    )
    ax.set_xlabel(x_col.replace("_", " "))
    ax.set_ylabel(y_col.replace("_", " "))
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="best")
    style_axes(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_path.with_suffix(f".{ext}"), bbox_inches="tight", dpi=220)
    plt.close(fig)


def plot_command(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out_dir / ".cache"))
    entropy_csv = Path(args.entropy_csv) if args.entropy_csv else out_dir / "optimization_headroom_entropy.csv"
    entropy = pd.read_csv(entropy_csv)
    perf = collect_performance(Path(args.experiments_root))
    merged = entropy.merge(perf, on="experiment_id", how="left")
    merged.to_csv(out_dir / "optimization_headroom_joined.csv", index=False)

    entropy_scopes = ("prompt", "trajectory", "content", "reasoning", "final", "format")
    x_cols = []
    for scope in entropy_scopes:
        for suffix in ("entropy_collapse", "optimization_headroom", "collapse_fraction"):
            col = f"{scope}_{suffix}"
            if col in merged.columns:
                x_cols.append(col)
    y_cols = [
        col
        for col in [
            "sft_eval_pct",
            "grpo_final_val_pct",
            "grpo_best_val_pct",
            "grpo_eval_pct",
            "grpo_final_val_minus_sft_pp",
            "grpo_eval_minus_sft_pp",
        ]
        if col in merged.columns
    ]
    stats = [corr_pair(merged, x, y) for x in x_cols for y in y_cols]
    write_json(out_dir / "optimization_headroom_correlations.json", stats)

    for x in x_cols:
        for y in y_cols:
            name = f"{x}__vs__{y}"
            plot_scatter(merged, x, y, out_dir / name, f"{x.replace('_', ' ')} vs {y.replace('_', ' ')}")
    print(f"[plot] wrote joined CSV and {len(stats)} correlation entries under {out_dir}", flush=True)


def add_compute_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("compute", help="Compute base/SFT policy entropy.")
    parser.add_argument("--experiments-root", default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--experiments-json", default="", help="Optional JSON list with experiment_dir/base_model/dataset fields.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-source", default="sft", help="sft, rl_train, val, or an explicit parquet path.")
    parser.add_argument("--max-samples", type=int, default=128, help="Samples per experiment. Use <=0 for all rows.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--metric-scope", choices=["prompt", "trajectory"], default="trajectory")
    parser.add_argument("--entropy-min", type=float, default=0.0, help="Entropy_min in nats. Default assumes deterministic optimum.")
    parser.add_argument("--gpu-ids", default="", help="CUDA_VISIBLE_DEVICES value, e.g. 2.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-response-tokens", type=int, default=2048)
    parser.add_argument("--max-sequence-tokens", type=int, default=6144)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.set_defaults(func=compute_command)


def add_plot_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("plot", help="Plot entropy headroom correlations.")
    parser.add_argument("--experiments-root", default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--entropy-csv", default="")
    parser.set_defaults(func=plot_command)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_compute_parser(subparsers)
    add_plot_parser(subparsers)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
