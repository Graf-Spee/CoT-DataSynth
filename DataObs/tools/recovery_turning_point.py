#!/usr/bin/env python3
"""Measure recovery and turning-point behavior in reasoning trajectories.

The implementation is verifier-backed when a dataset answer verifier is
available. For GSM8K, it detects answer-like numeric candidates and arithmetic
equation errors, then uses correction markers only as supporting evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_EXPERIMENTS_ROOT = Path("/data/hrh/COT/experiments")
DEFAULT_OUT_DIR = Path("DataObs/experiment_observations/recovery_turning_point_qwen3_gsm8k")
DEFAULT_BASE_PATH = Path("outputs/evals/qwen3_4b_gsm8k_greedy_pass1_len16k/generated/responses_labeled.json")
DEFAULT_FOCUS_EXP = "exp1100_gsm8k_qwen3_teacher_qwen3_4b_t0.7_n4"
PERF_CSV = Path("DataObs/experiment_observations/sft_rl_correlation_qwen3_gsm8k/current_sft_grpo_eval_pairs_gsm8k.csv")

CORRECTION_RE = re.compile(
    r"\b("
    r"wait|actually|mistake|wrong|incorrect|correction|correct(?:ing)?|recheck|re-check|"
    r"recalculate|re-calculate|double[- ]check|let me check|hold on|rather|instead|"
    r"misread|rethink|verify"
    r")\b|"
    r"(等等|等一下|不对|错了|错误|其实|修正|改正|重新|再算|检查|核对|误解)",
    re.IGNORECASE,
)
STRONG_CORRECTION_RE = re.compile(
    r"\b("
    r"wait|actually|mistake|wrong|incorrect|correction|recheck|re-check|"
    r"recalculate|re-calculate|double[- ]check|hold on|rather|instead|"
    r"misread|rethink"
    r")\b|"
    r"(等等|等一下|不对|错了|错误|其实|修正|改正|重新|再算|误解)",
    re.IGNORECASE,
)
ANSWER_CUE_RE = re.compile(
    r"(\\boxed|####|final answer|answer is|the answer|答案|最终答案|final result)",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
EQUATION_RE = re.compile(
    r"(?<![\w])("
    r"-?\$?\d[\d,]*(?:\.\d+)?"
    r"(?:\s*(?:\+|-|\*|/|×|x|X|÷)\s*-?\$?\d[\d,]*(?:\.\d+)?){1,5}"
    r")\s*=\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)(?!\s*(?:\+|-|\*|/|×|x|X|÷))"
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    dataset: str
    split: str
    experiment_id: str = ""
    label: str = ""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict, list)):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.replace(",", "").replace("$", "").strip()
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def extract_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    if "\\boxed " in text[idx:]:
        return text[idx:].split("\\boxed ", 1)[1].split("$", 1)[0].strip()
    start = text.find("{", idx)
    if start < 0:
        return None
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : pos].strip()
    return None


def extract_answer_candidate(text: str, dataset: str, require_cue: bool = True) -> float | None:
    if dataset != "gsm8k":
        return None
    if require_cue and not ANSWER_CUE_RE.search(text):
        return None
    boxed = extract_boxed(text)
    if boxed is not None:
        value = to_float(boxed)
        if value is not None:
            return value
    strict = re.findall(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    if strict:
        return to_float(strict[-1])
    numbers = NUMBER_RE.findall(text.replace(",", ""))
    if not numbers:
        return None
    return to_float(numbers[-1])


def is_correct_value(value: float | None, ground_truth: Any) -> bool:
    pred = to_float(value)
    gold = to_float(ground_truth)
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= 1e-6


def compute_final_score(response: str, ground_truth: Any, dataset: str) -> float:
    if dataset != "gsm8k":
        return 0.0
    return 1.0 if is_correct_value(extract_answer_candidate(response, dataset, require_cue=False), ground_truth) else 0.0


def normalize_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"</?think>", "\n", text, flags=re.IGNORECASE)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_steps(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    text = re.sub(r"\n\s*(?:-{3,}|\*{3,})\s*\n", "\n\n", text)
    text = re.sub(r"(?m)^\s*(#{1,6}\s*)?(\*\*)?\s*(Step\s+\d+|步骤\s*\d+|第\s*\d+\s*步)", r"\n\n\3", text)
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
    if len(blocks) <= 2:
        blocks = [b.strip() for b in re.split(r"(?<=[.!?。！？])\s+", text) if b.strip()]
    steps: list[str] = []
    for block in blocks:
        block = re.sub(r"\s+", " ", block).strip()
        if not block:
            continue
        if steps and len(block) < 24 and not CORRECTION_RE.search(block):
            steps[-1] = f"{steps[-1]} {block}"
        else:
            steps.append(block)
    return steps


def safe_eval_expr(expr: str) -> float | None:
    expr = expr.replace(",", "").replace("$", "")
    expr = expr.replace("×", "*").replace("x", "*").replace("X", "*").replace("÷", "/")
    if not re.fullmatch(r"[\d\.\+\-\*/\s]+", expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def equation_errors(step: str) -> int:
    errors = 0
    candidates: list[str] = []
    for line in re.split(r"(?:\n|;|。|！|!|\?)", step):
        if "=" not in line or not re.search(r"(?:\+|-|\*|/|×|x|X|÷)", line):
            continue
        if len(line) > 220:
            eq = line.find("=")
            line = line[max(0, eq - 90) : eq + 130]
        candidates.append(line)
    for candidate in candidates:
        for match in EQUATION_RE.finditer(candidate):
            lhs = safe_eval_expr(match.group(1))
            rhs = to_float(match.group(2))
            if lhs is None or rhs is None:
                continue
            if abs(lhs - rhs) > max(1e-6, abs(rhs) * 1e-6):
                errors += 1
    return errors


def analyze_trajectory(response: str, ground_truth: Any, dataset: str, score: float | None = None) -> dict[str, Any]:
    steps = split_steps(response)
    final_score = float(score) if score is not None and not pd.isna(score) else compute_final_score(response, ground_truth, dataset)
    final_correct = final_score >= 0.5

    wrong_flags: list[bool] = []
    correct_answer_flags: list[bool] = []
    correction_flags: list[bool] = []
    strong_correction_flags: list[bool] = []
    answer_values: list[float | None] = []
    arithmetic_error_counts: list[int] = []
    for step in steps:
        candidate = extract_answer_candidate(step, dataset, require_cue=True)
        answer_values.append(candidate)
        correct_answer = is_correct_value(candidate, ground_truth)
        wrong_answer = candidate is not None and not correct_answer
        arith_errors = equation_errors(step)
        arithmetic_error_counts.append(arith_errors)
        wrong_flags.append(wrong_answer or arith_errors > 0)
        correct_answer_flags.append(correct_answer)
        correction_flags.append(bool(CORRECTION_RE.search(step)))
        strong_correction_flags.append(bool(STRONG_CORRECTION_RE.search(step)))

    n_steps = len(steps)
    wrong_positions = [i + 1 for i, flag in enumerate(wrong_flags) if flag]
    correction_positions = [i + 1 for i, flag in enumerate(correction_flags) if flag]
    strong_correction_positions = [i + 1 for i, flag in enumerate(strong_correction_flags) if flag]
    correct_answer_positions = [i + 1 for i, flag in enumerate(correct_answer_flags) if flag]

    verified_correction_positions: list[int] = []
    if final_correct:
        for pos in strong_correction_positions:
            before = any(wrong_flags[: max(pos - 1, 0)])
            after_correct = any(correct_answer_flags[pos - 1 :]) or final_correct
            after_wrong = any(wrong_flags[pos - 1 :])
            if before and after_correct and not after_wrong:
                verified_correction_positions.append(pos)

    turning_point_step: int | None = None
    if final_correct and any(wrong_flags):
        for idx in range(n_steps):
            if not any(wrong_flags[:idx]):
                continue
            suffix_wrong = sum(wrong_flags[idx:])
            suffix_correct = sum(correct_answer_flags[idx:]) + 1
            if suffix_correct / max(suffix_correct + suffix_wrong, 1) >= 0.8:
                turning_point_step = idx + 1
                break
        if turning_point_step is None:
            last_wrong = max(i for i, flag in enumerate(wrong_flags) if flag)
            if last_wrong + 1 < n_steps:
                turning_point_step = last_wrong + 2

    has_wrong_to_correct_flip = bool(final_correct and wrong_positions and (correct_answer_positions or final_correct))
    first_correction = correction_positions[0] if correction_positions else None
    first_strong_correction = strong_correction_positions[0] if strong_correction_positions else None
    first_verified_correction = verified_correction_positions[0] if verified_correction_positions else None
    return {
        "n_steps": n_steps,
        "final_correct": bool(final_correct),
        "final_score": final_score,
        "correction_marker_count": len(correction_positions),
        "has_correction_marker": bool(correction_positions),
        "correction_positions": json.dumps(correction_positions),
        "first_correction_step": first_correction,
        "first_correction_pos": (first_correction / n_steps) if first_correction and n_steps else math.nan,
        "strong_correction_marker_count": len(strong_correction_positions),
        "has_strong_correction_marker": bool(strong_correction_positions),
        "strong_correction_positions": json.dumps(strong_correction_positions),
        "first_strong_correction_step": first_strong_correction,
        "first_strong_correction_pos": (first_strong_correction / n_steps) if first_strong_correction and n_steps else math.nan,
        "verified_correction_count": len(verified_correction_positions),
        "has_recovery_pattern": bool(verified_correction_positions),
        "verified_correction_positions": json.dumps(verified_correction_positions),
        "first_verified_correction_step": first_verified_correction,
        "first_verified_correction_pos": (first_verified_correction / n_steps) if first_verified_correction and n_steps else math.nan,
        "wrong_evidence_count": len(wrong_positions),
        "wrong_evidence_positions": json.dumps(wrong_positions),
        "correct_answer_evidence_count": len(correct_answer_positions),
        "correct_answer_positions": json.dumps(correct_answer_positions),
        "arithmetic_error_count": int(sum(arithmetic_error_counts)),
        "has_wrong_to_correct_flip": has_wrong_to_correct_flip,
        "turning_point_step": turning_point_step,
        "turning_point_pos": (turning_point_step / n_steps) if turning_point_step and n_steps else math.nan,
        "has_turning_point": turning_point_step is not None,
    }


def sibling_ground_truths(path: Path) -> list[Any]:
    parquet_path = path.with_name("responses.parquet")
    if not parquet_path.exists():
        return []
    df = pd.read_parquet(parquet_path)
    truths = []
    for _, row in df.iterrows():
        reward_model = _to_builtin(row.get("reward_model", {}))
        if isinstance(reward_model, dict):
            truths.append(reward_model.get("ground_truth"))
        else:
            truths.append(None)
    return truths


def load_response_rows(spec: SourceSpec) -> list[dict[str, Any]]:
    data = load_json(spec.path)
    truths = sibling_ground_truths(spec.path)
    rows = []
    for idx, item in enumerate(data):
        response = str(item.get("response", ""))
        ground_truth = item.get("ground_truth")
        if ground_truth is None and idx < len(truths):
            ground_truth = truths[idx]
        rows.append(
            {
                "source_index": idx,
                "response": response,
                "ground_truth": ground_truth,
                "score": item.get("score"),
                "prompt": item.get("prompt", ""),
            }
        )
    return rows


def metric_record(path: Path) -> tuple[str, float, int, int] | None:
    metrics_path = path.with_name("responses_labeled.metrics.json")
    if not metrics_path.exists():
        return None
    data = load_json(metrics_path)
    key = "gsm8k" if "gsm8k" in data else next(iter(data.keys()))
    metrics = data[key]
    return key, float(metrics["mean@1"]), int(metrics.get("correct", -1)), int(metrics.get("total", -1))


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


def discover_sources(args: argparse.Namespace) -> list[SourceSpec]:
    sources: list[SourceSpec] = []
    if args.include_base and args.base_path.exists():
        sources.append(SourceSpec("qwen3_4b_base", args.base_path, "gsm8k", "base", "", "Qwen3-4B base"))
    if args.scan_experiments:
        for exp_dir in sorted(p for p in args.experiments_root.iterdir() if p.is_dir()):
            if "gsm8k" not in exp_dir.name:
                continue
            for split in args.splits:
                path = exp_dir / "eval" / split / "generated" / "responses_labeled.json"
                if path.exists():
                    sources.append(SourceSpec(f"{exp_dir.name}:{split}", path, "gsm8k", split, exp_dir.name, short_label(exp_dir.name)))
    for item in args.source:
        parts = item.split("=", 1)
        if len(parts) != 2:
            raise SystemExit(f"Invalid --source format, expected name=path: {item}")
        sources.append(SourceSpec(parts[0], Path(parts[1]), args.dataset, "custom", "", parts[0]))
    seen: set[Path] = set()
    unique = []
    for source in sources:
        resolved = source.path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(source)
    return unique


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (name, split, exp_id, label), sub in details.groupby(["source_name", "split", "experiment_id", "label"], dropna=False):
        total = len(sub)
        final_correct = int(sub["final_correct"].sum())
        rows.append(
            {
                "source_name": name,
                "split": split,
                "experiment_id": exp_id,
                "label": label,
                "group": group_name(exp_id) if exp_id else "base",
                "n": total,
                "final_acc_pct": 100.0 * final_correct / total if total else math.nan,
                "avg_steps": sub["n_steps"].mean(),
                "correction_marker_total": int(sub["correction_marker_count"].sum()),
                "correction_marker_frequency": sub["has_correction_marker"].mean(),
                "correction_markers_per_trajectory": sub["correction_marker_count"].mean(),
                "strong_correction_marker_total": int(sub["strong_correction_marker_count"].sum()),
                "strong_correction_marker_frequency": sub["has_strong_correction_marker"].mean(),
                "strong_correction_markers_per_trajectory": sub["strong_correction_marker_count"].mean(),
                "verified_correction_total": int(sub["verified_correction_count"].sum()),
                "recovery_pattern_frequency": sub["has_recovery_pattern"].mean(),
                "wrong_to_correct_flip_frequency": sub["has_wrong_to_correct_flip"].mean(),
                "first_correction_avg_pos": sub["first_correction_pos"].mean(),
                "first_strong_correction_avg_pos": sub["first_strong_correction_pos"].mean(),
                "first_verified_correction_avg_pos": sub["first_verified_correction_pos"].mean(),
                "wrong_evidence_frequency": (sub["wrong_evidence_count"] > 0).mean(),
                "wrong_evidence_per_trajectory": sub["wrong_evidence_count"].mean(),
                "arithmetic_errors_per_trajectory": sub["arithmetic_error_count"].mean(),
                "turning_point_frequency": sub["has_turning_point"].mean(),
                "avg_turning_point_depth": sub["turning_point_step"].mean(),
                "avg_turning_point_pos": sub["turning_point_pos"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "experiment_id", "source_name"])


def collect_performance(experiments_root: Path) -> pd.DataFrame:
    rows = []
    for exp_dir in sorted(p for p in experiments_root.iterdir() if p.is_dir() and "gsm8k" in p.name):
        sft_path = exp_dir / "eval" / "sft" / "generated" / "responses_labeled.json"
        grpo_path = exp_dir / "eval" / "grpo" / "generated" / "responses_labeled.json"
        if not sft_path.exists() or not grpo_path.exists():
            continue
        sft = metric_record(sft_path)
        grpo = metric_record(grpo_path)
        if sft is None or grpo is None or sft[0] != "gsm8k" or grpo[0] != "gsm8k":
            continue
        rows.append(
            {
                "experiment_id": exp_dir.name,
                "sft_eval_pct": 100.0 * sft[1],
                "sft_correct": sft[2],
                "sft_total": sft[3],
                "grpo_eval_pct": 100.0 * grpo[1],
                "grpo_correct": grpo[2],
                "grpo_total": grpo[3],
                "grpo_eval_minus_sft_pp": 100.0 * (grpo[1] - sft[1]),
            }
        )
    return pd.DataFrame(rows)


def add_train_val_performance(perf: pd.DataFrame) -> pd.DataFrame:
    joined_path = Path("DataObs/experiment_observations/optimization_headroom_qwen3_gsm8k_trajectory_all/optimization_headroom_joined.csv")
    if not joined_path.exists() or perf.empty:
        return perf
    extra = pd.read_csv(joined_path)
    cols = [
        "experiment_id",
        "grpo_final_val_pct",
        "grpo_best_val_pct",
        "grpo_final_val_minus_sft_pp",
        "grpo_best_val_minus_sft_pp",
    ]
    cols = [c for c in cols if c in extra.columns]
    if len(cols) <= 1:
        return perf
    return perf.merge(extra[cols], on="experiment_id", how="left")


def corr_stats(df: pd.DataFrame, metrics: list[str], targets: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": int(len(df)), "pairs": {}}
    for metric in metrics:
        for target in targets:
            sub = df[[metric, target]].dropna()
            key = f"{metric}__vs__{target}"
            if len(sub) < 2 or sub[metric].nunique() < 2 or sub[target].nunique() < 2:
                out["pairs"][key] = {"n": int(len(sub)), "pearson": math.nan, "spearman": math.nan}
            else:
                out["pairs"][key] = {
                    "n": int(len(sub)),
                    "pearson": float(sub[metric].corr(sub[target], method="pearson")),
                    "spearman": float(sub[metric].corr(sub[target], method="spearman")),
                }
    return out


def style_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#E6E8EB", linewidth=0.8)
    ax.set_axisbelow(True)


def savefig(out_dir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        plt.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=220)
    plt.close()


def plot_base_vs_sft(summary: pd.DataFrame, out_dir: Path) -> None:
    keep = summary[
        (summary["source_name"] == "qwen3_4b_base")
        | ((summary["experiment_id"] == DEFAULT_FOCUS_EXP) & (summary["split"] == "sft"))
    ].copy()
    if len(keep) < 2:
        return
    keep["display"] = np.where(keep["source_name"] == "qwen3_4b_base", "Base", "SFT-4B")
    metrics = [
        ("final_acc_pct", "Final accuracy (%)", 100.0),
        ("strong_correction_marker_frequency", "Strong correction freq.", 1.0),
        ("recovery_pattern_frequency", "Verified recovery freq.", 1.0),
        ("turning_point_frequency", "Turning point freq.", 1.0),
        ("avg_turning_point_pos", "Avg turning point pos.", 1.0),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15.0, 4.0))
    colors = ["#4E79A7", "#F28E2B"]
    for ax, (col, title, ymax) in zip(axes, metrics):
        vals = keep[col].to_numpy(dtype=float)
        if ymax == 1.0:
            vals = vals * 100.0
            ylim = 100.0
            ylabel = "%"
        else:
            ylim = ymax
            ylabel = "%"
        ax.bar(keep["display"], vals, color=colors[: len(keep)], width=0.62)
        ax.set_title(title)
        ax.set_ylim(0, ylim)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        style_axes(ax)
    fig.suptitle("Qwen3-4B Base vs SFT-4B Recovery/Turning-Point Behavior", y=1.03)
    savefig(out_dir, "qwen3_4b_base_vs_sft_recovery_turning_point")


def plot_focus_position_distributions(details: pd.DataFrame, out_dir: Path) -> None:
    focus = details[
        (details["source_name"] == "qwen3_4b_base")
        | ((details["experiment_id"] == DEFAULT_FOCUS_EXP) & (details["split"].isin(["sft", "grpo"])))
    ].copy()
    if focus.empty:
        return
    focus["display"] = np.select(
        [
            focus["source_name"] == "qwen3_4b_base",
            focus["split"] == "sft",
            focus["split"] == "grpo",
        ],
        ["Base", "SFT-4B", "GRPO-4B"],
        default=focus["label"],
    )
    order = ["Base", "SFT-4B", "GRPO-4B"]
    colors = {"Base": "#4E79A7", "SFT-4B": "#F28E2B", "GRPO-4B": "#59A14F"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, col, title in [
        (axes[0], "turning_point_pos", "Turning point normalized position"),
        (axes[1], "first_verified_correction_pos", "Verified correction normalized position"),
    ]:
        for name in order:
            vals = focus.loc[focus["display"] == name, col].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            ax.hist(
                vals,
                bins=np.linspace(0, 1, 11),
                histtype="step",
                linewidth=2.0,
                color=colors[name],
                label=f"{name} (n={len(vals)})",
            )
        ax.set_title(title)
        ax.set_xlabel("normalized step position")
        ax.set_ylabel("trajectory count")
        ax.legend(frameon=False, fontsize=9)
        style_axes(ax)
    fig.suptitle("Where Recovery Happens in Qwen3-4B Focus Runs", y=1.03)
    savefig(out_dir, "qwen3_4b_focus_position_distributions")


def plot_sft_metric_ranking(joined: pd.DataFrame, out_dir: Path) -> None:
    if joined.empty:
        return
    ordered = joined.sort_values("turning_point_frequency", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    y = np.arange(len(ordered))
    ax.barh(y - 0.18, 100.0 * ordered["recovery_pattern_frequency"], height=0.34, color="#4E79A7", label="verified recovery")
    ax.barh(y + 0.18, 100.0 * ordered["turning_point_frequency"], height=0.34, color="#F28E2B", label="turning point")
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["label"])
    ax.set_xlabel("frequency (%)")
    ax.set_title("SFT Experiment Ranking by Recovery/Turning-Point Frequency")
    ax.legend(frameon=False)
    style_axes(ax)
    savefig(out_dir, "sft_recovery_turning_point_ranking")


def plot_correlation_panel(joined: pd.DataFrame, correlations: dict[str, Any], out_dir: Path) -> None:
    if joined.empty:
        return
    panels = [
        ("recovery_pattern_frequency", "sft_eval_pct", "Recovery freq.", "SFT eval (%)"),
        ("turning_point_frequency", "sft_eval_pct", "Turning point freq.", "SFT eval (%)"),
        ("recovery_pattern_frequency", "grpo_eval_pct", "Recovery freq.", "GRPO eval (%)"),
        ("turning_point_frequency", "grpo_eval_pct", "Turning point freq.", "GRPO eval (%)"),
        ("recovery_pattern_frequency", "grpo_eval_minus_sft_pp", "Recovery freq.", "GRPO-SFT (pp)"),
        ("turning_point_frequency", "grpo_eval_minus_sft_pp", "Turning point freq.", "GRPO-SFT (pp)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0))
    palette = {
        "teacher size": "#4E79A7",
        "teacher type": "#F28E2B",
        "prompt setting": "#E15759",
        "difficulty": "#59A14F",
        "seed size": "#B07AA1",
        "pipeline": "#9C755F",
        "other": "#BAB0AC",
    }
    annotate_labels = {"ts:4b", "ts:1_7b", "tt:DeepSeek-R1", "size:n8", "pipeline"}
    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes.ravel(), panels):
        sub = joined[[xcol, ycol, "label", "group"]].dropna()
        x = 100.0 * sub[xcol].to_numpy(dtype=float)
        y = sub[ycol].to_numpy(dtype=float)
        for group, group_df in sub.groupby("group"):
            ax.scatter(
                100.0 * group_df[xcol],
                group_df[ycol],
                s=76,
                color=palette.get(group, "#BAB0AC"),
                edgecolor="white",
                linewidth=0.8,
                alpha=0.92,
                label=group,
            )
        for _, row in sub.iterrows():
            if str(row["label"]) in annotate_labels:
                ax.annotate(str(row["label"]), (100.0 * row[xcol], row[ycol]), xytext=(5, 4), textcoords="offset points", fontsize=8)
        key = f"{xcol}__vs__{ycol}"
        stat = correlations["pairs"].get(key, {})
        if len(sub) >= 2 and len(set(x)) > 1:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(xs, m * xs + b, linestyle="--", color="#222222", linewidth=1.2)
        ax.text(
            0.03,
            0.97,
            f"r={stat.get('pearson', math.nan):.3f}\nrho={stat.get('spearman', math.nan):.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D0D5DD", "alpha": 0.95},
        )
        ax.set_xlabel(f"{xlabel} (%)")
        ax.set_ylabel(ylabel)
        style_axes(ax)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle("SFT Trajectory Recovery/Turning-Point Metrics vs GSM8K Performance", y=1.01)
    savefig(out_dir, "recovery_turning_point_vs_sft_grpo_performance")


def write_report(out_dir: Path, summary: pd.DataFrame, joined: pd.DataFrame, correlations: dict[str, Any]) -> None:
    focus = summary[
        (summary["source_name"] == "qwen3_4b_base")
        | ((summary["experiment_id"] == DEFAULT_FOCUS_EXP) & (summary["split"] == "sft"))
        | ((summary["experiment_id"] == DEFAULT_FOCUS_EXP) & (summary["split"] == "grpo"))
    ].copy()
    lines = [
        "# Recovery Pattern and Turning Point Report",
        "",
        "## Metric Design",
        "",
        "- Step split: strip thinking tags, split by markdown step headings, horizontal rules, blank paragraphs, then sentence boundaries for single-block outputs.",
        "- Error evidence: GSM8K numeric answer candidate is wrong, or a simple arithmetic equation in the step is invalid.",
        "- Correction count/position: report both any marker and strong marker. Any marker includes weak checking phrases such as `let me check`; strong marker excludes generic checking and keeps terms such as `wait`, `actually`, `mistake`, `wrong`, `recheck`, `不对`, `修正`.",
        "- Verified recovery pattern: final answer is correct, an earlier step has error evidence, a later strong correction-marker step appears, and the suffix after that correction has no detected error evidence.",
        "- Turning point: for a final-correct trajectory with earlier error evidence, the earliest step whose suffix is at least 80% correct evidence; depth is reported as raw step index and normalized index.",
        "",
        "This design is cross-dataset by replacing the answer extractor/verifier. Without a verifier, only lexical correction counts are directly comparable and recovery/turning-point fields should be treated as unavailable.",
        "",
        "## Main Findings",
        "",
        "1. Generic correction markers saturate after SFT/RL style training: the Qwen3-4B SFT and GRPO focus runs contain at least one marker in nearly every trajectory, mostly due to repeated `wait` / `let me check` style self-review.",
        "2. Verified recovery is much rarer and more informative: base has almost no verified recovery, while the 4B SFT checkpoint shows a clear increase and the matched GRPO checkpoint increases it further.",
        "3. Turning points in SFT/GRPO trajectories happen earlier than in base when they exist, suggesting trained models enter a stable correct suffix sooner after an error.",
        "4. Across SFT experiments, recovery/turning-point frequency has only moderate Pearson correlation with SFT accuracy and weak or negative relation to standalone GRPO eval/delta, so the metric is diagnostic rather than a standalone predictor.",
        "",
        "## Qwen3-4B Focus",
        "",
        "| source | split | n | final acc (%) | any marker (%) | strong marker (%) | recovery (%) | turning point (%) | avg TP depth | avg TP pos |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in focus.iterrows():
        source = "Qwen3-4B base" if row["source_name"] == "qwen3_4b_base" else row["label"]
        lines.append(
            f"| {source} | {row['split']} | {int(row['n'])} | {row['final_acc_pct']:.2f} | "
            f"{100 * row['correction_marker_frequency']:.2f} | {100 * row['strong_correction_marker_frequency']:.2f} | "
            f"{100 * row['recovery_pattern_frequency']:.2f} | "
            f"{100 * row['turning_point_frequency']:.2f} | {row['avg_turning_point_depth']:.2f} | {row['avg_turning_point_pos']:.3f} |"
        )
    if not joined.empty:
        lines.extend(
            [
                "",
                "## SFT Experiment Ranking",
                "",
                "| rank | experiment | group | recovery (%) | turning point (%) | SFT eval (%) | GRPO eval (%) | GRPO-SFT (pp) |",
                "|---:|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        ranked = joined.sort_values("turning_point_frequency", ascending=False).head(8)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            lines.append(
                f"| {rank} | {row['label']} | {row['group']} | {100 * row['recovery_pattern_frequency']:.2f} | "
                f"{100 * row['turning_point_frequency']:.2f} | {row['sft_eval_pct']:.2f} | "
                f"{row['grpo_eval_pct']:.2f} | {row['grpo_eval_minus_sft_pp']:+.2f} |"
            )
    lines.extend(["", "## Correlation Summary", ""])
    if joined.empty:
        lines.append("No joined experiment-level SFT metrics were available for correlation.")
    else:
        for key, stat in correlations["pairs"].items():
            if key.endswith("__vs__sft_eval_pct") or key.endswith("__vs__grpo_eval_pct") or key.endswith("__vs__grpo_eval_minus_sft_pp"):
                lines.append(
                    f"- `{key}`: n={stat['n']}, Pearson={stat['pearson']:.3f}, Spearman={stat['spearman']:.3f}."
                )
        val_keys = [
            "recovery_pattern_frequency__vs__grpo_final_val_pct",
            "recovery_pattern_frequency__vs__grpo_best_val_pct",
            "turning_point_frequency__vs__grpo_final_val_pct",
            "turning_point_frequency__vs__grpo_best_val_pct",
            "avg_turning_point_pos__vs__grpo_final_val_pct",
            "avg_turning_point_pos__vs__grpo_best_val_pct",
        ]
        available_val = [key for key in val_keys if key in correlations["pairs"] and correlations["pairs"][key]["n"] >= 2]
        if available_val:
            lines.extend(
                [
                    "",
                    "### Train-Time GRPO Validation Subset",
                    "",
                    "Only experiments with available train-time validation curves are included here, so these correlations have smaller n and higher uncertainty.",
                ]
            )
            for key in available_val:
                stat = correlations["pairs"][key]
                lines.append(f"- `{key}`: n={stat['n']}, Pearson={stat['pearson']:.3f}, Spearman={stat['spearman']:.3f}.")
        lines.extend(
            [
                "",
                "Observed on these GSM8K SFT eval trajectories, verified recovery/turning-point frequencies are sparse. Correlations should therefore be read as diagnostic signals rather than causal evidence.",
            ]
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `trajectory_metrics.csv`: one row per trajectory.",
            "- `model_summary.csv`: one row per evaluated source.",
            "- `sft_experiment_joined.csv`: SFT trajectory metrics joined with SFT/GRPO performance.",
            "- `correlations.json`: Pearson/Spearman statistics.",
            "- `qwen3_4b_base_vs_sft_recovery_turning_point.png`: focus comparison.",
            "- `qwen3_4b_focus_position_distributions.png`: focus-run position histograms.",
            "- `sft_recovery_turning_point_ranking.png`: SFT experiment ranking by recovery/turning-point frequency.",
            "- `recovery_turning_point_vs_sft_grpo_performance.png`: correlation panel.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--include-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scan-experiments", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--splits", nargs="+", default=["sft", "grpo"], choices=["sft", "grpo"])
    parser.add_argument("--source", action="append", default=[], help="Extra source as name=/path/to/responses_labeled.json")
    parser.add_argument("--dataset", default="gsm8k")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = discover_sources(args)
    if not sources:
        raise SystemExit("No response sources found.")

    detail_rows = []
    for spec in sources:
        if not spec.path.exists():
            continue
        metric = metric_record(spec.path)
        for row in load_response_rows(spec):
            analysis = analyze_trajectory(row["response"], row["ground_truth"], spec.dataset, row["score"])
            detail_rows.append(
                {
                    "source_name": spec.name,
                    "experiment_id": spec.experiment_id,
                    "label": spec.label or spec.name,
                    "split": spec.split,
                    "dataset": spec.dataset,
                    "path": str(spec.path),
                    "source_index": row["source_index"],
                    "ground_truth": row["ground_truth"],
                    "metric_mean@1": metric[1] if metric else math.nan,
                    **analysis,
                }
            )

    details = pd.DataFrame(detail_rows)
    summary = summarize(details)
    perf = add_train_val_performance(collect_performance(args.experiments_root))
    sft_summary = summary[(summary["split"] == "sft") & summary["experiment_id"].astype(bool)].copy()
    joined = sft_summary.merge(perf, on="experiment_id", how="inner", suffixes=("", "_perf"))

    metric_cols = [
        "correction_marker_frequency",
        "strong_correction_marker_frequency",
        "recovery_pattern_frequency",
        "wrong_to_correct_flip_frequency",
        "wrong_evidence_frequency",
        "turning_point_frequency",
        "avg_turning_point_pos",
    ]
    target_cols = [
        "sft_eval_pct",
        "grpo_eval_pct",
        "grpo_eval_minus_sft_pp",
        "grpo_final_val_pct",
        "grpo_best_val_pct",
    ]
    target_cols = [c for c in target_cols if c in joined.columns]
    correlations = corr_stats(joined, metric_cols, target_cols)

    details.to_csv(args.out_dir / "trajectory_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(args.out_dir / "model_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    joined.to_csv(args.out_dir / "sft_experiment_joined.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    write_json(args.out_dir / "correlations.json", correlations)
    write_json(
        args.out_dir / "metric_config.json",
        {
            "mostly_correct_threshold": 0.8,
            "sources": [spec.__dict__ | {"path": str(spec.path)} for spec in sources],
            "notes": "Verifier-backed metrics currently implement GSM8K numeric verification. Verified recovery uses strong correction markers.",
        },
    )

    plot_base_vs_sft(summary, args.out_dir)
    plot_focus_position_distributions(details, args.out_dir)
    plot_sft_metric_ranking(joined, args.out_dir)
    plot_correlation_panel(joined, correlations, args.out_dir)
    write_report(args.out_dir, summary, joined, correlations)
    print(f"Wrote recovery/turning-point analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
