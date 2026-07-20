#!/usr/bin/env python3
"""Stratify recovery/turning-point metrics by GSM8K difficulty buckets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path("DataObs/experiment_observations/recovery_turning_point_qwen3_gsm8k")
TRAJECTORY_CSV = OUT_DIR / "trajectory_metrics.csv"
PASSK_PER_PROBLEM = Path("outputs/evals/qwen3_4b_gsm8k_passk_k16_len16k/passk/passk_per_problem.json")
FOCUS_EXP = "exp1100_gsm8k_qwen3_teacher_qwen3_4b_t0.7_n4"


def load_passk_buckets() -> pd.DataFrame:
    data = json.loads(PASSK_PER_PROBLEM.read_text(encoding="utf-8"))
    df = pd.DataFrame(data).rename(columns={"index": "source_index", "num_correct": "base_k16_correct"})
    df = df[["source_index", "base_k16_correct"]].copy()
    df["difficulty_bucket"] = df["base_k16_correct"].map(
        lambda count: "easy_16of16" if count == 16 else ("hard_0of16" if count == 0 else "medium_1to15")
    )
    df["difficulty_label"] = df["difficulty_bucket"].map(
        {
            "easy_16of16": "easy: base 16/16 correct",
            "medium_1to15": "medium: base 1-15/16 correct",
            "hard_0of16": "hard: base 0/16 correct",
        }
    )
    return df


def summarize(grouped: Any) -> pd.DataFrame:
    rows = []
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {
            "n": int(len(sub)),
            "base_k16_correct_mean": float(sub["base_k16_correct"].mean()),
            "final_acc_pct": 100.0 * float(sub["final_correct"].mean()),
            "strong_marker_pct": 100.0 * float(sub["has_strong_correction_marker"].mean()),
            "wrong_evidence_pct": 100.0 * float((sub["wrong_evidence_count"] > 0).mean()),
            "recovery_pct": 100.0 * float(sub["has_recovery_pattern"].mean()),
            "turning_point_pct": 100.0 * float(sub["has_turning_point"].mean()),
            "avg_turning_point_pos": float(sub["turning_point_pos"].mean()),
        }
        rows.append((*keys, row))
    out_rows = []
    for item in rows:
        *keys, metrics = item
        out_rows.append({**{f"key_{idx}": key for idx, key in enumerate(keys)}, **metrics})
    return pd.DataFrame(out_rows)


def style_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#E6E8EB", linewidth=0.8)
    ax.set_axisbelow(True)


def savefig(name: str) -> None:
    for ext in ("png", "pdf"):
        plt.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=220)
    plt.close()


def plot_focus(focus: pd.DataFrame) -> None:
    source_order = ["Base", "SFT-4B", "GRPO-4B"]
    bucket_order = ["easy_16of16", "medium_1to15", "hard_0of16"]
    bucket_labels = ["easy\n16/16", "medium\n1-15/16", "hard\n0/16"]
    colors = {"Base": "#4E79A7", "SFT-4B": "#F28E2B", "GRPO-4B": "#59A14F"}
    metrics = [
        ("final_acc_pct", "Final accuracy (%)"),
        ("wrong_evidence_pct", "Wrong evidence (%)"),
        ("recovery_pct", "Verified recovery (%)"),
        ("turning_point_pct", "Turning point (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    x = np.arange(len(bucket_order))
    width = 0.24
    for ax, (col, title) in zip(axes.ravel(), metrics):
        for idx, source in enumerate(source_order):
            sub = focus[focus["source_display"] == source].set_index("difficulty_bucket").reindex(bucket_order)
            ax.bar(x + (idx - 1) * width, sub[col], width=width, color=colors[source], label=source)
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_labels)
        ax.set_ylabel("%")
        ax.set_title(title)
        style_axes(ax)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Recovery/Turning Point by Base Pass@16 Difficulty Bucket", y=1.02)
    savefig("difficulty_stratified_focus")


def plot_all_sft(all_sft: pd.DataFrame) -> None:
    bucket_order = ["easy_16of16", "medium_1to15", "hard_0of16"]
    bucket_labels = ["easy\n16/16", "medium\n1-15/16", "hard\n0/16"]
    sub = all_sft.set_index("difficulty_bucket").reindex(bucket_order)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    x = np.arange(len(bucket_order))
    width = 0.34
    ax.bar(x - width / 2, sub["recovery_pct"], width=width, color="#4E79A7", label="verified recovery")
    ax.bar(x + width / 2, sub["turning_point_pct"], width=width, color="#F28E2B", label="turning point")
    for idx, row in enumerate(sub.itertuples()):
        ax.text(idx - width / 2, row.recovery_pct + 0.5, f"n={int(row.n)}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylabel("frequency (%)")
    ax.set_title("All SFT Runs: Recovery/Turning Point by Difficulty")
    ax.legend(frameon=False)
    style_axes(ax)
    savefig("difficulty_stratified_all_sft")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    traj = pd.read_csv(TRAJECTORY_CSV)
    buckets = load_passk_buckets()
    joined = traj.merge(buckets, on="source_index", how="inner")

    focus = joined[
        (joined["source_name"] == "qwen3_4b_base")
        | ((joined["experiment_id"] == FOCUS_EXP) & (joined["split"].isin(["sft", "grpo"])))
    ].copy()
    focus["source_display"] = np.select(
        [focus["source_name"] == "qwen3_4b_base", focus["split"] == "sft", focus["split"] == "grpo"],
        ["Base", "SFT-4B", "GRPO-4B"],
        default=focus["source_name"],
    )
    focus_summary = summarize(focus.groupby(["source_display", "split", "difficulty_bucket", "difficulty_label"], dropna=False))
    focus_summary = focus_summary.rename(
        columns={"key_0": "source_display", "key_1": "split", "key_2": "difficulty_bucket", "key_3": "difficulty_label"}
    )

    sft = joined[(joined["split"] == "sft") & joined["experiment_id"].astype(bool)].copy()
    all_sft = summarize(sft.groupby(["difficulty_bucket", "difficulty_label"], dropna=False))
    all_sft = all_sft.rename(columns={"key_0": "difficulty_bucket", "key_1": "difficulty_label"})

    by_exp = summarize(sft.groupby(["label", "difficulty_bucket", "difficulty_label"], dropna=False))
    by_exp = by_exp.rename(columns={"key_0": "label", "key_1": "difficulty_bucket", "key_2": "difficulty_label"})

    focus_summary.to_csv(OUT_DIR / "difficulty_stratified_focus.csv", index=False)
    all_sft.to_csv(OUT_DIR / "difficulty_stratified_all_sft.csv", index=False)
    by_exp.to_csv(OUT_DIR / "difficulty_stratified_by_sft_experiment.csv", index=False)

    plot_focus(focus_summary)
    plot_all_sft(all_sft)
    print(f"Wrote difficulty stratification outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
