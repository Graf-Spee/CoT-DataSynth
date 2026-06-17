#!/usr/bin/env python3
"""Parse VERL GRPO logs into CSV metrics and optional curve plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RAY_PREFIX_RE = re.compile(r"\([^)]* pid=\d+\)\s*")
STEP_RE = re.compile(r"\bstep:(?P<step>\d+)\s+-\s+(?P<body>.*)")
PAIR_RE = re.compile(
    r"(?P<key>[A-Za-z0-9_./@+-]+):"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf|-inf)"
)

PREFERRED_COLUMNS = [
    "step",
    "training/global_step",
    "training/epoch",
    "critic/rewards/mean",
    "critic/score/mean",
    "critic/advantages/mean",
    "actor/entropy",
    "actor/kl_loss",
    "actor/kl_coef",
    "actor/pg_loss",
    "actor/pg_clipfrac",
    "actor/ppo_kl",
    "actor/grad_norm",
    "actor/lr",
    "response_length/mean",
    "response_length/max",
    "response_length/clip_ratio",
    "prompt_length/mean",
    "perf/max_memory_allocated_gb",
    "perf/max_memory_reserved_gb",
    "perf/cpu_memory_used_gb",
    "timing_s/step",
    "timing_s/testing",
    "perf/throughput",
]

DERIVED_COLUMNS = [
    "derived/train_accuracy_proxy",
    "derived/reward_rolling_mean",
    "derived/score_rolling_mean",
    "derived/entropy_rolling_mean",
    "derived/kl_loss_rolling_mean",
    "derived/ppo_kl_rolling_mean",
    "derived/grad_norm_rolling_mean",
    "derived/grad_norm_over_clip",
    "derived/grad_clip_hit",
]

PAPER_PLOTS = [
    {
        "stem": "reward",
        "title": "Reward / Accuracy",
        "ylabel": "Reward",
        "metrics": [
            ("critic/rewards/mean", "Train reward", "#4C78A8", "-", "o"),
            ("derived/reward_rolling_mean", "Train reward (MA)", "#1F4E79", "-", None),
            ("derived/val_accuracy_proxy", "Validation reward@1", "#F58518", "--", "s"),
        ],
        "extra": "validation_reward",
    },
    {
        "stem": "entropy",
        "title": "Policy Entropy",
        "ylabel": "Entropy",
        "metrics": [
            ("actor/entropy", "Entropy", "#54A24B", "-", "o"),
            ("derived/entropy_rolling_mean", "Entropy (MA)", "#2F6B2F", "-", None),
        ],
        "extra": None,
    },
    {
        "stem": "grad_norm",
        "title": "Gradient Norm",
        "ylabel": "Gradient norm",
        "metrics": [
            ("actor/grad_norm", "Grad norm", "#B279A2", "-", "o"),
            ("derived/grad_norm_rolling_mean", "Grad norm (MA)", "#7A3F6C", "-", None),
        ],
        "extra": None,
    },
    {
        "stem": "kl",
        "title": "KL Divergence",
        "ylabel": "KL",
        "metrics": [
            ("actor/kl_loss", "KL loss", "#E45756", "-", "o"),
            ("derived/kl_loss_rolling_mean", "KL loss (MA)", "#A12D2D", "-", None),
            ("actor/ppo_kl", "PPO KL", "#72B7B2", "--", "^"),
        ],
        "extra": None,
    },
]


def clean_line(line: str) -> str:
    line = ANSI_RE.sub("", line)
    line = line.replace("\r", "\n")
    line = RAY_PREFIX_RE.sub("", line)
    return line


def parse_float(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def parse_log(log_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            for line in clean_line(raw_line).splitlines():
                match = STEP_RE.search(line)
                if not match:
                    continue
                row: dict[str, float] = {"step": float(match.group("step"))}
                for pair in PAIR_RE.finditer(match.group("body")):
                    value = parse_float(pair.group("value"))
                    if value is not None:
                        row[pair.group("key")] = value
                rows.append(row)
    return rows


def deduplicate_rows(rows: list[dict[str, float]], mode: str) -> list[dict[str, float]]:
    if mode == "none":
        return rows
    by_step: dict[int, dict[str, float]] = {}
    order: list[int] = []
    for row in rows:
        step = int(row["step"])
        if step not in by_step:
            order.append(step)
            by_step[step] = row
            continue
        if mode == "last":
            by_step[step] = row
    return [by_step[step] for step in sorted(order)]


def collect_columns(rows: list[dict[str, float]]) -> list[str]:
    keys = {key for row in rows for key in row}
    val_reward_cols = sorted(
        key for key in keys if key.startswith("val-") and "/reward/" in key
    )
    derived = [key for key in DERIVED_COLUMNS if key in keys]
    preferred = [key for key in PREFERRED_COLUMNS if key in keys]
    remaining = sorted(keys - set(preferred) - set(derived) - set(val_reward_cols) - {"step"})
    columns = ["step"]
    columns.extend(key for key in preferred if key != "step")
    columns.extend(val_reward_cols)
    columns.extend(derived)
    columns.extend(remaining)
    return columns


def metric_values(rows: list[dict[str, float]], key: str) -> list[tuple[float, float]]:
    return [(row["step"], row[key]) for row in rows if key in row]


def rolling_mean(rows: list[dict[str, float]], key: str, window: int) -> dict[int, float]:
    values: list[float] = []
    result: dict[int, float] = {}
    for row in rows:
        if key not in row:
            continue
        values.append(row[key])
        result[int(row["step"])] = sum(values[-window:]) / min(len(values), window)
    return result


def looks_like_binary_score(rows: list[dict[str, float]]) -> bool:
    mins = [row["critic/score/min"] for row in rows if "critic/score/min" in row]
    maxs = [row["critic/score/max"] for row in rows if "critic/score/max" in row]
    if not mins or not maxs:
        return False
    return min(mins) >= 0.0 and max(maxs) <= 1.0


def add_derived_metrics(
    rows: list[dict[str, float]],
    window: int,
    grad_clip: float | None,
) -> list[dict[str, float]]:
    derived_rows = [dict(row) for row in rows]
    by_step = {int(row["step"]): row for row in derived_rows}
    rolling_specs = {
        "critic/rewards/mean": "derived/reward_rolling_mean",
        "critic/score/mean": "derived/score_rolling_mean",
        "actor/entropy": "derived/entropy_rolling_mean",
        "actor/kl_loss": "derived/kl_loss_rolling_mean",
        "actor/ppo_kl": "derived/ppo_kl_rolling_mean",
        "actor/grad_norm": "derived/grad_norm_rolling_mean",
    }
    for source, target in rolling_specs.items():
        for step, value in rolling_mean(derived_rows, source, window).items():
            by_step[step][target] = value

    if looks_like_binary_score(derived_rows):
        for row in derived_rows:
            if "critic/score/mean" in row:
                row["derived/train_accuracy_proxy"] = row["critic/score/mean"]

    val_reward_keys = sorted(
        key for row in derived_rows for key in row if key.startswith("val-") and "/reward/" in key
    )
    val_reward_keys = list(dict.fromkeys(val_reward_keys))
    if len(val_reward_keys) == 1:
        val_key = val_reward_keys[0]
        for row in derived_rows:
            if val_key in row:
                row["derived/val_accuracy_proxy"] = row[val_key]

    if grad_clip and grad_clip > 0:
        for row in derived_rows:
            if "actor/grad_norm" not in row:
                continue
            row["derived/grad_norm_over_clip"] = row["actor/grad_norm"] / grad_clip
            row["derived/grad_clip_hit"] = 1.0 if row["actor/grad_norm"] >= grad_clip else 0.0

    return derived_rows


def write_csv(rows: list[dict[str, float]], columns: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def summarize_series(rows: list[dict[str, float]], key: str) -> dict[str, float | int] | None:
    points = metric_values(rows, key)
    if not points:
        return None
    steps = [step for step, _ in points]
    values = [value for _, value in points]
    max_idx = max(range(len(values)), key=lambda idx: values[idx])
    min_idx = min(range(len(values)), key=lambda idx: values[idx])
    return {
        "count": len(values),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "first": values[0],
        "last": values[-1],
        "delta": values[-1] - values[0],
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": values[min_idx],
        "min_step": int(steps[min_idx]),
        "max": values[max_idx],
        "max_step": int(steps[max_idx]),
    }


def write_metric_summary(rows: list[dict[str, float]], output_path: Path) -> None:
    keys = sorted({key for row in rows for key in row if key != "step"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "metric",
            "count",
            "first_step",
            "last_step",
            "first",
            "last",
            "delta",
            "mean",
            "std",
            "min",
            "min_step",
            "max",
            "max_step",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in keys:
            summary = summarize_series(rows, key)
            if summary is None:
                continue
            writer.writerow({"metric": key, **summary})


def write_summary(rows: list[dict[str, float]], output_path: Path) -> None:
    latest = rows[-1]
    val_reward_keys = sorted(
        key for key in latest if key.startswith("val-") and "/reward/" in key
    )
    core_metrics = [
        "critic/rewards/mean",
        "critic/score/mean",
        "derived/train_accuracy_proxy",
        "derived/val_accuracy_proxy",
        "actor/entropy",
        "actor/kl_loss",
        "actor/ppo_kl",
        "actor/pg_loss",
        "actor/pg_clipfrac",
        "actor/pg_clipfrac_lower",
        "actor/grad_norm",
        "derived/grad_norm_over_clip",
        "derived/grad_clip_hit",
        "perf/max_memory_allocated_gb",
        "perf/max_memory_reserved_gb",
        "timing_s/step",
    ]
    summary = {
        "num_steps": len(rows),
        "first_step": int(rows[0]["step"]),
        "last_step": int(latest["step"]),
        "latest": {
            key: latest[key]
            for key in core_metrics
            if key in latest
        },
        "series_summary": {
            key: summarize_series(rows, key)
            for key in core_metrics
            if summarize_series(rows, key) is not None
        },
        "latest_validation_rewards": {key: latest[key] for key in val_reward_keys},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def setup_plot_env(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".plot_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))


def import_pyplot(output_dir: Path):
    setup_plot_env(output_dir)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local env
        print(f"[WARN] matplotlib unavailable, skip plots: {exc}", file=sys.stderr)
        return None
    return plt


def plot_main_metric_group(
    rows: list[dict[str, float]],
    config: dict,
    output_dir: Path,
) -> list[Path]:
    plt = import_pyplot(output_dir)
    if plt is None:
        return []

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 15,
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "lines.linewidth": 2.7,
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    plotted = False
    used_labels: set[str] = set()

    metrics = list(config["metrics"])
    if config.get("extra") == "validation_reward":
        val_keys = sorted(
            key
            for row in rows
            for key in row
            if key.startswith("val-") and "/reward/" in key
        )
        if val_keys:
            metrics = [item for item in metrics if item[0] != "derived/val_accuracy_proxy"]
        for idx, key in enumerate(dict.fromkeys(val_keys)):
            label = "Validation reward@1" if idx == 0 else key
            if key != "derived/val_accuracy_proxy":
                metrics.append((key, label, "#F58518", "--", "s"))

    for key, label, color, linestyle, marker in metrics:
        points = metric_values(rows, key)
        if not points:
            continue
        x_values, y_values = zip(*points)
        markevery = max(1, len(points) // 12)
        ax.plot(
            x_values,
            y_values,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=5.5 if marker else 0,
            markevery=markevery if marker else None,
            alpha=0.95,
            label=label if label not in used_labels else None,
        )
        used_labels.add(label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return []

    ax.set_title(config["title"], pad=12, weight="semibold")
    ax.set_xlabel("Training step")
    ax.set_ylabel(config["ylabel"])
    ax.grid(True, which="major", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    outputs = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{config['stem']}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_all(rows: list[dict[str, float]], output_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    for config in PAPER_PLOTS:
        outputs.extend(plot_main_metric_group(rows, config, output_dir))
    return outputs


def default_output_dir(log_path: Path) -> Path:
    if log_path.name == "grpo.log" and log_path.parent.name == "logs":
        return log_path.parent.parent / "grpo_metrics"
    return log_path.with_suffix("").parent / f"{log_path.stem}_metrics"


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse GRPO training log metrics.")
    parser.add_argument("--log", required=True, help="Path to grpo.log")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <experiment>/grpo_metrics for pipeline logs.",
    )
    parser.add_argument(
        "--deduplicate",
        choices=["last", "first", "none"],
        default="last",
        help="How to handle duplicate step lines. Default: last.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Rolling window size for derived smooth curves. Default: 5.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Grad clip threshold used to estimate clipping hits. Default follows ppo_trainer.yaml: 1.0.",
    )
    parser.add_argument("--plot", action="store_true", help="Write PNG curve plots.")
    args = parser.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    if not log_path.exists():
        raise SystemExit(f"[ERROR] log not found: {log_path}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(log_path)
    rows = deduplicate_rows(parse_log(log_path), args.deduplicate)
    if not rows:
        raise SystemExit(f"[ERROR] no GRPO step metrics found in: {log_path}")

    derived_rows = add_derived_metrics(rows, args.rolling_window, args.grad_clip)
    columns = collect_columns(rows)
    derived_columns = collect_columns(derived_rows)
    csv_path = output_dir / "grpo_metrics.csv"
    derived_csv_path = output_dir / "grpo_metrics_derived.csv"
    metric_summary_path = output_dir / "metric_summary.csv"
    summary_path = output_dir / "summary.json"
    write_csv(rows, columns, csv_path)
    write_csv(derived_rows, derived_columns, derived_csv_path)
    write_metric_summary(derived_rows, metric_summary_path)
    write_summary(derived_rows, summary_path)

    print(f"[OK] parsed {len(rows)} steps")
    print(f"[OK] csv: {csv_path}")
    print(f"[OK] derived csv: {derived_csv_path}")
    print(f"[OK] metric summary: {metric_summary_path}")
    print(f"[OK] summary: {summary_path}")

    if args.plot:
        plot_paths = plot_all(derived_rows, output_dir)
        if plot_paths:
            for path in plot_paths:
                print(f"[OK] plot: {path}")
        else:
            print("[WARN] no plots written")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
