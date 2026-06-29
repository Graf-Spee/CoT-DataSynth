#!/usr/bin/env python3
"""Build same-prompt, different-samples-per-prompt SFT subsets.

Given a distilled SFT pool with repeated rows per prompt, this script writes
nested subsets such as n1/n2/n4/n8 where every output contains the same prompt
set and only the number of sampled responses per prompt changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_sample_counts(value: str) -> list[int]:
    counts: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        count = int(item)
        if count <= 0:
            raise argparse.ArgumentTypeError(f"sample counts must be positive, got {count}")
        counts.append(count)
    if not counts:
        raise argparse.ArgumentTypeError("at least one sample count is required")
    return sorted(set(counts))


def output_path(output_dir: Path, prefix: str, count: int) -> Path:
    return output_dir / f"{prefix}_n{count}.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create nested same-prompt subsets with different numbers of responses per prompt."
    )
    parser.add_argument("--input", required=True, help="Distilled SFT pool parquet.")
    parser.add_argument("--output-dir", required=True, help="Directory for output parquet files.")
    parser.add_argument("--prefix", default="", help="Output filename prefix. Defaults to input stem.")
    parser.add_argument("--group-key", default="source_index", help="Column identifying the original prompt.")
    parser.add_argument("--sample-counts", type=parse_sample_counts, default=parse_sample_counts("1,2,4,8"))
    parser.add_argument("--seed", type=int, default=42, help="Seed for response sampling within each prompt.")
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="Limit to first/shuffled N eligible prompts. 0 means all eligible prompts.",
    )
    parser.add_argument(
        "--shuffle-prompts",
        action="store_true",
        help="Shuffle eligible prompt ids before applying --max-prompts.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow prompts with fewer than max(sample-counts) responses. This breaks identical prompt sets.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    prefix = args.prefix or input_path.stem
    max_count = max(args.sample_counts)

    if not input_path.exists():
        raise SystemExit(f"[ERROR] Input parquet not found: {input_path}")

    df = pd.read_parquet(input_path)
    if args.group_key not in df.columns:
        raise SystemExit(f"[ERROR] Missing group key column {args.group_key!r} in {input_path}")
    if df.empty:
        raise SystemExit(f"[ERROR] Input parquet is empty: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    planned_outputs = [output_path(output_dir, prefix, count) for count in args.sample_counts]
    if not args.overwrite:
        existing = [str(path) for path in planned_outputs if path.exists()]
        if existing:
            raise SystemExit("[ERROR] Output exists; pass --overwrite to replace:\n" + "\n".join(existing))

    counts = df.groupby(args.group_key, dropna=False).size()
    if args.allow_partial:
        eligible_prompt_ids = counts[counts > 0].index.to_list()
    else:
        eligible_prompt_ids = counts[counts >= max_count].index.to_list()
    if not eligible_prompt_ids:
        raise SystemExit(
            f"[ERROR] No prompts have enough responses for max sample count {max_count}."
        )

    if args.shuffle_prompts:
        rng = np.random.default_rng(args.seed)
        eligible_prompt_ids = list(rng.permutation(eligible_prompt_ids))
    else:
        eligible_prompt_ids = sorted(eligible_prompt_ids)

    if args.max_prompts > 0:
        eligible_prompt_ids = eligible_prompt_ids[: args.max_prompts]
    if not eligible_prompt_ids:
        raise SystemExit("[ERROR] --max-prompts selected zero prompts.")

    selected = df[df[args.group_key].isin(eligible_prompt_ids)].copy()

    # Randomize candidate order once, then take prefix lengths n=1,2,4,8. This
    # makes larger subsets strict supersets of smaller ones for each prompt.
    rng = np.random.default_rng(args.seed)
    selected["_sample_priority"] = rng.random(len(selected))
    selected = selected.sort_values([args.group_key, "_sample_priority"], kind="mergesort")
    selected["_sample_rank"] = selected.groupby(args.group_key, sort=False).cumcount()

    written: dict[str, dict[str, object]] = {}
    for count in args.sample_counts:
        subset = selected[selected["_sample_rank"] < count].drop(
            columns=["_sample_priority", "_sample_rank"]
        )
        path = output_path(output_dir, prefix, count)
        subset.to_parquet(path, index=False)
        written[f"n{count}"] = {
            "path": str(path),
            "rows": int(len(subset)),
            "prompts": int(subset[args.group_key].nunique(dropna=False)),
            "samples_per_prompt": count,
        }
        print(
            f"n{count}: prompts={written[f'n{count}']['prompts']} "
            f"rows={written[f'n{count}']['rows']} -> {path}"
        )

    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "prefix": prefix,
        "group_key": args.group_key,
        "sample_counts": args.sample_counts,
        "seed": args.seed,
        "max_prompts": args.max_prompts,
        "shuffle_prompts": bool(args.shuffle_prompts),
        "allow_partial": bool(args.allow_partial),
        "input_rows": int(len(df)),
        "input_prompts": int(df[args.group_key].nunique(dropna=False)),
        "eligible_prompts": int(len(eligible_prompt_ids)),
        "min_required_responses_per_prompt": None if args.allow_partial else max_count,
        "outputs": written,
    }
    summary_path = output_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
