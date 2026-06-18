#!/usr/bin/env python3
"""Split a seed parquet into same-prompt and disjoint-prompt experiment inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict, list)):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def prompt_to_text(prompt: Any) -> str:
    prompt = _to_builtin(prompt)
    if isinstance(prompt, str):
        return prompt.strip()
    if isinstance(prompt, dict):
        return str(prompt.get("content", "")).strip()
    if isinstance(prompt, list):
        parts: list[str] = []
        for item in prompt:
            item = _to_builtin(item)
            if isinstance(item, dict):
                role = str(item.get("role", "user"))
                content = str(item.get("content", ""))
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(prompt).strip()


def stable_hash(text: str) -> str:
    normalized = " ".join(text.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create same-prompt and disjoint-prompt seed splits.")
    parser.add_argument("--input", required=True, help="Input seed parquet with prompt column.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sft-ratio", type=float, default=0.5, help="Fraction of unique prompts assigned to SFT seed.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--prefix", default="", help="Optional output filename prefix.")
    args = parser.parse_args()

    if not (0.0 < args.sft_ratio < 1.0):
        parser.error("--sft-ratio must be in (0, 1).")

    df = pd.read_parquet(args.input)
    if args.prompt_key not in df.columns:
        raise SystemExit(f"[ERROR] Missing prompt column: {args.prompt_key}")

    df = df.copy()
    df["_prompt_text_for_split"] = df[args.prompt_key].map(prompt_to_text)
    df["_prompt_hash"] = df["_prompt_text_for_split"].map(stable_hash)

    unique_hashes = sorted(df["_prompt_hash"].unique())
    # Deterministic shuffle independent of Python's randomized hash seed.
    keyed = sorted(
        unique_hashes,
        key=lambda x: hashlib.sha256(f"{args.seed}:{x}".encode("utf-8")).hexdigest(),
    )
    n_sft = max(1, min(len(keyed) - 1, round(len(keyed) * args.sft_ratio)))
    sft_hashes = set(keyed[:n_sft])
    rl_hashes = set(keyed[n_sft:])

    sft_df = df[df["_prompt_hash"].isin(sft_hashes)].drop(columns=["_prompt_text_for_split", "_prompt_hash"])
    rl_df = df[df["_prompt_hash"].isin(rl_hashes)].drop(columns=["_prompt_text_for_split", "_prompt_hash"])
    overlap_df = df.drop(columns=["_prompt_text_for_split", "_prompt_hash"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.prefix}_" if args.prefix else ""
    sft_path = out_dir / f"{prefix}sft_seed.parquet"
    rl_path = out_dir / f"{prefix}rl_train.parquet"
    overlap_path = out_dir / f"{prefix}overlap_train.parquet"
    manifest_path = out_dir / f"{prefix}split_manifest.json"

    sft_df.to_parquet(sft_path, index=False)
    rl_df.to_parquet(rl_path, index=False)
    overlap_df.to_parquet(overlap_path, index=False)

    manifest = {
        "input": args.input,
        "prompt_key": args.prompt_key,
        "seed": args.seed,
        "sft_ratio": args.sft_ratio,
        "num_rows_input": int(len(df)),
        "num_unique_prompts": int(len(unique_hashes)),
        "num_rows_sft_seed": int(len(sft_df)),
        "num_rows_rl_train": int(len(rl_df)),
        "num_rows_overlap_train": int(len(overlap_df)),
        "num_unique_sft_prompts": int(len(sft_hashes)),
        "num_unique_rl_prompts": int(len(rl_hashes)),
        "prompt_overlap_sft_rl": 0,
        "outputs": {
            "sft_seed": str(sft_path),
            "rl_train": str(rl_path),
            "overlap_train": str(overlap_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
