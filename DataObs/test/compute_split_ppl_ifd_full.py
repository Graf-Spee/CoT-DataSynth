#!/usr/bin/env python3
"""
Compute per-example full PPL/IFD for one split and save results.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline_lib.advanced_metrics import compute_ifd_for_sample, compute_ppl_for_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _extract_prompt_text(item: Dict) -> str:
    prompt = item.get("prompt", None)
    if prompt is not None:
        if isinstance(prompt, list):
            return " ".join(str(p.get("content", "")) for p in prompt).strip()
        return str(prompt).strip()

    for key in ("question", "query", "instruction", "input"):
        value = item.get(key, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _extract_answer_text(item: Dict) -> str:
    extra_info = item.get("extra_info", {})
    if isinstance(extra_info, dict):
        answer = extra_info.get("answer", None)
        if answer is not None and str(answer).strip():
            return str(answer).strip()

    for key in ("answer", "output", "response", "solution"):
        value = item.get(key, None)
        if value is not None and str(value).strip():
            return str(value).strip()

    reward_model = item.get("reward_model", {})
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth", None)
        if ground_truth is not None and str(ground_truth).strip():
            return str(ground_truth).strip()
    return ""


def _load_split(path: str) -> List[Dict]:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p).to_dict("records")
    if p.suffix == ".jsonl":
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows
    if p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported split format: {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-example full PPL/IFD for one split")
    parser.add_argument("--split_path", required=True, help="Path to split parquet/jsonl/json")
    parser.add_argument("--model", required=True, help="HF model path/name")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--metrics", default="both", choices=["ppl", "ifd", "both"], help="Which metrics to compute")
    parser.add_argument("--output_file", required=True, help="Output file (.parquet or .csv)")
    parser.add_argument("--max_records", type=int, default=None, help="Optional cap for debug")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when max_records is set")
    args = parser.parse_args()

    logger.info("Loading split data: %s", args.split_path)
    data = _load_split(args.split_path)
    if args.max_records and len(data) > args.max_records:
        random.Random(args.seed).shuffle(data)
        data = data[: args.max_records]
        logger.info("Using max_records=%d after shuffle", args.max_records)

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model/tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    model.eval()

    compute_ppl = args.metrics in ("ppl", "both")
    compute_ifd = args.metrics in ("ifd", "both")

    rows: List[Dict] = []
    iterator = enumerate(data)
    if tqdm is not None:
        iterator = enumerate(tqdm(data, desc=f"Full {args.metrics.upper()}", unit="sample", dynamic_ncols=True))

    start = time.time()
    for i, item in iterator:
        prompt = _extract_prompt_text(item)
        answer = _extract_answer_text(item)
        rec = {
            "row_id": i,
            "prompt_len": len(prompt) if prompt else 0,
            "answer_len": len(answer) if answer else 0,
            "has_prompt": bool(prompt),
            "has_answer": bool(answer),
            "ppl": np.nan,
            "ifd": np.nan,
            "ppl_ok": False,
            "ifd_ok": False,
        }

        if compute_ppl and answer:
            try:
                ppl_val = compute_ppl_for_text(answer, model, tokenizer, device=args.device)
                if not np.isnan(ppl_val) and not np.isinf(ppl_val):
                    rec["ppl"] = float(ppl_val)
                    rec["ppl_ok"] = True
            except Exception:
                pass

        if compute_ifd and prompt and answer:
            try:
                ifd_val = compute_ifd_for_sample(prompt, answer, model, tokenizer, device=args.device)
                if not np.isnan(ifd_val) and not np.isinf(ifd_val) and ifd_val > 0:
                    rec["ifd"] = float(ifd_val)
                    rec["ifd_ok"] = True
            except Exception:
                pass

        rows.append(rec)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        raise ValueError("output_file must end with .parquet or .csv")

    summary = {
        "split_path": args.split_path,
        "model": args.model,
        "device": args.device,
        "metrics": args.metrics,
        "num_rows": int(len(df)),
        "ppl_valid": int(df["ppl_ok"].sum()),
        "ifd_valid": int(df["ifd_ok"].sum()),
        "ppl_mean": float(df.loc[df["ppl_ok"], "ppl"].mean()) if df["ppl_ok"].any() else None,
        "ifd_mean": float(df.loc[df["ifd_ok"], "ifd"].mean()) if df["ifd_ok"].any() else None,
        "elapsed_sec": round(time.time() - start, 2),
        "output_file": str(out_path),
    }

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Saved: %s", out_path)
    logger.info("Saved meta: %s", meta_path)
    logger.info("Summary: ppl_valid=%s ifd_valid=%s ppl_mean=%s ifd_mean=%s",
                summary["ppl_valid"], summary["ifd_valid"], summary["ppl_mean"], summary["ifd_mean"])


if __name__ == "__main__":
    main()

