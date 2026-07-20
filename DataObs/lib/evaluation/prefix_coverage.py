#!/usr/bin/env python3
"""Generate base samples and compute teacher-vs-base prefix coverage."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict, list)):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def _as_messages(prompt: Any) -> list[dict[str, str]]:
    prompt = _to_builtin(prompt)
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, dict):
        return [{"role": str(prompt.get("role", "user")), "content": str(prompt.get("content", ""))}]
    if isinstance(prompt, list):
        messages = []
        for item in prompt:
            item = _to_builtin(item)
            if isinstance(item, dict):
                content = str(item.get("content", ""))
                if content.strip():
                    messages.append({"role": str(item.get("role", "user")), "content": content})
            else:
                content = str(item)
                if content.strip():
                    messages.append({"role": "user", "content": content})
        if messages:
            return messages
    raise ValueError(f"Unsupported prompt type: {type(prompt).__name__}")


def _last_user_text(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return messages[-1].get("content", "") if messages else ""


def _append_gsm8k_suffix(prompt: Any) -> tuple[str, list[dict[str, str]]]:
    suffix = "Please reason step by step, and put your final answer within \\boxed{}."
    messages = _as_messages(prompt)
    question = _last_user_text(messages).strip()
    out = [dict(message) for message in messages]
    for idx in range(len(out) - 1, -1, -1):
        if out[idx].get("role") == "user":
            out[idx]["content"] = out[idx].get("content", "").rstrip() + "\n\n" + suffix
            break
    return question, out


def generate_base(args: argparse.Namespace) -> None:
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "offline")

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    if args.max_rows > 0:
        df = df.head(args.max_rows)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl.utils.reward_score import gsm8k

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    prompts: list[str] = []
    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        if args.append_suffix:
            question, messages = _append_gsm8k_suffix(row["prompt"])
        else:
            messages = _as_messages(row["prompt"])
            question = _last_user_text(messages).strip()
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
        )
        reward_model = _to_builtin(row["reward_model"])
        rows.append(
            {
                "source_index": int(idx),
                "question": question,
                "gold_answer": reward_model.get("ground_truth") if isinstance(reward_model, dict) else None,
                "dataset": "gsm8k",
                "data_source": row.get("data_source", "gsm8k"),
            }
        )

    llm = LLM(
        model=args.model_id,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, sampling_params)

    out_rows: list[dict[str, Any]] = []
    for row, output in zip(rows, outputs):
        for sample_idx, item in enumerate(output.outputs):
            answer = item.text.strip()
            score = gsm8k.compute_score(answer, row["gold_answer"])
            out_rows.append(
                {
                    **row,
                    "answer": answer,
                    "sample_index": sample_idx,
                    "teacher_score": float(score),
                    "teacher_filter_passed": bool(score >= args.correct_threshold),
                    "answer_char_len": len(answer),
                }
            )

    pd.DataFrame(out_rows).to_parquet(output_path, index=False)
    summary = {
        "model_id": args.model_id,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "num_input_rows": len(rows),
        "num_samples_per_prompt": args.num_samples,
        "num_candidates": len(out_rows),
        "pass_rate": float(np.mean([r["teacher_score"] for r in out_rows])) if out_rows else 0.0,
        "prompt_pass_rate": float(
            pd.DataFrame(out_rows).groupby("source_index")["teacher_filter_passed"].any().mean()
        )
        if out_rows
        else 0.0,
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "seed": args.seed,
            "gpu_ids": args.gpu_ids,
        },
        "elapsed_sec": time.time() - start_time,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"</?think>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _step_units(text: str) -> list[str]:
    text = _clean_text(text)
    text = re.sub(r"####.*$", "", text).strip()
    raw = re.split(r"(?:\n+|(?<=[.!?。！？])\s+|(?:-{3,})|(?:#+\s+))", text)
    units = []
    for unit in raw:
        unit = re.sub(r"\s+", " ", unit).strip().lower()
        unit = re.sub(r"^[*-]\s+", "", unit)
        if unit:
            units.append(unit)
    return units


def _lcp_len(a: list[Any], b: list[Any]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _load_candidate_frame(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "answer" in df.columns:
        return df.copy()
    if "responses" not in df.columns:
        raise ValueError(f"Expected either 'answer' or 'responses' column in {path}")

    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        responses = row["responses"]
        if hasattr(responses, "tolist"):
            responses = responses.tolist()
        reward_model = _to_builtin(row.get("reward_model", {}))
        gold_answer = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
        for sample_idx, answer in enumerate(list(responses)):
            rows.append(
                {
                    "source_index": int(idx),
                    "sample_index": int(sample_idx),
                    "answer": str(answer),
                    "gold_answer": gold_answer,
                    "teacher_score": float(row.get("score", 0.0)) if sample_idx == 0 else 0.0,
                    "teacher_filter_passed": bool(row.get("score", 0.0)) if sample_idx == 0 else False,
                }
            )
    return pd.DataFrame(rows)


def _select_teacher(df: pd.DataFrame, selector: str) -> pd.DataFrame:
    work = df.copy()
    if selector == "first_passed" and "teacher_filter_passed" in work.columns:
        work = work[work["teacher_filter_passed"].astype(bool)]
    work = work.sort_values(["source_index", "sample_index"])
    return work.groupby("source_index", as_index=False).first()


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def compute_coverage(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    base = _load_candidate_frame(args.base_candidates)
    teacher_raw = _load_candidate_frame(args.teacher_candidates)
    teacher = _select_teacher(teacher_raw, args.teacher_selector)

    base = base.sort_values(["source_index", "sample_index"])
    if args.max_base_samples > 0:
        base = base.groupby("source_index", as_index=False).head(args.max_base_samples)
    base_groups = {int(k): v for k, v in base.groupby("source_index")}

    rows: list[dict[str, Any]] = []
    for _, trow in teacher.iterrows():
        source_index = int(trow["source_index"])
        bgroup = base_groups.get(source_index)
        if bgroup is None or bgroup.empty:
            continue

        teacher_text = _clean_text(str(trow["answer"]))
        teacher_tokens = tokenizer.encode(teacher_text, add_special_tokens=False)
        teacher_steps = _step_units(teacher_text)

        best: dict[str, Any] | None = None
        for _, brow in bgroup.iterrows():
            base_text = _clean_text(str(brow["answer"]))
            base_tokens = tokenizer.encode(base_text, add_special_tokens=False)
            token_lcp = _lcp_len(teacher_tokens, base_tokens)
            base_steps = _step_units(base_text)
            step_lcp = _lcp_len(teacher_steps, base_steps)
            candidate = {
                "base_sample_index": int(brow["sample_index"]),
                "token_lcp": token_lcp,
                "token_prefix_ratio": token_lcp / max(1, len(teacher_tokens)),
                "step_lcp": step_lcp,
                "step_prefix_ratio": step_lcp / max(1, len(teacher_steps)),
                "teacher_tokens": len(teacher_tokens),
                "base_tokens": len(base_tokens),
                "teacher_steps": len(teacher_steps),
                "base_steps": len(base_steps),
            }
            if best is None or (
                candidate["token_lcp"],
                candidate["step_lcp"],
            ) > (best["token_lcp"], best["step_lcp"]):
                best = candidate
        assert best is not None
        rows.append(
            {
                "source_index": source_index,
                "teacher_sample_index": int(trow["sample_index"]),
                "teacher_score": float(trow.get("teacher_score", 0.0)),
                "teacher_filter_passed": bool(trow.get("teacher_filter_passed", False)),
                "first_divergence_token": int(best["token_lcp"] + 1),
                "first_divergence_step": int(best["step_lcp"] + 1),
                **best,
            }
        )

    per_problem = pd.DataFrame(rows)
    per_problem_path = output_dir / f"{args.name}.per_problem.csv"
    per_problem.to_csv(per_problem_path, index=False)

    summary = {
        "name": args.name,
        "base_candidates": str(args.base_candidates),
        "teacher_candidates": str(args.teacher_candidates),
        "teacher_selector": args.teacher_selector,
        "num_base_prompts": int(base["source_index"].nunique()),
        "num_teacher_prompts_raw": int(teacher_raw["source_index"].nunique()),
        "num_teacher_prompts_selected": int(teacher["source_index"].nunique()),
        "num_compared_prompts": int(len(per_problem)),
        "max_base_samples": args.max_base_samples,
        "token_prefix_ratio": _summarize(per_problem["token_prefix_ratio"].tolist()),
        "token_lcp": _summarize(per_problem["token_lcp"].tolist()),
        "first_divergence_token": _summarize(per_problem["first_divergence_token"].tolist()),
        "step_prefix_ratio": _summarize(per_problem["step_prefix_ratio"].tolist()),
        "step_lcp": _summarize(per_problem["step_lcp"].tolist()),
        "first_divergence_step": _summarize(per_problem["first_divergence_step"].tolist()),
    }
    summary_path = output_dir / f"{args.name}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    ax.hist(per_problem["token_prefix_ratio"] * 100.0, bins=40, color="#4E79A7", alpha=0.85)
    ax.set_xlabel("Best Token Prefix Coverage (%)")
    ax.set_ylabel("Problems")
    ax.set_title(args.name)
    fig.tight_layout()
    fig.savefig(output_dir / f"{args.name}.token_prefix_hist.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    ax.hist(per_problem["step_prefix_ratio"] * 100.0, bins=40, color="#59A14F", alpha=0.85)
    ax.set_xlabel("Best Step Prefix Coverage (%)")
    ax.set_ylabel("Problems")
    ax.set_title(args.name)
    fig.tight_layout()
    fig.savefig(output_dir / f"{args.name}.step_prefix_hist.png")
    plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate-base")
    gen.add_argument("--input-file", required=True)
    gen.add_argument("--output-file", required=True)
    gen.add_argument("--model-id", required=True)
    gen.add_argument("--gpu-ids", default="0")
    gen.add_argument("--trust-remote-code", action="store_true")
    gen.add_argument("--dtype", default="auto")
    gen.add_argument("--num-samples", type=int, default=4)
    gen.add_argument("--max-new-tokens", type=int, default=1024)
    gen.add_argument("--temperature", type=float, default=0.7)
    gen.add_argument("--top-p", type=float, default=0.95)
    gen.add_argument("--top-k", type=int, default=-1)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--tensor-parallel-size", type=int, default=1)
    gen.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    gen.add_argument("--max-model-len", type=int, default=4096)
    gen.add_argument("--max-num-seqs", type=int, default=256)
    gen.add_argument("--correct-threshold", type=float, default=0.99)
    gen.add_argument("--max-rows", type=int, default=0)
    gen.add_argument("--append-suffix", action="store_true")
    gen.add_argument("--enable-thinking", action="store_true")
    gen.set_defaults(func=generate_base)

    comp = sub.add_parser("compute")
    comp.add_argument("--name", required=True)
    comp.add_argument("--base-candidates", required=True)
    comp.add_argument("--teacher-candidates", required=True)
    comp.add_argument("--tokenizer", default="/data/pretrain_models/Qwen3-4B")
    comp.add_argument("--output-dir", required=True)
    comp.add_argument("--teacher-selector", choices=["first", "first_passed"], default="first_passed")
    comp.add_argument("--max-base-samples", type=int, default=4)
    comp.set_defaults(func=compute_coverage)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
