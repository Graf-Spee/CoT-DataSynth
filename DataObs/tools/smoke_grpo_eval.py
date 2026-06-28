#!/usr/bin/env python3
"""Smoke test GRPO-eval-only risks.

This checks the parts that differ from SFT eval:
1. Resolve a GRPO global_step.
2. Validate the actor checkpoint layout.
3. Merge GRPO LoRA adapter if needed.
4. Run a tiny generation + reward evaluation with the merged actor model.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATAOBS_DIR = REPO_ROOT / "DataObs"

REWARD_FUNCTIONS = {
    "gsm8k": REPO_ROOT / "verl" / "utils" / "reward_score" / "gsm8k.py",
    "math": REPO_ROOT / "verl" / "utils" / "reward_score" / "math_verify.py",
    "math-500": REPO_ROOT / "verl" / "utils" / "reward_score" / "math_verify.py",
    "aqua_rat": REPO_ROOT / "verl" / "utils" / "reward_score" / "multiple_choice.py",
    "arc-challenge": REPO_ROOT / "verl" / "utils" / "reward_score" / "multiple_choice.py",
    "commonsenseqa": REPO_ROOT / "verl" / "utils" / "reward_score" / "multiple_choice.py",
    "strategyqa": REPO_ROOT / "verl" / "utils" / "reward_score" / "truefalse.py",
    "mbpp": REPO_ROOT / "verl" / "utils" / "reward_score" / "mbpp.py",
    "mbppplus": REPO_ROOT / "verl" / "utils" / "reward_score" / "mbpp.py",
    "humaneval": REPO_ROOT / "verl" / "utils" / "reward_score" / "mbpp.py",
    "humanevalplus": REPO_ROOT / "verl" / "utils" / "reward_score" / "mbpp.py",
}


def sh(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def run(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[RUN] {sh(cmd)}")
    print(f"[LOG] {log_path}")
    with log_path.open("w", encoding="utf-8") as fout:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            fout.write(line)
        ret = proc.wait()
    if ret != 0:
        raise SystemExit(f"[ERROR] Command failed with exit code {ret}: {sh(cmd)}")


def is_full_hf_model_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    direct_files = [
        path / "model.safetensors",
        path / "pytorch_model.bin",
        path / "model.safetensors.index.json",
    ]
    return any(p.is_file() for p in direct_files) or any(path.glob("*.safetensors"))


def resolve_step(grpo_dir: Path, step: str | None) -> str:
    if step:
        return step

    tracker = grpo_dir / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        value = tracker.read_text(encoding="utf-8").strip()
        if value:
            return value

    candidates: list[int] = []
    for item in grpo_dir.glob("global_step_*"):
        suffix = item.name.removeprefix("global_step_")
        if item.is_dir() and suffix.isdigit():
            candidates.append(int(suffix))
    if not candidates:
        raise SystemExit(f"[ERROR] No global_step_* found under {grpo_dir}")
    return str(max(candidates))


def resolve_grpo_model(args: argparse.Namespace, env: dict[str, str], out_dir: Path) -> Path:
    grpo_dir = Path(args.grpo_dir).resolve()
    step = resolve_step(grpo_dir, args.step)
    actor_dir = grpo_dir / f"global_step_{step}" / "actor"
    hf_dir = actor_dir / "huggingface"
    lora_dir = actor_dir / "lora_adapter"
    merged_dir = actor_dir / "merged_model"

    print(f"[INFO] GRPO dir: {grpo_dir}")
    print(f"[INFO] Step: {step}")
    print(f"[INFO] Actor dir: {actor_dir}")

    if not actor_dir.is_dir():
        raise SystemExit(f"[ERROR] Actor dir not found: {actor_dir}")

    if (lora_dir / "adapter_model.safetensors").is_file():
        print(f"[OK] Found GRPO LoRA adapter: {lora_dir}")
        if not hf_dir.is_dir():
            raise SystemExit(f"[ERROR] Tokenizer/config dir not found: {hf_dir}")
        print(f"[OK] Found actor HuggingFace metadata: {hf_dir}")

        if is_full_hf_model_dir(merged_dir) and not args.force_merge:
            print(f"[OK] Reusing merged model: {merged_dir}")
            return merged_dir

        merge_script = DATAOBS_DIR / "lib" / "model_ops" / "merge_lora_qwen.py"
        if not merge_script.is_file():
            raise SystemExit(f"[ERROR] Merge script not found: {merge_script}")
        if not Path(args.base_model).is_dir():
            raise SystemExit(f"[ERROR] Base model not found: {args.base_model}")

        cmd = [
            sys.executable,
            str(merge_script),
            "--base",
            args.base_model,
            "--lora",
            str(lora_dir),
            "--tokenizer",
            str(hf_dir),
            "--output",
            str(merged_dir),
        ]
        run(cmd, out_dir / "logs" / "merge.log", env)
        if not is_full_hf_model_dir(merged_dir):
            raise SystemExit(f"[ERROR] Merge finished but output is not a full HF model dir: {merged_dir}")
        print(f"[OK] Merged GRPO model: {merged_dir}")
        return merged_dir

    if is_full_hf_model_dir(hf_dir):
        print(f"[OK] No LoRA adapter; using full actor HF model: {hf_dir}")
        return hf_dir

    raise SystemExit(
        "[ERROR] Could not find evaluable GRPO actor. "
        f"Expected {lora_dir}/adapter_model.safetensors or full HF model in {hf_dir}"
    )


def default_eval_data(grpo_dir: Path) -> Path:
    candidate = grpo_dir.parent / "rl_data" / "val_prepared.parquet"
    if candidate.is_file():
        return candidate
    raise SystemExit(
        "[ERROR] --eval-data was not provided and default val parquet was not found: "
        f"{candidate}"
    )


def make_smoke_eval_data(args: argparse.Namespace, out_dir: Path) -> Path:
    source = Path(args.eval_data).resolve() if args.eval_data else default_eval_data(Path(args.grpo_dir).resolve())
    if not source.is_file():
        raise SystemExit(f"[ERROR] Eval data not found: {source}")

    raw_smoke = out_dir / "generated" / "raw_smoke.parquet"
    prepared_smoke = out_dir / "generated" / "prepared_smoke.parquet"
    raw_smoke.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(source).head(args.num_rows)
    if df.empty:
        raise SystemExit(f"[ERROR] Eval data is empty: {source}")
    df.to_parquet(raw_smoke, index=False)

    prep_script = DATAOBS_DIR / "lib" / "evaluation" / "prepare_eval_data.py"
    cmd = [
        sys.executable,
        str(prep_script),
        "--dataset",
        args.dataset,
        "--input",
        str(raw_smoke),
        "--output",
        str(prepared_smoke),
        "--method",
        args.prompt_template_method,
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    print(f"[OK] Smoke eval data: {prepared_smoke} ({len(df)} rows)")
    return prepared_smoke


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    env["CONFIG_DIR"] = str(Path(args.config_dir).resolve())
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("WANDB_MODE", "offline")
    cuda_alloc_conf = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:True" in cuda_alloc_conf and not args.keep_pytorch_cuda_alloc_conf:
        print("[INFO] Unsetting PYTORCH_CUDA_ALLOC_CONF for vLLM generation; expandable_segments is incompatible with vLLM memory pool.")
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    return env


def normalize_dataset(name: str) -> str:
    key = name.lower().replace("_", "-")
    aliases = {
        "commonsenseqa": "commonsenseqa",
        "commonsense-qa": "commonsenseqa",
        "strategyqa": "strategyqa",
        "strategy-qa": "strategyqa",
        "mbpp-plus": "mbppplus",
        "humaneval-plus": "humanevalplus",
        "human-eval-plus": "humanevalplus",
        "aqua-rat": "aqua_rat",
        "arc": "arc-challenge",
    }
    return aliases.get(key, key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test GRPO eval-specific checkpoint and merge path.")
    parser.add_argument("--grpo-dir", required=True, help="GRPO output dir containing global_step_*.")
    parser.add_argument("--base-model", required=True, help="Base model used to merge GRPO LoRA adapters.")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--eval-data", default="", help="Prepared or raw eval parquet. Defaults to <exp>/rl_data/val_prepared.parquet.")
    parser.add_argument("--output-dir", default="", help="Smoke output dir. Defaults to <grpo-dir>/eval_smoke_step_<step>.")
    parser.add_argument("--step", default="", help="Specific GRPO step. Defaults to latest_checkpointed_iteration.txt.")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--config-dir", default=str(REPO_ROOT / "config"))
    parser.add_argument("--num-rows", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--response-length", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ray-num-cpus", type=int, default=16)
    parser.add_argument("--prompt-template-method", default="zeroshot", choices=["plain", "zeroshot", "fewshot"])
    parser.add_argument("--force-merge", action="store_true")
    parser.add_argument("--skip-generation", action="store_true", help="Only validate checkpoint layout and merge path.")
    parser.add_argument(
        "--keep-pytorch-cuda-alloc-conf",
        action="store_true",
        help="Keep PYTORCH_CUDA_ALLOC_CONF from the parent shell. Default unsets expandable_segments:True because vLLM v1 rejects it.",
    )
    args = parser.parse_args()

    args.dataset = normalize_dataset(args.dataset)
    reward_fn = REWARD_FUNCTIONS.get(args.dataset)
    if reward_fn is None or not reward_fn.is_file():
        raise SystemExit(f"[ERROR] No reward function mapping for dataset={args.dataset}")

    grpo_dir = Path(args.grpo_dir).resolve()
    step = resolve_step(grpo_dir, args.step)
    out_dir = Path(args.output_dir).resolve() if args.output_dir else grpo_dir / f"eval_smoke_step_{step}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(args)
    model_path = resolve_grpo_model(args, env, out_dir)
    if args.skip_generation:
        print("[OK] Checkpoint layout and merge path passed. Generation was skipped.")
        return 0

    smoke_data = make_smoke_eval_data(args, out_dir)
    generation_output = out_dir / "generated" / "responses.parquet"
    evaluation_output = out_dir / "generated" / "responses_labeled.json"

    gen_cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_generation",
        f"--config-path={env['CONFIG_DIR']}",
        "--config-name=generation",
        f"model.path={model_path}",
        "model.no_chat=false",
        f"data.path={smoke_data}",
        f"data.output_path={generation_output}",
        "data.prompt_key=prompt",
        "data.n_samples=1",
        f"data.batch_size={args.batch_size}",
        f"rollout.temperature={args.temperature}",
        f"rollout.seed={args.seed}",
        f"rollout.prompt_length={args.prompt_length}",
        f"rollout.response_length={args.response_length}",
        f"rollout.gpu_memory_utilization={args.gpu_memory_utilization}",
        f"rollout.max_num_batched_tokens={args.max_num_batched_tokens}",
        f"rollout.max_num_seqs={args.max_num_seqs}",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.device=cuda",
        f"ray_init.num_cpus={args.ray_num_cpus}",
    ]
    run(gen_cmd, out_dir / "logs" / "generation.log", env)
    if not generation_output.is_file():
        raise SystemExit(f"[ERROR] Generation output not found: {generation_output}")

    eval_cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_eval",
        f"--config-path={env['CONFIG_DIR']}",
        "--config-name=evaluation",
        f"data.path={generation_output}",
        f"data.output_path={evaluation_output}",
        "data.response_key=responses",
        "data.data_source_key=data_source",
        "data.reward_model_key=reward_model",
        f"custom_reward_function.path={reward_fn}",
        f"ray_init.num_cpus={args.ray_num_cpus}",
    ]
    run(eval_cmd, out_dir / "logs" / "evaluation.log", env)
    if not evaluation_output.is_file():
        raise SystemExit(f"[ERROR] Evaluation output not found: {evaluation_output}")

    print("\n[OK] GRPO eval smoke test passed.")
    print(f"[OK] Model: {model_path}")
    print(f"[OK] Responses: {generation_output}")
    print(f"[OK] Evaluation: {evaluation_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
