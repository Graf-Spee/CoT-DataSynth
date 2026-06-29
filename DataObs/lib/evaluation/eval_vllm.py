#!/usr/bin/env python3
"""Standalone SFT evaluation using vLLM directly (no verl dependency).

Usage examples:
  # Evaluate a full model checkpoint on gsm8k
  python DataObs/lib/evaluation/eval_vllm.py \
    --model-path /data/pretrain_models/Qwen3-4B \
    --base-model /data/pretrain_models/Qwen3-4B \
    --dataset gsm8k \
    --output-dir /data/hrh/COT/experiments/my_exp/eval/sft \
    --gpu-ids 5

  # Evaluate a LoRA checkpoint (auto-merges with base model)
  python DataObs/lib/evaluation/eval_vllm.py \
    --model-path /path/to/lora_checkpoint \
    --base-model /data/pretrain_models/Qwen3-4B \
    --dataset gsm8k \
    --output-dir /data/hrh/COT/experiments/my_exp/eval/sft \
    --gpu-ids 5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "arc-challenge": {
        "reward": "verl/utils/reward_score/multiple_choice.py",
        "eval_data": "/data/open_datasets/ai2_arc/ARC-Challenge/test-00000-of-00001.parquet",
    },
    "aqua_rat": {
        "reward": "verl/utils/reward_score/multiple_choice.py",
        "eval_data": "/data/open_datasets/aqua_rat/raw/test-00000-of-00001.parquet",
    },
    "commonsenseQA": {
        "reward": "verl/utils/reward_score/multiple_choice.py",
        "eval_data": "/data/open_datasets/CommonsenseQA/data/validation-00000-of-00001.parquet",
    },
    "gsm8k": {
        "reward": "verl/utils/reward_score/gsm8k.py",
        "eval_data": "/data/open_datasets/GSM8K/main/test-00000-of-00001.parquet",
    },
    "humaneval": {
        "reward": "verl/utils/reward_score/mbpp.py",
        "eval_data": "/data/open_datasets/humaneval/openai_humaneval/test-00000-of-00001.parquet",
    },
    "humanevalplus": {
        "reward": "verl/utils/reward_score/mbpp.py",
        "eval_data": "/data/open_datasets/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet",
    },
    "math-500": {
        "reward": "verl/utils/reward_score/math_verify.py",
        "eval_data": "/data/open_datasets/MATH-500/test.parquet",
    },
    "mbpp": {
        "reward": "verl/utils/reward_score/mbpp.py",
        "eval_data": "/data/open_datasets/mbpp/sanitized/test-00000-of-00001.parquet",
    },
    "mbppplus": {
        "reward": "verl/utils/reward_score/mbpp.py",
        "eval_data": "/data/open_datasets/mbppplus/data/test-00000-of-00001-d5781c9c51e02795.parquet",
    },
    "numinamath": {
        "reward": "verl/utils/reward_score/math_verify.py",
        "eval_data": "/data/open_datasets/NuminaMath-CoT/data/test-00000-of-00001.parquet",
    },
    "strategyQA": {
        "reward": "verl/utils/reward_score/truefalse.py",
        "eval_data": "/data/open_datasets/StrategyQA/data/test-00000-of-00001-bae602f3ee37f4ca.parquet",
    },
}


def _normalize_dataset(name: str) -> str:
    raw = name.strip()
    key = raw.lower().replace("_", "").replace("-", "")
    mapping = {
        "arc": "arc-challenge",
        "arcchallenge": "arc-challenge",
        "ai2arc": "arc-challenge",
        "aquarat": "aqua_rat",
        "commonsenseqa": "commonsenseQA",
        "csqa": "commonsenseQA",
        "gsm8k": "gsm8k",
        "humaneval": "humaneval",
        "humanevalplus": "humanevalplus",
        "livecodebench": "livecodebench",
        "lcb": "livecodebench",
        "math": "math",
        "math500": "math-500",
        "mbpp": "mbpp",
        "mbppplus": "mbppplus",
        "numinamath": "numinamath",
        "numina": "numinamath",
        "strategyqa": "strategyQA",
    }
    return mapping.get(key, raw)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())


def _load_module(module_path: Path, name: str = "module"):
    spec = importlib.util.spec_from_file_location(name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_lora_checkpoint(path: Path) -> bool:
    return (path / "adapter_model.safetensors").is_file()


def is_full_model(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    model_files = [
        path / "model.safetensors",
        path / "pytorch_model.bin",
        path / "model.safetensors.index.json",
    ]
    return any(p.is_file() for p in model_files) or any(path.glob("*.safetensors"))


def merge_lora(base_model: str, lora_path: str, tokenizer_path: str, output_path: str) -> Path:
    merge_script = REPO_ROOT / "DataObs" / "lib" / "model_ops" / "merge_lora_qwen.py"
    if not merge_script.exists():
        raise FileNotFoundError(f"LoRA merge script not found: {merge_script}")
    merge_fn = _load_module(merge_script, "merge_lora_qwen").merge_lora_weights
    merge_fn(
        base_path=base_model,
        lora_path=lora_path,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
        verbose=True,
    )
    return Path(output_path)


def resolve_model_path(checkpoint_path: Path, base_model: str, output_dir: Path) -> Path:
    """Resolve model path, merging LoRA if needed."""
    if not is_lora_checkpoint(checkpoint_path):
        return checkpoint_path

    merged_dir = output_dir / "merged_model"
    if is_full_model(merged_dir):
        print(f"[INFO] Reusing existing merged model: {merged_dir}")
        return merged_dir

    print(f"[INFO] Detected LoRA adapter, merging with base model: {base_model}")
    return merge_lora(
        base_model=base_model,
        lora_path=str(checkpoint_path),
        tokenizer_path=str(checkpoint_path),
        output_path=str(merged_dir),
    )


def prepare_eval_data(dataset: str, input_path: str, output_path: str, method: str = "zeroshot") -> Path:
    """Prepare eval parquet with prompt/reward_model/data_source columns."""
    prepare_script = REPO_ROOT / "DataObs" / "lib" / "evaluation" / "prepare_eval_data.py"
    if not prepare_script.exists():
        raise FileNotFoundError(f"Prepare eval data script not found: {prepare_script}")

    import subprocess
    subprocess.run(
        [
            sys.executable,
            str(prepare_script),
            "--dataset", dataset,
            "--input", input_path,
            "--output", output_path,
            "--method", method,
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    return Path(output_path)


def load_reward_functions(dataset: str):
    """Load compute_score and extract_pred for the given dataset."""
    ds_cfg = DATASET_CONFIG.get(dataset)
    if ds_cfg is None:
        raise ValueError(f"Unsupported dataset: {dataset}. Choices: {list(DATASET_CONFIG)}")

    reward_path = REPO_ROOT / ds_cfg["reward"]
    if not reward_path.exists():
        raise FileNotFoundError(f"Reward function not found: {reward_path}")

    module = _load_module(reward_path, f"reward_{dataset}")
    compute_score = getattr(module, "compute_score", None)
    extract_pred = getattr(module, "extract_pred", None)
    if compute_score is None:
        raise AttributeError(f"'compute_score' not found in {reward_path}")
    return compute_score, extract_pred


def run_vllm_eval(
    model_path: str,
    base_model: str,
    dataset: str,
    output_dir: str,
    gpu_ids: str,
    *,
    eval_data: Optional[str] = None,
    num_samples: int = 1,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_response_len: int = 1024,
    batch_size: int = 32,
    gpu_memory_utilization: float = 0.8,
    tensor_parallel_size: int = 1,
    prompt_template_method: str = "zeroshot",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "offline"

    ds_cfg = DATASET_CONFIG[dataset]
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model (merge LoRA if needed)
    checkpoint = Path(model_path).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Model path not found: {checkpoint}")
    resolved_model = resolve_model_path(checkpoint, base_model, out_dir)
    print(f"[INFO] Model path: {resolved_model}")

    # Prepare eval data
    eval_data_path = Path(eval_data).resolve() if eval_data else Path(ds_cfg["eval_data"])
    prepared_data = out_dir / f"prepared_{_safe_name(dataset)}.parquet"
    if not prepared_data.exists():
        prepare_eval_data(dataset, str(eval_data_path), str(prepared_data), method=prompt_template_method)

    # Load prompts
    df = pd.read_parquet(prepared_data)
    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples)

    # The prompt column may contain numpy arrays of chat messages, e.g.
    #   array([{"role": "user", "content": "..."}], dtype=object)
    # vLLM generate() expects strings, so apply the model's chat template.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(resolved_model), trust_remote_code=True)

    raw_prompts = df["prompt"].tolist()
    prompts: list[str] = []
    for p in raw_prompts:
        if isinstance(p, str):
            prompts.append(p)
        else:
            # numpy array of message dicts -> list of dicts
            messages = p.tolist() if hasattr(p, "tolist") else list(p)
            prompts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
    print(f"[INFO] Loaded {len(prompts)} prompts from {prepared_data}")

    # Load reward functions
    compute_score_fn, extract_pred_fn = load_reward_functions(dataset)

    # vLLM generation
    from vllm import LLM, SamplingParams

    print(f"[INFO] Loading model with vLLM (tp={tensor_parallel_size}, gpu_mem={gpu_memory_utilization})...")
    llm = LLM(
        model=str(resolved_model),
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        n=num_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_response_len,
        seed=seed,
    )

    print(f"[INFO] Generating responses (n={num_samples}, temp={temperature}, max_tokens={max_response_len})...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    print(f"[INFO] Generation done in {elapsed:.1f}s ({len(prompts)} prompts, {len(prompts) / elapsed:.1f} prompts/s)")

    # Extract responses and score
    all_responses = []
    all_scores = []
    for output in outputs:
        responses = [o.text for o in output.outputs]
        all_responses.append(responses)

        # Use first response for scoring
        pred_text = responses[0]
        ground_truth = df.iloc[len(all_scores)]["reward_model"]["ground_truth"]
        score = compute_score_fn(pred_text, ground_truth)
        all_scores.append(score)

    accuracy = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"[INFO] Accuracy: {accuracy:.4f} ({sum(all_scores)}/{len(all_scores)})")

    # Save results
    results_df = df.copy()
    results_df["responses"] = all_responses
    results_df["score"] = all_scores

    generated_dir = out_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    responses_path = generated_dir / "responses.parquet"
    results_df.to_parquet(responses_path, index=False)

    results_json = {
        "dataset": dataset,
        "model_path": str(resolved_model),
        "num_samples": len(prompts),
        "accuracy": accuracy,
        "num_correct": int(sum(all_scores)),
        "elapsed_sec": elapsed,
        "config": {
            "num_samples_per_prompt": num_samples,
            "temperature": temperature,
            "top_p": top_p,
            "max_response_len": max_response_len,
            "batch_size": batch_size,
        },
    }
    results_json_path = generated_dir / "results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    metrics_json = {
        dataset: {
            "test_score": accuracy,
            "accuracy": accuracy * 100,
            "correct": int(sum(all_scores)),
            "total": len(all_scores),
            "mean@1": accuracy,
            "pass@1/mean": accuracy,
            "pass@1/std": 0.0,
        }
    }
    metrics_json_path = generated_dir / "responses_labeled.metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    # Also write the verl-compatible labeled json
    labeled = []
    for i, output in enumerate(outputs):
        for j, o in enumerate(output.outputs):
            labeled.append({
                "prompt": prompts[i],
                "response": o.text,
                "score": float(compute_score_fn(o.text, df.iloc[i]["reward_model"]["ground_truth"])),
            })
    labeled_path = generated_dir / "responses_labeled.json"
    with open(labeled_path, "w", encoding="utf-8") as f:
        json.dump(labeled, f, indent=2, ensure_ascii=False)

    # Write eval log
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_log = logs_dir / "evaluation.log"
    with open(eval_log, "w", encoding="utf-8") as f:
        f.write(f"accuracy: {accuracy}\n")
        f.write(f"__DATAOBS_ACCURACY__={accuracy}\n")
        json.dump(results_json, f, indent=2)

    print(f"[INFO] Results saved to {out_dir}")
    print(f"[INFO] Accuracy: {accuracy:.4f}")

    return results_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SFT checkpoint with vLLM (no verl dependency)")
    parser.add_argument("--model-path", required=True, help="Model checkpoint path (full model or LoRA adapter)")
    parser.add_argument("--base-model", required=True, help="Base model path for LoRA merge or tokenizer")
    parser.add_argument("--dataset", required=True, help="Dataset name: gsm8k, math-500, arc-challenge, etc.")
    parser.add_argument("--output-dir", required=True, help="Directory for evaluation outputs")
    parser.add_argument("--gpu-ids", default="0", help="GPU IDs, e.g. '5' or '0,1'")
    parser.add_argument("--eval-data", default="", help="Override eval data parquet path")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--temperature", type=float, default=0.6, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--max-response-len", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--batch-size", type=int, default=32, help="vLLM batch size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="vLLM GPU memory utilization")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--prompt-template-method", default="zeroshot", choices=["plain", "zeroshot", "fewshot"])
    parser.add_argument("--max-samples", type=int, default=0, help="Limit eval to first N rows (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = _normalize_dataset(args.dataset)
    if dataset not in DATASET_CONFIG:
        print(f"[ERROR] Unsupported dataset: {args.dataset} (normalized: {dataset})")
        print(f"[ERROR] Supported: {list(DATASET_CONFIG)}")
        sys.exit(1)

    result = run_vllm_eval(
        model_path=args.model_path,
        base_model=args.base_model,
        dataset=dataset,
        output_dir=args.output_dir,
        gpu_ids=args.gpu_ids,
        eval_data=args.eval_data or None,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_response_len=args.max_response_len,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        prompt_template_method=args.prompt_template_method,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        seed=args.seed,
    )
    print(f"\nFinal accuracy: {result['accuracy']:.4f}")
