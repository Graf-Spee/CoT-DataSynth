#!/usr/bin/env python3
"""
Merge LoRA adapter weights into a base model and save a standalone model.

Supports model families used in this repo (e.g., Qwen / Llama) through
Transformers + PEFT standard APIs.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL_PATH = "/data/pretrain_models/Qwen2.5-3B-Instruct/"
DEFAULT_TOKENIZER_PATH = "/data/hjw/outputs/GSM8K/training/split_0/global_step_30"
DEFAULT_LORA_PATH = "/data/hjw/outputs/GSM8K/training/split_0/global_step_30"
DEFAULT_SAVE_PATH = "/data/hjw/outputs/GSM8K/training/split_0/checkpoint-last"


def _copy_missing_tokenizer_files(tokenizer_path: Path, output_path: Path) -> None:
    for src in tokenizer_path.iterdir():
        if not src.is_file():
            continue
        if "adapter" in src.name:
            continue
        dst = output_path / src.name
        if not dst.exists():
            shutil.copy2(src, dst)


def merge_lora_weights(
    base_path: str,
    lora_path: str,
    tokenizer_path: str,
    output_path: str,
    *,
    cuda_home_override: Optional[str] = "/usr/local/cuda-12.6",
    dtype: torch.dtype = torch.float32,
    device_map: str = "cuda:0",
    trust_remote_code: bool = True,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Merge LoRA weights into base model and save merged model.

    Returns a small metadata dict for callers.
    """
    base = Path(base_path)
    lora = Path(lora_path)
    tokenizer_dir = Path(tokenizer_path)
    output = Path(output_path)

    original_cuda_home = os.environ.get("CUDA_HOME")
    try:
        if cuda_home_override:
            os.environ["CUDA_HOME"] = cuda_home_override

        if verbose:
            print(f"[merge] base model: {base}")
            print(f"[merge] lora path: {lora}")
            print(f"[merge] tokenizer path: {tokenizer_dir}")
            print(f"[merge] output path: {output}")

        model = AutoModelForCausalLM.from_pretrained(
            str(base),
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir),
            trust_remote_code=trust_remote_code,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = PeftModel.from_pretrained(model, str(lora))
        model = model.merge_and_unload()

        output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output), safe_serialization=True)
        tokenizer.save_pretrained(str(output))
        _copy_missing_tokenizer_files(tokenizer_dir, output)

        num_params_m = sum(p.numel() for p in model.parameters()) / 1e6
        if verbose:
            print(f"[merge] done. merged model saved to: {output}")
            print(f"[merge] model params: {num_params_m:.2f}M")

        return {
            "base_path": str(base),
            "lora_path": str(lora),
            "tokenizer_path": str(tokenizer_dir),
            "output_path": str(output),
            "num_params_m": f"{num_params_m:.2f}",
        }
    finally:
        if original_cuda_home is None:
            os.environ.pop("CUDA_HOME", None)
        else:
            os.environ["CUDA_HOME"] = original_cuda_home


def verify_model(
    merged_model_path: str,
    *,
    test_input: str = "Solve: 1+1=",
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    trust_remote_code: bool = True,
) -> str:
    """Quick smoke test for merged model loading + generation."""
    model_dir = Path(merged_model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=trust_remote_code,
    )
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("[verify] input:", test_input)
    print("[verify] output:", result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model.")
    parser.add_argument("--base", default=DEFAULT_BASE_MODEL_PATH, help="Base model path")
    parser.add_argument("--lora", default=DEFAULT_LORA_PATH, help="LoRA adapter path")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_PATH, help="Tokenizer path")
    parser.add_argument("--output", default=DEFAULT_SAVE_PATH, help="Output path for merged model")
    parser.add_argument("--verify", action="store_true", help="Run a quick generation smoke test")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    merge_lora_weights(args.base, args.lora, args.tokenizer, args.output)
    if args.verify:
        verify_model(args.output)


if __name__ == "__main__":
    main()
