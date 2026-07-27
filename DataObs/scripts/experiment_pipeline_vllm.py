#!/usr/bin/env python3
"""Experiment pipeline with vLLM-based evaluation (no verl dependency).

This is a drop-in variant of experiment_pipeline.py where the sft_eval
and grpo_eval stages use vLLM directly for generation and scoring
instead of verl + Ray.

All other stages (distill, metrics, sft, grpo) are identical to the
original pipeline.

Usage:
  # SFT eval only with vLLM
  python DataObs/scripts/experiment_pipeline_vllm.py \
    --experiment-id exp1100_gsm8k_qwen3_teacher_qwen3_1_7b_t0.7_n4 \
    --dataset gsm8k \
    --base-model /data/pretrain_models/Qwen3-4B \
    --teacher-model /data/pretrain_models/Qwen3-1.7B \
    --output-dir /data/hrh/COT/experiments \
    --stages sft_eval \
    --gpu-ids 5 \
    --teacher-num-samples 4 \
    --teacher-temperature 0.7 \
    --teacher-top-p 0.95 \
    --teacher-do-sample \
    --teacher-batch-size 64 \
    --teacher-max-new-tokens 1024 \
    --teacher-tensor-parallel-size 1 \
    --teacher-gpu-memory-utilization 0.6 \
    --data-variant qwen3_teacher_qwen3_4b \
    --reasoning-source teacher \
    --sft-epochs 1 \
    --sft-arg data.train_batch_size=4 \
    --sft-arg data.micro_batch_size_per_gpu=2 \
    --sft-arg data.max_length=2048 \
    --eval-arg data.batch_size=8 \
    --sft-checkpoint /path/to/sft/checkpoint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure DataObs/scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from experiment_pipeline import (  # noqa: E402
    DISTILL_METHODS,
    Pipeline,
    REPO_ROOT,
    check_input,
    normalize_distill_args,
    parse_env_pairs,
    resolve_sft_checkpoint,
)

# Extend the base parser with vLLM-specific args, then restore the original
# Pipeline behavior via subclassing.


class PipelineVLLM(Pipeline):
    """Pipeline that uses vLLM for SFT and GRPO eval instead of verl."""

    @staticmethod
    def _is_full_hf_model_dir(path: Path) -> bool:
        if not path.is_dir() or not (path / "config.json").is_file():
            return False
        model_files = [
            path / "model.safetensors",
            path / "pytorch_model.bin",
            path / "model.safetensors.index.json",
        ]
        return any(p.is_file() for p in model_files) or any(path.glob("*.safetensors"))

    @staticmethod
    def _normalize_grpo_step(step: str) -> str:
        step = step.strip()
        return step.removeprefix("global_step_")

    def _latest_grpo_step_id(self) -> str | None:
        latest_file = self.grpo_dir / "latest_checkpointed_iteration.txt"
        if latest_file.is_file():
            step = self._normalize_grpo_step(latest_file.read_text(encoding="utf-8").strip())
            if step:
                return step

        candidates: list[tuple[int, str]] = []
        for item in self.grpo_dir.glob("global_step_*"):
            if not item.is_dir():
                continue
            step = self._normalize_grpo_step(item.name)
            if step.isdigit():
                candidates.append((int(step), step))
        if not candidates:
            return None
        return sorted(candidates)[-1][1]

    def stage_sft_eval(self) -> None:
        checkpoint = resolve_sft_checkpoint(
            self.sft_dir,
            Path(self.args.sft_checkpoint) if self.args.sft_checkpoint else None,
        )
        if not self.args.dry_run:
            check_input(str(checkpoint), "SFT checkpoint", True, self.args.dry_run)

        cmd = [
            sys.executable,
            str(REPO_ROOT / "DataObs" / "lib" / "evaluation" / "eval_vllm.py"),
            "--model-path", str(checkpoint),
            "--base-model", self.args.base_model,
            "--dataset", self.args.dataset,
            "--output-dir", str(self.sft_eval_dir),
            "--gpu-ids", self.args.eval_gpu_ids or self.args.gpu_ids,
            "--num-samples", str(self.args.eval_num_samples),
            "--temperature", str(self.args.eval_temperature),
            "--top-p", str(self.args.eval_top_p),
            "--max-response-len", str(self.args.eval_max_response_len),
            "--batch-size", str(self.args.eval_batch_size),
            "--gpu-memory-utilization", str(self.args.eval_gpu_memory_utilization),
            "--tensor-parallel-size", str(self.args.eval_tensor_parallel_size),
            "--prompt-template-method", self.args.eval_prompt_template_method,
        ]

        if self.args.eval_max_samples > 0:
            cmd += ["--max-samples", str(self.args.eval_max_samples)]

        if self.args.eval_seed:
            cmd += ["--seed", str(self.args.eval_seed)]

        self.run_cmd("sft_eval", cmd)
        self.results["sft_eval_dir"] = str(self.sft_eval_dir)

    def stage_grpo_eval(self) -> None:
        check_input(str(self.grpo_dir), "GRPO dir", True, self.args.dry_run)
        eval_env = parse_env_pairs(self.args.eval_env)

        if self.args.grpo_step:
            step_id = self._normalize_grpo_step(str(self.args.grpo_step))
        else:
            step_id = self._latest_grpo_step_id()

        if not step_id:
            if not self.args.dry_run:
                raise SystemExit(f"[ERROR] Could not find latest checkpoint under: {self.grpo_dir}")
            step_id = "DRY_RUN_STEP"

        actor_dir = self.grpo_dir / f"global_step_{step_id}" / "actor"
        actor_hf_dir = actor_dir / "huggingface"
        actor_lora_dir = actor_dir / "lora_adapter"
        merged_model_dir = actor_dir / "merged_model"

        if not self.args.dry_run:
            check_input(str(actor_dir), "GRPO actor checkpoint dir", True, self.args.dry_run)

        force_merge = eval_env.get("FORCE_MERGE", "0") == "1"
        has_lora = (actor_lora_dir / "adapter_model.safetensors").is_file()
        has_full_actor = self._is_full_hf_model_dir(actor_hf_dir)

        if has_lora or (self.args.dry_run and not has_full_actor):
            model_for_eval = merged_model_dir
            if not self._is_full_hf_model_dir(merged_model_dir) or force_merge:
                if not self.args.dry_run:
                    check_input(str(actor_hf_dir), "GRPO actor tokenizer/config dir", True, self.args.dry_run)
                    check_input(str(actor_lora_dir), "GRPO actor LoRA adapter dir", True, self.args.dry_run)
                    check_input(self.args.base_model, "base model", True, self.args.dry_run)

                merge_cmd = [
                    sys.executable,
                    str(REPO_ROOT / "DataObs" / "lib" / "model_ops" / "merge_lora_qwen.py"),
                    "--base", self.args.base_model,
                    "--lora", str(actor_lora_dir),
                    "--tokenizer", str(actor_hf_dir),
                    "--output", str(merged_model_dir),
                ]
                merge_env = dict(eval_env)
                merge_env["CUDA_VISIBLE_DEVICES"] = self.args.eval_gpu_ids or self.args.gpu_ids
                self.run_cmd("grpo_merge", merge_cmd, merge_env)
        elif has_full_actor:
            model_for_eval = actor_hf_dir
        else:
            raise SystemExit(
                "[ERROR] Could not find evaluable GRPO actor model. "
                f"Expected LoRA adapter at {actor_lora_dir} or full HF model at {actor_hf_dir}"
            )

        if not self.args.dry_run:
            check_input(str(model_for_eval), "GRPO model for eval", True, self.args.dry_run)

        cmd = [
            sys.executable,
            str(REPO_ROOT / "DataObs" / "lib" / "evaluation" / "eval_vllm.py"),
            "--model-path", str(model_for_eval),
            "--base-model", self.args.base_model,
            "--dataset", self.args.dataset,
            "--output-dir", str(self.grpo_eval_dir),
            "--gpu-ids", self.args.eval_gpu_ids or self.args.gpu_ids,
            "--num-samples", str(self.args.eval_num_samples),
            "--temperature", str(self.args.eval_temperature),
            "--top-p", str(self.args.eval_top_p),
            "--max-response-len", str(self.args.eval_max_response_len),
            "--batch-size", str(self.args.eval_batch_size),
            "--gpu-memory-utilization", str(self.args.eval_gpu_memory_utilization),
            "--tensor-parallel-size", str(self.args.eval_tensor_parallel_size),
            "--prompt-template-method", self.args.eval_prompt_template_method,
        ]

        if self.args.eval_max_samples > 0:
            cmd += ["--max-samples", str(self.args.eval_max_samples)]

        if self.args.eval_seed:
            cmd += ["--seed", str(self.args.eval_seed)]

        self.run_cmd("grpo_eval", cmd, eval_env)
        self.results["grpo_eval_dir"] = str(self.grpo_eval_dir)
        self.results["grpo_eval_step"] = step_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experiment stages with vLLM-based SFT evaluation."
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", default="/data/hrh/COT/experiments")
    parser.add_argument(
        "--stages", default="sft,sft_eval,grpo,grpo_eval",
        help="Comma-separated stages: distill,metrics,sft,sft_eval,grpo,grpo_eval",
    )
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--data-variant", default="")
    parser.add_argument("--reasoning-source", default="")
    parser.add_argument("--model-name", default="")

    # Distillation
    parser.add_argument("--distill-method", default="teacher_correctness_filter", choices=DISTILL_METHODS)
    parser.add_argument("--distill-dataset", default="")
    parser.add_argument("--distill-input", default="")
    parser.add_argument("--distill-output", default="")
    parser.add_argument("--teacher-model", default="")
    parser.add_argument("--distill-gpu-ids", default="")
    parser.add_argument("--teacher-num-samples", type=int, default=1)
    parser.add_argument("--teacher-temperature", type=float, default=0.7)
    parser.add_argument("--teacher-top-p", type=float, default=0.95)
    parser.add_argument("--teacher-batch-size", type=int, default=128)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=8192)
    parser.add_argument("--teacher-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--teacher-gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument(
        "--teacher-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--teacher-correct-threshold", type=float, default=0.99)
    parser.add_argument("--teacher-do-sample", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-teacher-filter", action="store_true")
    parser.add_argument("--answer-aug-use-original-metamath-prompt", action="store_true")
    parser.add_argument("--rephrase-num-cots", type=int, default=1)
    parser.add_argument("--rephrase-max-new-tokens", type=int, default=512)
    parser.add_argument("--backward-question-max-new-tokens", type=int, default=1024)
    parser.add_argument("--consistency-max-new-tokens", type=int, default=1024)
    parser.add_argument("--smoke-num-rows", type=int, default=0)

    # Metrics
    parser.add_argument("--metrics-data", default="")
    parser.add_argument("--metrics-model", default="")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--similarity-type", default="jaccard")
    parser.add_argument("--metrics-arg", action="append", default=[])

    # SFT
    parser.add_argument("--sft-data", default="")
    parser.add_argument("--sft-val-data", default="")
    parser.add_argument("--sft-output-dir", default="")
    parser.add_argument("--sft-checkpoint", default="")
    parser.add_argument("--sft-gpu-ids", default="")
    parser.add_argument("--sft-epochs", type=int, default=1)
    parser.add_argument("--sft-arg", action="append", default=[])
    parser.add_argument("--sft-env", action="append", default=[])

    # Eval (vLLM-specific)
    parser.add_argument("--eval-gpu-ids", default="")
    parser.add_argument("--sft-eval-output-dir", default="")
    parser.add_argument("--grpo-eval-output-dir", default="")
    parser.add_argument("--eval-arg", action="append", default=[])
    parser.add_argument("--eval-env", action="append", default=[])
    # vLLM generation params for eval
    parser.add_argument("--eval-num-samples", type=int, default=1,
                        help="Number of samples per prompt for vLLM eval")
    parser.add_argument("--eval-temperature", type=float, default=0.0,
                        help="Temperature for vLLM eval generation")
    parser.add_argument("--eval-top-p", type=float, default=1.0,
                        help="Top-p for vLLM eval generation")
    parser.add_argument("--eval-max-response-len", type=int, default=16384,
                        help="Max tokens for vLLM eval generation")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Batch size for vLLM eval")
    parser.add_argument("--eval-gpu-memory-utilization", type=float, default=0.8,
                        help="vLLM GPU memory utilization for eval")
    parser.add_argument("--eval-tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size for vLLM eval")
    parser.add_argument("--eval-prompt-template-method", default="zeroshot",
                        choices=["plain", "zeroshot", "fewshot"])
    parser.add_argument("--eval-max-samples", type=int, default=0,
                        help="Limit eval to first N rows (0=all)")
    parser.add_argument("--eval-seed", type=int, default=42,
                        help="Random seed for eval generation")

    # GRPO
    parser.add_argument("--rl-train-data", default="")
    parser.add_argument("--rl-val-data", default="")
    parser.add_argument("--rl-train-from-distill-kept", action="store_true")
    parser.add_argument("--grpo-output-dir", default="")
    parser.add_argument("--grpo-gpu-ids", default="")
    parser.add_argument("--grpo-step", default="")
    parser.add_argument("--grpo-arg", action="append", default=[])
    parser.add_argument("--grpo-env", action="append", default=[])

    args = parser.parse_args()
    return normalize_distill_args(args)


if __name__ == "__main__":
    PipelineVLLM(parse_args()).run()
