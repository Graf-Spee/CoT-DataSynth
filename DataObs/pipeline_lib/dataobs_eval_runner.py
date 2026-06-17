"""
Python-native DataObs evaluation runner.

This module mirrors scripts/eval_dataobs.sh without shell orchestration, and
returns structured evaluation results directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import ray
from dotenv import load_dotenv
from omegaconf import OmegaConf

from verl.trainer.main_generation import run_generation
from verl.trainer.main_eval import run_evaluation_with_config
from verl.utils.eval.apply_prompt_template import apply_prompt_template

logger = logging.getLogger(__name__)


@dataclass
class EvalRunResult:
    success: bool
    accuracy: Optional[float]
    dataset_name: str
    model_path: str
    eval_output_dir: str
    generation_output: str
    labeled_output: str
    # results_file: Optional[str]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "accuracy": self.accuracy,
            "dataset_name": self.dataset_name,
            "model_path": self.model_path,
            "eval_output_dir": self.eval_output_dir,
            "generation_output": self.generation_output,
            "labeled_output": self.labeled_output,
            # "results_file": self.results_file,
            "details": self.details,
        }


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())


def _repo_root(repo_dir: Optional[str]) -> Path:
    if repo_dir:
        return Path(repo_dir).resolve()

    fallback = Path(__file__).resolve().parents[2]
    config_env = fallback / "config" / "bash_config.env"
    if config_env.exists():
        load_dotenv(config_env)
    env_repo = os.getenv("REPO_DIR")
    if env_repo:
        return Path(env_repo).resolve()
    return fallback


def _resolve_config_dir(repo_root: Path, config_dir: Optional[str]) -> Path:
    if config_dir:
        return Path(config_dir).resolve()
    env_dir = os.getenv("CONFIG_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    return repo_root / "config"


def _apply_runtime_env(gpu_id: str) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "offline"
    gpu_list = [x.strip() for x in gpu_id.split(",") if x.strip()]
    return max(1, len(gpu_list))


_RUNTIME_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "WANDB_MODE",
)


def _snapshot_runtime_env() -> Dict[str, Optional[str]]:
    return {key: os.environ.get(key) for key in _RUNTIME_ENV_KEYS}


def _restore_runtime_env(snapshot: Dict[str, Optional[str]]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _dataset_config(repo_root: Path, data_name: str) -> Dict[str, Any]:
    normalized = data_name.strip().lower()
    reward_root = repo_root / "verl" / "utils" / "reward_score"

    mapping: Dict[str, Dict[str, Any]] = {
        "arc-challenge": {
            "reward": reward_root / "multiple_choice.py",
            "eval_data": "/data/open_datasets/ai2_arc/ARC-Challenge/test-00000-of-00001.parquet",
        },
        "aqua-rat": {
            "reward": reward_root / "multiple_choice.py",
            "eval_data": "/data/open_datasets/aqua_rat/raw/test-00000-of-00001.parquet",
        },
        "commonsenseqa": {
            "reward": reward_root / "multiple_choice.py",
            "eval_data": "/data/open_datasets/CommonsenseQA/data/validation-00000-of-00001.parquet",
        },
        "gsm8k": {
            "reward": reward_root / "gsm8k.py",
            "eval_data": "/data/open_datasets/GSM8K/main/test-00000-of-00001.parquet",
        },
        "humaneval": {
            "reward": reward_root / "mbpp.py",
            "eval_data": "/data/open_datasets/humaneval/openai_humaneval/test-00000-of-00001.parquet",
            "calc_maj": False,
        },
        "humanevalplus": {
            "reward": reward_root / "mbpp.py",
            "eval_data": "/data/open_datasets/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet",
            "calc_maj": False,
        },
        "math-500": {
            "reward": reward_root / "math_verify.py",
            "eval_data": "/data/open_datasets/MATH-500/test.parquet",
        },
        "mbpp": {
            "reward": reward_root / "mbpp.py",
            "eval_data": "/data/open_datasets/mbpp/sanitized/test-00000-of-00001.parquet",
            "calc_maj": False,
        },
        "mbppplus": {
            "reward": reward_root / "mbpp.py",
            "eval_data": "/data/open_datasets/mbppplus/data/test-00000-of-00001-d5781c9c51e02795.parquet",
            "calc_maj": False,
        },
        "numinamath": {
            "reward": reward_root / "math_verify.py",
            "eval_data": "/data/open_datasets/NuminaMath-CoT/data/test-00000-of-00001.parquet",
        },
        "strategyqa": {
            "reward": reward_root / "truefalse.py",
            "eval_data": "/data/open_datasets/StrategyQA/data/test-00000-of-00001-bae602f3ee37f4ca.parquet",
        },
        "bfcl": {
            "is_bfcl": True,
        },
    }

    if normalized not in mapping:
        supported = ", ".join(sorted(mapping.keys()))
        raise ValueError(f"Unsupported dataset '{data_name}'. Supported datasets: {supported}")
    out = dict(mapping[normalized])
    out.setdefault("calc_maj", True)
    return out


def _load_merge_lora_fn(repo_root: Path):
    merge_script = repo_root / "scripts" / "lora_model_merge" / "merge_lora_qwen.py"
    if not merge_script.exists():
        raise FileNotFoundError(f"LoRA merge script not found: {merge_script}")
    spec = importlib.util.spec_from_file_location("merge_lora_qwen_module", str(merge_script))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from: {merge_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "merge_lora_weights"):
        raise AttributeError(f"'merge_lora_weights' not found in: {merge_script}")
    return module.merge_lora_weights


def _choose_model_path(repo_root: Path, checkpoint_path: Path, base_model: str, eval_output_dir: Path) -> Path:
    adapter_file = checkpoint_path / "adapter_model.safetensors"
    if not adapter_file.exists():
        return checkpoint_path

    merged_model_path = eval_output_dir / "merged_model"
    has_model = (merged_model_path / "config.json").exists() and (
        any(merged_model_path.glob("*.safetensors"))
        or (merged_model_path / "pytorch_model.bin").exists()
        or (merged_model_path / "model.safetensors.index.json").exists()
    )
    if not has_model:
        merge_lora_fn = _load_merge_lora_fn(repo_root)
        merge_lora_fn(
            base_path=base_model,
            lora_path=str(checkpoint_path),
            tokenizer_path=str(checkpoint_path),
            output_path=str(merged_model_path),
            verbose=True,
        )
    return merged_model_path


def _extract_accuracy(results: Dict[str, Any]) -> Optional[float]:
    if "overall" in results and "accuracy" in results["overall"]:
        return float(results["overall"]["accuracy"])
    for _, metrics in results.items():
        if isinstance(metrics, dict) and "accuracy" in metrics:
            return float(metrics["accuracy"])
    return None


def _prepare_generation_input_data(
    dataset_name: str,
    prompt_template_method: str,
    source_data_path: Path,
    isolated_eval_dir: Path,
) -> Path:
    """
    Build a temporary parquet for generation input under isolated_eval_dir.
    Prefer dataset-specific prompt templating; fallback to plain parquet copy.
    """
    prepared_path = isolated_eval_dir / f"prepared_{_safe_name(dataset_name)}.parquet"
    prepared_path.parent.mkdir(parents=True, exist_ok=True)

    if source_data_path.suffix != ".parquet":
        raise ValueError(f"Eval data must be a parquet file, got: {source_data_path}")

    try:
        apply_prompt_template(
            dataset_name=dataset_name,
            method=prompt_template_method,
            input_parquet_path=str(source_data_path),
            output_parquet_path=str(prepared_path),
        )
        logger.info(f"Prepared eval data with prompt template: {prepared_path}")
    except ValueError as e:
        # Unsupported dataset in template registry: keep behavior by using original parquet content.
        logger.info(
            "Prompt template not applied for dataset '%s' (%s). Fallback to parquet copy.",
            dataset_name,
            e,
        )
        shutil.copy2(source_data_path, prepared_path)
    return prepared_path


def _limit_parquet_rows(parquet_path: Path, max_samples: Optional[int]) -> None:
    if max_samples is None:
        return
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")

    dataframe = pd.read_parquet(parquet_path)
    limited_dataframe = dataframe.head(max_samples).copy()
    limited_dataframe.to_parquet(parquet_path, index=False)
    logger.info("Limited eval data to %s rows at %s", len(limited_dataframe), parquet_path)


# def _update_training_results_file(
#     training_output_dir: Optional[str],
#     dataset_name: str,
#     accuracy: Optional[float],
# ) -> Optional[Path]:
#     if not training_output_dir:
#         return None
#     out_dir = Path(training_output_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     result_file = out_dir / "training_results.json"

#     payload: Dict[str, Any] = {}
#     if result_file.exists():
#         try:
#             payload = json.loads(result_file.read_text(encoding="utf-8"))
#         except Exception:
#             payload = {}

#     by_dataset = payload.get("test_accuracy_by_dataset")
#     if not isinstance(by_dataset, dict):
#         by_dataset = {}
#     by_dataset[dataset_name] = accuracy
#     payload["test_accuracy_by_dataset"] = by_dataset
#     payload["last_eval_dataset"] = dataset_name
#     payload["last_eval_accuracy"] = accuracy
#     result_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
#     return result_file


def run_dataobs_evaluation(
    checkpoint_path: str,
    base_model: str,
    data_name: str,
    eval_output_dir: str,
    gpu_id: str = "0",
    *,
    repo_dir: Optional[str] = None,
    config_dir: Optional[str] = None,
    eval_data_path: Optional[str] = None,
    # training_output_dir: Optional[str] = None,
    prompt_template_method: str = "zeroshot",
    max_samples: Optional[int] = None,
    generation_batch_size: int = 32,
    generation_temperature: float = 0.6,
    generation_seed: int = 42,
    generation_prompt_length: int = 512,
    generation_response_length: int = 1024,
    generation_gpu_memory_utilization: float = 0.8,
    ray_num_cpus: int = 48,
    shutdown_ray: bool = True,
) -> EvalRunResult:
    runtime_env_snapshot = _snapshot_runtime_env()
    prepared_eval_data_path: Optional[Path] = None
    try:
        repo_root = _repo_root(repo_dir)
        config_root = _resolve_config_dir(repo_root, config_dir)
        ds_cfg = _dataset_config(repo_root, data_name)

        checkpt = Path(checkpoint_path).resolve()
        if not checkpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpt}")
        if not config_root.exists():
            raise FileNotFoundError(f"Config directory not found: {config_root}")

        n_gpus_per_node = _apply_runtime_env(gpu_id)

        eval_output_root = Path(eval_output_dir).resolve()
        eval_output_root.mkdir(parents=True, exist_ok=True)

        model_path = _choose_model_path(repo_root, checkpt, base_model, eval_output_root)

        # Isolate generation and evaluation artifacts by dataset name to avoid collisions.
        isolated_eval_dir = eval_output_root / _safe_name(data_name)
        generated_dir = isolated_eval_dir / "generated"
        logs_dir = isolated_eval_dir / "logs"
        generated_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        if ds_cfg.get("is_bfcl"):
            raise NotImplementedError(
                "BFCL function-path is not implemented in this runner. "
                "Use scripts/eval_bfcl_dataobs.py if BFCL is required."
            )

        generation_output = generated_dir / "responses.parquet"
        labeled_output = generated_dir / "responses_labeled.json"

        source_eval_data_path = Path(eval_data_path).resolve() if eval_data_path else Path(ds_cfg["eval_data"])
        prepared_eval_data_path = _prepare_generation_input_data(
            dataset_name=data_name,
            prompt_template_method=prompt_template_method,
            source_data_path=source_eval_data_path,
            isolated_eval_dir=isolated_eval_dir,
        )
        _limit_parquet_rows(prepared_eval_data_path, max_samples=max_samples)

        generation_cfg = OmegaConf.load(config_root / "generation.yaml")
        generation_cfg.model.path = str(model_path)
        generation_cfg.model.no_chat = False
        generation_cfg.data.path = str(prepared_eval_data_path)
        generation_cfg.data.output_path = str(generation_output)
        generation_cfg.data.prompt_key = "prompt"
        generation_cfg.data.n_samples = 1
        generation_cfg.data.batch_size = generation_batch_size
        generation_cfg.rollout.temperature = generation_temperature
        generation_cfg.rollout.seed = generation_seed
        generation_cfg.rollout.prompt_length = generation_prompt_length
        generation_cfg.rollout.response_length = generation_response_length
        generation_cfg.rollout.gpu_memory_utilization = generation_gpu_memory_utilization
        generation_cfg.trainer.n_gpus_per_node = n_gpus_per_node
        generation_cfg.trainer.nnodes = 1
        generation_cfg.trainer.device = "cuda"
        generation_cfg.ray_init.num_cpus = ray_num_cpus

        run_generation(generation_cfg)
        if not generation_output.exists():
            raise RuntimeError(f"Generation output missing: {generation_output}")

        eval_cfg = OmegaConf.load(config_root / "evaluation.yaml")
        eval_cfg.data.path = str(generation_output)
        eval_cfg.data.output_path = str(labeled_output)
        eval_cfg.data.response_key = "responses"
        eval_cfg.data.data_source_key = "data_source"
        eval_cfg.data.reward_model_key = "reward_model"
        eval_cfg.custom_reward_function.path = str(ds_cfg["reward"])
        eval_cfg.custom_reward_function.name = "compute_score"
        eval_cfg.custom_reward_function.pred_name = "extract_pred"
        eval_cfg.custom_reward_function.calc_maj = bool(ds_cfg["calc_maj"])
        eval_cfg.ray_init.num_cpus = ray_num_cpus

        results = run_evaluation_with_config(eval_cfg)
        accuracy = _extract_accuracy(results)

        eval_log = logs_dir / "evaluation.log"
        with eval_log.open("w", encoding="utf-8") as f:
            if accuracy is None:
                f.write("__DATAOBS_ACCURACY__=N/A\n")
            else:
                f.write(f"__DATAOBS_ACCURACY__={accuracy}\n")
                f.write(f"accuracy: {accuracy}\n")
            f.write(json.dumps(results, ensure_ascii=False, indent=2))
            f.write("\n")

        # # Change #3: isolate persisted accuracy by dataset instead of overwriting single key.
        # result_file = _update_training_results_file(training_output_dir, data_name, accuracy)

        return EvalRunResult(
            success=True,
            accuracy=accuracy,
            dataset_name=data_name,
            model_path=str(model_path),
            eval_output_dir=str(isolated_eval_dir),
            generation_output=str(generation_output),
            labeled_output=str(labeled_output),
            # results_file=str(result_file) if result_file else None,
            details={"metrics": results},
        )
    finally:
        try:
            if prepared_eval_data_path is not None and prepared_eval_data_path.exists():
                prepared_eval_data_path.unlink()
                logger.info(f"Removed temporary eval data file: {prepared_eval_data_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary eval data file {prepared_eval_data_path}: {e}")
        if shutdown_ray and ray.is_initialized():
            ray.shutdown()
        _restore_runtime_env(runtime_env_snapshot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DataObs evaluation directly from Python, with an optional smoke test profile."
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Checkpoint or merged model directory to evaluate. Defaults to --base-model for this smoke test.",
    )
    parser.add_argument(
        "--base-model",
        default="/data/pretrain_models/Qwen2.5-0.5B-Instruct",
        help="Base model path used for LoRA merge or direct evaluation.",
    )
    parser.add_argument("--data-name", default="commonsenseqa", help="Dataset name used to select reward logic.")
    parser.add_argument(
        "--eval-output-dir",
        default="/data/nas/hjw/dataobs_eval_smoke",
        help="Directory where evaluation artifacts will be written.",
    )
    parser.add_argument("--gpu-id", default="1", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--repo-dir", default=None, help="Optional repo root override.")
    parser.add_argument("--config-dir", default=None, help="Optional config directory override.")
    parser.add_argument(
        "--eval-data-path",
        default=None,
        help="Optional parquet path to evaluate. Override this to point to a custom smoke-test shard.",
    )
    parser.add_argument(
        "--prompt-template-method",
        default="zeroshot",
        help="Prompt template method passed to apply_prompt_template.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4,
        help="Limit evaluation to the first N rows for smoke testing. Set to 0 to run the full eval data.",
    )
    parser.add_argument("--generation-batch-size", type=int, default=4, help="Generation batch size for the run.")
    parser.add_argument(
        "--generation-temperature",
        type=float,
        default=0.0,
        help="Generation temperature. Smoke test defaults to deterministic decoding.",
    )
    parser.add_argument("--generation-seed", type=int, default=42, help="Generation seed.")
    parser.add_argument("--generation-prompt-length", type=int, default=512, help="Prompt token budget.")
    parser.add_argument("--generation-response-length", type=int, default=1024, help="Response token budget.")
    parser.add_argument(
        "--generation-gpu-memory-utilization",
        type=float,
        default=0.8,
        help="vLLM GPU memory utilization target.",
    )
    parser.add_argument("--ray-num-cpus", type=int, default=48, help="Ray CPU count for the run.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    checkpoint_path = args.checkpoint_path or args.base_model

    max_samples = args.max_samples
    if max_samples is not None and max_samples <= 0:
        max_samples = None

    result = run_dataobs_evaluation(
        checkpoint_path=checkpoint_path,
        base_model=args.base_model,
        data_name=args.data_name,
        eval_output_dir=args.eval_output_dir,
        gpu_id=args.gpu_id,
        repo_dir=args.repo_dir,
        config_dir=args.config_dir,
        eval_data_path=args.eval_data_path,
        prompt_template_method=args.prompt_template_method,
        max_samples=max_samples,
        generation_batch_size=args.generation_batch_size,
        generation_temperature=args.generation_temperature,
        generation_seed=args.generation_seed,
        generation_prompt_length=args.generation_prompt_length,
        generation_response_length=args.generation_response_length,
        generation_gpu_memory_utilization=args.generation_gpu_memory_utilization,
        ray_num_cpus=args.ray_num_cpus,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
