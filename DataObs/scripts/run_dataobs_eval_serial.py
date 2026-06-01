#!/usr/bin/env python3
"""
Serial runner for DataObs evaluation-only pipeline.

Runs data_obs_pipeline_test.py multiple times with different eval_data_name values,
avoiding concurrent writes to evaluation_split_accuracy.csv.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def build_command(args: argparse.Namespace, eval_data_name: str) -> list[str]:
    cmd = [
        sys.executable,
        "data_obs_pipeline_test.py",
        "--data_path",
        "None",
        "--model_id",
        args.model_id,
        "--output_dir",
        args.output_dir,
        "--gpu_ids",
        args.gpu_ids,
        "--gpus_per_split",
        str(args.gpus_per_split),
        "--data_name",
        args.data_name,
        "--seed",
        str(args.seed),
        "--only_evaluation",
        "--splits_dir",
        args.splits_dir,
        "--eval_data_name",
        eval_data_name,
    ]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run DataObs evaluation-only pipeline serially for multiple datasets."
    )
    parser.add_argument(
        "--eval_data_names",
        nargs="+",
        required=True,
        help="One or more eval dataset names, e.g. math-500 numinamath mbpp",
    )
    parser.add_argument(
        "--model_id",
        default="/data/pretrain_models/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--output_dir", default="/data/hjw/outputs")
    parser.add_argument("--gpu_ids", default="0,1")
    parser.add_argument("--gpus_per_split", type=int, default=2)
    parser.add_argument("--data_name", default="MATH-CoT-Llama8B")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument(
        "--splits_dir",
        default="/data/hjw/outputs/MATH-CoT-Llama8B/splits/",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue remaining datasets even if one run fails.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    failures: list[tuple[str, int]] = []

    for idx, eval_name in enumerate(args.eval_data_names, start=1):
        cmd = build_command(args, eval_name)
        print(f"[{idx}/{len(args.eval_data_names)}] Running eval_data_name={eval_name}")
        print("  " + shlex.join(cmd))
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            capture_output=True,
            text=True,
        )

        print(f"----- Output for {eval_name} (stdout) -----")
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        else:
            print("[empty]")
        print(f"----- Output for {eval_name} (stderr) -----")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
        else:
            print("[empty]")
        print(f"----- End output for {eval_name} -----")

        if result.returncode == 0:
            print(f"[OK] {eval_name}")
            continue

        print(f"[FAILED] {eval_name} (exit code={result.returncode})")
        failures.append((eval_name, result.returncode))
        if not args.continue_on_error:
            break

    if failures:
        print("\nFailed datasets:")
        for name, code in failures:
            print(f"  - {name}: exit code {code}")
        return 1

    print("\nAll evaluations completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
