import argparse
import os
import subprocess
import tempfile

import pandas as pd

from verl.utils.reward_score.mbpp import compute_score


PROCESSED_PATHS = {
    "humaneval": "/data/open_datasets/humaneval/openai_humaneval/processed/test.parquet",
    "humanevalplus": "/data/open_datasets/humanevalplus/processed/test.parquet",
    "mbppplus": "/data/open_datasets/mbppplus/processed/test.parquet",
}


def _pick_solution(row: pd.Series, dataset: str) -> str:
    if dataset in {"humaneval", "humanevalplus"}:
        prompt_header = str(row.get("prompt_original", row.get("prompt", "")))
        return f"{prompt_header}{str(row.get('canonical_solution', ''))}"
    return str(row.get("code", ""))


def _make_response(solution_code: str) -> str:
    return f"```python\n{solution_code}\n```"


def run_compute_score_smoke(dataset: str, df: pd.DataFrame, n: int) -> float:
    total = 0.0
    for i in range(min(n, len(df))):
        row = df.iloc[i]
        gt = row["reward_model"]["ground_truth"]
        sol = _pick_solution(row, dataset)
        total += compute_score(_make_response(sol), gt, method="strict")
    return total / min(n, len(df))


def run_main_eval_smoke(dataset: str, df: pd.DataFrame, n: int) -> tuple[int, int, float]:
    n = min(n, len(df))
    tmp_df = df.iloc[:n].copy()
    tmp_df["responses"] = [[_make_response(_pick_solution(row, dataset))] for _, row in tmp_df.iterrows()]

    with tempfile.TemporaryDirectory(prefix=f"mock_eval_{dataset}_") as td:
        input_parquet = os.path.join(td, "responses.parquet")
        output_json = os.path.join(td, "responses_labeled.json")
        tmp_df.to_parquet(input_parquet, index=False)

        cmd = [
            "python3",
            "-m",
            "verl.trainer.main_eval",
            "--config-path=/home/hjw/CoT-Data-verl/CoT-DataSynth/config",
            "--config-name=evaluation",
            f"data.path={input_parquet}",
            f"data.output_path={output_json}",
            "data.response_key=responses",
            "data.data_source_key=data_source",
            "data.reward_model_key=reward_model",
            "custom_reward_function.path=verl/utils/reward_score/mbpp.py",
            "custom_reward_function.calc_maj=false",
            "ray_init.num_cpus=1",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "main_eval smoke test failed\n"
                f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )

        labeled = pd.read_json(output_json)
        scores = labeled["reward_scores"].tolist()
        correct = sum(1 for row in scores if row and row[0] >= 0.99)
        return correct, n, correct / n if n else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only mock evaluation smoke test for codegen datasets")
    parser.add_argument("--dataset", choices=["humaneval", "humanevalplus", "mbppplus", "all"], default="all")
    parser.add_argument("--n", type=int, default=8, help="Number of samples for smoke test")
    parser.add_argument("--path-humaneval", default=None, help="Override processed parquet path for humaneval")
    parser.add_argument("--path-humanevalplus", default=None, help="Override processed parquet path for humanevalplus")
    parser.add_argument("--path-mbppplus", default=None, help="Override processed parquet path for mbppplus")
    args = parser.parse_args()

    datasets = ["humaneval", "humanevalplus", "mbppplus"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        path = PROCESSED_PATHS[ds]
        if ds == "humaneval" and args.path_humaneval:
            path = args.path_humaneval
        elif ds == "humanevalplus" and args.path_humanevalplus:
            path = args.path_humanevalplus
        elif ds == "mbppplus" and args.path_mbppplus:
            path = args.path_mbppplus
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Processed file not found: {path}\n"
                f"Please run: python3 scripts/data_preprocess/codegen_prompt_gt_convert.py --dataset {ds}"
            )

        df = pd.read_parquet(path)
        acc_compute = run_compute_score_smoke(ds, df, args.n)
        correct, total, acc_main_eval = run_main_eval_smoke(ds, df, args.n)

        print(f"[{ds}] samples={total}")
        print(f"[{ds}] compute_score strict acc={acc_compute:.4f}")
        print(f"[{ds}] main_eval acc={correct}/{total} ({acc_main_eval:.4f})")
        print("-" * 60)


if __name__ == "__main__":
    main()
