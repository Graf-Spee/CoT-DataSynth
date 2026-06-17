"""
DataObs Pipeline: End-to-end data analysis workflow
Splits dataset, computes metrics, trains models, and analyzes correlations
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List

import pandas as pd
import os
# Add lib to path
# sys.path.insert(0, str(Path(__file__).parent / "lib"))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.data_process.data_obs import DataSplitter, GPUAllocator, ResultCollector, DatasetMetrics
from lib.metrics.data_metrics import compute_data_statistics, compute_data_quality_metrics
from lib.metrics.advanced_metrics import (
    compute_dataset_diversity,
    compute_dataset_entropy,
    compute_ppl_metrics,
    compute_ifd_metrics,
    SimilarityType,
)
from lib.training.training_pipeline import TrainingPipeline
from lib.training.training_pipeline_parallel import TrainingPipelineParallel
from lib.analysis.analysis_pipeline import CorrelationAnalyzer, AnalysisVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> List[dict]:
    """Load data from parquet or jsonl file"""
    data_path = Path(data_path)

    if data_path.suffix == '.parquet':
        import pyarrow.parquet as pq
        table = pq.read_table(data_path)
        data = table.to_pandas().to_dict('records')
    elif data_path.suffix == '.jsonl':
        data = []
        with open(data_path) as f:
            for line in f:
                data.append(json.loads(line))
    elif data_path.suffix == '.json':
        with open(data_path) as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    logger.info(f"Loaded {len(data)} samples from {data_path}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description='DataObs Pipeline: Data analysis and training workflow'
    )
    parser.add_argument('--data_path', required=True, help='Path to dataset (parquet/jsonl/json)')
    parser.add_argument('--data_name', default='MATH-CoT', help='Dataset name for output subdirectory')
    parser.add_argument('--model_id', required=True, help='Model ID for training')
    parser.add_argument('--output_dir', required=True, help='Output directory for experiment')
    parser.add_argument('--splits_dir', default=None, help='Path to existing splits directory (use with --skip_split)')
    parser.add_argument('--train_script', default='DataObs/lib/training/sft_dataobs.sh', help='Training script path')
    parser.add_argument('--val_data_path', default='/data/open_datasets/MATH-500/test-processed.parquet', help='Path to val dataset (parquet)')
    parser.add_argument('--eval_data_name', default='MATH-500', help='Name of evaluation dataset')
    parser.add_argument('--prompt_template_method', default='zeroshot', help='Prompt template method for evaluation data preparation')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu_ids', default='0,1,2,3,4,5,6,7', help='Available GPU IDs (comma-separated)')
    parser.add_argument('--gpus_per_split', type=int, default=1, help='GPUs per training split')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of training epochs per split')
    parser.add_argument('--parallel', action='store_true', help='Deprecated. Parallel pipeline is no longer used')
    
    parser.add_argument('--n_splits', type=int, default=10, help='Number of data splits')
    parser.add_argument('--similarity_type', default='jaccard',
                        help='Similarity type for diversity: jaccard, levenshtein, cosine, jaro_winkler, ngram, bertouch, bleu, rouge')
    parser.add_argument('--compute_on', default='both', choices=['prompt', 'answer', 'both'], help='What to compute diversity on')
    parser.add_argument('--model', default=None, help='Model for PPL and IFD computation')
    parser.add_argument('--ppl_ifd_sample_ratio', type=float, default=0.01,
                        help='Sampling ratio for PPL/IFD (default: 0.01). Use 1.0 for full.')
    parser.add_argument('--ppl_ifd_max_samples', type=int, default=None,
                        help='Optional hard cap for PPL/IFD sample size (overrides ratio)')
    parser.add_argument('--ppl_ifd_sample_seed', type=int, default=42,
                        help='Random seed for PPL/IFD sampling')
    
    parser.add_argument('--skip_split', action='store_true', help='Skip data splitting (use existing splits)')
    parser.add_argument('--skip_metrics', action='store_true', help='Skip metric computation')
    parser.add_argument('--skip_training', action='store_true', help='Skip training phase')
    parser.add_argument('--skip_analysis', action='store_true', help='Skip analysis phase')
    parser.add_argument('--do_evaluation', action='store_true', help='Conduct separate evaluation phase')
    parser.add_argument('--only_evaluation', action='store_true', help='Only Conduct separate evaluation phase')

    args = parser.parse_args()

    if args.parallel:
        logger.warning("--parallel is deprecated and ignored. Running sequential TrainingPipeline.")

    # Setup paths
    output_dir = Path(args.output_dir)
    if args.data_name is None:
        args.data_name = Path(args.data_path).stem
    output_dir = output_dir / args.data_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read CoT-DataSynth directory from config/bash_config.env
    current_script_dir = Path(__file__).resolve().parent
    config_env_path = current_script_dir / "../../config/bash_config.env" 
    load_dotenv(config_env_path)
    cot_datasynth_dir = os.getenv("REPO_DIR")
    if cot_datasynth_dir is None:
        raise ValueError("Could not find CoT-DataSynth directory. Please specify REPO_DIR in config/bash_config.env.")
    else:
        print(f"cot_datasynth_dir: {cot_datasynth_dir}")

    logger.info(f"Output directory: {output_dir}") 
    logger.info(f"CoT-DataSynth directory: {cot_datasynth_dir}")

    # Phase 1: Data Splitting
    if not args.skip_split and not args.only_evaluation:
        logger.info("=" * 50)
        logger.info("Phase 1: Data Splitting")
        logger.info("=" * 50)

        data = load_data(args.data_path)
        splitter = DataSplitter(args.n_splits, str(output_dir))
        splits = splitter.split_dataset(data, seed=args.seed)

        for split_id, split_data in enumerate(splits):
            splitter.save_split(split_data, split_id, format="parquet")

        logger.info(f"Saved {args.n_splits} splits to {output_dir}/splits/")
        splits_dir = output_dir / "splits"
    else:
        logger.info("Skipping data splitting (using existing splits)")
        splitter = DataSplitter(args.n_splits, str(output_dir))
        # 如果指定了 splits_dir，使用它；否则用默认路径
        if args.splits_dir:
            splits_dir = Path(args.splits_dir)
            logger.info(f"Using splits from: {splits_dir}")
            # 自动检测 splits 数量 (支持 parquet)
            existing_splits = list(splits_dir.glob("split_*.parquet"))
            if existing_splits:
                args.n_splits = max(int(f.stem.split("_")[1]) for f in existing_splits) + 1
                logger.info(f"Auto-detected {args.n_splits} splits")
        else:
            splits_dir = output_dir / "splits"

    # Phase 2: Compute Data Metrics
    if not args.skip_metrics and not args.only_evaluation:
        logger.info("=" * 50)
        logger.info("Phase 2: Computing Data Metrics")
        logger.info("=" * 50)

        # Convert similarity type
        try:
            sim_type = SimilarityType[args.similarity_type.upper()]
        except KeyError:
            logger.warning(f"Unknown similarity type: {args.similarity_type}, using jaccard")
            sim_type = SimilarityType.JACCARD

        metrics_list = []
        for split_id in range(args.n_splits):
            split_file = splits_dir / f"split_{split_id}.parquet"
            if not split_file.exists():
                logger.warning(f"Split file not found: {split_file}")
                continue

            split_data = load_data(str(split_file))

            # Compute all metrics
            metrics = {}

            # 1. statistics
            logger.info(f"Split {split_id}: computing statistics...")
            metrics.update(compute_data_statistics(split_data))

            # 2. quality
            logger.info(f"Split {split_id}: computing quality...")
            metrics.update(compute_data_quality_metrics(split_data))

            # 3. diversity (支持多种相似度函数)
            logger.info(f"Split {split_id}: computing diversity with {args.similarity_type}...")
            metrics.update(compute_dataset_diversity(
                split_data,
                similarity_type=sim_type,
                compute_on=args.compute_on
            ))

            # 4. entropy
            logger.info(f"Split {split_id}: computing entropy...")
            metrics.update(compute_dataset_entropy(split_data))

            # 5. PPL (需要 model)
            if args.model:
                logger.info(f"Split {split_id}: computing PPL with {args.model}...")
                try:
                    metrics.update(compute_ppl_metrics(
                        split_data,
                        model_name=args.model,
                        sample_ratio=args.ppl_ifd_sample_ratio,
                        max_samples=args.ppl_ifd_max_samples,
                        random_seed=args.ppl_ifd_sample_seed,
                    ))
                except Exception as e:
                    logger.warning(f"Failed to compute PPL: {e}")

            # 6. IFD (需要 model)
            if args.model:
                logger.info(f"Split {split_id}: computing IFD with {args.model}...")
                try:
                    metrics.update(compute_ifd_metrics(
                        split_data,
                        model_name=args.model,
                        sample_ratio=args.ppl_ifd_sample_ratio,
                        max_samples=args.ppl_ifd_max_samples,
                        random_seed=args.ppl_ifd_sample_seed,
                    ))
                except Exception as e:
                    logger.warning(f"Failed to compute IFD: {e}")

            dataset_metrics = DatasetMetrics(
                split_id=split_id,
                num_samples=len(split_data),
                metrics=metrics
            )
            splitter.save_metrics(dataset_metrics)

            # 展开 metrics 字典到顶层
            metrics_dict = dataset_metrics.to_dict()
            flattened = {
                'split_id': metrics_dict['split_id'],
                'num_samples': metrics_dict['num_samples']
            }
            flattened.update(metrics_dict['metrics'])
            metrics_list.append(flattened)

        # Save summary
        metrics_df = pd.DataFrame(metrics_list)
        metrics_csv = output_dir / "data_metrics_summary.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        logger.info(f"Saved metrics summary to {metrics_csv}")

    # Phase 3: Training
    if not args.skip_training and not args.only_evaluation:
        logger.info("=" * 50)
        logger.info("Phase 3: Training")
        logger.info("=" * 50)

        # Setup GPU allocation
        gpu_ids = [int(g) for g in args.gpu_ids.split(',')]
        allocator = GPUAllocator(gpu_ids, args.gpus_per_split)
        gpu_allocations = allocator.allocate(args.n_splits)

        # 如果有 --splits_dir，自动检测 splits 数量
        if args.splits_dir:
            splits_path = Path(args.splits_dir)
            # 自动检测 splits 数量 (支持 parquet)
            existing_splits = list(splits_path.glob("split_*.parquet"))
            if existing_splits:
                args.n_splits = max(int(f.stem.split("_")[1]) for f in existing_splits) + 1
                splits_dir = splits_path
                logger.info(f"Auto-detected {args.n_splits} splits from {splits_dir}")
        else:
            splits_dir = output_dir / "splits"

        

        training_pipeline = TrainingPipeline(str(output_dir), cot_datasynth_dir)

        # Prepare training configs
        base_config = {}

        configs = training_pipeline.prepare_training_configs(
            str(splits_dir),
            gpu_allocations,
            base_config,
            args.n_splits,
            args.model_id
        )

        # Run trainings
        logger.info(f"Running {len(configs)} trainings...")

        results = training_pipeline.run_all_trainings(
            configs,
            args.train_script,
            parallel=False,
            timeout=None,
            val_data_path=args.val_data_path,
            eval_data_name=args.eval_data_name,
            prompt_template_method=args.prompt_template_method,
            num_epochs=args.num_epochs
        )

        logger.info(f"Training results: {results}")

    # Phase 3.5: Evaluation (optional, can be run independently)
    if args.do_evaluation or args.only_evaluation:
        logger.info("=" * 50)
        logger.info("Phase 3.5: Evaluation on Test Set")
        logger.info("=" * 50)

        training_pipeline = TrainingPipeline(str(output_dir), cot_datasynth_dir)
        
        gpu_ids = [int(g) for g in args.gpu_ids.split(',')]
        allocator = GPUAllocator(gpu_ids, args.gpus_per_split)
        gpu_allocations = allocator.allocate(args.n_splits)
        eval_summary_rows = []

        logger.info(f"Running evaluation for {args.n_splits} splits...")
        logger.info(f"GPU allocations: {gpu_allocations}")

        for split_id in range(args.n_splits):
            split_output_dir = training_pipeline.training_dir / f"split_{split_id}"
            if split_output_dir.exists():
                gpu_id = gpu_allocations[split_id] if gpu_allocations[split_id] else [0]
                logger.info(f"Evaluating split {split_id} on GPU {gpu_id}...")

                config = {
                    'output_dir': str(split_output_dir),
                    'eval_output_dir': str(training_pipeline.eval_dir / f"split_{split_id}"),
                    'gpu_ids': gpu_id,
                    'base_model_id': args.model_id,
                }

                eval_success = training_pipeline.run_evaluation(
                    config,
                    args.eval_data_name,
                    prompt_template_method=args.prompt_template_method,
                )

                split_accuracy = None
                results_file = split_output_dir / "training_results.json"
                if results_file.exists():
                    try:
                        with open(results_file, 'r', encoding='utf-8') as f:
                            split_results = json.load(f)
                        split_accuracy = split_results.get("test_accuracy")
                    except Exception as e:
                        logger.warning(f"Failed to read {results_file}: {e}")

                eval_summary_rows.append({
                    "eval_data_name": args.eval_data_name,
                    "split_id": split_id,
                    "accuracy": split_accuracy,
                    "eval_success": bool(eval_success),
                })
            else:
                logger.warning(f"Split {split_id} output directory not found: {split_output_dir}")
                eval_summary_rows.append({
                    "eval_data_name": args.eval_data_name,
                    "split_id": split_id,
                    "accuracy": None,
                    "eval_success": False,
                })

        eval_summary_path = output_dir / "evaluation_split_accuracy.csv"

        split_accuracy_map = {
            int(row["split_id"]): row.get("accuracy")
            for row in eval_summary_rows
        }

        detected_split_ids = set()
        for split_dir in training_pipeline.training_dir.glob("split_*"):
            try:
                detected_split_ids.add(int(split_dir.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
        if not detected_split_ids:
            detected_split_ids = set(split_accuracy_map.keys())

        current_max_split = max(detected_split_ids) if detected_split_ids else -1
        current_row = {"dataset_name": args.eval_data_name}
        for split_id in range(current_max_split + 1):
            current_row[f"split_{split_id}"] = split_accuracy_map.get(split_id)

        current_values = [
            value for value in current_row.values()
            if isinstance(value, (int, float))
        ]
        if current_values:
            current_avg = float(sum(current_values) / len(current_values))
            current_var = float(sum((v - current_avg) ** 2 for v in current_values) / len(current_values))
        else:
            current_avg = None
            current_var = None
        current_row["avg"] = current_avg
        current_row["var"] = current_var

        new_df = pd.DataFrame([current_row])

        if eval_summary_path.exists():
            existing_df = pd.read_csv(eval_summary_path)

            existing_split_cols = [
                c for c in existing_df.columns
                if c.startswith("split_") and c.split("_")[-1].isdigit()
            ]
            existing_max_split = (
                max(int(c.split("_")[1]) for c in existing_split_cols)
                if existing_split_cols else -1
            )
            final_max_split = max(existing_max_split, current_max_split)
            final_columns = (
                ["dataset_name"]
                + [f"split_{i}" for i in range(final_max_split + 1)]
                + ["avg", "var"]
            )

            existing_df = existing_df.reindex(columns=final_columns)
            new_df = new_df.reindex(columns=final_columns)
            merged_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            final_columns = (
                ["dataset_name"]
                + [f"split_{i}" for i in range(current_max_split + 1)]
                + ["avg", "var"]
            )
            merged_df = new_df.reindex(columns=final_columns)

        merged_df.to_csv(eval_summary_path, index=False)
        logger.info(f"Saved split evaluation summary to {eval_summary_path}")

    # Phase 4: Analysis
    if not args.skip_analysis and not args.only_evaluation:
        logger.info("=" * 50)
        logger.info("Phase 4: Analysis and Visualization")
        logger.info("=" * 50)

        # Load data metrics
        metrics_csv = output_dir / "data_metrics_summary.csv"
        if not metrics_csv.exists():
            logger.warning(f"Metrics file not found: {metrics_csv}")
            return

        data_metrics_df = pd.read_csv(metrics_csv)

        # 展开 metrics 列（如果存在）
        if 'metrics' in data_metrics_df.columns:
            import ast
            # 将 metrics 字符串转换为字典
            metrics_expanded = data_metrics_df['metrics'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            # 展开为多列
            metrics_df = pd.json_normalize(metrics_expanded)
            # 删除原始 metrics 列，添加展开的列
            data_metrics_df = data_metrics_df.drop('metrics', axis=1)
            data_metrics_df = pd.concat([data_metrics_df, metrics_df], axis=1)

        # 自动收集训练结果
        logger.info("Collecting training results...")

        # 确保 training_pipeline 已初始化 (如果跳过了训练)
        if args.skip_training:
            training_pipeline = TrainingPipeline(str(output_dir), cot_datasynth_dir)

        training_results = training_pipeline.collect_training_results(
            args.n_splits,
            metric_keys=['train_loss', 'val_loss', 'val_accuracy']
        )

        if training_results:
            # 保存为 CSV
            training_results_df = pd.DataFrame(training_results).T
            training_results_df.index.name = 'split_id'
            training_results_csv = output_dir / "training_results.csv"
            training_results_df.to_csv(training_results_csv)
            logger.info(f"Saved training results to {training_results_csv}")
        else:
            logger.warning("No training results found")
            return
            # # 创建 dummy 数据用于演示
            # training_results_df = pd.DataFrame({
            #     'split_id': range(args.n_splits),
            #     'accuracy': [0.5 + 0.05 * i for i in range(args.n_splits)],
            #     'loss': [1.0 - 0.05 * i for i in range(args.n_splits)]
            # })
            # training_results_df = training_results_df.set_index('split_id')

        # Compute correlations - 保存到 observation 目录
        obs_dir = output_dir / "observation"
        obs_dir.mkdir(parents=True, exist_ok=True)

        analyzer = CorrelationAnalyzer(str(obs_dir))
        correlations = analyzer.compute_correlations(data_metrics_df, training_results_df)
        correlations.to_csv(obs_dir / "correlations.csv", index=False)
        logger.info(f"Saved correlations to {obs_dir}/correlations.csv")

        # 生成强相关性统计表
        strong_corr = analyzer.get_strong_correlations_summary(correlations, threshold=0.6)
        strong_corr.to_csv(obs_dir / "strong_correlations_0.6.csv", index=False)
        logger.info(f"Found {len(strong_corr)} strong correlations (|corr| >= 0.6)")

        # 生成中等相关性统计表
        medium_corr = analyzer.get_strong_correlations_summary(correlations, threshold=0.4)
        medium_corr.to_csv(obs_dir / "strong_correlations_0.4.csv", index=False)
        logger.info(f"Found {len(medium_corr)} medium correlations (|corr| >= 0.4)")

        # Generate visualizations
        visualizer = AnalysisVisualizer(str(obs_dir))
        visualizer.plot_correlation_heatmap(data_metrics_df, training_results_df)
        visualizer.plot_scatter_matrix(data_metrics_df, training_results_df)
        visualizer.plot_metrics_overview(data_metrics_df, training_results_df)
        visualizer.plot_training_convergence(training_results_df)

        logger.info("Analysis completed!")

    logger.info("=" * 50)
    logger.info("Pipeline completed!")
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
