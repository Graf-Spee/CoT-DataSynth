"""DataObs library."""

from importlib import import_module

__all__ = [
    # Core classes
    'DataSplitter',
    'DatasetMetrics',
    'GPUAllocator',
    'TrainingConfig',
    'TrainingResult',
    'ResultCollector',
    # Basic metrics
    'compute_data_statistics',
    'compute_data_quality_metrics',
    'compute_difficulty_metrics',
    'compute_all_data_metrics',
    # Training and analysis
    'TrainingPipeline',
    'CorrelationAnalyzer',
    'AnalysisVisualizer',
    # Advanced metrics
    'compute_text_similarity_matrix',
    'compute_dataset_diversity',
    'compute_dataset_entropy',
    'MultiRewardDifficultyAssessor',
    'RewardFunctionConfig',
    'compute_advanced_data_metrics',
    'reward_exact_match',
    'reward_partial_match',
    'reward_contains',
    'reward_similarity',
    # Diversity framework
    'SimilarityType',
    'DiversityConfig',
    'SimilarityFunction',
    'get_similarity_function',
    'compute_diversity_with_similarity',
]

_LAZY_IMPORTS = {
    "DataSplitter": (".data_process.data_obs", "DataSplitter"),
    "DatasetMetrics": (".data_process.data_obs", "DatasetMetrics"),
    "GPUAllocator": (".data_process.data_obs", "GPUAllocator"),
    "TrainingConfig": (".data_process.data_obs", "TrainingConfig"),
    "TrainingResult": (".data_process.data_obs", "TrainingResult"),
    "ResultCollector": (".data_process.data_obs", "ResultCollector"),
    "jsonl_to_parquet": (".data_process.data_obs", "jsonl_to_parquet"),
    "batch_jsonl_to_parquet": (".data_process.data_obs", "batch_jsonl_to_parquet"),
    "compute_data_statistics": (".metrics.data_metrics", "compute_data_statistics"),
    "compute_data_quality_metrics": (".metrics.data_metrics", "compute_data_quality_metrics"),
    "compute_difficulty_metrics": (".metrics.data_metrics", "compute_difficulty_metrics"),
    "compute_all_data_metrics": (".metrics.data_metrics", "compute_all_data_metrics"),
    "TrainingPipeline": (".training.training_pipeline", "TrainingPipeline"),
    "CorrelationAnalyzer": (".analysis.analysis_pipeline", "CorrelationAnalyzer"),
    "AnalysisVisualizer": (".analysis.analysis_pipeline", "AnalysisVisualizer"),
    "compute_text_similarity_matrix": (".metrics.advanced_metrics", "compute_text_similarity_matrix"),
    "compute_dataset_diversity": (".metrics.advanced_metrics", "compute_dataset_diversity"),
    "compute_dataset_entropy": (".metrics.advanced_metrics", "compute_dataset_entropy"),
    "MultiRewardDifficultyAssessor": (".metrics.advanced_metrics", "MultiRewardDifficultyAssessor"),
    "RewardFunctionConfig": (".metrics.advanced_metrics", "RewardFunctionConfig"),
    "compute_advanced_data_metrics": (".metrics.advanced_metrics", "compute_advanced_data_metrics"),
    "reward_exact_match": (".metrics.advanced_metrics", "reward_exact_match"),
    "reward_partial_match": (".metrics.advanced_metrics", "reward_partial_match"),
    "reward_contains": (".metrics.advanced_metrics", "reward_contains"),
    "reward_similarity": (".metrics.advanced_metrics", "reward_similarity"),
    "SimilarityType": (".metrics.advanced_metrics", "SimilarityType"),
    "DiversityConfig": (".metrics.advanced_metrics", "DiversityConfig"),
    "SimilarityFunction": (".metrics.advanced_metrics", "SimilarityFunction"),
    "get_similarity_function": (".metrics.advanced_metrics", "get_similarity_function"),
    "compute_diversity_with_similarity": (".metrics.advanced_metrics", "compute_diversity_with_similarity"),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
