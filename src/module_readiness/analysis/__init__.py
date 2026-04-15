__all__ = [
    "AggregationResult",
    "DegreeAggregationResult",
    "ScoringResult",
    "build_retrieval_candidate_pool",
    "build_indicators",
    "build_degree_outputs",
    "compute_scores",
    "evaluate_retrieval_labels",
    "grid_search_retrieval_thresholds",
    "split_labeled_retrieval_dataset",
    "split_modules_for_evaluation",
]


def __getattr__(name: str):
    if name in {"DegreeAggregationResult", "build_degree_outputs"}:
        from .degrees import DegreeAggregationResult, build_degree_outputs

        return {
            "DegreeAggregationResult": DegreeAggregationResult,
            "build_degree_outputs": build_degree_outputs,
        }[name]
    if name in {"AggregationResult", "build_indicators"}:
        from .aggregation import AggregationResult, build_indicators

        return {
            "AggregationResult": AggregationResult,
            "build_indicators": build_indicators,
        }[name]
    if name in {
        "build_retrieval_candidate_pool",
        "evaluate_retrieval_labels",
        "grid_search_retrieval_thresholds",
        "split_labeled_retrieval_dataset",
        "split_modules_for_evaluation",
    }:
        from .retrieval_eval import (
            build_retrieval_candidate_pool,
            evaluate_retrieval_labels,
            grid_search_retrieval_thresholds,
            split_labeled_retrieval_dataset,
            split_modules_for_evaluation,
        )

        return {
            "build_retrieval_candidate_pool": build_retrieval_candidate_pool,
            "evaluate_retrieval_labels": evaluate_retrieval_labels,
            "grid_search_retrieval_thresholds": grid_search_retrieval_thresholds,
            "split_labeled_retrieval_dataset": split_labeled_retrieval_dataset,
            "split_modules_for_evaluation": split_modules_for_evaluation,
        }[name]
    if name in {"ScoringResult", "compute_scores"}:
        from .scoring import ScoringResult, compute_scores

        return {
            "ScoringResult": ScoringResult,
            "compute_scores": compute_scores,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
