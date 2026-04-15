"""Build labeling pools and evaluate retrieval quality for module-to-job matching.

Example workflow:
    python3 scripts/evaluate_retrieval.py export-pool --sample-size 100 --output outputs/retrieval_label_pool.csv
    # Manually fill the `relevance` column in the exported CSV.
    python3 scripts/evaluate_retrieval.py evaluate --labels outputs/retrieval_label_pool.csv --k 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from module_readiness import run_pipeline  # noqa: E402
from module_readiness.analysis import (  # noqa: E402
    build_retrieval_candidate_pool,
    evaluate_retrieval_labels,
    grid_search_retrieval_thresholds,
    split_labeled_retrieval_dataset,
    split_modules_for_evaluation,
)


def _print_frame(title: str, frame: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    if frame.empty:
        print("(no rows)")
        return
    print(frame.to_string(index=False))


def _load_state(*, quick: bool) -> object:
    print("Running pipeline to load jobs, modules, and retrieval artifacts...")
    return run_pipeline(quick=quick)


def _export_pool(args: argparse.Namespace) -> None:
    state = _load_state(quick=args.quick)
    pool = build_retrieval_candidate_pool(
        jobs=state.jobs,
        modules=state.modules,
        retrieval=state.retrieval,
        module_codes=args.module_codes,
        sample_size=args.sample_size,
        seed=args.seed,
        per_mode_top_k=args.per_mode_top_k,
        final_top_k=args.final_top_k,
    )
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(output_path, index=False)

    module_count = pool["module_code"].nunique() if not pool.empty else 0
    per_module_counts = pool.groupby("module_code").size() if not pool.empty else pd.Series(dtype=int)
    print(f"Wrote candidate pool for {module_count} modules to {output_path}")
    if args.final_top_k > 0:
        print(
            f"Each module is represented by up to {args.final_top_k} pooled jobs after "
            "mode-balanced pooling across hybrid, BM25, and embedding retrieval."
        )
    else:
        print(
            "Each module keeps the full union of the per-ranker candidates after "
            "mode-balanced pooling across hybrid, BM25, and embedding retrieval."
        )
    if not per_module_counts.empty:
        print(
            "Rows per module: "
            f"min={int(per_module_counts.min())}, "
            f"median={float(per_module_counts.median()):.1f}, "
            f"max={int(per_module_counts.max())}"
        )
    print("Fill the `relevance` column with graded labels, for example 0, 1, 2, 3.")
    _print_frame("Candidate Pool Preview", pool.head(15))


def _build_split_manifest(
    train_modules: pd.DataFrame,
    test_modules: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split_name, frame in (("train", train_modules), ("test", test_modules)):
        manifest = frame.copy()
        manifest["split"] = split_name
        frames.append(manifest)

    manifest = pd.concat(frames, ignore_index=True)
    ordered_columns = [
        "split",
        "module_code",
        "module_title",
        "module_faculty",
        "faculty",
        "module_profile",
        "module_description",
    ]
    ordered_columns = [column for column in ordered_columns if column in manifest.columns]
    return manifest[ordered_columns].sort_values(["split", "module_code"]).reset_index(drop=True)


def _export_split_pools(args: argparse.Namespace) -> None:
    state = _load_state(quick=args.quick)
    train_modules, test_modules = split_modules_for_evaluation(
        state.modules,
        module_codes=args.module_codes,
        sample_size=args.sample_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    train_codes = train_modules["module_code"].astype(str).tolist()
    test_codes = test_modules["module_code"].astype(str).tolist()

    train_pool = build_retrieval_candidate_pool(
        jobs=state.jobs,
        modules=state.modules,
        retrieval=state.retrieval,
        module_codes=train_codes,
        seed=args.seed,
        per_mode_top_k=args.per_mode_top_k,
        final_top_k=args.final_top_k,
    )
    test_pool = build_retrieval_candidate_pool(
        jobs=state.jobs,
        modules=state.modules,
        retrieval=state.retrieval,
        module_codes=test_codes,
        seed=args.seed,
        per_mode_top_k=args.per_mode_top_k,
        final_top_k=args.final_top_k,
    )

    split_manifest = _build_split_manifest(train_modules, test_modules)

    train_output = Path(args.train_output).expanduser().resolve()
    test_output = Path(args.test_output).expanduser().resolve()
    split_output = Path(args.split_output).expanduser().resolve()
    for path in (train_output, test_output, split_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    train_pool.to_csv(train_output, index=False)
    test_pool.to_csv(test_output, index=False)
    split_manifest.to_csv(split_output, index=False)

    print(
        "Exported module split and labeling pools: "
        f"{len(train_codes)} train modules, {len(test_codes)} test modules."
    )
    print(f"Saved split manifest to {split_output}")
    print(f"Saved train labeling pool to {train_output}")
    print(f"Saved test labeling pool to {test_output}")
    print(
        f"Train rows: {len(train_pool)}; "
        f"Test rows: {len(test_pool)}"
    )
    _print_frame("Module Split Preview", split_manifest.head(15))
    _print_frame("Train Pool Preview", train_pool.head(10))
    _print_frame("Test Pool Preview", test_pool.head(10))


def _evaluate(args: argparse.Namespace) -> None:
    labels_path = Path(args.labels).expanduser().resolve()
    labels = pd.read_csv(labels_path)
    state = _load_state(quick=args.quick)
    summary, details = evaluate_retrieval_labels(
        labels=labels,
        jobs=state.jobs,
        modules=state.modules,
        retrieval=state.retrieval,
        k=args.k,
    )

    _print_frame(f"Retrieval Summary @ {args.k}", summary)
    _print_frame("Per-Module Details", details.head(20))
    print(
        "\nNote: precision/recall treat jobs with relevance >= 2 as relevant, and "
        "unlabeled jobs are treated as non-relevant."
    )

    if args.summary_output:
        summary_output = Path(args.summary_output).expanduser().resolve()
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_output, index=False)
        print(f"Saved summary metrics to {summary_output}")
    if args.details_output:
        details_output = Path(args.details_output).expanduser().resolve()
        details_output.parent.mkdir(parents=True, exist_ok=True)
        details.to_csv(details_output, index=False)
        print(f"Saved per-module metrics to {details_output}")


def _grid_values(values: list[float] | None, fallback: float) -> list[float]:
    if values:
        return [float(value) for value in values]
    return [float(fallback)]


def _grid_search_thresholds(args: argparse.Namespace) -> None:
    labels_path = Path(args.labels).expanduser().resolve()
    labels = pd.read_csv(labels_path)
    state = _load_state(quick=args.quick)

    results = grid_search_retrieval_thresholds(
        labels=labels,
        jobs=state.jobs,
        modules=state.modules,
        retrieval=state.retrieval,
        k=int(args.k),
        bm25_min_scores=_grid_values(args.bm25_min_scores, state.config.bm25_min_score),
        bm25_relative_mins=_grid_values(args.bm25_relative_mins, state.config.bm25_relative_min),
        embedding_min_similarities=_grid_values(
            args.embedding_min_similarities,
            state.config.embedding_min_similarity,
        ),
        embedding_relative_mins=_grid_values(
            args.embedding_relative_mins,
            state.config.embedding_relative_min,
        ),
    )

    _print_frame("Threshold Grid Search Results", results.head(20))
    best_per_mode = results.groupby("mode", as_index=False).head(1).reset_index(drop=True)
    _print_frame("Best Thresholds Per Mode", best_per_mode)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"Saved threshold grid-search results to {output_path}")


def _split_labeled_dataset(args: argparse.Namespace) -> None:
    labels_path = Path(args.labels).expanduser().resolve()
    labels = pd.read_csv(labels_path)
    train_labels, test_labels, split_manifest = split_labeled_retrieval_dataset(
        labels,
        test_size=args.test_size,
        seed=args.seed,
    )

    train_output = Path(args.train_output).expanduser().resolve()
    test_output = Path(args.test_output).expanduser().resolve()
    split_output = Path(args.split_output).expanduser().resolve()
    for path in (train_output, test_output, split_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    train_labels.to_csv(train_output, index=False)
    test_labels.to_csv(test_output, index=False)
    split_manifest.to_csv(split_output, index=False)

    print(
        "Split labeled dataset by module: "
        f"{train_labels['module_code'].nunique()} train modules, "
        f"{test_labels['module_code'].nunique()} test modules."
    )
    print(f"Saved train labels to {train_output}")
    print(f"Saved test labels to {test_output}")
    print(f"Saved split manifest to {split_output}")
    _print_frame("Split Manifest Preview", split_manifest.head(15))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export module-job labeling pools and evaluate hybrid/BM25/embedding retrieval."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export-pool",
        help="Create a pooled module-job candidate CSV for manual labeling.",
    )
    export_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the CSV file that will contain the module-job candidate pool.",
    )
    export_parser.add_argument(
        "--module-codes",
        nargs="*",
        default=None,
        help="Optional explicit module codes to export. If omitted, the script samples modules.",
    )
    export_parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of modules to export when --module-codes is not supplied.",
    )
    export_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for module sampling.",
    )
    export_parser.add_argument(
        "--per-mode-top-k",
        type=int,
        default=10,
        help="How many jobs to retrieve per ranker before forming the pooled labeling shortlist.",
    )
    export_parser.add_argument(
        "--final-top-k",
        type=int,
        default=0,
        help=(
            "Maximum pooled jobs to keep per module in the exported labeling sheet. "
            "Use 0 to keep the full union across rankers."
        ),
    )
    export_parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the quick pipeline variant. Use only for debugging, not final evaluation.",
    )
    export_parser.set_defaults(func=_export_pool)

    split_parser = subparsers.add_parser(
        "export-split-pools",
        help="Sample modules once, split them into train/test sets, and export separate labeling pools.",
    )
    split_parser.add_argument(
        "--train-output",
        type=str,
        required=True,
        help="Path to the CSV file that will contain the train labeling pool.",
    )
    split_parser.add_argument(
        "--test-output",
        type=str,
        required=True,
        help="Path to the CSV file that will contain the test labeling pool.",
    )
    split_parser.add_argument(
        "--split-output",
        type=str,
        required=True,
        help="Path to the CSV manifest containing the sampled module split.",
    )
    split_parser.add_argument(
        "--module-codes",
        nargs="*",
        default=None,
        help="Optional explicit module codes to split. If omitted, the script samples modules first.",
    )
    split_parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of modules to sample before splitting when --module-codes is not supplied.",
    )
    split_parser.add_argument(
        "--test-size",
        type=int,
        default=40,
        help="Number of sampled modules to place in the test split.",
    )
    split_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for module sampling and split assignment.",
    )
    split_parser.add_argument(
        "--per-mode-top-k",
        type=int,
        default=10,
        help="How many jobs to retrieve per ranker before forming each pooled labeling shortlist.",
    )
    split_parser.add_argument(
        "--final-top-k",
        type=int,
        default=0,
        help=(
            "Maximum pooled jobs to keep per module in each exported labeling sheet. "
            "Use 0 to keep the full union across rankers."
        ),
    )
    split_parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the quick pipeline variant. Use only for debugging, not final evaluation.",
    )
    split_parser.set_defaults(func=_export_split_pools)

    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Score a manually labeled module-job file with nDCG@k, Precision@k, and Recall@k.",
    )
    eval_parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="CSV containing at least module_code, job_id, and relevance columns.",
    )
    eval_parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Cutoff for ranking metrics such as nDCG@k.",
    )
    eval_parser.add_argument(
        "--summary-output",
        type=str,
        default="",
        help="Optional path to save the aggregated summary metrics CSV.",
    )
    eval_parser.add_argument(
        "--details-output",
        type=str,
        default="",
        help="Optional path to save the per-module metrics CSV.",
    )
    eval_parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the quick pipeline variant. Use only if your labeled modules are present in quick mode.",
    )
    eval_parser.set_defaults(func=_evaluate)

    split_labels_parser = subparsers.add_parser(
        "split-labeled-dataset",
        help="Split an already labeled retrieval CSV into train/test files by module_code.",
    )
    split_labels_parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to the labeled retrieval CSV to split.",
    )
    split_labels_parser.add_argument(
        "--train-output",
        type=str,
        required=True,
        help="Path to the CSV file that will contain the train labels.",
    )
    split_labels_parser.add_argument(
        "--test-output",
        type=str,
        required=True,
        help="Path to the CSV file that will contain the test labels.",
    )
    split_labels_parser.add_argument(
        "--split-output",
        type=str,
        required=True,
        help="Path to the CSV manifest containing the labeled module split.",
    )
    split_labels_parser.add_argument(
        "--test-size",
        type=int,
        default=40,
        help="Number of unique modules to place in the test split.",
    )
    split_labels_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split assignment.",
    )
    split_labels_parser.set_defaults(func=_split_labeled_dataset)

    grid_parser = subparsers.add_parser(
        "grid-search-thresholds",
        help="Evaluate multiple retrieval-threshold combinations against one labeled CSV.",
    )
    grid_parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="CSV containing at least module_code, job_id, and relevance columns.",
    )
    grid_parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Cutoff for ranking metrics such as nDCG@k.",
    )
    grid_parser.add_argument(
        "--bm25-min-scores",
        nargs="*",
        type=float,
        default=None,
        help="Grid of absolute BM25 score thresholds. Defaults to the current config value.",
    )
    grid_parser.add_argument(
        "--bm25-relative-mins",
        nargs="*",
        type=float,
        default=None,
        help="Grid of relative BM25 score thresholds. Defaults to the current config value.",
    )
    grid_parser.add_argument(
        "--embedding-min-similarities",
        nargs="*",
        type=float,
        default=None,
        help="Grid of absolute embedding similarity thresholds. Defaults to the current config value.",
    )
    grid_parser.add_argument(
        "--embedding-relative-mins",
        nargs="*",
        type=float,
        default=None,
        help="Grid of relative embedding similarity thresholds. Defaults to the current config value.",
    )
    grid_parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to save the threshold grid-search results CSV.",
    )
    grid_parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the quick pipeline variant. Use only if your labeled modules are present in quick mode.",
    )
    grid_parser.set_defaults(func=_grid_search_thresholds)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
