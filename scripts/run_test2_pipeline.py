"""Run the full module readiness pipeline and print a diagnostic summary.

Usage:
    .venv/bin/python scripts/run_test2_pipeline.py
    .venv/bin/python scripts/run_test2_pipeline.py --quick
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
# Let the script import from `src/` directly when run from the repository checkout.
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from module_readiness import ModuleReadinessQueryAPI, run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the module readiness pipeline end-to-end")
    parser.add_argument("--quick", action="store_true", help="Cap modules at 350 for a faster run")
    parser.add_argument(
        "--query",
        type=str,
        default="entry level data analyst with python and sql",
        help="Sample query for the preview at the end",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Rows to show in previews")
    args = parser.parse_args()

    mode = "quick" if args.quick else "full"
    print(f"Running pipeline in {mode} mode...")
    if not args.quick:
        print("Full mode uses the configured module corpus and can take a while.")
    state = run_pipeline(quick=args.quick)
    api = ModuleReadinessQueryAPI(state)

    # Show a compact preview of the main outputs so the script doubles as a smoke test.
    print("\n=== Diagnostics ===")
    print(json.dumps(state.diagnostics, indent=2))

    print("\n=== Top module role scores ===")
    cols = [c for c in ["module_code", "module_title", "role_family", "role_score"]
            if c in state.module_role_scores.columns]
    print(
        state.module_role_scores[cols]
        .sort_values("role_score", ascending=False)
        .head(args.top_k)
        .to_string(index=False)
    )

    print(f"\n=== Job search: '{args.query}' ===")
    jobs_result = api.search_jobs(args.query, exp_max=2, top_k=min(5, args.top_k))
    print(jobs_result.to_string(index=False) if not jobs_result.empty else "(no results)")

    print(f"\n=== Module recommendations: '{args.query}' ===")
    modules_result = api.recommend_relevant_modules(args.query, top_k=min(5, args.top_k))
    print(modules_result.to_string(index=False) if not modules_result.empty else "(no results)")


if __name__ == "__main__":
    main()
