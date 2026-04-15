"""Run query examples against a completed pipeline state.

Runs the full pipeline once, then demonstrates the three query methods:
  - search_jobs
  - recommend_relevant_modules
  - get_module_role_profile

Usage:
    .venv/bin/python scripts/run_test2_queries.py
    .venv/bin/python scripts/run_test2_queries.py --query "data science internship" --role-family "Data & Analytics"
    .venv/bin/python scripts/run_test2_queries.py --quick
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
# Let the script import from `src/` directly when run from the repository checkout.
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from module_readiness import ModuleReadinessQueryAPI, run_pipeline  # noqa: E402


SAMPLE_QUERIES = [
    "entry level policy research role",
    "data analyst python sql",
    "software engineer machine learning",
]
SAMPLE_MODULE = "CS3244"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run query examples against the pipeline")
    parser.add_argument(
        "--query",
        type=str,
        default=SAMPLE_QUERIES[0],
        help="Natural language job query",
    )
    parser.add_argument(
        "--role-family",
        type=str,
        default="",
        help="Optional role family filter for module recommendations",
    )
    parser.add_argument(
        "--module",
        type=str,
        default=SAMPLE_MODULE,
        help="Module code to profile",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--quick", action="store_true", help="Cap modules at 350 for faster run")
    args = parser.parse_args()

    print("Running pipeline...")
    state = run_pipeline(quick=args.quick)
    api = ModuleReadinessQueryAPI(state)

    role_family = args.role_family.strip() or None

    # Each section demonstrates a different query surface over the same precomputed state.
    print(f"\n=== Job search: '{args.query}' ===")
    jobs_result = api.search_jobs(args.query, exp_max=2, top_k=args.top_k)
    print(jobs_result.to_string(index=False) if not jobs_result.empty else "(no results)")

    print(f"\n=== Module recommendations: '{args.query}'"
          + (f" (role: {role_family})" if role_family else "") + " ===")
    modules_result = api.recommend_relevant_modules(
        args.query,
        top_k=args.top_k,
        role_family=role_family,
    )
    print(modules_result.to_string(index=False) if not modules_result.empty else "(no results)")

    print(f"\n=== Module role profile: {args.module} ===")
    profile = api.get_module_role_profile(args.module, top_families=5)
    print(profile.to_string(index=False) if not profile.empty else f"(module {args.module} not found in scored results)")


if __name__ == "__main__":
    main()
