"""
Extract per-module preclusion relationships from raw_modules and write
outputs/module_preclusions.csv.

Each output row represents one (module_code, precluded_module_code) pair.

Columns
-------
module_code         : the source module  (e.g. CS3244)
precluded_code      : the module it precludes  (e.g. CS2109S)
has_wildcard        : True if the precluded token contained a % wildcard
raw_preclusion_rule : the original preclusionRule string (for audit)

Usage
-----
    python -m src.module_readiness.ingestion.extract_preclusions
    # or from project root:
    python src/module_readiness/ingestion/extract_preclusions.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils.db_utils import get_engine  # noqa: E402
from sqlalchemy import text  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "module_preclusions.csv"

# Matches a module-code token: 2-6 uppercase letters followed by digits/letters/%.
# No trailing \b so that the % wildcard suffix (non-word char) is captured.
_MODULE_TOKEN_RE = re.compile(r"\b([A-Z]{2,6}\d[A-Z0-9]*%?)")


def _extract_precluded_codes(rule: str) -> list[tuple[str, bool]]:
    """
    Parse a preclusionRule string and return a list of (code, has_wildcard) pairs.

    The rule format looks like:
        PROGRAM_TYPES IF_IN Undergraduate Degree THEN COURSES (1) CS2109S:D, CS3263:D
        PROGRAM_TYPES IF_IN ... THEN (COURSES (1) ACC3604:D AND COURSES (1) LC2008:D)

    Strategy: extract all tokens that look like module codes (letter prefix + digits),
    strip the optional grade suffix (e.g. ":D"), and deduplicate.
    Tokens that are clearly structural keywords (PROGRAM_TYPES, IF_IN, THEN, COURSES,
    PROGRAMS, COHORT_YEARS, SUBJECTS, MUST_NOT_BE_IN, AND, OR, NOT) are ignored.
    """
    if not rule or not isinstance(rule, str):
        return []

    _KEYWORDS = {
        "PROGRAM", "PROGRAMS", "PROGRAM_TYPES", "IF_IN", "THEN", "COURSES",
        "COHORT_YEARS", "SUBJECTS", "MUST_NOT_BE_IN", "AND", "OR", "NOT",
        "UNDERGRADUATE", "GRADUATE", "DEGREE", "COURSEWORK", "RESEARCH",
        "HONORS", "HONOURS",
    }

    results: list[tuple[str, bool]] = []
    seen: set[str] = set()

    for match in _MODULE_TOKEN_RE.finditer(rule):
        token = match.group(1)
        # Strip grade suffix like :D, :C
        token = token.split(":")[0]
        upper = token.upper()

        if upper in _KEYWORDS:
            continue
        # Skip pure-numeric tokens and very short ones (likely not module codes)
        if re.fullmatch(r"\d+", upper):
            continue
        # Must start with at least 2 letters then a digit to be a module code
        if not re.match(r"^[A-Z]{2,6}\d", upper):
            continue

        if upper not in seen:
            seen.add(upper)
            has_wildcard = "%" in upper
            results.append((upper, has_wildcard))

    return results


def build_preclusions(modules_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the raw_modules DataFrame, return a long-form preclusion table.
    Uses preclusionRule when available, falls back to parsing the human-readable
    preclusion text otherwise.
    """
    rows: list[dict] = []

    for _, mod in modules_df.iterrows():
        code = str(mod.get("moduleCode", "")).strip()
        if not code:
            continue

        rule = str(mod.get("preclusionRule", "") or "").strip()
        fallback = str(mod.get("preclusion", "") or "").strip()

        source = rule if rule else fallback
        if not source:
            continue

        pairs = _extract_precluded_codes(source)
        for precluded_code, has_wildcard in pairs:
            # Skip self-references (module precluding itself — common with wildcard stems)
            stem = precluded_code.rstrip("%")
            if stem and code.startswith(stem.rstrip("%")) and has_wildcard:
                # e.g. CS1010% on CS1010E — skip self-family entries if they match
                pass  # keep these; they're informative (other variants in the family)
            rows.append({
                "module_code": code,
                "precluded_code": precluded_code,
                "has_wildcard": has_wildcard,
            })

    df = pd.DataFrame(rows, columns=["module_code", "precluded_code", "has_wildcard"])
    return df.drop_duplicates(subset=["module_code", "precluded_code"]).reset_index(drop=True)


def main() -> None:
    print("Loading raw_modules from database...")
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text('SELECT "moduleCode", "preclusionRule", "preclusion" FROM raw_modules')
        )
        modules_df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
    print(f"  {len(modules_df):,} modules loaded")

    print("Extracting preclusions...")
    preclusions_df = build_preclusions(modules_df)

    n_modules = preclusions_df["module_code"].nunique()
    n_pairs = len(preclusions_df)
    n_wildcard = preclusions_df["has_wildcard"].sum()
    print(f"  {n_pairs:,} preclusion pairs across {n_modules:,} modules ({n_wildcard:,} with wildcards)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    preclusions_df.to_csv(OUTPUT_PATH, index=False)
    print(f"  Written to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
