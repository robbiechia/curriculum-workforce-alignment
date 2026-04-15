"""
Database utilities: connection management, read/write helpers, and bulk data loaders.

Quick usage from any module_readiness script:

    from data_utils.db_utils import read_table, read_sql, write_table

    jobs = read_table("raw_jobs")
    modules = read_sql("SELECT * FROM raw_modules WHERE faculty = 'Computing'")
    write_table(results_df, "my_output_table")

Data setup (check DB status and load missing tables):
    bash src/data_utils/data_setup.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
 

# ---------------------------------------------------------------------------
# .env loader (no external deps)
# ---------------------------------------------------------------------------

def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

_engine = None


def get_engine():
    """Return a cached SQLAlchemy engine built from DATABASE_URL in the environment."""
    global _engine
    if _engine is not None:
        return _engine

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise EnvironmentError(
            "DATABASE_URL not set. Add it to .env — see src/data_utils/DATA_SETUP.md for instructions."
        )

    try:
        from sqlalchemy import create_engine
    except ImportError:
        raise ImportError("sqlalchemy not installed. Run: pip install sqlalchemy psycopg2-binary")

    _engine = create_engine(db_url)
    return _engine


# ---------------------------------------------------------------------------
# Read / write helpers
# ---------------------------------------------------------------------------

def read_table(table_name: str, engine=None):
    """
    Load an entire database table into a DataFrame.

    Args:
        table_name: Name of the table (e.g. "raw_jobs", "raw_modules").
        engine: SQLAlchemy engine. Defaults to get_engine().

    Returns:
        pandas.DataFrame
    """

    engine = engine or get_engine()
    return pd.read_sql_table(table_name, engine)


def read_sql(query: str, engine=None, params=None):
    """
    Run a SQL query and return results as a DataFrame.

    Args:
        query: SQL string (e.g. "SELECT * FROM raw_jobs WHERE salary_min > 3000").
        engine: SQLAlchemy engine. Defaults to get_engine().
        params: Optional dict of query parameters for parameterised queries.

    Returns:
        pandas.DataFrame
    """
    import pandas as pd
    engine = engine or get_engine()
    return pd.read_sql(query, engine, params=params)


def write_table(df, table_name: str, if_exists: str = "replace", engine=None) -> int:
    """
    Write a DataFrame to a database table.

    Args:
        df: pandas.DataFrame to write.
        table_name: Target table name.
        if_exists: "replace" (default) drops and recreates, "append" adds rows,
                   "fail" raises if table exists.
        engine: SQLAlchemy engine. Defaults to get_engine().

    Returns:
        Number of rows written.
    """
    engine = engine or get_engine()
    df_to_write = df.copy()
    _serialize_complex_values(df_to_write)
    df_to_write.to_sql(table_name, engine, if_exists=if_exists, index=False)
    return len(df_to_write)


def write_logged_table(df, table_name: str, if_exists: str = "replace", engine=None) -> int:
    """Write a DataFrame to a table and update the shared load log."""
    engine = engine or get_engine()
    row_count = write_table(df, table_name=table_name, if_exists=if_exists, engine=engine)
    _log_load(table_name, row_count, engine)
    return row_count


# ---------------------------------------------------------------------------
# Load log
# ---------------------------------------------------------------------------

def _log_load(table_name: str, row_count: int, engine=None) -> None:
    """Upsert a row into _data_log recording when a table was last loaded."""
    engine = engine or get_engine()
    try:
        from sqlalchemy import text
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS _data_log (
                    table_name TEXT PRIMARY KEY,
                    loaded_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    row_count  INTEGER
                )
            """))
            conn.execute(text("""
                INSERT INTO _data_log (table_name, loaded_at, row_count)
                VALUES (:t, NOW(), :r)
                ON CONFLICT (table_name) DO UPDATE
                    SET loaded_at = NOW(), row_count = EXCLUDED.row_count
            """), {"t": table_name, "r": row_count})
    except Exception as e:
        print(f"  [warn] _data_log update failed for {table_name}: {e}")


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------

def _serialize_complex_values(df) -> None:
    """Serialize every list or dict cell to a JSON string (cell-level, no sampling)."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v
            )


# ---------------------------------------------------------------------------
# Bulk raw jobs loader
# ---------------------------------------------------------------------------

def _flatten_job(data: dict) -> dict:
    # Flatten the nested MSF JSON structure into one relational row per job posting.
    salary = data.get("salary") or {}
    company = data.get("postedCompany") or {}
    meta = data.get("metadata") or {}
    return {
        "job_id": meta.get("jobPostId") or data.get("uuid"),
        "title": data.get("title"),
        "description": data.get("description"),
        "ssoc_code": data.get("ssocCode"),
        "ssec_eqa": data.get("ssecEqa") if isinstance(data.get("ssecEqa"), str) else (data.get("ssecEqa") or {}).get("ssecEqa"),
        "salary_min": salary.get("minimum"),
        "salary_max": salary.get("maximum"),
        "salary_type": (salary.get("type") or {}).get("salaryType"),
        "min_experience_years": data.get("minimumYearsExperience"),
        "num_vacancies": data.get("numberOfVacancies"),
        "status": (data.get("status") or {}).get("jobStatus") if isinstance(data.get("status"), dict) else data.get("status"),
        "company_uen": company.get("uen"),
        "company_name": company.get("name"),
        "skills": json.dumps([s.get("skill") for s in (data.get("skills") or [])]),
        "categories": json.dumps([c.get("category") for c in (data.get("categories") or [])]),
        "employment_types": json.dumps(data.get("employmentTypes") or []),
        "position_levels": json.dumps(data.get("positionLevels") or []),
        "posted_at": meta.get("createdAt"),
        "deleted_at": meta.get("deletedAt"),
    }


def load_raw_jobs(engine=None) -> int:
    """Flatten all MSF job JSON files and load them into the raw_jobs table."""
    import pandas as pd
    engine = engine or get_engine()

    msf_dir = PROJECT_ROOT / "data" / "MSF_data"
    files = sorted(msf_dir.glob("*.json"))
    if not files:
        print(f"  [skip] No JSON files found in {msf_dir}")
        return 0

    rows = []
    for f in files:
        try:
            with open(f) as fh:
                rows.append(_flatten_job(json.load(fh)))
        except Exception as e:
            print(f"  [warn] {f.name}: {e}")

    df = pd.DataFrame(rows)
    df.to_sql("raw_jobs", engine, if_exists="replace", index=False)
    _log_load("raw_jobs", len(df), engine)
    print(f"  raw_jobs: {len(df):,} rows")
    return len(df)


# ---------------------------------------------------------------------------
# Bulk raw modules loader
# ---------------------------------------------------------------------------

def _pick_nusmods_year(override: Optional[str]) -> Path:
    base = PROJECT_ROOT / "data" / "nusmods"
    if override:
        year_dir = base / override
        if not year_dir.exists():
            sys.exit(f"ERROR: NUSMods cache not found for year '{override}' at {year_dir}")
        return year_dir
    years = sorted(base.iterdir()) if base.exists() else []
    if not years:
        sys.exit(f"ERROR: No NUSMods cache found at {base}. Run scrape_nusmods.py first.")
    return years[-1]


def load_raw_modules(engine=None, year_override: Optional[str] = None) -> int:
    """Merge NUSMods module list + details and load them into the raw_modules table."""
    import pandas as pd
    engine = engine or get_engine()

    year_dir = _pick_nusmods_year(year_override)
    print(f"  NUSMods year: {year_dir.name}")

    list_path = year_dir / "moduleList.json"
    if not list_path.exists():
        print(f"  [skip] moduleList.json missing in {year_dir}")
        return 0

    with open(list_path) as f:
        module_list = json.load(f)

    list_df = pd.DataFrame(module_list)

    detail_rows = []
    modules_dir = year_dir / "modules"
    # The module list provides the catalogue skeleton; per-module JSON files contain
    # the richer description / prerequisite fields used by the pipeline.
    if modules_dir.exists():
        for jf in sorted(modules_dir.glob("*.json")):
            try:
                with open(jf) as fh:
                    detail_rows.append(json.load(fh))
            except Exception as e:
                print(f"  [warn] {jf.name}: {e}")

    if detail_rows:
        detail_df = pd.DataFrame(detail_rows)
        df = list_df.merge(detail_df, on="moduleCode", how="left", suffixes=("", "_detail"))
        if "title_detail" in df.columns:
            df.drop(columns=["title_detail"], inplace=True)
    else:
        df = list_df

    _serialize_complex_values(df)
    df.to_sql("raw_modules", engine, if_exists="replace", index=False)
    _log_load("raw_modules", len(df), engine)
    print(f"  raw_modules: {len(df):,} rows")
    return len(df)


# ---------------------------------------------------------------------------
# Bulk pipeline outputs loader (schema-agnostic)
# ---------------------------------------------------------------------------

def load_pipeline_outputs(engine=None) -> int:
    """Auto-discover and load all parquet/csv files in outputs/ into the database."""
    import pandas as pd
    engine = engine or get_engine()

    outputs_dir = PROJECT_ROOT / "outputs"
    if not outputs_dir.exists():
        print(f"  [skip] outputs/ directory not found")
        return 0

    parquet_files = {f.stem: f for f in outputs_dir.glob("*.parquet")}
    csv_files = {f.stem: f for f in outputs_dir.glob("*.csv")}
    all_stems = set(parquet_files) | set(csv_files)

    if not all_stems:
        print("  [skip] No .parquet or .csv files found in outputs/")
        return 0

    total = 0
    for stem in sorted(all_stems):
        path = parquet_files.get(stem) or csv_files[stem]
        try:
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            _serialize_complex_values(df)
            df.to_sql(stem, engine, if_exists="replace", index=False)
            _log_load(stem, len(df), engine)
            print(f"  {stem}: {len(df):,} rows")
            total += len(df)
        except Exception as e:
            print(f"  [warn] {stem}: {e}")

    return total


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Load data into shared PostgreSQL database")
    parser.add_argument("--raw-only", action="store_true", help="Load only raw inputs")
    parser.add_argument("--outputs-only", action="store_true", help="Load only pipeline outputs")
    parser.add_argument("--year", help="NUSMods academic year to load (e.g. 2023-2024)")
    args = parser.parse_args()

    try:
        engine = get_engine()
    except EnvironmentError as e:
        sys.exit(f"ERROR: {e}")

    do_raw = not args.outputs_only
    do_outputs = not args.raw_only

    print("Connecting to database...")
    try:
        with engine.connect():
            pass
    except Exception as e:
        sys.exit(f"ERROR: Could not connect to database.\n{e}")

    print("Connected.\n")

    if do_raw:
        print("Loading raw inputs...")
        load_raw_jobs(engine)
        load_raw_modules(engine, year_override=args.year)
        print()

    if do_outputs:
        print("Loading pipeline outputs...")
        load_pipeline_outputs(engine)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
