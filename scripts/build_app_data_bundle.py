from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "outputs"
DEFAULT_DEST_DIR = PROJECT_ROOT / "app_data"

TABLES_TO_COPY = (
    "degree_summary",
    "degree_skill_supply",
    "job_role_map",
    "jobs_clean",
    "module_preclusions",
    "module_role_scores",
    "module_summary",
    "modules_clean",
)

DEGREE_MODULE_MAP_COLUMNS = [
    "degree_id",
    "bucket_id",
    "faculty",
    "faculty_code",
    "degree",
    "major",
    "primary_major",
    "curriculum_type",
    "module_type",
    "module_credits",
    "is_unrestricted_elective",
    "module_code",
    "module_found",
    "module_title",
    "module_credit",
]

MODULE_JOB_EVIDENCE_COLUMNS = [
    "module_code",
    "job_id",
    "job_title",
    "company",
    "rrf_score",
    "evidence_terms",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a slim runtime app_data bundle from existing outputs/ artifacts."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--dest-dir", type=Path, default=DEFAULT_DEST_DIR)
    args = parser.parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    dest_dir = args.dest_dir.expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    for stem in TABLES_TO_COPY:
        df = _read_table(source_dir, stem)
        _write_table(df, dest_dir / f"{stem}.parquet")

    degree_module_map = _read_table(
        source_dir,
        "degree_module_map",
        usecols=DEGREE_MODULE_MAP_COLUMNS,
    )
    degree_module_map = _coerce_bool(
        degree_module_map,
        ["is_unrestricted_elective", "module_found"],
    )
    degree_module_map = degree_module_map.loc[
        ~degree_module_map["is_unrestricted_elective"]
    ].reset_index(drop=True)
    _write_table(degree_module_map, dest_dir / "degree_module_map.parquet")

    module_job_evidence = _read_table(
        source_dir,
        "module_job_evidence",
        usecols=MODULE_JOB_EVIDENCE_COLUMNS,
    )
    _write_table(module_job_evidence, dest_dir / "module_job_evidence.parquet")


def _read_table(source_dir: Path, stem: str, usecols: list[str] | None = None) -> pd.DataFrame:
    parquet_path = source_dir / f"{stem}.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        if usecols is not None:
            missing = [column for column in usecols if column not in df.columns]
            if missing:
                raise KeyError(f"{parquet_path.name} is missing columns: {missing}")
            df = df[usecols].copy()
        return df

    csv_path = source_dir / f"{stem}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, usecols=usecols, keep_default_na=False)

    raise FileNotFoundError(f"Could not find {stem}.csv or {stem}.parquet under {source_dir}")


def _coerce_bool(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    true_values = {"true", "1", "yes", "y"}
    for column in columns:
        if column in out.columns:
            out[column] = out[column].map(
                lambda value: str(value).strip().lower() in true_values
            )
    return out


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[write] {path.relative_to(PROJECT_ROOT)} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
