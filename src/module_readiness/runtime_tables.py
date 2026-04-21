from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import pandas as pd


def candidate_runtime_dirs(*directories: Path) -> list[Path]:
    """Return ordered unique data directories for the deployed app."""
    resolved: list[Path] = []
    seen: set[Path] = set()
    for directory in directories:
        path = Path(directory).resolve()
        if path in seen:
            continue
        seen.add(path)
        resolved.append(path)
    return resolved


def read_runtime_table(table_name: str, data_dirs: Sequence[Path]) -> pd.DataFrame:
    """Read a runtime app table from app_data/ or outputs/."""
    stem = Path(table_name).stem
    for data_dir in data_dirs:
        pickle_path = data_dir / f"{stem}.pkl.gz"
        if pickle_path.exists():
            return _normalize_frame(pd.read_pickle(pickle_path, compression="gzip"))

        # Prefer CSV over parquet so the deployed app does not require pyarrow.
        csv_path = data_dir / f"{stem}.csv"
        if csv_path.exists():
            return _normalize_frame(_read_csv_with_fallback(csv_path))

        parquet_path = data_dir / f"{stem}.parquet"
        if parquet_path.exists():
            try:
                return _normalize_frame(pd.read_parquet(parquet_path))
            except ImportError:
                pass

    searched = ", ".join(str(Path(path).resolve()) for path in data_dirs)
    raise FileNotFoundError(f"Could not find runtime table '{stem}' in: {searched}")


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False)
    except pd.errors.ParserError:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh, skipinitialspace=True))
        if not rows:
            return pd.DataFrame()
        header = [str(value).strip() for value in rows[0]]
        body = [
            (([str(value).strip() for value in row]) + [""] * len(header))[: len(header)]
            for row in rows[1:]
        ]
        return pd.DataFrame(body, columns=header)


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(column).strip() for column in out.columns]
    for column in out.columns:
        if _should_strip_values(out[column]):
            out[column] = out[column].map(_strip_if_string)
    return out


def _should_strip_values(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _strip_if_string(value: object) -> object:
    if isinstance(value, str):
        return value.strip()
    return value
