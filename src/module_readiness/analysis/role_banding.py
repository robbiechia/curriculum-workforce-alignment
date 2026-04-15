from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODERATE_PERCENTILE = 0.60
STRONG_PERCENTILE = 0.85
GLOBAL_MODERATE_FLOOR = 0.20
GLOBAL_STRONG_FLOOR = 0.30
MIN_STRONG_GAP = 0.05
STRONG_SUPPORT_MIN = 5.0


def read_csv_loose(path: Path) -> pd.DataFrame:
    """Read a CSV defensively, tolerating inconsistent quoting/field counts."""
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except pd.errors.ParserError:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh, skipinitialspace=True))
        if not rows:
            return pd.DataFrame()
        header = [str(v).strip() for v in rows[0]]
        body = [
            (([str(v).strip() for v in row]) + [""] * len(header))[: len(header)]
            for row in rows[1:]
        ]
        df = pd.DataFrame(body, columns=header)
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(lambda v: str(v).strip())
    return df


def format_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "(no rows)"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: f"{v:.{digits}f}")
    return out.to_string(index=False)


@dataclass(frozen=True)
class RoleBand:
    role_family_name: str
    moderate_threshold: float
    strong_threshold: float
    strong_support_min: float = STRONG_SUPPORT_MIN


def load_module_role_scores() -> pd.DataFrame:
    df = read_csv_loose(OUTPUTS_DIR / "module_role_scores.csv").copy()
    for col in ["role_score", "evidence_job_count", "support_weight"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_role_band_thresholds(
    module_role_scores: pd.DataFrame,
    moderate_percentile: float = MODERATE_PERCENTILE,
    strong_percentile: float = STRONG_PERCENTILE,
    global_moderate_floor: float = GLOBAL_MODERATE_FLOOR,
    global_strong_floor: float = GLOBAL_STRONG_FLOOR,
    min_strong_gap: float = MIN_STRONG_GAP,
    strong_support_min: float = STRONG_SUPPORT_MIN,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    grouped = module_role_scores.dropna(subset=["role_family_name", "role_score"]).groupby("role_family_name")
    for role, grp in grouped:
        scores = grp["role_score"].dropna()
        moderate_threshold = max(float(scores.quantile(moderate_percentile)), global_moderate_floor)
        strong_threshold = max(float(scores.quantile(strong_percentile)), global_strong_floor)
        if strong_threshold <= moderate_threshold:
            strong_threshold = moderate_threshold + min_strong_gap
        rows.append(
            {
                "role_family_name": str(role),
                "n_rows": int(len(scores)),
                "moderate_threshold": moderate_threshold,
                "strong_threshold": strong_threshold,
                "strong_support_min": float(strong_support_min),
                "role_median": float(scores.quantile(0.50)),
                "role_p60": float(scores.quantile(0.60)),
                "role_p85": float(scores.quantile(0.85)),
                "role_p90": float(scores.quantile(0.90)),
            }
        )
    return pd.DataFrame(rows).sort_values("role_family_name").reset_index(drop=True)


def role_band_lookup(
    module_role_scores: pd.DataFrame,
    **kwargs,
) -> dict[str, RoleBand]:
    thresholds = compute_role_band_thresholds(module_role_scores, **kwargs)
    return {
        str(row["role_family_name"]): RoleBand(
            role_family_name=str(row["role_family_name"]),
            moderate_threshold=float(row["moderate_threshold"]),
            strong_threshold=float(row["strong_threshold"]),
            strong_support_min=float(row["strong_support_min"]),
        )
        for _, row in thresholds.iterrows()
    }


def classify_role_score(
    score: float,
    band: RoleBand,
    evidence_job_count: float | None = None,
) -> str:
    if pd.isna(score):
        return "Weak"
    if float(score) >= band.strong_threshold:
        if evidence_job_count is None or pd.isna(evidence_job_count) or float(evidence_job_count) >= band.strong_support_min:
            return "Strong"
    if float(score) >= band.moderate_threshold:
        return "Moderate"
    return "Weak"


def summarize_band_counts(module_role_scores: pd.DataFrame, lookup: dict[str, RoleBand]) -> pd.DataFrame:
    df = module_role_scores.copy()
    df["band"] = df.apply(
        lambda r: classify_role_score(
            float(r["role_score"]),
            lookup[str(r["role_family_name"])],
            r.get("evidence_job_count"),
        ),
        axis=1,
    )
    return (
        df.groupby("band", observed=False)
        .agg(
            n=("role_score", "size"),
            mean_score=("role_score", "mean"),
            median_score=("role_score", "median"),
            mean_support=("evidence_job_count", "mean"),
        )
        .reset_index()
    )
