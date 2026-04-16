from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config.settings import PipelineConfig
from ..retrieval import HybridRetrievalEngine

OTHER_NEAR_TIE_MARGIN = 0.02

# Number of top-ranked jobs used to compute the concentration ratio.
# A higher window would dilute the signal from the very best matches.
CONCENTRATION_WINDOW = 5

# A concentration ratio above this value suggests the module scores are clustered
# at the top, with a long tail of weak matches — a bimodal-like distribution.
BIMODALITY_CONCENTRATION_THRESHOLD = 1.5

# IQR must also exceed this floor for a distribution to be flagged as bimodal;
# prevents flagging when all scores are tightly bunched near zero.
BIMODALITY_IQR_THRESHOLD = 0.3


@dataclass
class ScoringResult:
    module_job_scores: pd.DataFrame
    module_ssoc5_scores: pd.DataFrame
    module_role_scores: pd.DataFrame
    matrices: Dict[str, np.ndarray]
    diagnostics: Dict[str, float | str]


def _safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def _support_weight(evidence_job_count: float, support_prior: float) -> float:
    evidence = max(0.0, float(evidence_job_count))
    prior = max(0.0, float(support_prior))
    if evidence <= 0:
        return 0.0
    if prior <= 0:
        return 1.0
    return evidence / (evidence + prior)


def _aggregate_group_scores(
    module_job_scores: pd.DataFrame,
    top_k: int,
    group_cols: List[str],
    support_prior: float,
) -> pd.DataFrame:
    """Aggregate per-job RRF scores into a per-module-group alignment score.

    For each (module, group) pair — where group is either a role family or an
    SSOC code — takes the top-K retrieved jobs by RRF score and averages them.
    The raw average is then dampened by a support weight that penalises groups
    backed by very few matching jobs (see _support_weight).

    Also computes distributional statistics (Q1/median/Q3/IQR, concentration
    ratio, bimodality flag) so downstream consumers can assess confidence.

    Args:
        module_job_scores: Long-format evidence table (one row per module-job pair).
        top_k: Number of top jobs to average per module-group pair.
        group_cols: Columns that define the grouping dimension (e.g. role_family).
        support_prior: Controls how aggressively low-evidence pairs are penalised.

    Returns:
        DataFrame with one row per (module, group) and aggregated score columns.
    """
    if module_job_scores.empty:
        return pd.DataFrame(
            columns=[
                "module_code",
                "module_title",
                *group_cols,
                "raw_role_score",
                "support_weight",
                "role_score",
                "q1",
                "median",
                "q3",
                "iqr",
                "concentration_ratio",
                "bimodality_flag",
                "evidence_job_count",
            ]
        )

    rows: List[Dict[str, float | str]] = []
    grouped = module_job_scores.groupby(["module_code", "module_title", *group_cols], as_index=False)

    for _, group in grouped:
        # Keep only the strongest evidence jobs for this module-group pairing before
        # computing its aggregate alignment score.
        g = group.sort_values("rrf_score", ascending=False).head(top_k)
        values = g["rrf_score"].to_numpy(dtype=float)

        evidence_job_count = float(len(g))
        raw_role_score = float(values.mean()) if values.size else 0.0
        support_weight = _support_weight(evidence_job_count, support_prior)
        role_score = raw_role_score * support_weight
        q1 = _safe_quantile(values, 0.25)
        median = _safe_quantile(values, 0.50)
        q3 = _safe_quantile(values, 0.75)
        iqr = q3 - q1

        top_window = values[:CONCENTRATION_WINDOW] if values.size >= CONCENTRATION_WINDOW else values
        topk_mean = float(values.mean()) if values.size else 0.0
        top_window_mean = float(top_window.mean()) if top_window.size else 0.0
        concentration_ratio = (top_window_mean / topk_mean) if topk_mean > 0 else 0.0
        bimodality_flag = int(
            (concentration_ratio > BIMODALITY_CONCENTRATION_THRESHOLD)
            and (iqr > BIMODALITY_IQR_THRESHOLD)
        )

        row: Dict[str, float | str] = {
            "module_code": str(g["module_code"].iloc[0]),
            "module_title": str(g["module_title"].iloc[0]),
            "raw_role_score": raw_role_score,
            "support_weight": support_weight,
            "role_score": role_score,
            "q1": q1,
            "median": median,
            "q3": q3,
            "iqr": iqr,
            "concentration_ratio": concentration_ratio,
            "bimodality_flag": bimodality_flag,
            "evidence_job_count": evidence_job_count,
        }
        for col in group_cols:
            row[col] = str(g[col].iloc[0])
        rows.append(row)

    return pd.DataFrame(rows)


def _apply_other_near_tie_preference(
    module_role_scores: pd.DataFrame,
    margin: float = OTHER_NEAR_TIE_MARGIN,
) -> pd.DataFrame:
    """Demote the "Other" role cluster when a named cluster is nearly as strong.

    Without this adjustment, a module with a weak but broad signal could be
    labelled "Other" in the summary even though a named cluster (e.g. "Data &
    Analytics") scores almost as high. The function adds a selection_score
    column that suppresses "Other" whenever the gap to the best named cluster
    is within `margin`, ensuring summaries prefer interpretable labels.

    Args:
        module_role_scores: Output of _aggregate_group_scores for role families.
        margin: Score gap within which "Other" is demoted. Defaults to
                OTHER_NEAR_TIE_MARGIN (0.02).

    Returns:
        DataFrame with selection_score and role_rank_within_module columns added.
    """
    if module_role_scores.empty:
        return module_role_scores.copy()

    out = module_role_scores.copy()
    out["selection_score"] = out["role_score"].astype(float)

    for module_code, group in out.groupby("module_code"):
        named = group[group["role_family"] != "Other"]
        if named.empty:
            continue

        best_named_score = float(named["role_score"].max())
        other_idx = group.index[group["role_family"] == "Other"]
        if len(other_idx) == 0:
            continue

        for idx in other_idx:
            # Prefer a named role cluster over "Other" whenever the two are effectively
            # tied, so module summaries remain interpretable.
            other_score = float(out.at[idx, "role_score"])
            if other_score <= best_named_score + float(margin):
                out.at[idx, "selection_score"] = best_named_score - 1e-9

    out["role_rank_within_module"] = out.groupby("module_code")["selection_score"].rank(
        ascending=False,
        method="dense",
    )
    return out.sort_values(
        ["module_code", "selection_score", "role_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def compute_scores(
    config: PipelineConfig,
    jobs: pd.DataFrame,
    modules: pd.DataFrame,
    retrieval: HybridRetrievalEngine,
) -> ScoringResult:
    """Run the full module-to-job scoring pipeline and return aggregated results.

    For every module, retrieves the top-N most relevant jobs using the hybrid
    engine, then aggregates the resulting RRF scores into two views:
      - module_ssoc5_scores: grouped by fine-grained SSOC occupation codes
      - module_role_scores: grouped by curated role family clusters

    The role score view additionally applies near-tie preference so that named
    clusters are preferred over "Other" in downstream summaries.

    Args:
        config: Pipeline configuration (top_k, retrieval_top_n, support_prior, etc.).
        jobs: Enriched jobs DataFrame with role family and SSOC assignments.
        modules: Consolidated modules DataFrame.
        retrieval: Pre-built HybridRetrievalEngine with BM25 and embedding indices.

    Returns:
        ScoringResult with per-job evidence, per-SSOC scores, per-role scores,
        the raw RRF matrix, and diagnostics.
    """
    top_n = max(int(config.top_k) * 4, int(config.retrieval_top_n))
    module_rows: List[Dict[str, float | str]] = []
    rrf_score_matrix = np.zeros((len(modules), len(jobs)), dtype=float)

    for i, module in modules.iterrows():
        # This evidence table is the bridge between retrieval and aggregation: every row
        # is one retrieved job supporting a given module.
        ranking = retrieval.rank_jobs_from_module(i, top_k=top_n)
        if ranking.empty:
            continue

        for _, match in ranking.iterrows():
            j = int(match["index"])
            job_row = jobs.iloc[j]

            fallback_family = str(job_row.get("role_family", "Other"))
            fallback_family_name = str(job_row.get("role_family_name", fallback_family)).strip()
            if not fallback_family_name:
                fallback_family_name = fallback_family
            broad_family = str(job_row.get("broad_family", "Other")).strip() or "Other"

            ssoc_5d = str(job_row.get("ssoc_5d", "")).strip()
            ssoc_4d = str(job_row.get("ssoc_4d", "")).strip()
            ssoc_5d_name = str(job_row.get("ssoc_5d_name", "")).strip()
            ssoc_4d_name = str(job_row.get("ssoc_4d_name", "")).strip()

            if not ssoc_5d:
                ssoc_5d = fallback_family
            if not ssoc_4d:
                ssoc_4d = ssoc_5d[:4] if ssoc_5d.isdigit() and len(ssoc_5d) >= 4 else fallback_family
            if not ssoc_5d_name:
                ssoc_5d_name = fallback_family_name
            if not ssoc_4d_name:
                ssoc_4d_name = fallback_family_name

            rrf_score = float(match["rrf_score"])
            rrf_score_matrix[i, j] = rrf_score

            module_rows.append(
                {
                    "module_code": module["module_code"],
                    "module_title": module["module_title"],
                    "job_id": job_row["job_id"],
                    "job_title": job_row["title"],
                    "company": job_row["company"],
                    "role_family": fallback_family,
                    "role_family_name": fallback_family_name,
                    "broad_family": broad_family,
                    "ssoc_4d": ssoc_4d,
                    "ssoc_4d_name": ssoc_4d_name,
                    "ssoc_5d": ssoc_5d,
                    "ssoc_5d_name": ssoc_5d_name,
                    "primary_category": job_row["primary_category"],
                    "bm25_score": float(match["bm25_score"]),
                    "embedding_score": float(match["embedding_score"]),
                    "bm25_rank": match["bm25_rank"],
                    "embedding_rank": match["embedding_rank"],
                    "rrf_score": rrf_score,
                    "evidence_terms": str(match.get("evidence_terms", "")),
                }
            )

    module_job_scores = pd.DataFrame(
        module_rows,
        columns=[
            "module_code",
            "module_title",
            "job_id",
            "job_title",
            "company",
            "role_family",
            "role_family_name",
            "broad_family",
            "ssoc_4d",
            "ssoc_4d_name",
            "ssoc_5d",
            "ssoc_5d_name",
            "primary_category",
            "bm25_score",
            "embedding_score",
            "bm25_rank",
            "embedding_rank",
            "rrf_score",
            "evidence_terms",
        ],
    )

    module_ssoc5_scores = _aggregate_group_scores(
        module_job_scores=module_job_scores,
        top_k=int(config.top_k),
        group_cols=["ssoc_4d", "ssoc_4d_name", "ssoc_5d", "ssoc_5d_name"],
        support_prior=float(config.role_support_prior),
    )
    module_role_scores = _aggregate_group_scores(
        module_job_scores=module_job_scores,
        top_k=int(config.top_k),
        group_cols=["role_family", "role_family_name", "broad_family"],
        support_prior=float(config.role_support_prior),
    )
    module_role_scores = _apply_other_near_tie_preference(module_role_scores)

    diagnostics: Dict[str, float | str] = {
        "scores_module_job_rows": float(len(module_job_scores)),
        "scores_module_ssoc5_rows": float(len(module_ssoc5_scores)),
        "scores_module_role_rows": float(len(module_role_scores)),
        "scores_ranker": "bm25+embeddings+rrf",
    }

    matrices = {
        "rrf_score": rrf_score_matrix,
    }

    return ScoringResult(
        module_job_scores=module_job_scores,
        module_ssoc5_scores=module_ssoc5_scores,
        module_role_scores=module_role_scores,
        matrices=matrices,
        diagnostics=diagnostics,
    )
