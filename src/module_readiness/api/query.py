from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..orchestration.pipeline import ModuleReadinessState


class ModuleReadinessQueryAPI:
    """Query interface over a completed pipeline run.

    All methods operate on the in-memory ``ModuleReadinessState`` produced by
    ``run_pipeline()`` — no database calls are made here.  The retrieval engine
    and all score tables are already built; this class is purely a read layer.
    """

    def __init__(self, state: "ModuleReadinessState"):
        self.state = state

    def search_jobs(self, natural_language_query: str, exp_max: int = 2, top_k: int = 10) -> pd.DataFrame:
        """Retrieve the most relevant jobs for a free-text query.

        Encodes the query and runs hybrid retrieval against the job corpus.
        Only jobs with ``experience_years <= exp_max`` are considered, which
        keeps results relevant to recent graduates.

        Args:
            natural_language_query: Plain English description, e.g. "data analyst with SQL".
            exp_max: Maximum years of experience to include (default 2).
            top_k: Number of jobs to return.

        Returns:
            DataFrame with one row per job, including RRF score, BM25/embedding
            component scores, role family, and evidence terms.
        """
        # Search over the already-scored / normalized job corpus rather than hitting the
        # raw DB tables again.
        allowed_indices = self.state.jobs[
            self.state.jobs["experience_years"].fillna(9999) <= int(exp_max)
        ].index.to_numpy(dtype=int)
        ranking = self.state.retrieval.rank_jobs_from_text(
            natural_language_query,
            top_k=top_k,
            allowed_indices=allowed_indices,
        )
        if ranking.empty:
            return pd.DataFrame()

        rows = []
        for _, match in ranking.iterrows():
            idx = int(match["index"])
            role_family = self.state.jobs.at[idx, "role_family"]
            role_family_name = (
                self.state.jobs.at[idx, "role_family_name"]
                if "role_family_name" in self.state.jobs.columns
                else role_family
            )
            rows.append(
                {
                    "job_id": self.state.jobs.at[idx, "job_id"],
                    "title": self.state.jobs.at[idx, "title"],
                    "company": self.state.jobs.at[idx, "company"],
                    "broad_family": self.state.jobs.at[idx, "broad_family"]
                    if "broad_family" in self.state.jobs.columns
                    else "Other",
                    "role_family": role_family,
                    "role_family_name": role_family_name,
                    "primary_category": self.state.jobs.at[idx, "primary_category"],
                    "score": float(match["rrf_score"]),
                    "bm25_score": float(match["bm25_score"]),
                    "embedding_score": float(match["embedding_score"]),
                    "bm25_rank": match["bm25_rank"],
                    "embedding_rank": match["embedding_rank"],
                    "evidence_terms": str(match.get("evidence_terms", "")),
                }
            )

        return pd.DataFrame(rows)

    def recommend_relevant_modules(
        self,
        query_or_job_ids: str | Sequence[str],
        top_k: int = 10,
        role_family: str | None = None,
        allowed_module_codes: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Recommend modules relevant to a query or a set of specific job IDs.

        Accepts two input forms:
        - A string — encoded and used directly as the retrieval query.
        - A list of job ID strings — the engine aggregates evidence across all
          supplied jobs before ranking modules, giving a richer signal than a
          single text query when you already have a target job set.

        ``role_family`` and ``allowed_module_codes`` both narrow the candidate
        pool before retrieval runs, not after — so ``top_k`` refers to the
        final output size, not a post-filter of a larger retrieval.

        Args:
            query_or_job_ids: Free-text query or a list of job ID strings.
            top_k: Number of modules to return.
            role_family: If set, only modules that scored against this role
                         family are considered.
            allowed_module_codes: Explicit allowlist of module codes, useful
                                  for restricting results to a degree's curriculum.

        Returns:
            DataFrame with one row per module, including similarity score, role
            family assignment, and evidence terms.
        """
        allowed_indices: np.ndarray | None = None
        allowed_codes: set[str] | None = None
        if allowed_module_codes:
            allowed_codes = {str(code).strip().upper() for code in allowed_module_codes if str(code).strip()}
            if not allowed_codes:
                return pd.DataFrame()

        if role_family:
            # Optional role-family filtering narrows the candidate module set before
            # the actual retrieval call.
            role_mask = self.state.module_role_scores["role_family"] == role_family
            if "role_family_name" in self.state.module_role_scores.columns:
                role_mask = role_mask | (self.state.module_role_scores["role_family_name"] == role_family)
            role_codes = {
                str(code).strip().upper()
                for code in self.state.module_role_scores[role_mask]["module_code"].tolist()
                if str(code).strip()
            }
            allowed_codes = role_codes if allowed_codes is None else (allowed_codes & role_codes)
            if allowed_codes is not None and not allowed_codes:
                return pd.DataFrame()

        if allowed_codes is not None:
            allowed_indices = self.state.modules[
                self.state.modules["module_code"].astype(str).str.upper().isin(allowed_codes)
            ].index.to_numpy(dtype=int)
            if allowed_indices.size == 0:
                return pd.DataFrame()

        retrieval_top_k = max(int(top_k), min(int(self.state.config.retrieval_top_n), int(top_k) * 5))
        if isinstance(query_or_job_ids, str):
            ranking = self.state.retrieval.rank_modules_from_text(
                query_or_job_ids,
                top_k=retrieval_top_k,
                allowed_indices=allowed_indices,
            )
        else:
            ids = set(query_or_job_ids)
            subset = self.state.jobs[self.state.jobs["job_id"].isin(ids)]
            indices = subset.index.tolist()
            if not indices:
                return pd.DataFrame()
            ranking = self.state.retrieval.rank_modules_from_job_indices(
                indices,
                top_k=retrieval_top_k,
                allowed_indices=allowed_indices,
            )

        if ranking.empty:
            return pd.DataFrame()

        rows = []
        for _, match in ranking.iterrows():
            idx = int(match["index"])
            module_code = self.state.modules.iloc[idx]["module_code"]

            summary_row = self.state.module_summary[
                self.state.module_summary["module_code"] == module_code
            ]
            top_role = (
                summary_row["top_role_family"].iloc[0]
                if (not summary_row.empty and "top_role_family" in summary_row.columns)
                else (
                    summary_row["top_role_cluster"].iloc[0]
                    if (not summary_row.empty and "top_role_cluster" in summary_row.columns)
                    else "NA"
                )
            )
            top_role_name = (
                summary_row["top_role_family_name"].iloc[0]
                if (not summary_row.empty and "top_role_family_name" in summary_row.columns)
                else top_role
            )
            top_broad_family = (
                summary_row["top_broad_family"].iloc[0]
                if (not summary_row.empty and "top_broad_family" in summary_row.columns)
                else "NA"
            )
            if top_broad_family == "NA":
                top_broad_family = "Other"
            top_role_score = (
                float(summary_row["top_role_score"].iloc[0]) if not summary_row.empty else np.nan
            )

            rows.append(
                {
                    "module_code": module_code,
                    "module_title": self.state.modules.iloc[idx]["module_title"],
                    "similarity_score": float(match["rrf_score"]),
                    "bm25_score": float(match["bm25_score"]),
                    "embedding_score": float(match["embedding_score"]),
                    "top_broad_family": top_broad_family,
                    "top_role_family": top_role,
                    "top_role_family_name": top_role_name,
                    "top_role_score": top_role_score,
                    "evidence_terms": str(match.get("evidence_terms", "")),
                }
            )
            if len(rows) >= top_k:
                break

        return pd.DataFrame(rows)

    def get_module_role_profile(self, module_code: str, top_families: int = 5) -> pd.DataFrame:
        """Return the top role families for a single module, sorted by score.

        Looks up precomputed scores from the pipeline — no retrieval is run.
        Useful for understanding a module's alignment profile across all roles,
        not just its single best match.
        """
        # This is a simple lookup over precomputed module-role scores.
        subset = self.state.module_role_scores[
            self.state.module_role_scores["module_code"].str.upper() == module_code.upper()
        ].copy()
        if subset.empty:
            return subset

        return subset.sort_values("role_score", ascending=False).head(top_families).reset_index(drop=True)
