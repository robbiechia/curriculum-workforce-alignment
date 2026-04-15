from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..orchestration.pipeline import ModuleReadinessState


class ModuleReadinessQueryAPI:
    def __init__(self, state: "ModuleReadinessState"):
        self.state = state

    def search_jobs(self, natural_language_query: str, exp_max: int = 2, top_k: int = 10) -> pd.DataFrame:
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
        # This is a simple lookup over precomputed module-role scores.
        subset = self.state.module_role_scores[
            self.state.module_role_scores["module_code"].str.upper() == module_code.upper()
        ].copy()
        if subset.empty:
            return subset

        return subset.sort_values("role_score", ascending=False).head(top_families).reset_index(drop=True)
