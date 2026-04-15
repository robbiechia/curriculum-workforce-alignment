from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class AggregationResult:
    module_role_scores: pd.DataFrame
    module_summary: pd.DataFrame
    module_gap_summary: pd.DataFrame
    diagnostics: Dict[str, float | str]


def _normalize_series(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    v_min = float(values.min())
    v_max = float(values.max())
    if math.isclose(v_min, v_max):
        return pd.Series(np.full(len(values), 0.5), index=values.index)
    return (values - v_min) / (v_max - v_min)


def _degree_profile(degree: str, profile_keywords: Dict[str, List[str]]) -> str:
    text = (degree or "").lower()
    best = "general"
    best_hits = 0
    for profile, keywords in profile_keywords.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits > best_hits:
            best_hits = hits
            best = profile
    return best


def _build_ssoc4_fallback_labels(module_ssoc5_scores: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["module_code", "ssoc4_fallback_name"]
    if module_ssoc5_scores is None or module_ssoc5_scores.empty:
        return pd.DataFrame(columns=columns)

    required = {"module_code", "ssoc_4d", "ssoc_4d_name", "role_score"}
    if not required.issubset(module_ssoc5_scores.columns):
        return pd.DataFrame(columns=columns)

    fallback = module_ssoc5_scores.copy()
    fallback["module_code"] = fallback["module_code"].astype(str)
    fallback["ssoc_4d"] = fallback["ssoc_4d"].astype(str).str.strip()
    fallback["ssoc_4d_name"] = fallback["ssoc_4d_name"].astype(str).str.strip()
    fallback = fallback[
        (fallback["ssoc_4d"] != "")
        & (fallback["ssoc_4d_name"] != "")
        & (fallback["ssoc_4d_name"].str.lower() != "nan")
    ].copy()
    if fallback.empty:
        return pd.DataFrame(columns=columns)

    agg_spec: Dict[str, str] = {"role_score": "max"}
    if "evidence_job_count" in fallback.columns:
        agg_spec["evidence_job_count"] = "sum"

    fallback = fallback.groupby(
        ["module_code", "ssoc_4d", "ssoc_4d_name"],
        as_index=False,
    ).agg(agg_spec)

    sort_cols = ["module_code", "role_score", "ssoc_4d_name"]
    ascending = [True, False, True]
    if "evidence_job_count" in fallback.columns:
        sort_cols = ["module_code", "role_score", "evidence_job_count", "ssoc_4d_name"]
        ascending = [True, False, False, True]

    # This fallback is only for display naming in summaries; the underlying role-cluster
    # assignment remains unchanged.
    fallback = (
        fallback.sort_values(sort_cols, ascending=ascending)
        .groupby("module_code", as_index=False)
        .head(1)
        .rename(
            columns={
                "ssoc_4d_name": "ssoc4_fallback_name",
            }
        )
    )

    return fallback[columns].reset_index(drop=True)


def _build_gap_summary(
    jobs: pd.DataFrame,
    modules: pd.DataFrame,
    module_role_scores: pd.DataFrame,
    top_roles_per_module: int = 2,
) -> pd.DataFrame:
    if jobs.empty or modules.empty or module_role_scores.empty:
        return pd.DataFrame(
            columns=[
                "role_family",
                "role_family_name",
                "skill",
                "demand_score",
                "supply_score",
                "gap_score",
                "gap_type",
            ]
        )

    demand_by_role_skill: Dict[str, Counter] = defaultdict(Counter)
    role_name_map: Dict[str, str] = {}

    for _, row in jobs.iterrows():
        role = str(row.get("role_family", "Other"))
        role_name = str(row.get("role_family_name", role)).strip() or role
        role_name_map[role] = role_name
        for skill in row.get("technical_skills", []):
            demand_by_role_skill[role][str(skill)] += 1.0

    # Only let each module contribute to its strongest role labels when approximating
    # supply, otherwise every low-confidence role match would dilute the gap view.
    ranking_col = "selection_score" if "selection_score" in module_role_scores.columns else "role_score"
    top_assignments = (
        module_role_scores.sort_values(["module_code", ranking_col, "role_score"], ascending=[True, False, False])
        .groupby("module_code")
        .head(top_roles_per_module)
    )

    module_role_map: Dict[str, set[str]] = defaultdict(set)
    for _, row in top_assignments.iterrows():
        module_role_map[str(row["module_code"])].add(str(row["role_family"]))

    module_skill_map = {
        str(row["module_code"]): set(row.get("technical_skills", []))
        for _, row in modules.iterrows()
    }

    supply_by_role_skill: Dict[str, Counter] = defaultdict(Counter)
    for module_code, roles in module_role_map.items():
        skills = module_skill_map.get(module_code, set())
        for role in roles:
            for skill in skills:
                supply_by_role_skill[role][skill] += 1.0

    rows: List[Dict[str, float | str]] = []
    for role, demand_counter in demand_by_role_skill.items():
        if not demand_counter:
            continue
        supply_counter = supply_by_role_skill.get(role, Counter())

        demand_total = sum(demand_counter.values()) or 1.0
        supply_total = sum(supply_counter.values()) or 1.0

        for skill, demand_value in demand_counter.most_common(15):
            demand_score = float(demand_value / demand_total)
            supply_score = float(supply_counter.get(skill, 0.0) / supply_total)
            gap_score = demand_score - supply_score
            rows.append(
                {
                    "role_family": role,
                    "role_family_name": role_name_map.get(role, role),
                    "skill": skill,
                    "demand_score": demand_score,
                    "supply_score": supply_score,
                    "gap_score": gap_score,
                    "gap_type": "undersupply" if gap_score > 0 else "oversupply",
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["role_family", "gap_score"], ascending=[True, False]).reset_index(
        drop=True
    )


def build_indicators(
    jobs: pd.DataFrame,
    modules: pd.DataFrame,
    module_role_scores: pd.DataFrame,
    role_rules: Dict[str, object],
    module_ssoc5_scores: pd.DataFrame | None = None,
) -> AggregationResult:
    out = module_role_scores.copy()

    if out.empty:
        empty_summary = pd.DataFrame(
            columns=[
                "module_code",
                "module_title",
                "module_profile",
                "module_source",
                "top_broad_family",
                "top_role_family",
                "top_role_family_name",
                "top_role_score",
                "top_role_family_name_source",
            ]
        )
        return AggregationResult(
            module_role_scores=out,
            module_summary=empty_summary,
            module_gap_summary=pd.DataFrame(),
            diagnostics={"indicators_module_rows": 0.0},
        )

    modules_meta = modules[["module_code", "module_title", "module_profile", "module_source"]].copy()

    ranking_col = "selection_score" if "selection_score" in out.columns else "role_score"
    best_role = out.sort_values([ranking_col, "role_score"], ascending=[False, False]).groupby("module_code").head(1)
    role_cols = ["module_code", "role_family", "role_score"]
    if "broad_family" in best_role.columns:
        role_cols.append("broad_family")
    if "role_family_name" in best_role.columns:
        role_cols.append("role_family_name")

    module_summary = modules_meta.merge(
        best_role[role_cols],
        on="module_code",
        how="left",
    ).rename(
        columns={
            "broad_family": "top_broad_family",
            "role_family": "top_role_family",
            "role_family_name": "top_role_family_name",
            "role_score": "top_role_score",
        }
    )
    module_summary["top_role_family_name"] = (
        module_summary["top_role_family_name"].fillna(module_summary["top_role_family"])
    )
    module_summary["top_role_family_name_source"] = "role_family"

    ssoc4_fallbacks = _build_ssoc4_fallback_labels(module_ssoc5_scores)
    module_summary = module_summary.merge(ssoc4_fallbacks, on="module_code", how="left")

    # Keep the curated role cluster for analysis, but swap in a more informative SSOC-4
    # label for display when the cluster would otherwise just be "Other".
    fallback_mask = (
        module_summary["top_role_family"].astype(str).eq("Other")
        & module_summary["ssoc4_fallback_name"].fillna("").astype(str).str.strip().ne("")
        & module_summary["ssoc4_fallback_name"].fillna("").astype(str).ne("Other")
    )
    module_summary.loc[fallback_mask, "top_role_family_name"] = module_summary.loc[
        fallback_mask, "ssoc4_fallback_name"
    ]
    module_summary.loc[fallback_mask, "top_role_family_name_source"] = "ssoc4_fallback"
    module_summary = module_summary.drop(columns=["ssoc4_fallback_name"], errors="ignore")

    module_gap_summary = _build_gap_summary(jobs, modules, out)

    diagnostics: Dict[str, float | str] = {
        "indicators_module_rows": float(len(module_summary)),
        "indicators_role_rows": float(len(out)),
        "indicators_gap_rows": float(len(module_gap_summary)),
    }

    return AggregationResult(
        module_role_scores=out,
        module_summary=module_summary,
        module_gap_summary=module_gap_summary,
        diagnostics=diagnostics,
    )
