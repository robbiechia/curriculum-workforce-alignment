"""
NUS Curriculum Readiness Dashboard — MOE Edition
=================================================
Three analytical tabs:

  1. Curriculum Analysis   — For a degree × role, score the Common Curriculum
                             and Primary Major separately. Show every module and
                             its alignment score. Recommend high-value modules
                             not currently in the curriculum.

  2. Skill Requirements    — For a role, show ALL skills demanded by job postings
                             (technical + soft), with a faculty-level coverage
                             heatmap showing where the system covers each skill.

  3. Skill Gaps            — For a degree × role, show every demanded skill and
                             whether the required curriculum covers it. For each
                             gap, recommend top modules that address it.

Primary metric: module role_score — the alignment of a module to a role family,
derived from hybrid BM25 + embedding retrieval against real job postings and
dampened by evidence count (support_weight). Validated against human labels
(0–3 scale; threshold between 1 and 2). Scores ≥ 0.55 correspond to ~62% of
labelled job pairs being rated genuinely relevant by annotators.
"""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.module_readiness.analysis.role_banding import (
    RoleBand,
    classify_role_score,
    role_band_lookup,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = APP_ROOT / "outputs"

ROLE_ORDER = [
    "Software Engineering", "Data Science / Analytics", "AI / ML",
    "Cloud / Infrastructure / DevOps", "Business Analysis / Product / Project",
    "Finance / Risk / Compliance", "Accounting / Audit / Tax",
    "Marketing / Communications", "Sales / Business Development", "HR / Talent",
    "Legal / IP", "Education / Training", "Engineering / Manufacturing",
    "Electronics / Embedded / Systems", "Social / Healthcare / Community",
    "Operations / Admin", "Cybersecurity",
]

FACULTY_FULL = {
    "BIZ": "Business", "CDE": "Design & Eng", "CDE/SOC": "CDE / Computing",
    "SOC": "Computing", "CHS": "Humanities & Sciences", "LAW": "Law",
    "MED": "Medicine", "DEN": "Dentistry", "YST": "Music (YST)",
}

# Curriculum type grouping
COMMON_TYPES = re.compile(
    r"Common Curriculum|Faculty Requirements|Healthcare Common|"
    r"Pre-Clinical Phase|Specialisation Requirements",
    re.IGNORECASE,
)
PRIMARY_TYPES = re.compile(r"Primary Major Requirements", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except pd.errors.ParserError:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh, skipinitialspace=True))
        if not rows:
            return pd.DataFrame()
        header = [str(v).strip() for v in rows[0]]
        body = [
            (([str(v).strip() for v in row]) + [""] * len(header))[:len(header)]
            for row in rows[1:]
        ]
        df = pd.DataFrame(body, columns=header)
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(lambda v: str(v).strip())
    return df


def _parse_list(s: object) -> list[str]:
    if not s or str(s).strip() in ("", "nan"):
        return []
    try:
        result = ast.literal_eval(str(s))
        if isinstance(result, list):
            return [str(x).strip().lower() for x in result if str(x).strip()]
    except Exception:
        pass
    return [x.strip().lower() for x in str(s).split(";") if x.strip()]


def _parse_codes(s: object) -> list[str]:
    """Parse semicolon-separated module codes."""
    if not s or str(s).strip() in ("", "nan"):
        return []
    return [c.strip() for c in str(s).split(";") if c.strip()]


def _coerce_bool(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    trues = {"true", "1", "yes", "y"}
    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(lambda v: str(v).strip().lower() in trues)
    return df


@st.cache_data(show_spinner=False)
def _load_all() -> dict:
    # Core outputs
    mm = _read_csv(OUTPUTS_DIR / "degree_module_map.csv")
    mm = _coerce_bool(mm, ["is_unrestricted_elective", "module_found"])
    mm["is_common"] = mm["curriculum_type"].str.contains(COMMON_TYPES, na=False)
    mm["is_primary"] = mm["curriculum_type"].str.contains(PRIMARY_TYPES, na=False)

    summary = _read_csv(OUTPUTS_DIR / "degree_summary.csv")
    mrs = _read_csv(OUTPUTS_DIR / "module_role_scores.csv")
    mrs["role_score"] = pd.to_numeric(mrs["role_score"], errors="coerce").fillna(0.0)
    mrs["evidence_job_count"] = pd.to_numeric(mrs["evidence_job_count"], errors="coerce").fillna(0.0)

    prec = _read_csv(OUTPUTS_DIR / "module_preclusions.csv")
    prec = _coerce_bool(prec, ["has_wildcard"])

    mods = _read_csv(OUTPUTS_DIR / "modules_clean.csv")
    mods["tech_skills"] = mods["technical_skills"].map(_parse_list)
    mods["soft_skills_parsed"] = mods["soft_skills"].map(_parse_list)

    jrm = _read_csv(OUTPUTS_DIR / "job_role_map.csv")
    jobs = _read_csv(OUTPUTS_DIR / "jobs_clean.csv")
    jobs["tech_list"] = jobs["technical_skills"].map(_parse_list)
    jobs["soft_list"] = jobs["soft_skills"].map(_parse_list)

    dss = _read_csv(OUTPUTS_DIR / "degree_skill_supply.csv")
    dss["supply_score"] = pd.to_numeric(dss["supply_score"], errors="coerce").fillna(0.0)

    return dict(mm=mm, summary=summary, mrs=mrs, prec=prec, mods=mods,
                jrm=jrm, jobs=jobs, dss=dss)


# ---------------------------------------------------------------------------
# Derived / cached helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _build_preclusion_sets(prec: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    """
    Return dict:
      module_code -> {"exact": set[str], "prefix": set[str]}

    Exact preclusions are symmetric module-to-module exclusions.
    Prefix preclusions represent wildcard families such as DAO1704%.
    """
    result: dict[str, dict[str, set[str]]] = {}

    def _ensure(code: str) -> dict[str, set[str]]:
        return result.setdefault(code, {"exact": set(), "prefix": set()})

    for _, row in prec.iterrows():
        src = str(row["module_code"]).strip()
        tgt = str(row["precluded_code"]).strip()
        wild = bool(row["has_wildcard"])
        if not src or not tgt:
            continue
        if wild:
            prefix = tgt.rstrip("%").strip()
            if prefix:
                _ensure(src)["prefix"].add(prefix)
        else:
            _ensure(src)["exact"].add(tgt)
            _ensure(tgt)["exact"].add(src)
    return result


def _precluded_by_curriculum(
    curriculum_codes: set[str],
    candidate_code: str,
    prec_sets: dict[str, dict[str, set[str]]],
) -> bool:
    """True if candidate is precluded by anything in the curriculum (or vice-versa)."""
    if candidate_code in curriculum_codes:
        return True
    candidate_info = prec_sets.get(candidate_code, {"exact": set(), "prefix": set()})
    candidate_exact = candidate_info["exact"]
    candidate_prefix = candidate_info["prefix"]
    for curr_code in curriculum_codes:
        if curr_code in candidate_exact:
            return True
        curr_info = prec_sets.get(curr_code, {"exact": set(), "prefix": set()})
        curr_exact = curr_info["exact"]
        curr_prefix = curr_info["prefix"]

        if candidate_code in curr_exact:
            return True
        if any(candidate_code.startswith(prefix) for prefix in curr_prefix):
            return True
        if any(curr_code.startswith(prefix) for prefix in candidate_prefix):
            return True
    return False


def _recommendation_threshold(
    primary_best: pd.DataFrame,
    role_scores: pd.DataFrame,
) -> tuple[float, str]:
    """
    Compute a role-specific recommendation threshold without a hardcoded floor.

    Priority:
      1. degree-specific primary-major cutoff
         - 10th-best positive primary-major slot pick, or
         - weakest positive primary-major slot pick if <10 exist
      2. role-wide 90th percentile floor
      4. SCORE_MODERATE fallback if no role-score evidence exists
    """
    primary_threshold: float | None = None
    primary_basis: str | None = None
    positive_primary = (
        pd.to_numeric(primary_best.get("role_score", pd.Series(dtype=float)), errors="coerce")
        .dropna()
    )
    positive_primary = positive_primary[positive_primary > 0].sort_values(ascending=False).reset_index(drop=True)
    if not positive_primary.empty:
        if len(positive_primary) >= 10:
            primary_threshold = float(positive_primary.iloc[9])
            primary_basis = "10th-best positive primary-major module"
        else:
            primary_threshold = float(positive_primary.iloc[-1])
            primary_basis = "weakest positive primary-major module"

    all_scores = pd.to_numeric(role_scores.get("role_score", pd.Series(dtype=float)), errors="coerce").dropna()
    all_scores = all_scores[all_scores > 0]
    if not all_scores.empty:
        role_floor = float(all_scores.quantile(0.90))
        if primary_threshold is None:
            return role_floor, "90th percentile of module scores for this role"
        return max(primary_threshold, role_floor), (
            f"value of {primary_basis}"
        )

    return 0.40, "moderate fallback (no role-score evidence)"


@st.cache_data(show_spinner=False)
def _build_role_band_lookup(mrs: pd.DataFrame) -> dict[str, RoleBand]:
    return role_band_lookup(mrs)


@st.cache_data(show_spinner=False)
def _build_module_skill_map(_mods: pd.DataFrame) -> pd.DataFrame:
    """Long table: (module_code, skill, skill_type)."""
    tech = _mods[["module_code", "tech_skills"]].explode("tech_skills").rename(
        columns={"tech_skills": "skill"}
    ).assign(skill_type="technical")
    soft = _mods[["module_code", "soft_skills_parsed"]].explode("soft_skills_parsed").rename(
        columns={"soft_skills_parsed": "skill"}
    ).assign(skill_type="soft")
    all_skills = pd.concat([tech, soft], ignore_index=True)
    all_skills = all_skills[all_skills["skill"].str.strip().str.len() > 0].drop_duplicates()
    return all_skills


@st.cache_data(show_spinner=False)
def _build_degree_skill_segment_coverage(_mm: pd.DataFrame, _mods: pd.DataFrame) -> pd.DataFrame:
    mod_skill_map = _build_module_skill_map(_mods)
    required = _mm[
        _mm["module_found"] &
        ~_mm["is_unrestricted_elective"]
    ][["degree_id", "faculty_code", "primary_major", "module_code", "is_common", "is_primary"]].copy()

    coverage = required.merge(mod_skill_map, on="module_code", how="inner")
    if coverage.empty:
        return pd.DataFrame(
            columns=[
                "degree_id", "faculty_code", "primary_major", "skill",
                "covered_all", "covered_common", "covered_primary",
            ]
        )

    grouped = (
        coverage.groupby(["degree_id", "faculty_code", "primary_major", "skill"])
        .agg(
            covered_common=("is_common", "max"),
            covered_primary=("is_primary", "max"),
        )
        .reset_index()
    )
    grouped["covered_common"] = grouped["covered_common"].astype(bool)
    grouped["covered_primary"] = grouped["covered_primary"].astype(bool)
    grouped["covered_all"] = grouped["covered_common"] | grouped["covered_primary"]
    return grouped


@st.cache_data(show_spinner=False)
def _build_degree_skill_segment_intensity(_mm: pd.DataFrame, _mods: pd.DataFrame) -> pd.DataFrame:
    mod_skill_map = _build_module_skill_map(_mods)
    required = _mm[
        _mm["module_found"] &
        ~_mm["is_unrestricted_elective"]
    ][["degree_id", "faculty_code", "primary_major", "module_code", "is_common", "is_primary"]].copy()

    if required.empty:
        return pd.DataFrame(
            columns=[
                "degree_id", "faculty_code", "primary_major", "skill",
                "covered_all", "covered_common", "covered_primary",
                "share_all", "share_common", "share_primary",
                "modules_all", "modules_common", "modules_primary",
                "total_modules_all", "total_modules_common", "total_modules_primary",
            ]
        )

    totals_all = (
        required.groupby(["degree_id", "faculty_code", "primary_major"])["module_code"]
        .nunique()
        .rename("total_modules_all")
        .reset_index()
    )
    totals_common = (
        required[required["is_common"]]
        .groupby(["degree_id", "faculty_code", "primary_major"])["module_code"]
        .nunique()
        .rename("total_modules_common")
        .reset_index()
    )
    totals_primary = (
        required[required["is_primary"]]
        .groupby(["degree_id", "faculty_code", "primary_major"])["module_code"]
        .nunique()
        .rename("total_modules_primary")
        .reset_index()
    )

    coverage = required.merge(mod_skill_map, on="module_code", how="inner").drop_duplicates(
        ["degree_id", "faculty_code", "primary_major", "module_code", "skill"]
    )
    if coverage.empty:
        return pd.DataFrame(
            columns=[
                "degree_id", "faculty_code", "primary_major", "skill",
                "covered_all", "covered_common", "covered_primary",
                "share_all", "share_common", "share_primary",
                "modules_all", "modules_common", "modules_primary",
                "total_modules_all", "total_modules_common", "total_modules_primary",
            ]
        )

    grouped = (
        coverage.groupby(["degree_id", "faculty_code", "primary_major", "skill"])
        .agg(
            modules_all=("module_code", "nunique"),
            modules_common=("is_common", "sum"),
            modules_primary=("is_primary", "sum"),
        )
        .reset_index()
    )
    grouped["covered_all"] = grouped["modules_all"] > 0
    grouped["covered_common"] = grouped["modules_common"] > 0
    grouped["covered_primary"] = grouped["modules_primary"] > 0

    grouped = grouped.merge(totals_all, on=["degree_id", "faculty_code", "primary_major"], how="left")
    grouped = grouped.merge(totals_common, on=["degree_id", "faculty_code", "primary_major"], how="left")
    grouped = grouped.merge(totals_primary, on=["degree_id", "faculty_code", "primary_major"], how="left")
    for col in ["total_modules_all", "total_modules_common", "total_modules_primary"]:
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce").fillna(0)

    grouped["share_all"] = grouped["modules_all"] / grouped["total_modules_all"].where(grouped["total_modules_all"] > 0, 1)
    grouped["share_common"] = grouped["modules_common"] / grouped["total_modules_common"].where(grouped["total_modules_common"] > 0, 1)
    grouped["share_primary"] = grouped["modules_primary"] / grouped["total_modules_primary"].where(grouped["total_modules_primary"] > 0, 1)
    return grouped


@st.cache_data(show_spinner=False)
def _build_role_skills(_jobs: pd.DataFrame, _jrm: pd.DataFrame) -> pd.DataFrame:
    """Long table: (role_family_name, skill, skill_type, job_count, total_jobs, demand_pct)."""
    role_col = "role_family" if "role_family" in _jrm.columns else "role_cluster"
    j = _jobs[["job_id", "tech_list", "soft_list"]].merge(
        _jrm[["job_id", role_col]], on="job_id", how="inner"
    )
    rows = []
    for role, grp in j.groupby(role_col):
        total = grp["job_id"].nunique()
        tech_exp = grp.explode("tech_list").rename(columns={"tech_list": "skill"})
        soft_exp = grp.explode("soft_list").rename(columns={"soft_list": "skill"})
        for df, stype in [(tech_exp, "technical"), (soft_exp, "soft")]:
            df = df[df["skill"].str.strip().str.len() > 0]
            freq = df.groupby("skill")["job_id"].nunique().reset_index()
            freq.columns = ["skill", "job_count"]
            freq["skill_type"] = stype
            freq["role_family_name"] = role
            freq["total_jobs"] = total
            freq["demand_pct"] = freq["job_count"] / total
            rows.append(freq)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _build_job_counts(jrm: pd.DataFrame) -> dict[str, int]:
    if jrm.empty:
        return {}
    role_col = "role_family" if "role_family" in jrm.columns else "role_cluster"
    return jrm[role_col].value_counts().to_dict()


@st.cache_data(show_spinner=False)
def _build_top_roles_per_module(
    mrs: pd.DataFrame,
    band_lookup: dict[str, RoleBand],
    top_n: int = 3,
) -> pd.DataFrame:
    cols = ["module_code", "role_family_name", "role_score", "evidence_job_count"]
    keep = [c for c in cols if c in mrs.columns]
    if not keep:
        return pd.DataFrame(columns=["module_code", "top_role_families"])
    base = mrs[keep].copy()
    base["role_score"] = pd.to_numeric(base["role_score"], errors="coerce").fillna(0.0)
    if "evidence_job_count" in base.columns:
        base["evidence_job_count"] = pd.to_numeric(base["evidence_job_count"], errors="coerce").fillna(0.0)
    else:
        base["evidence_job_count"] = 0.0

    def _band_label(row: pd.Series) -> str:
        role = str(row["role_family_name"])
        band = band_lookup.get(role)
        return classify_role_score(
            float(row["role_score"]),
            band,
            float(row["evidence_job_count"]),
        ) if band else "Weak"

    band_rank = {"Strong": 0, "Moderate": 1, "Weak": 2}
    base["alignment_band"] = base.apply(_band_label, axis=1)
    base["band_rank"] = base["alignment_band"].map(lambda v: band_rank.get(str(v), 3))
    base = base.sort_values(
        ["module_code", "band_rank", "role_score", "evidence_job_count", "role_family_name"],
        ascending=[True, True, False, False, True],
    )

    def _join_roles(grp: pd.DataFrame) -> list[tuple[str, float, float, str]]:
        top = grp.head(top_n)
        return [
            (
                str(r["role_family_name"]),
                float(r["role_score"]),
                float(r.get("evidence_job_count", 0.0)),
                str(r.get("alignment_band", "Weak")),
            )
            for _, r in top.iterrows()
        ]

    out = (
        base.groupby("module_code")
        .apply(_join_roles)
        .reset_index(name="top_role_families")
    )
    return out


def _ordered_roles(available: set[str]) -> list[str]:
    return [r for r in ROLE_ORDER if r in available] + sorted(
        r for r in available if r not in ROLE_ORDER and r != "Other"
    )


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------

def _score_color(score: float, band: RoleBand | None = None, evidence_job_count: float | None = None) -> str:
    if band is None:
        if score >= 0.55:
            return "#2E7D32"
        if score >= 0.40:
            return "#F57F17"
        return "#C62828"
    label = classify_role_score(score, band, evidence_job_count)
    if label == "Strong":
        return "#2E7D32"
    if label == "Moderate":
        return "#F57F17"
    return "#C62828"


def _score_badge(score: float, band: RoleBand | None = None, evidence_job_count: float | None = None) -> str:
    c = _score_color(score, band, evidence_job_count)
    if band is None:
        label = "Strong" if score >= 0.55 else "Moderate" if score >= 0.40 else "Weak"
    else:
        label = classify_role_score(score, band, evidence_job_count)
    return (
        f'<span style="background:{c};color:#fff;padding:2px 9px;'
        f'border-radius:10px;font-size:0.74em;font-weight:700">{label} {score:.3f}</span>'
    )


def _score_chip(score: float, band: RoleBand | None = None, evidence_job_count: float | None = None) -> str:
    c = _score_color(score, band, evidence_job_count)
    return (
        f'<span style="background:{c}22;color:{c};border:1px solid {c}55;'
        f'padding:1px 7px;border-radius:6px;font-size:0.76em;font-weight:700">'
        f'{score:.3f}</span>'
    )


def _module_card(
    code: str,
    title: str,
    score: float,
    band: RoleBand | None = None,
    evidence_job_count: float | None = None,
    curriculum_type: str = "",
) -> str:
    c = _score_color(score, band, evidence_job_count)
    pct = int(min(score / 0.70, 1.0) * 100)  # 0.70 = rough max, fills bar
    subtitle = f'<div style="font-size:0.70em;color:#999;margin-top:1px">{curriculum_type}</div>' if curriculum_type else ""
    return (
        f'<div style="display:flex;align-items:center;gap:10px;padding:7px 12px;'
        f'margin:3px 0;background:#fafafa;border-radius:8px;border-left:3px solid {c}">'
        f'<div style="flex:1;min-width:0">'
        f'<span style="font-size:0.82em;font-weight:700;color:#111">{code}</span>'
        f'<span style="font-size:0.78em;color:#555;margin-left:6px">{title[:55]}</span>'
        f'{subtitle}'
        f'</div>'
        f'<div style="flex:0 0 90px;background:#eee;border-radius:4px;height:5px">'
        f'<div style="background:{c};border-radius:4px;height:5px;width:{pct}%"></div>'
        f'</div>'
        f'<span style="font-size:0.82em;font-weight:700;color:{c};min-width:38px;text-align:right">'
        f'{score:.3f}</span>'
        f'</div>'
    )


def _skill_row(skill: str, job_count: int, total_jobs: int, covered: bool,
               is_tech: bool, modules: list[str] | None = None, modules_html: str = "") -> str:
    demand_pct = job_count / max(total_jobs, 1)
    bar_pct = int(min(demand_pct / 0.5, 1.0) * 100)  # 50% demand fills bar
    type_color = "#1565C0" if is_tech else "#6A1B9A"
    type_label = "tech" if is_tech else "soft"
    if covered:
        bg, border, icon = "#f1f9f1", "#43A047", "✓"
    else:
        bg, border, icon = "#fff8f8", "#E53935", "✗"
    bar_color = "#43A047" if covered else "#E53935"

    resolved_modules_html = modules_html
    if covered and not resolved_modules_html and modules:
        badges = "".join(
            f'<span style="background:#e8f5e9;color:#1B5E20;border:1px solid #a5d6a7;'
            f'border-radius:4px;padding:1px 5px;font-size:0.66em;font-weight:600;'
            f'margin-right:3px">{m}</span>'
            for m in modules[:4]
        )
        resolved_modules_html = f'<div style="margin-top:2px">{badges}</div>'

    return (
        f'<div style="padding:6px 10px;margin:2px 0;background:{bg};'
        f'border-radius:7px;border-left:3px solid {border}">'
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<span style="font-size:0.75em;font-weight:700;color:{type_color};'
        f'background:{type_color}18;border-radius:4px;padding:0px 5px">{type_label}</span>'
        f'<span style="flex:1;font-size:0.83em;font-weight:600;color:#111">{icon} {skill}</span>'
        f'<div style="flex:0 0 70px;background:#ddd;border-radius:3px;height:4px">'
        f'<div style="background:{bar_color};border-radius:3px;height:4px;width:{bar_pct}%"></div>'
        f'</div>'
        f'<span style="font-size:0.74em;color:#666;white-space:nowrap">'
        f'{job_count} / {total_jobs} jobs ({demand_pct:.0%})</span>'
        f'</div>'
        f'{resolved_modules_html}'
        f'</div>'
    )


def _recommendation_card(
    code: str,
    title: str,
    score: float,
    band: RoleBand | None = None,
    evidence_job_count: float | None = None,
    skills_covered: list[str] | None = None,
) -> str:
    c = _score_color(score, band, evidence_job_count)
    badges = ""
    if skills_covered:
        badges = "".join(
            f'<span style="background:#E3F2FD;color:#0D47A1;border:1px solid #90CAF9;'
            f'border-radius:4px;padding:1px 5px;font-size:0.67em;font-weight:600;'
            f'margin-right:3px;margin-top:2px;display:inline-block">{s}</span>'
            for s in skills_covered[:6]
        )
    return (
        f'<div style="height:100%;box-sizing:border-box;padding:8px 12px;margin:0;'
        f'background:#F8F9FA;border-radius:8px;border-left:3px solid {c};'
        f'display:flex;flex-direction:column;justify-content:center">'
        f'<div style="display:flex;align-items:center;gap:10px">'
        f'<span style="font-size:0.82em;font-weight:700;color:#111">{code}</span>'
        f'<span style="flex:1;font-size:0.78em;color:#444">{title[:60]}</span>'
        f'{_score_chip(score, band, evidence_job_count)}'
        f'</div>'
        f'<div style="flex-wrap:wrap">{badges}</div>'
        f'</div>'
    )


def _legend_scores(band: RoleBand, role: str) -> str:
    items = [
        (_score_color(band.strong_threshold, band, band.strong_support_min),
         f"Strong  ≥ {band.strong_threshold:.2f}",
         f"Strongly aligned for {role} roles; identified in ≥ {band.strong_support_min:.0f} jobs"),
        (_score_color(band.moderate_threshold, band),
         f"Moderate  {band.moderate_threshold:.2f} – {band.strong_threshold - 0.001:.2f}",
         f"Moderate relative alignment for {role}"),
        (_score_color(max(band.moderate_threshold - 0.05, 0.0), band),
         f"Weak  < {band.moderate_threshold:.2f}",
         f"Not particularly relevant relative to other modules for {role}"),
    ]
    parts = "".join(
        f'<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:8px">'
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        f'background:{c};margin-top:4px;flex-shrink:0"></span>'
        f'<span style="font-size:0.78em;color:#555">'
        f'<strong>{label}</strong><br>'
        f'<span style="color:#999">{note}</span></span></div>'
        for c, label, note in items
    )
    explanation = (
        f'<div style="font-size:0.78em;color:#555;line-height:1.5;text-align:right">'
        f'<strong>How bands are calculated</strong><br>'
        f'Strong: ≥ 85% of role family scores<br>'
        f'Moderate: between 60% & 85% of role family scores<br>'
        f'Weak: &lt; 60% of role family scores'
        f'</div>'
    )
    return (
        f'<div style="background:#fafafa;border:1px solid #eee;border-radius:8px;'
        f'padding:10px 14px;margin-bottom:14px">'
        f'<div style="font-size:0.74em;font-weight:700;color:#888;text-transform:uppercase;'
        f'letter-spacing:0.05em;margin-bottom:6px">Alignment Score Guide</div>'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:24px;flex-wrap:nowrap">'
        f'<div style="display:flex;flex-direction:column;flex:0 0 52%;min-width:0">{parts}</div>'
        f'<div style="display:flex;justify-content:flex-end;flex:0 0 44%;min-width:0">{explanation}</div>'
        f'</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Sidebar: all filters live here, called once from main()
# ---------------------------------------------------------------------------

def _render_sidebar(
    summary: pd.DataFrame,
    available_roles: set[str],
) -> tuple[str, str, str, str]:
    """Render sidebar filters once. Returns (degree_id, degree_label, faculty_code, role)."""
    st.sidebar.markdown("## Filters")

    st.sidebar.markdown("### Role Family")
    roles = _ordered_roles(available_roles)
    sel_role = st.sidebar.selectbox("Role", roles, key="role")

    st.sidebar.markdown("### Degree")
    fac_opts = (
        summary[["faculty", "faculty_code"]].drop_duplicates()
        .assign(label=lambda d: d.apply(
            lambda r: f"{r['faculty_code']} · {FACULTY_FULL.get(str(r['faculty_code']), r['faculty'])}",
            axis=1,
        ))
        .sort_values("faculty_code")
    )
    sel_fac_label = st.sidebar.selectbox("Faculty", fac_opts["label"].tolist(), key="fac")
    sel_fac = str(fac_opts.loc[fac_opts["label"] == sel_fac_label, "faculty"].iloc[0])
    filtered = summary[summary["faculty"].astype(str) == sel_fac].sort_values("primary_major")
    majors = filtered["primary_major"].astype(str).drop_duplicates().tolist()
    sel_major = st.sidebar.selectbox("Programme", majors, key="major")
    row = filtered[filtered["primary_major"].astype(str) == sel_major].iloc[0]
    degree_id = str(row["degree_id"])
    degree_label = str(row["primary_major"])
    faculty_code = str(row.get("faculty_code", ""))

    st.sidebar.caption(
        "Degree and role selections apply to all Curriculum Analysis and Skill Gaps tabs. "
    )

    return degree_id, degree_label, faculty_code, sel_role


# ---------------------------------------------------------------------------
# Tab 1 — Curriculum Analysis
# ---------------------------------------------------------------------------

def _compute_n_slots(bucket_df: pd.DataFrame) -> int:
    """Return number of modules a student must pick from this bucket (credit-based)."""
    total = pd.to_numeric(bucket_df["module_credits"].iloc[0], errors="coerce")
    per_mod = pd.to_numeric(bucket_df["module_credit"], errors="coerce").dropna()
    if per_mod.empty or pd.isna(total) or float(total) <= 0:
        return 1
    mode_val = float(per_mod.mode().iloc[0])
    return max(1, round(float(total) / mode_val)) if mode_val > 0 else 1


def _curriculum_segment_chart(
    seg_label: str,
    seg_df: pd.DataFrame,
    role: str,
    color: str,
    band: RoleBand,
    primary_picked: "pd.DataFrame | None" = None,
    prec_sets: "dict[str, dict[str, set[str]]] | None" = None,
) -> "tuple[float, pd.DataFrame]":
    """
    Pick the best n_slots modules per requirement bucket (credit-based slot count).

    Primary mode (primary_picked=None):
      - For each bucket, pick top n_slots found modules by role_score.

    Common mode (primary_picked provided):
      - For each bucket, check if any primary pick (or precluded equivalent) appears
        in the bucket's option list. Those slots are treated as satisfied by the
        primary pick. Remaining slots get the best-scoring found module.

    Returns (segment_score, picks_df).
    """
    if seg_df.empty:
        st.info(f"No modules found in {seg_label} for this degree.")
        return 0.0, pd.DataFrame()

    found = seg_df[seg_df["module_found"]].copy()
    n_choosable = found["module_code"].nunique()

    # Total credit-based required slots across all buckets
    total_slots = 0
    for _, bgrp in seg_df.groupby("bucket_id"):
        total_slots += _compute_n_slots(bgrp)

    if found.empty:
        st.warning(f"No matched modules in {seg_label}.")
        return 0.0, pd.DataFrame()

    primary_codes: set[str] = (
        set(primary_picked["module_code"].dropna()) if primary_picked is not None else set()
    )

    all_picks: list[pd.DataFrame] = []
    # Tracks modules chosen in earlier buckets (primary mode only) so we never
    # pick the same module twice across different requirement groups.
    picked_globally: set[str] = set()

    for bid, bgrp in found.sort_values("bucket_id").groupby("bucket_id"):
        n_slots = _compute_n_slots(bgrp)
        n_found = len(bgrp)
        bucket_codes = set(bgrp["module_code"])
        picks_for_bucket: list[pd.Series] = []

        if primary_codes:
            # ── Common mode: check primary overlap first ──────────────────
            # Direct: primary pick code appears in this bucket
            direct_overlap = bucket_codes & primary_codes

            # Preclusion: a bucket module is precluded by a primary pick
            prec_overlap: set[str] = set()
            if prec_sets:
                for p_code in primary_codes:
                    p_info = prec_sets.get(p_code, {"exact": set(), "prefix": set()})
                    p_exact = p_info["exact"]
                    p_prefix = p_info["prefix"]
                    for b_code in bucket_codes:
                        if b_code in p_exact or any(b_code.startswith(pfx) for pfx in p_prefix):
                            prec_overlap.add(p_code)

            # Slots satisfied by primary picks (direct overlap)
            overlap_codes_used: set[str] = set()
            for code in list(direct_overlap)[:n_slots]:
                row = bgrp[bgrp["module_code"] == code].iloc[0].copy()
                row["from_primary"] = True
                picks_for_bucket.append(row)
                overlap_codes_used.add(code)

            # Slots satisfied via preclusion (primary pick precludes a bucket option)
            remaining_via_prec = n_slots - len(picks_for_bucket)
            for p_code in list(prec_overlap)[:remaining_via_prec]:
                if p_code in overlap_codes_used:
                    continue
                p_rows = primary_picked[primary_picked["module_code"] == p_code]
                if p_rows.empty:
                    continue
                row = p_rows.iloc[0].copy()
                row["from_primary"] = True
                row["bucket_id"] = bid
                picks_for_bucket.append(row)
                overlap_codes_used.add(p_code)

            # Fill remaining slots with best-scored found modules not already picked
            remaining = n_slots - len(picks_for_bucket)
            if remaining > 0:
                rest = (
                    bgrp[~bgrp["module_code"].isin(overlap_codes_used)]
                    .sort_values("role_score", ascending=False)
                    .head(remaining)
                )
                for _, r in rest.iterrows():
                    r = r.copy()
                    r["from_primary"] = False
                    picks_for_bucket.append(r)
        else:
            # ── Primary mode ──────────────────────────────────────────────
            if n_slots >= n_found:
                # Forced bucket: all modules must be taken — skip preclusion.
                # (Curriculum designers intentionally require every option even if
                #  general preclusion rules exist between some of them.)
                for _, r in bgrp.sort_values("role_score", ascending=False).iterrows():
                    r = r.copy()
                    r["from_primary"] = False
                    picks_for_bucket.append(r)
            else:
                # Choice bucket: greedy best-first, honouring preclusions both
                # within this bucket and against modules already chosen globally.
                for _, r in bgrp.sort_values("role_score", ascending=False).iterrows():
                    code = str(r["module_code"])
                    in_bucket_picked = {str(p["module_code"]) for p in picks_for_bucket}
                    if prec_sets and _precluded_by_curriculum(
                        picked_globally | in_bucket_picked, code, prec_sets
                    ):
                        continue
                    r = r.copy()
                    r["from_primary"] = False
                    picks_for_bucket.append(r)
                    if len(picks_for_bucket) >= n_slots:
                        break

            # Track chosen codes globally so later choice buckets avoid them
            for p in picks_for_bucket:
                picked_globally.add(str(p["module_code"]))

        if picks_for_bucket:
            all_picks.append(pd.DataFrame(picks_for_bucket))

    if not all_picks:
        st.warning(f"No picks could be assembled for {seg_label}.")
        return 0.0, pd.DataFrame()

    picks_df = pd.concat(all_picks, ignore_index=True)
    picks_df["role_score"] = pd.to_numeric(picks_df["role_score"], errors="coerce").fillna(0.0)

    n_from_primary = int(picks_df.get("from_primary", pd.Series([False] * len(picks_df))).sum())
    n_with_data = int((picks_df["role_score"] > 0).sum())
    seg_score = float(picks_df["role_score"].mean())

    # ── Stats row ──────────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns([1.2, 1.4, 1.4, 1.4])
    s1.markdown(
        f'<div style="text-align:center;padding:10px 6px;background:{color}10;'
        f'border-radius:10px;border:1px solid {color}33">'
        f'<div style="font-size:0.68em;color:#888;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.04em">Avg Score</div>'
        f'<div style="font-size:1.7em;font-weight:900;color:{color};line-height:1.1">'
        f'{seg_score:.3f}</div>'
        f'<div style="margin-top:2px">{_score_badge(seg_score, band)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    stat_rows = [
        (s2, n_choosable,  "total choosable modules", seg_label),
        (s3, n_with_data,  "with alignment data",     f"for {role}"),
        (s4, total_slots,  "required slots",           f"in {seg_label}"),
    ]
    for col, val, label, sub in stat_rows:
        col.markdown(
            f'<div style="text-align:center;padding:10px 6px;background:#fafafa;'
            f'border-radius:10px;border:1px solid #e8e8e8">'
            f'<div style="font-size:1.6em;font-weight:800;color:#333;line-height:1.1">{val}</div>'
            f'<div style="font-size:0.72em;color:#555;margin-top:3px">{label}</div>'
            f'<div style="font-size:0.67em;color:#aaa">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if primary_codes and n_from_primary > 0:
        st.caption(
            f"**{n_from_primary}** of {total_slots} slots satisfied by Primary Major picks "
            f"(shown with ↑ prefix). Remaining slots use the best-scoring option available."
        )

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

    # ── Bar chart: one row per credit slot ────────────────────────────────
    evidence_series = pd.to_numeric(
        picks_df.get("evidence_job_count", pd.Series([float("nan")] * len(picks_df))),
        errors="coerce",
    )
    picks_df["bar_color"] = [
        _score_color(score, band, evidence)
        for score, evidence in zip(picks_df["role_score"], evidence_series)
    ]
    picks_df["module_title_disp"] = picks_df.get(
        "module_title", pd.Series([""] * len(picks_df))
    ).fillna("").astype(str)
    from_primary_col = picks_df.get("from_primary", pd.Series([False] * len(picks_df)))
    picks_df["label"] = picks_df.apply(
        lambda r: (
            f"↑ {r['module_code']}  {r['module_title_disp'][:40]}"
            if r.get("from_primary", False)
            else f"{r['module_code']}  {r['module_title_disp'][:42]}"
        ),
        axis=1,
    )
    picks_df["module_type_label"] = picks_df.get(
        "module_type", pd.Series([""] * len(picks_df))
    ).fillna("").astype(str)

    chart_df = picks_df.sort_values("role_score", ascending=False).reset_index(drop=True)

    fig = go.Figure(
        go.Bar(
            x=chart_df["role_score"],
            y=chart_df["label"],
            orientation="h",
            marker_color=chart_df["bar_color"].tolist(),
            text=chart_df["role_score"].map(lambda s: f"{s:.3f}"),
            textposition="outside",
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                "Requirement: %{customdata[2]}<br>"
                "Score: %{x:.3f}<extra></extra>"
            ),
            customdata=chart_df[["module_code", "module_title_disp", "module_type_label"]].fillna("").values,
        )
    )
    fig.add_vline(
        x=band.strong_threshold,
        line_dash="dot",
        line_color="#000000",
        annotation_text=f"Strong ≥{band.strong_threshold:.2f}",
        annotation_font_size=10,
        annotation_font_color="#000000",
        annotation_bgcolor="rgba(255,255,255,0.92)",
        annotation_bordercolor="#000000",
        annotation_borderwidth=1,
        annotation_y=1.08,
        annotation_position="top right",
    )
    fig.add_vline(
        x=band.moderate_threshold,
        line_dash="dot",
        line_color="#000000",
        annotation_text=f"Moderate ≥{band.moderate_threshold:.2f}",
        annotation_font_size=10,
        annotation_font_color="#000000",
        annotation_bgcolor="rgba(255,255,255,0.92)",
        annotation_bordercolor="#000000",
        annotation_borderwidth=1,
        annotation_y=1.08,
        annotation_position="top left",
    )
    n_bars = len(chart_df)
    fig.update_layout(
        height=max(280, n_bars * 36),
        margin={"t": 68, "b": 10, "l": 0, "r": 70},
        xaxis={"title": "Alignment Score", "range": [0, 0.85]},
        yaxis={"title": "", "tickfont": {"size": 10.5}, "autorange": "reversed"},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, width="stretch")

    primary_note = " (↑ = satisfied by Primary Major pick)" if primary_codes else ""
    st.caption(        
        f"Best-scoring available module shown per requirement.{primary_note} "
        f"Hover to see requirement type."
    )
    return seg_score, picks_df


def _render_curriculum_analysis(
    data: dict,
    degree_id: str,
    degree_label: str,
    faculty_code: str,
    sel_role: str,
) -> None:
    mm: pd.DataFrame = data["mm"]
    mrs: pd.DataFrame = data["mrs"]
    prec: pd.DataFrame = data["prec"]
    mods: pd.DataFrame = data["mods"]
    band_lookup = _build_role_band_lookup(mrs)
    role_band = band_lookup[sel_role]

    st.subheader(f"Curriculum Analysis")
    st.markdown(
        f'<div style="font-size:1.05em;font-weight:600;color:#333;margin-bottom:4px">'
        f'Major: {degree_label} &nbsp;·&nbsp; '
        f'Role Family: <span style="color:#1565C0">{sel_role}</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Our model assigns an alignment score to each module based on how well its content  "
        "matches the skills and requirements of the respective job postings. "
        f"Role Family specific thresholds for {sel_role} are as follows: "
    )
    st.markdown(_legend_scores(role_band, sel_role), unsafe_allow_html=True)

    # Get modules for this degree, non-UE — keep all bucket rows (no dedup by module_code)
    deg_mm = mm[
        (mm["degree_id"] == degree_id) &
        ~mm["is_unrestricted_elective"]
    ].copy()

    # Role scores for selected role
    role_scores = mrs[mrs["role_family_name"] == sel_role][
        ["module_code", "module_title", "role_score", "evidence_job_count"]
    ].drop_duplicates("module_code")

    # Join role scores onto every bucket row; flatten module_title collision
    deg_scored = deg_mm.merge(role_scores, on="module_code", how="left")
    deg_scored["role_score"] = deg_scored["role_score"].fillna(0.0)
    if "module_title_y" in deg_scored.columns:
        deg_scored["module_title"] = deg_scored["module_title_y"].fillna(
            deg_scored.get("module_title_x", pd.Series("", index=deg_scored.index)).fillna("")
        )
    elif "module_title" not in deg_scored.columns:
        deg_scored["module_title"] = ""

    # Build preclusion sets once (used by both segment charts)
    prec_sets = _build_preclusion_sets(prec)

    primary_df = deg_scored[deg_scored["is_primary"]].copy()
    common_df = deg_scored[deg_scored["is_common"]].copy()

    # ── Section 1: Primary Major (first — drives common curriculum heuristic) ──
    st.divider()
    st.markdown(
        f'<div style="font-size:1.0em;font-weight:700;color:#6A1B9A;margin-bottom:8px">'
        f'Primary Major Requirements</div>',
        unsafe_allow_html=True,
    )
    primary_score, primary_best = _curriculum_segment_chart(
        "Primary Major Requirements", primary_df, sel_role, "#6A1B9A", role_band,
        prec_sets=prec_sets,
    )

    # ── Section 2: Common Curriculum (uses primary picks to fill overlapping slots) ──
    st.divider()
    st.markdown(
        f'<div style="font-size:1.0em;font-weight:700;color:#1565C0;margin-bottom:8px">'
        f'Common Curriculum</div>',
        unsafe_allow_html=True,
    )
    common_score, common_best = _curriculum_segment_chart(
        "Common Curriculum", common_df, sel_role, "#1565C0", role_band,
        primary_picked=primary_best if not primary_best.empty else None,
        prec_sets=prec_sets,
    )

    # ── Score Comparison Card ─────────────────────────────────────────────────
    st.divider()
    cmp1, cmp2 = st.columns(2)
    for col, label, score, color in [
        (cmp1, "Primary Major", primary_score, "#6A1B9A"),
        (cmp2, "Common Curriculum", common_score, "#1565C0"),
    ]:
        col.markdown(
            f'<div style="text-align:center;padding:14px;border-radius:10px;'
            f'border:2px solid {color}44;background:{color}08">'
            f'<div style="font-size:0.80em;font-weight:700;color:{color};'
            f'text-transform:uppercase;letter-spacing:0.05em">{label}</div>'
            f'<div style="font-size:2.0em;font-weight:900;color:{color};margin:4px 0">'
            f'{score:.3f}</div>'
            f'<div style="font-size:0.72em;color:#999">Mean alignment to {sel_role}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Section 3: Recommended Modules ───────────────────────────────────────
    st.divider()
    st.markdown(
        f'<div style="font-size:1.18em;font-weight:800;color:#2E7D32;margin-bottom:6px">'
        f'Recommended Modules outside existing curriculum (Recommendations)</div>',
        unsafe_allow_html=True,
    )

    threshold, threshold_basis = _recommendation_threshold(primary_best, role_scores)

    st.caption(
        f"This is a set of the top 20 aligned modules that are not in this degree's required curriculum with alignment ≥ **{threshold:.3f}** "
        f"({threshold_basis}). "
        f"Preclusions from the current curriculum are excluded. You may consider these for inclusion or as recommended electives. "
    )

    # All modules listed in this degree's curriculum (for preclusion check)
    curriculum_codes = set(deg_scored["module_code"].dropna().unique())
    # prec_sets already built above

    # Candidates: all modules with high role score, not in curriculum
    candidates = role_scores[
        (role_scores["role_score"] >= threshold) &
        (~role_scores["module_code"].isin(curriculum_codes))
    ].sort_values("role_score", ascending=False)

    recs = []
    for _, row in candidates.iterrows():
        code = str(row["module_code"])
        if not _precluded_by_curriculum(curriculum_codes, code, prec_sets):
            recs.append(row)
        if len(recs) >= 20:
            break

    if not recs:
        st.info("No additional high-scoring modules found outside preclusion constraints.")
    else:
        recs_df = pd.DataFrame(recs)
        # Join module title from mods if missing
        if "module_title" not in recs_df.columns or recs_df["module_title"].isna().all():
            title_map = mods.set_index("module_code")["module_title"].to_dict() if "module_title" in mods.columns else {}
            recs_df["module_title"] = recs_df["module_code"].map(title_map).fillna("")
        recs_df = recs_df.sort_values("role_score", ascending=False).reset_index(drop=True)
        rec_cards = "".join(
            _recommendation_card(
                str(r["module_code"]),
                str(r.get("module_title", "")),
                float(r["role_score"]),
                role_band,
                r.get("evidence_job_count"),
                [],
            )
            for _, r in recs_df.iterrows()
        )
        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));'
            f'gap:10px;align-items:start">{rec_cards}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 2 — Skill Requirements Overview
# ---------------------------------------------------------------------------

def _render_skill_requirements(data: dict, degree_label: str, sel_role: str) -> None:
    mm: pd.DataFrame = data["mm"]
    mods: pd.DataFrame = data["mods"]
    jrm: pd.DataFrame = data["jrm"]
    jobs: pd.DataFrame = data["jobs"]

    role_skills = _build_role_skills(jobs, jrm)
    seg_coverage = _build_degree_skill_segment_coverage(mm, mods)
    seg_intensity = _build_degree_skill_segment_intensity(mm, mods)
    job_counts = _build_job_counts(jrm)

    st.subheader("Skill Requirements Overview")
    total_jobs = job_counts.get(sel_role, 0)
    st.markdown(
        f'<div style="font-size:1.05em;font-weight:600;color:#333;margin-bottom:4px">'
        f'Major: {degree_label} &nbsp;·&nbsp; '
        f'Role Family: <span style="color:#1565C0">{sel_role}</span> &nbsp;·&nbsp; '
        f'{total_jobs:,} job postings analysed</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "This page displays an overview of all relevant skills extracted from job postings for this role. "
        "Refer to the heatmap to explore whether faculties and programmes cover these skills in their curriculum. "
        "(XX %) shows the proportion of postings that mention the skill. For specific skills-modules matching, refer to Skill Gaps tab"
    )

    role_df = role_skills[role_skills["role_family_name"] == sel_role].copy()
    coverage_options = [
        "All required curriculum",
        "Common Curriculum",
        "Primary Major",
    ]
    coverage_key = "sr_coverage_basis"
    if coverage_key not in st.session_state:
        st.session_state[coverage_key] = "All required curriculum"
    selected_basis = str(st.session_state[coverage_key])
    if selected_basis not in coverage_options:
        selected_basis = "All required curriculum"
        st.session_state[coverage_key] = selected_basis
    coverage_col = {
        "All required curriculum": "covered_all",
        "Common Curriculum": "covered_common",
        "Primary Major": "covered_primary",
    }[selected_basis]

    # ── Coverage heatmap: skills × faculties ─────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:1.15rem;font-weight:700;margin-bottom:14px'>"
        "Which Faculties and Programmes cover these skills?"
        "</div>",
        unsafe_allow_html=True,
    )
    all_skills_for_role = role_df["skill"].unique()
    coverage_role = seg_coverage[seg_coverage["skill"].isin(all_skills_for_role)].copy()
    coverage_role["covered"] = coverage_role[coverage_col].astype(bool)
    intensity_role = seg_intensity[seg_intensity["skill"].isin(all_skills_for_role)].copy()

    if coverage_role.empty:
        st.info("No curriculum skill coverage data found for this role.")
        return

    role_df = role_df.sort_values(["skill_type", "demand_pct"], ascending=[True, False]).copy()
    skill_type_lookup = (
        role_df.drop_duplicates("skill")
        .set_index("skill")["skill_type"]
        .to_dict()
    )
    demand_ranked = role_df.sort_values("demand_pct", ascending=False)["skill"].tolist()
    tech_order = [s for s in demand_ranked if skill_type_lookup.get(s) == "technical"]
    soft_order = [s for s in demand_ranked if skill_type_lookup.get(s) == "soft"]
    demand_order = tech_order + soft_order
    drill_key = "sr_drill_faculty"
    if drill_key not in st.session_state:
        st.session_state[drill_key] = None

    available_faculties = sorted(coverage_role["faculty_code"].dropna().astype(str).unique())
    selected_faculty = st.session_state[drill_key]
    if selected_faculty not in available_faculties:
        selected_faculty = None
        st.session_state[drill_key] = None

    with st.container(border=True):
        basis_label_col, basis_buttons_col = st.columns([1.2, 6.8], vertical_alignment="center")
        with basis_label_col:
            st.markdown("**Curriculum Type**")
        with basis_buttons_col:
            basis_cols = st.columns(len(coverage_options))
            for idx, option in enumerate(coverage_options):
                button_type = "primary" if selected_basis == option else "secondary"
                with basis_cols[idx]:
                    if st.button(
                        option,
                        key=f"sr_basis_{option}",
                        width="stretch",
                        type=button_type,
                    ):
                        st.session_state[coverage_key] = option
                        st.rerun()

    with st.container(border=True):
        faculty_label_col, faculty_buttons_col = st.columns([1.2, 6.8], vertical_alignment="center")
        with faculty_label_col:
            st.markdown("**Faculty**")
        with faculty_buttons_col:
            faculty_options = ["All faculties"] + available_faculties
            split_idx = (len(faculty_options) + 1) // 2
            faculty_rows = [
                faculty_options[:split_idx],
                faculty_options[split_idx:],
            ]
            for row_idx, faculty_row in enumerate(faculty_rows):
                if not faculty_row:
                    continue
                nav_cols = st.columns(len(faculty_row))
                for idx, fac in enumerate(faculty_row):
                    is_selected = (
                        (fac == "All faculties" and selected_faculty is None)
                        or (fac != "All faculties" and selected_faculty == fac)
                    )
                    button_type = "primary" if is_selected else "secondary"
                    with nav_cols[idx]:
                        if st.button(
                            fac,
                            key=f"sr_fac_{row_idx}_{fac}",
                            width="stretch",
                            type=button_type,
                        ):
                            st.session_state[drill_key] = None if fac == "All faculties" else fac
                            st.rerun()

    scale_enabled = False
    if selected_faculty:
        toggle_col, legend_col = st.columns([3.8, 4.2], vertical_alignment="center")
        with toggle_col:
            scale_enabled = st.toggle(
                "Toggle Scale: module-teaching proportion",
                value=False,
                key="sr_toggle_scale",
            )
        with legend_col:
            if not scale_enabled:
                st.markdown(
                    """
                    <div style="display:flex;align-items:center;justify-content:flex-end;gap:14px;flex-wrap:wrap;margin:0">
                        <span style="display:inline-flex;align-items:center;gap:6px;font-size:0.82em;color:#555">
                            <span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#C8E6C9;border:1px solid #9CCC65"></span>
                            Yes
                        </span>
                        <span style="display:inline-flex;align-items:center;gap:6px;font-size:0.82em;color:#555">
                            <span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#FFCDD2;border:1px solid #EF9A9A"></span>
                            No
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    elif "sr_toggle_scale" in st.session_state:
        st.session_state["sr_toggle_scale"] = False
        st.markdown(
            """
            <div style="display:flex;align-items:center;justify-content:flex-end;gap:14px;flex-wrap:wrap;margin:10px 0 6px 0">
                <span style="display:inline-flex;align-items:center;gap:6px;font-size:0.82em;color:#555">
                    <span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#C8E6C9;border:1px solid #9CCC65"></span>
                    Yes
                </span>
                <span style="display:inline-flex;align-items:center;gap:6px;font-size:0.82em;color:#555">
                    <span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#FFCDD2;border:1px solid #EF9A9A"></span>
                    No
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if selected_faculty:
        scope_label = f"Programme: {selected_faculty} · {FACULTY_FULL.get(selected_faculty, selected_faculty)}"
        view_df = coverage_role[coverage_role["faculty_code"].astype(str) == selected_faculty].copy()
        if scale_enabled:
            intensity_df = intensity_role[intensity_role["faculty_code"].astype(str) == selected_faculty].copy()
            share_col = {
                "All required curriculum": "share_all",
                "Common Curriculum": "share_common",
                "Primary Major": "share_primary",
            }[selected_basis]
            modules_col = {
                "All required curriculum": "modules_all",
                "Common Curriculum": "modules_common",
                "Primary Major": "modules_primary",
            }[selected_basis]
            total_modules_col = {
                "All required curriculum": "total_modules_all",
                "Common Curriculum": "total_modules_common",
                "Primary Major": "total_modules_primary",
            }[selected_basis]
            pivot = (
                intensity_df.groupby(["skill", "primary_major"])[share_col]
                .max()
                .unstack("primary_major", fill_value=0.0)
            )
            hover_matrix = (
                intensity_df.groupby(["skill", "primary_major"])
                .agg(
                    modules=(modules_col, "max"),
                    total_modules=(total_modules_col, "max"),
                    share=(share_col, "max"),
                )
                .reset_index()
            )
            hover_lookup = {
                (str(r["skill"]), str(r["primary_major"])): (
                    f'{int(r["modules"])} of {int(r["total_modules"])} modules '
                    f'({float(r["share"]):.0%})'
                )
                for _, r in hover_matrix.iterrows()
            }
        else:
            pivot = (
                view_df.groupby(["skill", "primary_major"])["covered"]
                .max()
                .astype(int)
                .unstack("primary_major", fill_value=0)
            )
            hover_lookup = {
                (str(skill), str(programme)): ("Yes" if val else "No")
                for skill, row in pivot.iterrows()
                for programme, val in row.items()
            }
        entity_type = "programme"
        x_labels = sorted(pivot.columns.tolist())
    else:
        pivot = (
            coverage_role.groupby(["skill", "faculty_code"])["covered"]
            .max()
            .astype(int)
            .unstack("faculty_code", fill_value=0)
        )
        entity_type = "faculty"
        x_labels = sorted(pivot.columns.tolist())
        scope_label = "Faculty: All faculties"
        hover_lookup = {
            (str(skill), str(fac)): ("Yes" if val else "No")
            for skill, row in pivot.iterrows()
            for fac, val in row.items()
        }

    demand_order = [s for s in demand_order if s in pivot.index]
    pivot = pivot.reindex(demand_order)

    top_n_hm = 40
    if len(pivot) > top_n_hm:
        pivot_hm = pivot.head(top_n_hm)
        skills_note = f"Showing top {top_n_hm} skills by demand."
    else:
        pivot_hm = pivot
        skills_note = ""
    tech_count = int(sum(skill_type_lookup.get(str(skill)) == "technical" for skill in pivot_hm.index))
    soft_count = int(sum(skill_type_lookup.get(str(skill)) == "soft" for skill in pivot_hm.index))

    label_parts = [
        f"Curriculum Type: {selected_basis}",
        scope_label,
    ]
    if selected_faculty and scale_enabled:
        label_parts.append("Scale: On")
    if skills_note:
        label_parts.append(skills_note)
    st.caption(" · ".join(label_parts))

    if entity_type == "faculty":
        display_cols = {
            col: f"{col}<br>{FACULTY_FULL.get(str(col), str(col))}"
            for col in x_labels
            if col in pivot_hm.columns
        }
    else:
        display_cols = {
            col: str(col)
            for col in x_labels
            if col in pivot_hm.columns
        }

    pivot_hm = pivot_hm[[col for col in x_labels if col in pivot_hm.columns]]
    pivot_display = pivot_hm.rename(columns=display_cols)

    # Row annotations: demand pct
    skill_demand = role_df.set_index("skill")["demand_pct"].to_dict()
    y_labels = [
        f"{s}  ({skill_demand.get(s, 0):.0%})"
        for s in pivot_display.index
    ]
    hover_customdata = [
        [hover_lookup.get((str(skill), str(col)), "No") for col in pivot_display.columns]
        for skill in pivot_display.index
    ]
    if selected_faculty and scale_enabled:
        st.markdown(
            """
            <div style="display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap;margin:6px 0 10px 0">
                <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
                    <span style="font-size:0.82em;font-weight:600;color:#555">Proportion of Modules that teach skill</span>
                    <div style="display:flex;flex-direction:column;gap:4px">
                        <div style="width:220px;height:12px;border-radius:999px;border:1px solid #ddd;background:
                            linear-gradient(90deg,
                                #FFCDD2 0%,
                                #FFE0B2 20%,
                                #FFF9C4 45%,
                                #C8E6C9 72%,
                                #66BB6A 100%)"></div>
                        <div style="display:flex;justify-content:space-between;width:220px">
                            <span style="font-size:0.78em;color:#777">Low</span>
                            <span style="font-size:0.78em;color:#777">High</span>
                        </div>
                    </div>
                </div>
                <div style="display:flex;align-items:center;gap:6px;font-size:0.82em;color:#555">
                    <span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:#E0E0E0;border:1px solid #BDBDBD"></span>
                    No coverage
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        colorscale = [
            [0, "#E0E0E0"],
            [0.000001, "#FFCDD2"],
            [0.15, "#FFE0B2"],
            [0.4, "#FFF9C4"],
            [0.7, "#C8E6C9"],
            [1, "#66BB6A"],
        ]
        showscale = False
        colorbar = None
        hover_label = "Coverage intensity"
    else:
        colorscale = [[0, "#FFCDD2"], [0.4999, "#FFCDD2"], [0.5, "#C8E6C9"], [1, "#C8E6C9"]]
        showscale = False
        colorbar = None
        hover_label = f"Covered by this {entity_type}"

    fig_hm = go.Figure(
        go.Heatmap(
            z=pivot_display.values,
            x=pivot_display.columns.tolist(),
            y=y_labels,
            colorscale=colorscale,
            zmin=0, zmax=1,
            showscale=showscale,
            colorbar=colorbar,
            xgap=3, ygap=2,
            hovertemplate=(
                "<b>%{y}</b><br>%{x}<br>"
                f"{hover_label}: "
                "%{customdata}<extra></extra>"
            ),
            customdata=hover_customdata,
        )
    )
    fig_hm.update_layout(
        height=max(400, len(pivot_hm) * 22),
        margin={"t": 20, "b": 100, "l": 0, "r": 0},
        xaxis={"tickangle": -35, "tickfont": {"size": 9.5}},
        yaxis={"tickfont": {"size": 10}, "autorange": "reversed"},
    )
    st.plotly_chart(fig_hm, width="stretch")


# ---------------------------------------------------------------------------
# Tab 3 — Skill Gaps
# ---------------------------------------------------------------------------

def _render_skill_gaps(
    data: dict,
    degree_id: str,
    degree_label: str,
    faculty_code: str,
    sel_role: str,
) -> None:
    mm: pd.DataFrame = data["mm"]
    mrs: pd.DataFrame = data["mrs"]
    prec: pd.DataFrame = data["prec"]
    mods: pd.DataFrame = data["mods"]
    jrm: pd.DataFrame = data["jrm"]
    jobs: pd.DataFrame = data["jobs"]
    dss: pd.DataFrame = data["dss"]

    role_skills = _build_role_skills(jobs, jrm)
    mod_skill_map = _build_module_skill_map(mods)
    job_counts = _build_job_counts(jrm)
    prec_sets = _build_preclusion_sets(prec)
    band_lookup = _build_role_band_lookup(mrs)
    role_band = band_lookup[sel_role]

    st.subheader("Skill Gaps")
    total_jobs = job_counts.get(sel_role, 0)
    st.markdown(
        f'<div style="font-size:1.05em;font-weight:600;color:#333;margin-bottom:4px">'
        f'Major: {degree_label} &nbsp;·&nbsp; '
        f'Role Family: <span style="color:#1565C0">{sel_role}</span>'
        f' &nbsp;·&nbsp; {total_jobs:,} job postings</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "This page displays all skills demanded by job postings for this role. "
        "Coverage = whether the required curriculum (excluding Unrestricted Electives) "
        "contains at least one module with this skill. "
    )

    # ── Get demanded skills for this role ──────────────────────────────────
    role_df = role_skills[role_skills["role_family_name"] == sel_role].sort_values(
        "demand_pct", ascending=False
    ).reset_index(drop=True)
    if role_df.empty:
        st.warning(f"No skill data found for {sel_role}.")
        return

    # ── Get degree's required curriculum modules (non-UE) ──────────────────
    deg_mm = mm[
        (mm["degree_id"] == degree_id) &
        mm["module_found"] &
        ~mm["is_unrestricted_elective"]
    ].drop_duplicates("module_code")
    curriculum_codes = set(deg_mm["module_code"].unique())

    # Skills covered by curriculum: use degree_skill_supply
    deg_dss = dss[dss["degree_id"] == degree_id].copy()
    covered_skills = set(deg_dss[deg_dss["supply_score"] > 0]["skill"].str.lower().unique())

    # Build skill coverage table
    role_df["covered"] = role_df["skill"].isin(covered_skills)
    # Get covering module codes for covered skills
    skill_module_map = deg_dss[deg_dss["supply_score"] > 0].set_index("skill")[
        "module_codes"
    ].to_dict()
    role_mrs = mrs[mrs["role_family_name"] == sel_role].copy()
    role_mrs["role_score"] = pd.to_numeric(role_mrs["role_score"], errors="coerce").fillna(0.0)
    role_mrs["evidence_job_count"] = pd.to_numeric(role_mrs["evidence_job_count"], errors="coerce").fillna(0.0)
    role_score_lookup = (
        role_mrs.sort_values(["module_code", "role_score"], ascending=[True, False])
        .drop_duplicates("module_code")
        .set_index("module_code")[["role_score", "evidence_job_count"]]
        .to_dict("index")
    )

    def _covered_module_review_badges(module_codes: list[str]) -> str:
        if not module_codes:
            return '<span style="font-size:0.74em;color:#999">No module codes listed</span>'
        badges = []
        for code in module_codes[:8]:
            info = role_score_lookup.get(code)
            if info:
                score = float(info.get("role_score", 0.0))
                evidence = float(info.get("evidence_job_count", 0.0))
                color = _score_color(score, role_band, evidence)
                if color == "#2E7D32":
                    bg, fg, border = "#E8F5E9", "#1B5E20", "#A5D6A7"
                elif color == "#F57F17":
                    bg, fg, border = "#FFF3E0", "#A84300", "#FFCC80"
                else:
                    bg, fg, border = "#FFEBEE", "#B71C1C", "#EF9A9A"
                badges.append(
                    f'<span style="background:{bg};color:{fg};border:1px solid {border};'
                    f'border-radius:999px;padding:2px 8px;font-size:0.68em;font-weight:700;'
                    f'margin:0 6px 6px 0;display:inline-block">{code} ({score:.3f})</span>'
                )
            else:
                badges.append(
                    f'<span style="background:#F5F5F5;color:#555;border:1px solid #D0D0D0;'
                    f'border-radius:999px;padding:2px 8px;font-size:0.68em;font-weight:700;'
                    f'margin:0 6px 6px 0;display:inline-block">{code}</span>'
                )
        return "".join(badges)

    # Summary metrics
    n_total = len(role_df)
    n_covered = role_df["covered"].sum()
    n_missing = n_total - n_covered

    with st.container(border=True):
        st.markdown("#### Top Skill Gaps")
        st.caption(
            "Quick summary of demanded skills covered by the required curriculum "
            "(excluding Unrestricted Electives)."
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total skills demanded", n_total)
        m2.metric("Covered by curriculum", int(n_covered))
        m3.metric("Not covered", int(n_missing))
        m4.metric(
            "Coverage rate",
            f"{n_covered / n_total:.0%}" if n_total > 0 else "—",
        )

    st.divider()

    title_col, btn1, btn2, btn3 = st.columns([5.2, 1, 1, 1])
    with title_col:
        st.markdown("#### Skill-Module Gaps (Recommendations)")

    filter_key = "sg_skill_filter"
    if filter_key not in st.session_state:
        st.session_state[filter_key] = "Both"

    with btn1:
        if st.button(
            "Both",
            key="sg_filter_both",
            width="stretch",
            type="primary" if st.session_state[filter_key] == "Both" else "secondary",
        ):
            st.session_state[filter_key] = "Both"
    with btn2:
        if st.button(
            "Technical",
            key="sg_filter_technical",
            width="stretch",
            type="primary" if st.session_state[filter_key] == "Technical" else "secondary",
        ):
            st.session_state[filter_key] = "Technical"
    with btn3:
        if st.button(
            "Soft",
            key="sg_filter_soft",
            width="stretch",
            type="primary" if st.session_state[filter_key] == "Soft" else "secondary",
        ):
            st.session_state[filter_key] = "Soft"

    skill_filter = st.session_state[filter_key]
    st.caption(f"Showing: {skill_filter} skills")

    filtered_role_df = role_df.copy()
    if skill_filter == "Technical":
        filtered_role_df = filtered_role_df[filtered_role_df["skill_type"] == "technical"]
    elif skill_filter == "Soft":
        filtered_role_df = filtered_role_df[filtered_role_df["skill_type"] == "soft"]

    not_cov_df = filtered_role_df[~filtered_role_df["covered"]].reset_index(drop=True)
    cov_df = filtered_role_df[filtered_role_df["covered"]].reset_index(drop=True)

    st.markdown(
        """
        <div style="font-size:0.78em;color:#6b7280;line-height:1.55;margin-top:-4px;margin-bottom:8px">
            <div>This is an overview of skills demanded by the role and what modules teach them.</div>
            <div>Priority 1 - Uncovered skills with no recommended modules (left) may warrant urgent attention.</div>
            <div>Priority 2 - Uncovered skills with recommended modules (right) can be addressed by reviewing the recommended modules for inclusion or as electives.</div>
            <div>Priority 3 - Covered skills (bottom) are already taught by the curriculum but may warrant a review of the teaching approach or depth.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    role_mrs = role_mrs.sort_values("role_score", ascending=False)
    mod_skill_lookup = mod_skill_map.groupby("module_code")["skill"].apply(set).to_dict()

    rec_candidates = role_mrs[
        ~role_mrs["module_code"].isin(curriculum_codes)
    ].copy()
    rec_candidates = rec_candidates[
        rec_candidates["module_code"].map(
            lambda c: not _precluded_by_curriculum(curriculum_codes, c, prec_sets)
        )
    ]

    def _skill_recommendations(skill: str) -> pd.DataFrame:
        return rec_candidates[
            rec_candidates["module_code"].map(
                lambda c: skill in mod_skill_lookup.get(c, set())
            )
        ].head(3)

    priority_1: list[tuple[pd.Series, pd.DataFrame]] = []
    priority_2: list[tuple[pd.Series, pd.DataFrame]] = []
    for _, skill_row in not_cov_df.iterrows():
        skill_name = str(skill_row["skill"])
        skill_recs = _skill_recommendations(skill_name)
        if skill_recs.empty:
            priority_1.append((skill_row, skill_recs))
        else:
            priority_2.append((skill_row, skill_recs))

    p1_col, p2_col = st.columns(2)

    with p1_col:
        st.markdown(
            f'<div style="font-size:0.86em;font-weight:700;color:#B71C1C;'
            f'padding:6px 12px;background:#FFEBEE;border-radius:8px;margin-bottom:8px">'
            f'Priority 1 &nbsp;·&nbsp; Uncovered skills with no recommended modules &nbsp;·&nbsp; {len(priority_1)}</div>',
            unsafe_allow_html=True,
        )
        if not priority_1:
            st.success("No uncovered skills fall into Priority 1.")
        else:
            parts = []
            for skill_row, _ in priority_1:
                skill = str(skill_row["skill"])
                is_tech = str(skill_row["skill_type"]) == "technical"
                parts.append(
                    _skill_row(
                        skill,
                        int(skill_row["job_count"]),
                        total_jobs,
                        covered=False,
                        is_tech=is_tech,
                    )
                )
            st.markdown(
                f'<div style="max-height:320px;overflow-y:auto">{"".join(parts)}</div>',
                unsafe_allow_html=True,
            )

    with p2_col:
        st.markdown(
            f'<div style="font-size:0.86em;font-weight:700;color:#A84300;'
            f'padding:6px 12px;background:#FFF3E0;border-radius:8px;margin-bottom:8px">'
            f'Priority 2 &nbsp;·&nbsp; Uncovered skills with recommended modules &nbsp;·&nbsp; {len(priority_2)}</div>',
            unsafe_allow_html=True,
        )
        if not priority_2:
            st.success("No uncovered skills fall into Priority 2.")
        else:
            parts = []
            for skill_row, skill_recs in priority_2:
                skill = str(skill_row["skill"])
                is_tech = str(skill_row["skill_type"]) == "technical"
                parts.append(
                    _skill_row(
                        skill,
                        int(skill_row["job_count"]),
                        total_jobs,
                        covered=False,
                        is_tech=is_tech,
                    )
                )
                rec_html = "".join(
                    f'<div style="margin-left:22px;margin-top:1px;max-width:88%">'
                    f'{_recommendation_card(str(r["module_code"]), str(r["module_title"]), float(r["role_score"]), skills_covered=[])}'
                    f'</div>'
                    for _, r in skill_recs.iterrows()
                )
                parts.append(
                    f'<div style="font-size:0.69em;color:#888;font-weight:700;'
                    f'padding:2px 12px;margin-top:1px">↑ Recommended modules</div>'
                    f'{rec_html}'
                )
            st.markdown(
                f'<div style="max-height:320px;overflow-y:auto">{"".join(parts)}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.86em;font-weight:700;color:#1B5E20;'
        f'padding:6px 12px;background:#E8F5E9;border-radius:8px;margin-bottom:8px">'
        f'Priority 3 &nbsp;·&nbsp; Covered skills taught by modules &nbsp;·&nbsp; {len(cov_df)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "These skills are already covered by the required curriculum. review "
        "the modules currently teaching them as necessary."
    )
    if cov_df.empty:
        st.success("No covered skills available for Priority 3 review.")
    else:
        parts = []
        for _, skill_row in cov_df.iterrows():
            skill = str(skill_row["skill"])
            is_tech = str(skill_row["skill_type"]) == "technical"
            raw_codes = skill_module_map.get(skill, "")
            modules = _parse_codes(raw_codes)
            parts.append(
                _skill_row(
                    skill,
                    int(skill_row["job_count"]),
                    total_jobs,
                    covered=True,
                    is_tech=is_tech,
                    modules=modules,
                    modules_html=f'<div style="margin-top:2px">{_covered_module_review_badges(modules)}</div>',
                )
            )
        st.markdown(
            f'<div style="max-height:320px;overflow-y:auto">{"".join(parts)}</div>',
            unsafe_allow_html=True,
        )

    st.divider()


def _render_module_details(
    data: dict,
    degree_id: str,
    degree_label: str,
    sel_role: str,
) -> None:
    mm: pd.DataFrame = data["mm"]
    mods: pd.DataFrame = data["mods"]
    mrs: pd.DataFrame = data["mrs"]

    st.subheader("Module Details")
    st.markdown(
        f'<div style="font-size:1.05em;font-weight:600;color:#333;margin-bottom:4px">'
        f'Major: {degree_label} &nbsp;·&nbsp; '
        f'Role Family: <span style="color:#1565C0">{sel_role}</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Browse all modules in the catalogue. Modules from the selected degree and modules "
        "most aligned to the selected role are surfaced first, but search covers everything."
    )
    band_lookup = _build_role_band_lookup(mrs)

    deg_mm = mm[
        (mm["degree_id"] == degree_id)
        & (mm["module_found"])
        & (~mm["is_unrestricted_elective"])
    ].copy()
    selected_degree_codes = set(deg_mm["module_code"].dropna().astype(str).unique())
    selected_role_codes = set(
        mrs[mrs["role_family_name"] == sel_role]["module_code"].dropna().astype(str).unique()
    )
    top_roles = _build_top_roles_per_module(mrs, band_lookup)
    mods_subset = mods.copy()

    display_df = (
        mods_subset[
            [
                "module_code",
                "module_title",
                "module_credit",
                "module_description",
                "technical_skills",
                "soft_skills",
            ]
        ]
        .drop_duplicates("module_code")
        .merge(top_roles, on="module_code", how="left")
        .sort_values("module_code")
        .reset_index(drop=True)
    )

    display_df["module_credit"] = pd.to_numeric(display_df["module_credit"], errors="coerce")
    display_df["technical_skills_list"] = display_df["technical_skills"].map(_parse_list)
    display_df["soft_skills_list"] = display_df["soft_skills"].map(_parse_list)
    display_df["all_skills_text"] = display_df.apply(
        lambda r: " | ".join(r["technical_skills_list"] + r["soft_skills_list"]),
        axis=1,
    )
    display_df["search_text"] = display_df.apply(
        lambda r: " ".join(
            [
                str(r.get("module_code", "")),
                str(r.get("module_title", "")),
                str(r.get("module_description", "")),
                str(r.get("all_skills_text", "")),
                str(r.get("top_role_families", "")),
            ]
        ).lower(),
        axis=1,
    )
    display_df["in_selected_degree"] = display_df["module_code"].astype(str).isin(selected_degree_codes)
    display_df["matches_selected_role"] = display_df["top_role_families"].map(
        lambda roles: any(str(item[0]) == sel_role for item in roles) if isinstance(roles, list) else False
    )
    display_df["selected_role_rank"] = display_df["top_role_families"].map(
        lambda roles: next(
            (
                idx for idx, item in enumerate(roles)
                if isinstance(item, (list, tuple)) and len(item) > 0 and str(item[0]) == sel_role
            ),
            99,
        ) if isinstance(roles, list) else 99
    )
    display_df = display_df.sort_values(
        ["in_selected_degree", "matches_selected_role", "selected_role_rank", "module_code"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    search = st.text_input(
        "Search modules",
        value="",
        placeholder="Try CS2103, analytics, python, communication, Software Engineering...",
        key="module_details_search",
    ).strip().lower()

    if search:
        display_df = display_df[display_df["search_text"].str.contains(search, na=False)].copy()
        result_caption = (
            f"Showing {len(display_df)} search results from the full module catalogue. "
            f"Selected-major and {sel_role}-linked modules are shown first."
        )
    else:
        default_codes = selected_degree_codes | selected_role_codes
        display_df = display_df[display_df["module_code"].astype(str).isin(default_codes)].copy()
        display_df = display_df.head(100).copy()
        result_caption = (
            f"Showing top {len(display_df)} linked modules for the selected major or role family. "
            "Use search to access the full module catalogue."
        )

    st.caption(result_caption)
    if display_df.empty:
        st.warning("No modules match the current search.")
        return

    def _skill_badges(skills: list[str], bg: str, fg: str, border: str) -> str:
        if not skills:
            return '<span style="font-size:0.74em;color:#999">None listed</span>'
        return "".join(
            f'<span style="background:{bg};color:{fg};border:1px solid {border};'
            f'border-radius:999px;padding:2px 8px;font-size:0.70em;font-weight:600;'
            f'margin:0 5px 5px 0;display:inline-block">{skill}</span>'
            for skill in skills[:12]
        )

    def _role_badges(roles: object) -> str:
        if not roles or str(roles).strip() in ("", "nan"):
            return '<span style="font-size:0.74em;color:#999">No role-family scores available.</span>'
        badges = []
        for item in roles if isinstance(roles, list) else []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            role_name = str(item[0])
            score = float(item[1])
            evidence = float(item[2]) if len(item) > 2 and item[2] is not None else None
            band_label = str(item[3]) if len(item) > 3 else ""
            band = band_lookup.get(role_name)
            color = _score_color(score, band, evidence)
            if color == "#2E7D32":
                bg, fg, border = "#E8F5E9", "#1B5E20", "#A5D6A7"
            elif color == "#F57F17":
                bg, fg, border = "#FFF3E0", "#A84300", "#FFCC80"
            else:
                bg, fg, border = "#FFEBEE", "#B71C1C", "#EF9A9A"
            badges.append(
                f'<span style="background:{bg};color:{fg};border:1px solid {border};'
                f'border-radius:999px;padding:3px 9px;font-size:0.70em;font-weight:700;'
                f'margin:0 6px 6px 0;display:inline-block">{role_name} ({float(score):.3f})'
                f'{" · " + band_label if band_label else ""}</span>'
            )
        if not badges:
            return '<span style="font-size:0.74em;color:#999">No role-family scores available.</span>'
        return "".join(badges)

    cards = []
    for _, row in display_df.iterrows():
        credits = (
            f'{float(row["module_credit"]):.0f} MC'
            if pd.notna(row["module_credit"])
            else "MC not listed"
        )
        description = str(row.get("module_description", "")).strip() or "No description available."
        top_roles_html = _role_badges(row.get("top_role_families"))
        tech_html = _skill_badges(row["technical_skills_list"], "#E3F2FD", "#0D47A1", "#90CAF9")
        soft_html = _skill_badges(row["soft_skills_list"], "#F3E5F5", "#6A1B9A", "#CE93D8")
        cards.append(
            f'<div style="padding:14px 16px;margin:0 0 12px 0;background:#fff;border:1px solid #e8e8e8;'
            f'border-radius:12px;box-shadow:0 1px 2px rgba(0,0,0,0.03)">'
            f'<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;flex-wrap:wrap">'
            f'<div>'
            f'<div style="font-size:1.0em;font-weight:800;color:#111">{row["module_code"]}</div>'
            f'<div style="font-size:0.92em;color:#444;margin-top:2px">{row["module_title"]}</div>'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:flex-end">'
            f'<div style="font-size:0.78em;font-weight:700;color:#1565C0;background:#E3F2FD;'
            f'border:1px solid #BBDEFB;border-radius:999px;padding:4px 10px;white-space:nowrap">{credits}</div>'
            f'</div>'
            f'</div>'
            f'<div style="margin-top:10px;font-size:0.82em;line-height:1.55;color:#333">{description}</div>'
            f'<div style="margin-top:12px;font-size:0.72em;font-weight:800;color:#666;text-transform:uppercase;letter-spacing:0.04em">Technical Skills</div>'
            f'<div style="margin-top:6px">{tech_html}</div>'
            f'<div style="margin-top:10px;font-size:0.72em;font-weight:800;color:#666;text-transform:uppercase;letter-spacing:0.04em">Soft Skills</div>'
            f'<div style="margin-top:6px">{soft_html}</div>'
            f'<div style="margin-top:10px;font-size:0.72em;font-weight:800;color:#666;text-transform:uppercase;letter-spacing:0.04em">Top 3 Role Families</div>'
            f'<div style="margin-top:6px">{top_roles_html}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="max-height:980px;overflow-y:auto;padding-right:6px">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Curriculum Readiness Dashboard — MOE",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        div[data-baseweb="tab-list"] {
            gap: 12px;
            display: flex;
        }
        div[data-baseweb="tab-list"] > button[role="tab"] {
            flex: 1 1 0;
            min-width: 0;
            justify-content: center;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📊 Curriculum Readiness Dashboard")
    st.caption(
        "MOE policy review tool. Analyses how NUS degree curricula prepare graduates "
        "for Singapore's job market using module-to-job alignment scores validated against "
        "human-annotated relevance labels. Use the Streamlit page selector in the sidebar "
        "to switch between this dashboard and the Natural-Language Job Assistant. "
        "How to navigate the tabs on this page:"
    )
    st.markdown(
        """
        <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:10px 0 18px 0">
            <div style="background:#F3E5F5;border:1px solid #E1BEE7;border-radius:12px;padding:12px 14px">
                <div style="font-size:0.86em;font-weight:700;color:#6A1B9A;margin-bottom:4px">Curriculum Analysis</div>
                <div style="font-size:0.80em;color:#555">See how strongly the required curriculum aligns to the selected role, split into Primary Major and Common Curriculum.</div>
            </div>
            <div style="background:#E3F2FD;border:1px solid #90CAF9;border-radius:12px;padding:12px 14px">
                <div style="font-size:0.86em;font-weight:700;color:#1565C0;margin-bottom:4px">Skill Requirements</div>
                <div style="font-size:0.80em;color:#555">An overview of the skills employers ask for, and which faculties or programmes cover those skills in their curriculum.</div>
            </div>
            <div style="background:#FFF3E0;border:1px solid #FFCC80;border-radius:12px;padding:12px 14px">
                <div style="font-size:0.86em;font-weight:700;color:#A84300;margin-bottom:4px">Skill Gaps</div>
                <div style="font-size:0.80em;color:#555">Identify demanded skills that are not covered, and review modules that could help address those gaps.</div>
            </div>
            <div style="background:#E8F5E9;border:1px solid #A5D6A7;border-radius:12px;padding:12px 14px">
                <div style="font-size:0.86em;font-weight:700;color:#1B5E20;margin-bottom:4px">Module Details</div>
                <div style="font-size:0.80em;color:#555">Browse the degree’s required modules, inspect descriptions and skills, and review the top role families each module supports.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading data…"):
        data = _load_all()

    role_skills = _build_role_skills(data["jobs"], data["jrm"])
    available_roles = set(role_skills["role_family_name"].unique())

    degree_id, degree_label, faculty_code, sel_role = _render_sidebar(
        data["summary"], available_roles
    )

    tabs = st.tabs([
        "📚  Curriculum Analysis",
        "🔍  Skill Requirements",
        "📊  Skill Gaps",
        "🧩  Module Details",
    ])

    with tabs[0]:
        _render_curriculum_analysis(data, degree_id, degree_label, faculty_code, sel_role)

    with tabs[1]:
        _render_skill_requirements(data, degree_label, sel_role)

    with tabs[2]:
        _render_skill_gaps(data, degree_id, degree_label, faculty_code, sel_role)

    with tabs[3]:
        _render_module_details(data, degree_id, degree_label, sel_role)


if __name__ == "__main__":
    main()
