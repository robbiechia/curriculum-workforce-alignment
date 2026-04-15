from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import pandas as pd

from ..config.settings import PipelineConfig


def _escape_md(text: object) -> str:
    s = str(text)
    return s.replace("|", "\\|").replace("\n", " ").strip()


def _fmt_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return _escape_md(value)


def _markdown_table(df: pd.DataFrame, cols: List[str], n: int = 10) -> str:
    if df.empty:
        return "No rows available."

    view = df[cols].head(n).copy()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"

    rows = []
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(_fmt_value(row[c]) for c in cols) + " |")

    return "\n".join([header, sep] + rows)


def _gap_sections_markdown(gaps: pd.DataFrame, top_role_families: int = 6, top_skills: int = 6) -> str:
    if "gap_type" not in gaps.columns:
        return "No undersupply rows available."

    under = gaps[gaps["gap_type"] == "undersupply"].copy()
    if under.empty:
        return "No undersupply rows available."

    role_col = "role_family_name" if "role_family_name" in under.columns else "role_family"
    # Sort families by the single strongest undersupply signal so the report starts
    # with the most actionable problem areas.
    family_order = (
        under.groupby(role_col)["gap_score"]
        .max()
        .sort_values(ascending=False)
        .head(top_role_families)
        .index
    )

    sections: List[str] = []
    for family in family_order:
        family_rows = (
            under[under[role_col] == family]
            .sort_values(["gap_score", "demand_score"], ascending=[False, False])
            .head(top_skills)
            .copy()
        )
        family_rows = family_rows.rename(
            columns={
                "skill": "Skill",
                "demand_score": "Demand",
                "supply_score": "Supply",
                "gap_score": "Gap",
            }
        )

        sections.append(f"### {family}")
        sections.append(_markdown_table(family_rows, ["Skill", "Demand", "Supply", "Gap"], n=top_skills))
        sections.append("")

    return "\n".join(sections).strip()


def build_policy_brief(
    module_summary: pd.DataFrame,
    module_role_scores: pd.DataFrame,
    module_gap_summary: pd.DataFrame,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # The policy brief intentionally collapses the full outputs into a short executive
    # summary rather than exposing every diagnostics column.
    top_modules = module_summary.sort_values("top_role_score", ascending=False).head(12).copy()
    if "top_role_family_name" in top_modules.columns:
        top_modules["Top Role Family"] = top_modules["top_role_family_name"].astype(str)
    else:
        top_modules["Top Role Family"] = top_modules["top_role_family"]

    top_modules = top_modules.rename(
        columns={
            "module_code": "Module",
            "module_title": "Title",
            "top_role_score": "Role Score",
        }
    )
    top_modules.insert(0, "Rank", range(1, len(top_modules) + 1))

    if "gap_type" in module_gap_summary.columns:
        under = module_gap_summary[module_gap_summary["gap_type"] == "undersupply"].copy()
    else:
        under = pd.DataFrame(
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
    top_3 = under.sort_values("gap_score", ascending=False).head(3)
    top_3 = top_3.rename(
        columns={
            "role_family": "Role Family Code",
            "role_family_name": "Role Family",
            "skill": "Skill",
            "demand_score": "Demand",
            "supply_score": "Supply",
            "gap_score": "Gap",
        }
    )
    top_3_cols = [col for col in ["Role Family Code", "Role Family", "Skill", "Demand", "Supply", "Gap"] if col in top_3.columns]
    if (
        "Role Family Code" in top_3.columns
        and "Role Family" in top_3.columns
        and top_3["Role Family Code"].astype(str).equals(top_3["Role Family"].astype(str))
    ):
        top_3_cols = [col for col in top_3_cols if col != "Role Family Code"]

    lines = [
        "# Policy Findings: Module-First Job Readiness",
        "",
        f"Generated on: {ts}",
        "",
        "## Why this matters",
        "This report highlights where module content aligns well with early-career degree-level jobs and where high-demand skills are under-covered.",
        "",
        "## Snapshot summary",
        f"- Modules scored: {len(module_summary)}",
        f"- Module-role rows: {len(module_role_scores)}",
        f"- Undersupply rows: {len(under)}",
        "",
        "## Top aligned modules",
        _markdown_table(
            top_modules,
            ["Rank", "Module", "Title", "Top Role Family", "Role Score"],
            n=12,
        ),
        "",
        "## Highest undersupply signals (overall)",
        _markdown_table(top_3, top_3_cols, n=3),
        "",
        "## Undersupply gaps by role family",
        _gap_sections_markdown(module_gap_summary),
        "",
        "## Recommended actions for stakeholders",
        "1. Curriculum teams: strengthen modules for skills with repeated high undersupply gaps.",
        "2. Career services: use module-role evidence to guide module pathway choices.",
        "3. Policy teams: prioritize intervention for role families showing persistent undersupply.",
        "",
        "## Notes on interpretation",
        "- High role score means stronger alignment to current job demand for that role family.",
        "- Low score in unrelated families often reflects specialization, not poor module quality.",
        "- Role families are curated role clusters derived from SSOC plus title/skill rules.",
        "- Pairwise module-job alignment is computed with BM25 and sentence embeddings fused by reciprocal rank fusion.",
    ]

    return "\n".join(lines)


def build_technical_report(
    diagnostics: Dict[str, float | str],
    config: PipelineConfig,
) -> str:
    lines = [
        "# Technical Report (Test2)",
        "",
        "## Objective",
        "Measure module-to-job readiness for early-career university-level jobs and provide explainable outputs for policy and curriculum use.",
        "",
        "## Final method choices",
        "- Scope filter: minimumYearsExperience <= 2 and ssecEqa = 70.",
        "- Role grouping: jobs keep SSOC-5 and SSOC-4 codes for auditability, but final reporting uses curated role clusters.",
        "- Skills governance: SkillsFuture Skills Framework mapping for technical skills, plus broad soft-skill inference from workload.",
        "- Fit computation: hybrid retrieval over title/description text plus extracted technical skills using BM25 and sentence embeddings.",
        "- Rank fusion: reciprocal rank fusion (RRF) combines BM25 and embedding rankings into one score.",
        "- Aggregation: top-K module-job matches are averaged within curated role clusters and support-adjusted by evidence count.",
        "",
        "## Formula summary",
        "- retrieval_text = normalized(title/description text + extracted technical skills)",
        "- RRFScore = RRF(rank_BM25(retrieval_text), rank_Embedding(retrieval_text))",
        "- SSOC5RawScore(module, ssoc5) = mean_topK(RRFScore, K)",
        "- SupportWeight(n) = n / (n + role_support_prior)",
        "- RoleClusterScore(module, cluster) = mean_topK(RRFScore within cluster, K) * SupportWeight(evidence_job_count)",
        "",
        "## End-to-end workflow",
        "1. Read source tables from PostgreSQL (`raw_jobs`, `raw_modules`, `skillsfuture_mapping`, `ssoc2024_definitions`).",
        "2. Filter jobs to early-career, degree-level scope (`minimumYearsExperience <= 2`, `ssecEqa = 70`).",
        "3. Assign each job to a curated role cluster using SSOC plus title/skill split rules.",
        "4. Prepare modules from NUSMods data and build module text from title/description/workload metadata.",
        "5. Normalize skills with alias rules + SkillsFuture mapping and split into technical vs transferable signals.",
        "6. Consolidate module variants (for example, suffix variants such as `ACC1701A/B`) into base module codes.",
        "7. Build retrieval text for jobs and modules, then compute BM25 and sentence-embedding similarities.",
        "8. Fuse BM25 and embedding rankings with Reciprocal Rank Fusion (RRF) to get module-job fit evidence.",
        "9. Aggregate top-K evidence into module-level scores by SSOC and curated role clusters with support weighting.",
        "10. Compute demand-supply skill gaps per role family and generate final outputs/reports.",
        "",
        "## Runtime diagnostics",
        "```json",
        str(diagnostics),
        "```",
        "",
        "## Default configuration",
        f"- top_k: {config.top_k}",
        f"- bm25_k1: {config.bm25_k1}",
        f"- bm25_b: {config.bm25_b}",
        f"- rrf_k: {config.rrf_k}",
        f"- retrieval_top_n: {config.retrieval_top_n}",
        f"- bm25_min_score: {config.bm25_min_score}",
        f"- bm25_relative_min: {config.bm25_relative_min}",
        f"- embedding_min_similarity: {config.embedding_min_similarity}",
        f"- embedding_relative_min: {config.embedding_relative_min}",
        f"- role_support_prior: {config.role_support_prior}",
        f"- embedding_model_name: {config.embedding_model_name}",
    ]
    return "\n".join(lines)


def build_plain_language_report() -> str:
    lines = [
        "# Plain-Language Justification (Test2)",
        "",
        "## What this project is doing",
        "We are checking whether individual university modules teach skills that employers ask for in real entry-level jobs.",
        "",
        "## What we filtered and why",
        "- We kept jobs requiring 0 to 2 years of experience so the data matches fresh graduates.",
        "- We used `ssecEqa = 70` to keep university-level jobs only.",
        "",
        "## Why we used SSOC",
        "SSOC gives a consistent occupation backbone. In this iteration, SSOC remains in the job evidence for traceability, but final reporting uses curated role clusters that better reflect job-skill similarity.",
        "",
        "## Why SkillsFuture is included",
        "SkillsFuture Skills Framework is used as a reference taxonomy so technical skills and transferable skills are labeled consistently. This reduces ad-hoc manual labeling.",
        "",
        "## How to read high and low scores",
        "- High score in a role family means the module retrieval text aligns strongly with employer demand in that family.",
        "- Low score in unrelated families usually means specialization, not poor module quality.",
        "",
        "## What stakeholders can do with results",
        "- Curriculum teams can identify missing high-demand skills.",
        "- Career teams can guide students to modules that align with target job families.",
        "- Policy teams can monitor systemic skill gaps by family.",
    ]
    return "\n".join(lines)


def write_reports(
    config: PipelineConfig,
    diagnostics: Dict[str, float | str],
    module_summary: pd.DataFrame,
    module_role_scores: pd.DataFrame,
    module_gap_summary: pd.DataFrame,
) -> None:
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    policy_brief = build_policy_brief(module_summary, module_role_scores, module_gap_summary)
    technical_report = build_technical_report(diagnostics, config)
    plain_report = build_plain_language_report()

    (config.output_dir / "policy_brief.md").write_text(policy_brief, encoding="utf-8")
    (config.reports_dir / "technical_report_test2.md").write_text(technical_report, encoding="utf-8")
    (config.reports_dir / "plain_language_justification_test2.md").write_text(
        plain_report,
        encoding="utf-8",
    )
