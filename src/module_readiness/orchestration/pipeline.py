from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
from loguru import logger
import pandas as pd

from data_utils.db_utils import get_engine, read_table, write_logged_table
from ..analysis import (
    AggregationResult,
    DegreeAggregationResult,
    ScoringResult,
    build_degree_outputs,
    build_indicators,
    compute_scores,
)
from ..config import PipelineConfig, read_yaml_json, write_dataframe_with_fallback
from ..ingestion.extract_preclusions import build_preclusions
from ..ingestion import JobsIngestResult, ModulesIngestResult, load_jobs, load_nus_modules
from ..processing import (
    RoleFamilyResult,
    assign_role_families,
    apply_skill_taxonomy,
    consolidate_module_variants,
    load_skill_aliases,
)
from ..reporting import write_reports
from ..retrieval import HybridRetrievalEngine, RetrievalArtifacts, build_retrieval_artifacts


@dataclass
class ModuleReadinessState:
    # This is the shared in-memory view of the whole pipeline run. Scripts, query APIs,
    # and the Streamlit app all read from this state instead of recomputing results.
    config: PipelineConfig
    role_rules: Dict[str, object]

    jobs: pd.DataFrame
    modules: pd.DataFrame

    module_job_scores: pd.DataFrame
    module_ssoc5_scores: pd.DataFrame
    module_role_scores: pd.DataFrame
    module_summary: pd.DataFrame
    module_gap_summary: pd.DataFrame

    retrieval_artifacts: RetrievalArtifacts
    retrieval: HybridRetrievalEngine
    skill_channel_map: Dict[str, str]
    known_skills: list[str]

    diagnostics: Dict[str, float | str]
    degree_module_map: pd.DataFrame = field(default_factory=pd.DataFrame)
    degree_requirement_buckets: pd.DataFrame = field(default_factory=pd.DataFrame)
    degree_skill_supply: pd.DataFrame = field(default_factory=pd.DataFrame)
    degree_role_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    degree_ssoc5_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    degree_role_skill_gaps: pd.DataFrame = field(default_factory=pd.DataFrame)
    degree_ssoc5_skill_gaps: pd.DataFrame = field(default_factory=pd.DataFrame)
    degree_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    degree_plan_expansion_audit: pd.DataFrame = field(default_factory=pd.DataFrame)


def _build_job_role_map_frame(jobs: pd.DataFrame) -> pd.DataFrame:
    jobs_export = _build_jobs_clean_frame(jobs)
    role_cols = [
        "job_id",
        "title",
        "role_cluster",
        "role_cluster_source",
        "broad_family",
        "ssoc_code",
        "ssoc_4d",
        "ssoc_4d_name",
        "ssoc_5d",
        "ssoc_5d_name",
        "primary_category",
    ]
    role_cols = [col for col in role_cols if col in jobs_export.columns]
    return jobs_export.reindex(columns=role_cols).copy()


def _build_jobs_clean_frame(jobs: pd.DataFrame) -> pd.DataFrame:
    return jobs.drop(
        columns=[
            "role_family",
            "role_family_name",
            "broad_family_name",
            "role_family_source",
        ],
        errors="ignore",
    ).copy()


def _save_intermediate_tables(
    config: PipelineConfig,
    jobs: pd.DataFrame,
    modules: pd.DataFrame,
    diagnostics: Dict[str, float | str],
) -> None:
    jobs_export = _build_jobs_clean_frame(jobs)
    role_map = _build_job_role_map_frame(jobs_export)

    jobs_path, jobs_parquet = write_dataframe_with_fallback(
        jobs_export,
        config.output_dir / "jobs_clean.parquet",
    )
    modules_path, modules_parquet = write_dataframe_with_fallback(
        modules,
        config.output_dir / "modules_clean.parquet",
    )
    role_path, role_parquet = write_dataframe_with_fallback(
        role_map,
        config.output_dir / "job_role_map.parquet",
    )

    # Keep CSV snapshots in sync for easy inspection.
    jobs_export.to_csv(config.output_dir / "jobs_clean.csv", index=False)
    modules.to_csv(config.output_dir / "modules_clean.csv", index=False)
    role_map.to_csv(config.output_dir / "job_role_map.csv", index=False)

    diagnostics["jobs_clean_written_path"] = str(jobs_path)
    diagnostics["modules_clean_written_path"] = str(modules_path)
    diagnostics["job_role_map_written_path"] = str(role_path)
    diagnostics["jobs_clean_parquet_written"] = "yes" if jobs_parquet else "no"
    diagnostics["modules_clean_parquet_written"] = "yes" if modules_parquet else "no"
    diagnostics["job_role_map_parquet_written"] = "yes" if role_parquet else "no"


def _persist_output_tables_to_db(
    diagnostics: Dict[str, float | str],
    *,
    jobs: pd.DataFrame,
    modules: pd.DataFrame,
    module_job_scores: pd.DataFrame,
    module_ssoc5_scores: pd.DataFrame,
    module_role_scores: pd.DataFrame,
    module_gap_summary: pd.DataFrame,
    module_summary: pd.DataFrame,
    degree_module_map: pd.DataFrame,
    degree_requirement_buckets: pd.DataFrame,
    degree_skill_supply: pd.DataFrame,
    degree_role_scores: pd.DataFrame,
    degree_ssoc5_scores: pd.DataFrame,
    degree_role_skill_gaps: pd.DataFrame,
    degree_ssoc5_skill_gaps: pd.DataFrame,
    degree_summary: pd.DataFrame,
    degree_plan_expansion_audit: pd.DataFrame,
    top10_by_role_family: pd.DataFrame,
    persist_degree_outputs_to_db: bool,
) -> None:
    # persist_degree_outputs_to_db controls both the core module tables and the degree
    # tables. When False the pipeline runs fully local with no DB connection attempted.
    diagnostics["db_outputs_persisted"] = "yes" if persist_degree_outputs_to_db else "no"
    diagnostics["db_degree_outputs_persisted"] = "yes" if persist_degree_outputs_to_db else "no"
    if not persist_degree_outputs_to_db:
        diagnostics["db_output_tables_count"] = 0.0
        return

    engine = get_engine()
    jobs_export = _build_jobs_clean_frame(jobs)
    output_tables = {
        "jobs_clean": jobs_export,
        "modules_clean": modules,
        "job_role_map": _build_job_role_map_frame(jobs_export),
        "module_job_evidence": module_job_scores,
        "module_ssoc5_scores": module_ssoc5_scores,
        "module_role_scores": module_role_scores,
        "module_gap_summary": module_gap_summary,
        "module_summary": module_summary,
        "top10_by_role_family": top10_by_role_family,
        "degree_requirement_buckets": degree_requirement_buckets,
        "degree_module_map": degree_module_map,
        "degree_skill_supply": degree_skill_supply,
        "degree_role_scores": degree_role_scores,
        "degree_ssoc5_scores": degree_ssoc5_scores,
        "degree_role_skill_gaps": degree_role_skill_gaps,
        "degree_ssoc5_skill_gaps": degree_ssoc5_skill_gaps,
        "degree_summary": degree_summary,
    }
    if not degree_plan_expansion_audit.empty:
        output_tables["degree_plan_expansion_audit"] = degree_plan_expansion_audit
    diagnostics["db_output_tables_count"] = float(len(output_tables) + 1)
    output_tables["diagnostics"] = pd.DataFrame([diagnostics])

    for table_name, frame in output_tables.items():
        write_logged_table(frame, table_name=table_name, engine=engine)


def _pipeline_progress_logger(total_steps: int):
    current_step = 0

    def step(label: str) -> None:
        nonlocal current_step
        current_step += 1
        logger.info(f"[{current_step}/{total_steps}] {label}")

    return step


def _ensure_module_preclusions(config: PipelineConfig) -> tuple[Path, bool]:
    output_path = config.output_dir / "module_preclusions.csv"
    if output_path.exists():
        return output_path, False

    raw_modules = read_table("raw_modules")
    if raw_modules.empty:
        raise RuntimeError("Cannot build module_preclusions.csv because raw_modules is empty.")

    required_cols = ["moduleCode", "preclusionRule", "preclusion"]
    missing = [col for col in required_cols if col not in raw_modules.columns]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            f"Cannot build module_preclusions.csv because raw_modules is missing columns: {missing_list}"
        )

    preclusions_df = build_preclusions(raw_modules[required_cols].copy())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preclusions_df.to_csv(output_path, index=False)
    return output_path, True


def run_pipeline(
    config_path: Path | None = None,
    quick: bool = False,
    config: "PipelineConfig | None" = None,
) -> ModuleReadinessState:
    if config is None:
        config = PipelineConfig.from_file(config_path)
    if quick:
        # Quick mode keeps the workflow the same, but dials down the corpus size and
        # retrieval breadth so developers can iterate faster.
        config.top_k = min(30, int(config.top_k))
        if config.nusmods_max_modules is None or int(config.nusmods_max_modules) > 350:
            config.nusmods_max_modules = 350
        config.nusmods_fetch_workers = min(8, int(config.nusmods_fetch_workers))
        config.retrieval_top_n = min(max(int(config.retrieval_top_n), 120), 240)
    logger.info(f"Running pipeline with config: {config}")
    progress = _pipeline_progress_logger(total_steps=19)
    progress("Reading config and YAML files")
    role_rules = read_yaml_json(config.role_rules_file)
    cluster_rules = read_yaml_json(config.role_clusters_file)
    skill_aliases = load_skill_aliases(config.skill_aliases_file)
    # The runtime path is: load raw-ish DB tables -> normalize/enrich -> build retrieval
    # artifacts -> score -> aggregate -> export app-facing outputs.
    progress("Loading jobs")
    jobs_result: JobsIngestResult = load_jobs(config, skill_aliases)
    progress("Assigning role families")
    role_result: RoleFamilyResult = assign_role_families(
        jobs_result.jobs,
        role_rules,
        cluster_rules=cluster_rules,
    )
    progress("Loading NUS modules")
    modules_result: ModulesIngestResult = load_nus_modules(config, role_rules)
    progress("Ensuring module preclusions output")
    preclusions_path, preclusions_generated = _ensure_module_preclusions(config)
    progress("Applying skill taxonomy")
    taxonomy_result = apply_skill_taxonomy(
        config,
        role_rules,
        role_result.jobs,
        modules_result.modules,
    )
    # Consolidate module variants (e.g. ACC1701A/B/C/D -> ACC1701) so retrieval,
    # degree mappings, and output tables all speak the same module-code language.
    progress("Consolidating module variants")
    consolidated_modules = consolidate_module_variants(taxonomy_result.modules)
    progress("Building retrieval artifacts")
    retrieval_artifacts = build_retrieval_artifacts(
        config=config,
        jobs=taxonomy_result.jobs,
        modules=consolidated_modules,
    )
    progress("Building retrieval engine")
    retrieval = HybridRetrievalEngine(config=config, artifacts=retrieval_artifacts)
    progress("Computing scores")
    scoring_result: ScoringResult = compute_scores(
        config=config,
        jobs=taxonomy_result.jobs,
        modules=consolidated_modules,
        retrieval=retrieval,
    )
    progress("Building indicators")
    aggregation_result: AggregationResult = build_indicators(
        jobs=taxonomy_result.jobs,
        modules=consolidated_modules,
        module_role_scores=scoring_result.module_role_scores,
        role_rules=role_rules,
        module_ssoc5_scores=scoring_result.module_ssoc5_scores,
    )
    progress("Building degree outputs")
    # Degree outputs are an aggregation layer over the module outputs; they do not rerun
    # ingestion or retrieval.
    degree_result: DegreeAggregationResult = build_degree_outputs(
        config=config,
        jobs=taxonomy_result.jobs,
        modules=consolidated_modules,
        module_summary=aggregation_result.module_summary,
        module_role_scores=aggregation_result.module_role_scores,
        module_ssoc5_scores=scoring_result.module_ssoc5_scores,
        raw_modules=taxonomy_result.modules,
    )
    progress("Building diagnostics")
    diagnostics: Dict[str, float | str] = {}
    diagnostics.update(jobs_result.diagnostics)
    diagnostics.update(role_result.diagnostics)
    diagnostics.update(modules_result.diagnostics)
    diagnostics.update(taxonomy_result.diagnostics)
    diagnostics.update(retrieval_artifacts.diagnostics)
    diagnostics.update(scoring_result.diagnostics)
    diagnostics.update(aggregation_result.diagnostics)
    diagnostics.update(degree_result.diagnostics)
    diagnostics["module_preclusions_path"] = str(preclusions_path)
    diagnostics["module_preclusions_generated"] = "yes" if preclusions_generated else "no"
    progress("Writing scoring and degree CSV outputs")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # These CSVs are the main hand-off artifacts for reports, dashboards, and any
    # future web application.
    scoring_result.module_job_scores.to_csv(config.output_dir / "module_job_evidence.csv", index=False)
    aggregation_result.module_role_scores.to_csv(
        config.output_dir / "module_role_scores.csv", index=False
    )
    aggregation_result.module_summary.to_csv(config.output_dir / "module_summary.csv", index=False)
    degree_result.degree_requirement_buckets.to_csv(
        config.output_dir / "degree_requirement_buckets.csv", index=False
    )
    degree_result.degree_module_map.to_csv(config.output_dir / "degree_module_map.csv", index=False)
    degree_result.degree_skill_supply.to_csv(config.output_dir / "degree_skill_supply.csv", index=False)
    degree_result.degree_summary.to_csv(config.output_dir / "degree_summary.csv", index=False)
    if not degree_result.degree_plan_expansion_audit.empty:
        degree_result.degree_plan_expansion_audit.to_csv(
            config.output_dir / "degree_plan_expansion_audit.csv", index=False
        )
    progress("Writing clean intermediate outputs")
    _save_intermediate_tables(
        config=config,
        jobs=taxonomy_result.jobs,
        modules=consolidated_modules,
        diagnostics=diagnostics,
    )
    progress("Skipping database persistence")
    diagnostics["db_outputs_persisted"] = "no"
    diagnostics["db_degree_outputs_persisted"] = "no"
    diagnostics["db_output_tables_count"] = 0.0
    progress("Writing reports")
    write_reports(
        config=config,
        diagnostics=diagnostics,
        module_summary=aggregation_result.module_summary,
        module_role_scores=aggregation_result.module_role_scores,
        module_gap_summary=aggregation_result.module_gap_summary,
    )
    progress("Writing diagnostics")
    (config.output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )
    progress("Returning pipeline state")
    return ModuleReadinessState(
        config=config,
        role_rules=role_rules,
        jobs=taxonomy_result.jobs,
        modules=consolidated_modules,
        module_job_scores=scoring_result.module_job_scores,
        module_ssoc5_scores=scoring_result.module_ssoc5_scores,
        module_role_scores=aggregation_result.module_role_scores,
        module_summary=aggregation_result.module_summary,
        module_gap_summary=aggregation_result.module_gap_summary,
        retrieval_artifacts=retrieval_artifacts,
        retrieval=retrieval,
        skill_channel_map=taxonomy_result.skill_channel_map,
        known_skills=sorted(
            set(taxonomy_result.skill_channel_map.keys())
            | set(sum(taxonomy_result.jobs["technical_skills"].tolist(), []))
            | set(sum(taxonomy_result.jobs["soft_skills"].tolist(), []))
        ),
        diagnostics=diagnostics,
        degree_requirement_buckets=degree_result.degree_requirement_buckets,
        degree_module_map=degree_result.degree_module_map,
        degree_skill_supply=degree_result.degree_skill_supply,
        degree_summary=degree_result.degree_summary,
        degree_plan_expansion_audit=degree_result.degree_plan_expansion_audit,
    )
