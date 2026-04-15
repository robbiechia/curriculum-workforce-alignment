# Important Files Directory (Test2 Repo)

This guide is a quick map of the important files in this repository.
Use it to know where to look depending on what you are trying to do.

Generated on: 2026-03-29

## Quick Start

If you only read five files first:

1. `README.md` - project overview and how to run.
2. `src/module_readiness/pipeline.py` - end-to-end orchestration.
3. `outputs/module_summary.csv` - top-level module outcomes.
4. `outputs/module_ssoc5_scores.csv` - module scoring at SSOC-5 granularity.
5. `reports/technical_report_test2.md` - technical method summary.

## Pipeline Map

`config/*` -> `scripts/run_test2_pipeline.py` -> `src/module_readiness/*` -> `outputs/*` -> `reports/*`

## Root Files

| File | Importance | What It Is | When To Open |
| --- | --- | --- | --- |
| `README.md` | Critical | Main project documentation and usage notes. | Start here for setup, context, and run commands. |
| `requirements.txt` | Important | Python dependency list. | Installing or reproducing environment. |
| `.gitignore` | Important | Git ignore rules for generated/cache files. | Checking what should or should not be committed. |
| `CHANGELOG.md` | Reference | Change history notes. | Reviewing historical updates. |
| `MODULE_CONSOLIDATION_REPORT.md` | Reference | Notes on module variant consolidation behavior. | Understanding ACC/CS variant merge choices. |

## Config Files

| File | Importance | What It Is | When To Open |
| --- | --- | --- | --- |
| `config/pipeline_config.yaml` | Critical | Runtime parameters (weights, top_k, paths). | Changing scoring behavior or run settings. |
| `config/role_family_rules.yaml` | Critical | Role assignment rules and mappings. | Adjusting occupation grouping logic. |
| `config/skill_aliases.yaml` | Important | Skill normalization aliases. | Fixing inconsistent skill labels. |
| `config/skillsfuture_mapping.csv` | Critical | Skill channel mapping (technical/transferable). | Reviewing skill taxonomy behavior. |
| `config/ssoc2024-detailed-definitions.xlsx` | Critical | Official SSOC code to title mapping (4d/5d). | Validating readable SSOC names in outputs. |
| `config/jobsandskills-skillsfuture-unique-skills-list.xlsx` | Important | SkillsFuture unique skills reference file. | Checking raw skill reference content. |
| `config/jobsandskills-skillsfuture-tsc-to-unique-skills-mapping.xlsx` | Important | TSC to unique skills mapping reference. | Auditing skill mapping provenance. |
| `config/jobsandskills-skillsfuture-skills-framework-dataset.xlsx` | Important | Skills Framework dataset reference. | Deep taxonomy traceability checks. |

## Scripts

| File | Importance | What It Is | When To Open |
| --- | --- | --- | --- |
| `scripts/run_test2_pipeline.py` | Critical | Main script to run the full pipeline. | Generating fresh outputs/reports. |
| `scripts/run_test2_queries.py` | Important | Query/demo script for searching and recommending. | Sanity-checking results interactively. |

## Source Code (`src/module_readiness`)

| File | Importance | What It Is | When To Open |
| --- | --- | --- | --- |
| `src/module_readiness/__init__.py` | Reference | Package exports and public entrypoints. | Seeing what APIs are exposed. |
| `src/module_readiness/settings.py` | Critical | Config schema, loading, and path resolution. | Changing config defaults and IO behavior. |
| `src/module_readiness/pipeline.py` | Critical | End-to-end pipeline orchestration and output writing. | Understanding complete execution flow. |
| `src/module_readiness/ingest_jobs.py` | Critical | Job data ingestion, filtering, and preprocessing. | Investigating job sample composition. |
| `src/module_readiness/ingest_nusmods.py` | Critical | NUS module ingestion and module metadata assembly. | Debugging module inputs and cache/API behavior. |
| `src/module_readiness/role_families.py` | Critical | SSOC parsing and role family assignment, name mapping. | Occupation grouping and SSOC naming issues. |
| `src/module_readiness/skill_taxonomy.py` | Critical | Skill channel assignment into technical vs transferable. | Skill classification logic checks. |
| `src/module_readiness/features.py` | Critical | Vectorization and feature artifact construction. | Text/similarity feature engineering checks. |
| `src/module_readiness/scoring.py` | Critical | Module-job scoring, SSOC-5 aggregation, SSOC-4 roll-up. | Core score formula/debugging. |
| `src/module_readiness/aggregation.py` | Critical | Higher-level summaries and gap indicators. | Interpreting summary and gap outputs. |
| `src/module_readiness/reporting.py` | Important | Markdown report generation for policy/technical docs. | Editing report wording/structure. |
| `src/module_readiness/query_api.py` | Important | Query interface for search and recommendation views. | Building demos and result exploration tools. |
| `src/module_readiness/consolidate_variants.py` | Important | Consolidates module variants into parent codes. | Reviewing module deduplication effects. |

## Outputs (`outputs`)

These are generated artifacts. Re-run pipeline if they look stale.

| File | Importance | What It Is | When To Open |
| --- | --- | --- | --- |
| `outputs/diagnostics.json` | Critical | Run diagnostics, row counts, and key pipeline stats. | First check after each run for sanity. |
| `outputs/jobs_clean.csv` | Important | Cleaned jobs table with derived columns. | Auditing post-filter job data. |
| `outputs/jobs_clean.parquet` | Important | Same as `jobs_clean.csv` in parquet format. | Fast loading for analysis workflows. |
| `outputs/job_role_map.csv` | Critical | Job to role-family/SSOC mapping with readable names. | Validating occupation assignment quality. |
| `outputs/job_role_map.parquet` | Important | Same mapping table in parquet format. | High-performance downstream analysis. |
| `outputs/modules_clean.csv` | Important | Cleaned module catalog with skill channels. | Auditing module-side inputs. |
| `outputs/modules_clean.parquet` | Important | Same module table in parquet format. | Fast loading for analysis workflows. |
| `outputs/module_job_evidence.csv` | Critical | Evidence-level module x job scoring rows. | Tracing why a module received its score. |
| `outputs/module_ssoc5_scores.csv` | Critical | Aggregated module x SSOC-5 scoring table. | Detailed occupation-level outcome review. |
| `outputs/module_role_scores.csv` | Critical | Final module x SSOC-4 role-family scoring table. | Main role-family ranking analysis. |
| `outputs/module_summary.csv` | Critical | One-row-per-module top role summary. | High-level module prioritization. |
| `outputs/module_gap_summary.csv` | Critical | Skill demand-supply gaps by role family. | Curriculum and policy action planning. |
| `outputs/top10_by_role_family.csv` | Important | Top modules within each role family. | Quick per-family shortlist view. |
| `outputs/policy_brief.md` | Important | Auto-generated policy-facing markdown brief. | Sharing concise findings with stakeholders. |

## Reports (`reports`)

| File | Importance | What It Is | When To Open |
| --- | --- | --- | --- |
| `reports/technical_report_test2.md` | Critical | Technical method and diagnostic summary for current run. | Method validation and technical review. |
| `reports/plain_language_justification_test2.md` | Critical | Non-technical rationale of approach and interpretation. | Communicating to non-technical audiences. |
| `reports/technical_report_deep_dive.md` | Important | Deep mathematical/feature explanation write-up. | Detailed model mechanics and formula review. |
| `reports/one_page_teammate_guide.md` | Important | Quick onboarding guide for teammates. | Fast orientation for new collaborators. |
| `reports/app_report.md` | Reference | App/reporting narrative artifact. | Reviewing app-facing explanation content. |

## Tests (`tests`)

| File | Importance | What It Is | When To Open |
| --- | --- | --- | --- |
| `tests/test_ingest_jobs.py` | Critical | Validates job ingestion/filtering behavior. | Checking data eligibility regressions. |
| `tests/test_scoring.py` | Critical | Validates scoring formula and weighting behavior. | Checking score correctness after changes. |
| `tests/test_skill_taxonomy.py` | Critical | Validates skill aliasing and channel split behavior. | Checking taxonomy consistency regressions. |
| `tests/test_behavioral_sanity.py` | Important | Behavioral guardrail tests on output patterns. | Catching unintended ranking shifts. |

## How To Use This Guide

- If score values look wrong, start with `scoring.py`, then inspect `module_job_evidence.csv` and `module_ssoc5_scores.csv`.
- If occupation labels look wrong, inspect `role_families.py`, `config/ssoc2024-detailed-definitions.xlsx`, and `job_role_map.csv`.
- If skill gaps look odd, inspect `skill_taxonomy.py`, `aggregation.py`, and `module_gap_summary.csv`.
