# Data Contracts

## 1) Purpose
Document authoritative input/output data contracts, required tables, key columns, and assumptions for safe operation and modification.

## 2) Intended audience
- Engineers changing ingestion, scoring, or aggregation logic
- Engineers debugging DB/table issues
- Reviewers validating schema assumptions

## 3) When to read this
Read before modifying pipeline stages or dashboard table usage.

## 4) Prerequisites / assumed knowledge
- SQL table basics
- pandas dataframe conventions
- Familiarity with [02-system-architecture.md](./02-system-architecture.md)

## 5) High-level summary
Canonical source of truth is the live PostgreSQL database.  
Pipeline stages read source tables and produce output tables that are written back to DB and mirrored to CSV.

## 6) Main content sections

### 6.1 Source input contracts (DB)

#### `raw_jobs`
Purpose:
- Source postings corpus for job-side retrieval and demand signals

Required fields used in code:
- `job_id`
- `title`
- `description`
- `skills` (JSON string or list)
- `categories` (JSON string or list)
- `min_experience_years`
- `ssec_eqa`
- `ssoc_code`
- `company_name`
- `salary_min`, `salary_max`, `salary_type`
- `posted_at`, `deleted_at`

Transformation contract:
- Pipeline normalizes text (`title_clean`, `description_clean`, `job_text`)
- Filters to early-career and degree-level scope

#### `raw_modules`
Purpose:
- Source module catalog and metadata

Required fields used in code:
- `moduleCode`
- `title`
- `description`
- `additionalInformation`
- `prerequisite`, `preclusion`
- `department`, `faculty`
- `moduleCredit`
- `workload` (JSON string/list)
- `acadYear` (if present)

Transformation contract:
- Undergraduate-only filter (module levels 1-4 by code pattern)
- Module text assembly for retrieval

#### `skillsfuture_mapping`
Purpose:
- Skill normalization channel map (`technical` vs `transferable`)

Required fields:
- `skill_norm`
- `channel`
- `framework_cluster`
- `skillsfuture_note`

#### `ssoc2024_definitions`
Purpose:
- SSOC code-to-title lookup

Required fields:
- `SSOC 2024`
- `SSOC 2024 Title`

Usage:
- Enrich `ssoc_4d_name` and `ssoc_5d_name`

### 6.2 Configuration contracts
Primary config file:
- `config/pipeline_config.yaml`

High-impact parameters:
- Job filter scope: `exp_max`, `primary_ssec_eqa`
- Retrieval behavior: `bm25_*`, `embedding_*`, `rrf_k`, `retrieval_top_n`
- Aggregation behavior: `top_k`, `role_support_prior`
- Degree behavior: `degree_support_prior`, `degree_role_top_n`, `degree_demand_skill_top_n`
- Paths: mapping/rule files and output directories

Rule/config files:
- `config/role_family_rules.yaml`
- `config/role_clusters.yaml`
- `config/skill_aliases.yaml`
- `degree_mapping/degree_mapping_AY2425.csv`

### 6.3 Intermediate and output contracts

#### Core cleaned/intermediate tables
- `jobs_clean`
  - Includes normalized job text, clustered role labels, parsed skill channels
- `modules_clean`
  - Includes normalized module rows after variant consolidation and skill inference
- `job_role_map`
  - Compact role/SSOC mapping export per job

#### Module-level output tables
- `module_job_evidence`
  - One row per module-job evidence match with retrieval scores
  - Key columns: `module_code`, `job_id`, `role_family`, `ssoc_5d`, `bm25_score`, `embedding_score`, `rrf_score`
- `module_ssoc5_scores`
  - Aggregated module-to-SSOC(4d/5d) scores
  - Key columns: `module_code`, `ssoc_4d`, `ssoc_5d`, `role_score`, `support_weight`, `evidence_job_count`
- `module_role_scores`
  - Aggregated module-to-role-cluster scores
  - Key columns: `module_code`, `role_family`, `broad_family`, `role_score`, `support_weight`, `selection_score`
- `module_summary`
  - One row per module with top role labels/scores
  - Key columns: `module_code`, `top_role_cluster`, `top_role_family_name`, `top_role_score`
- `module_gap_summary`
  - Role-family skill demand-supply deltas
  - Key columns: `role_family`, `skill`, `demand_score`, `supply_score`, `gap_score`, `gap_type`
- `top10_by_role_family`
  - Convenience ranking table for top modules per role family

#### Degree-level output tables
- `degree_module_map`
  - Expanded degree-to-required-module rows with module matching metadata
- `degree_skill_supply`
  - Degree-level technical skill supply derived from matched required modules
- `degree_role_scores`
  - Degree-level role-cluster alignment scores
- `degree_ssoc5_scores`
  - Degree-level occupation alignment scores
- `degree_role_skill_gaps`
  - Degree vs role-cluster skill-demand gaps
- `degree_ssoc5_skill_gaps`
  - Degree vs occupation skill-demand gaps
- `degree_summary`
  - One-row degree summary including match coverage and top aligned targets

#### Operational table
- `diagnostics`
  - Single-row run diagnostics persisted to DB

### 6.4 File output mirrors
CSV files under `outputs/` mirror output tables.  
Dashboard should rely on DB tables, not CSV files, for canonical runtime consumption.

### 6.5 Dashboard table contract
`streamlit_app.py` requires the following DB tables to exist:
- `degree_summary`
- `degree_module_map`
- `degree_skill_supply`
- `degree_role_scores`
- `degree_ssoc5_scores`
- `degree_role_skill_gaps`
- `degree_ssoc5_skill_gaps`

### 6.6 Data quality and assumptions
- Source schema shape is assumed stable for required fields.
- JSON-like columns may arrive as stringified JSON or native lists; ingestion code handles both.
- Role and skill labels are rule-driven and can shift when config files change.
- Module code normalization removes trailing single uppercase letter during variant consolidation.

### 6.7 Change impact guidance
If changing:
- Source ingestion fields: update ingestion modules + tests + this contract doc.
- Output schema: update pipeline persistence list + dashboard + query/report consumers.
- Config parameters: update runbook and deep-dive docs.

## 7) Key workflows or examples
Schema drift check workflow:
1. Validate source tables (`raw_jobs`, `raw_modules`, `skillsfuture_mapping`, `ssoc2024_definitions`) in DB.
2. Run pipeline and inspect `outputs/diagnostics.json`.
3. Confirm required dashboard tables exist and are populated.

## 8) Common pitfalls / gotchas
- Assuming table names from CSV filenames without checking DB persistence list.
- Forgetting that list-valued columns may be JSON strings after DB round-trips.
- Confusing `module_code` raw variants with consolidated base codes.

## 9) Troubleshooting or FAQ
Q: Which artifact is canonical: CSV or DB table?  
A: DB table is canonical for runtime. CSV is inspection/export mirror.

Q: Why are there fewer modules in retrieval than in module ingest diagnostics?  
A: Variant consolidation reduces module count before retrieval/scoring.

## 10) Related documents / next reads
- Pipeline internals: [04-pipeline-deep-dive.md](./04-pipeline-deep-dive.md)
- Formula-level calculations: [11-calculation-reference.md](./11-calculation-reference.md)
- Operations: [05-operations-runbook.md](./05-operations-runbook.md)
- Debugging: [06-troubleshooting.md](./06-troubleshooting.md)

## Explain this system in 5 minutes
This project reads four source DB tables (`raw_jobs`, `raw_modules`, `skillsfuture_mapping`, `ssoc2024_definitions`), transforms and scores module-job alignment, then writes a family of module-level and degree-level output tables back to DB. CSV files are generated as mirrors for inspection. The Streamlit dashboard depends on a subset of degree-level output tables and fails early if they are missing.

## 11) Source basis
- Verified from code/config/artifacts:
  - `src/data_utils/db_utils.py`
  - `src/module_readiness/ingestion/*.py`
  - `src/module_readiness/orchestration/pipeline.py`
  - `src/module_readiness/analysis/*.py`
  - `streamlit_app.py`
  - `outputs/*.csv`, `outputs/diagnostics.json`
- Inferred from implementation:
  - Change-impact guidance prioritization
- Not verified / needs confirmation:
  - DB-level constraints/indexes outside repository-managed SQL

## Confidence rating
High

## Validation checklist
- [x] Source and output table contracts documented
- [x] Required dashboard tables listed
- [x] Canonical DB vs CSV mirror distinction explicit
- [x] Change impact guidance included
