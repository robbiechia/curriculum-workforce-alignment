# Pipeline Deep Dive

## 1) Purpose
Provide a stage-by-stage, code-accurate explanation of how the pipeline works and where to modify it safely.

## 2) Intended audience
- Engineers implementing or reviewing pipeline changes
- Teammates who need technical depth for presentation defense

## 3) When to read this
Read after architecture and data contracts:
- [02-system-architecture.md](./02-system-architecture.md)
- [03-data-contracts.md](./03-data-contracts.md)

## 4) Prerequisites / assumed knowledge
- Python, pandas, and dataclass patterns
- Basic information retrieval concepts

## 5) High-level summary
The pipeline is a deterministic staged workflow orchestrated by `run_pipeline(...)` in `orchestration/pipeline.py`.  
It computes module alignment scores from retrieval evidence, then aggregates to degree-level outputs and persists both artifacts and DB tables.

## 6) Main content sections

### 6.1 Entry points and responsibilities
Primary script:
- `scripts/run_test2_pipeline.py`
  - Calls `run_pipeline(quick=...)`
  - Prints diagnostics and sample query previews

Primary orchestrator:
- `src/module_readiness/orchestration/pipeline.py`
  - Defines `ModuleReadinessState`
  - Executes stages in fixed order
  - Persists CSV + DB outputs
  - Writes markdown reports and diagnostics

### 6.2 Stage-by-stage execution

#### Stage A: config and rule load
Files:
- `config/settings.py`
- `config/pipeline_config.yaml`
- `config/role_family_rules.yaml`
- `config/role_clusters.yaml`
- `config/skill_aliases.yaml`

Responsibilities:
- Resolve paths and defaults
- Initialize runtime knobs
- Load role and skill mapping rules

#### Stage B: job ingestion
File:
- `ingestion/jobs.py`

Responsibilities:
- Read `raw_jobs` from DB
- Parse JSON-like skills/categories
- Normalize text and skill tokens
- Filter corpus by `experience_years <= exp_max` and `ssec_eqa == primary_ssec_eqa`

Key output:
- filtered jobs dataframe with normalized columns + diagnostics

#### Stage C: role family / cluster assignment
File:
- `processing/role_families.py`

Responsibilities:
- Parse SSOC codes into 5d and 4d prefixes
- Apply cluster mapping precedence:
  1. `ssoc5_exact_map`
  2. `ssoc4_exact_map`
  3. `split_rules` keyword/category filters
  4. legacy family fallback
  5. `Other`
- Enrich SSOC names using `ssoc2024_definitions` (fallback to inferred labels if missing)

Key output:
- jobs with `role_cluster`, `role_family`, `broad_family`, SSOC label columns

#### Stage D: module ingestion
File:
- `ingestion/modules.py`

Responsibilities:
- Read `raw_modules` from DB
- Select undergraduate modules only (code-level filter)
- Construct `module_text` from title/description/context/workload/profile seeds
- Attach metadata (`module_faculty`, `module_department`, credits, etc.)

Notes:
- Uses `module_prefix_profiles` from role rules for profile seeding

#### Stage E: skill taxonomy application
File:
- `processing/skill_taxonomy.py`

Responsibilities:
- Build alias map from `skill_aliases.yaml`
- Load `skillsfuture_mapping` from DB
- Split job skills into `technical_skills` and `soft_skills`
- Infer module technical mentions from controlled vocabulary
- Infer module soft skills from text cues + workload weak signals

#### Stage F: module variant consolidation
File:
- `processing/module_variants.py`

Responsibilities:
- Normalize module codes by removing trailing single uppercase letter
- Merge list fields (`technical_skills`, `soft_skills`) across variants
- Produce consolidated module rows

#### Stage G: retrieval artifacts and engine
Files:
- `retrieval/text.py`
- `retrieval/embeddings.py`
- `retrieval/fusion.py`
- `retrieval/engine.py`

Responsibilities:
- Build retrieval text:
  - jobs: `job_text` + technical skill section
  - modules: `module_title + module_description` + technical skill section
- Tokenize for BM25 corpus
- Build embeddings using sentence-transformer model
- Cache embeddings on disk
- Rank with BM25 + embedding + RRF fusion

Important ranking mechanics:
- Independent BM25/embedding candidate thresholding
- Reciprocal rank fusion with `rrf_k`
- Mode-specific ranking support (`hybrid`, `bm25`, `embedding`)

#### Stage H: scoring
File:
- `analysis/scoring.py`

Responsibilities:
- For each module, retrieve top matching jobs
- Build module-job evidence table with ranking metadata
- Aggregate to:
  - module-SSOC scores
  - module-role-cluster scores
- Apply support shrinkage:
  - `support_weight = n / (n + prior)`
  - `role_score = raw_role_score * support_weight`
- Apply near-tie rule preferring named role over `Other`

#### Stage I: module indicators and gap summary
File:
- `analysis/aggregation.py`

Responsibilities:
- Build one-row-per-module summary (`module_summary`)
- Build role-family skill demand-supply gaps (`module_gap_summary`)
- Optional SSOC4 fallback naming for display when top cluster is `Other`

#### Stage J: degree outputs
File:
- `analysis/degrees.py`

Responsibilities:
- Load degree mapping CSV
- Expand required modules into `degree_module_map`
- Build degree skill supply
- Aggregate degree role and occupation scores
- Build degree role/occupation skill-gap tables
- Build degree summary coverage/leaderboard table

#### Stage K: persistence and reporting
Files:
- `orchestration/pipeline.py`
- `reporting/reports.py`
- `data_utils/db_utils.py`

Responsibilities:
- Write output CSVs to `outputs/`
- Persist output tables to DB
- Persist diagnostics table to DB and JSON file
- Generate markdown reports:
  - `outputs/policy_brief.md`
  - `reports/technical_report_test2.md`
  - `reports/plain_language_justification_test2.md`

### 6.3 Major collaboration points between modules
- Ingestion -> processing:
  - jobs/modules normalized before role/skill enrichment
- Processing -> retrieval:
  - skill-tagged corpus feeds retrieval text creation
- Retrieval -> scoring:
  - ranking evidence becomes module-job evidence rows
- Scoring -> aggregation/degrees:
  - support-weighted scores become summary and gap products

### 6.4 Configuration and extension points
Common extension points:
- New role clusters/split rules: `config/role_clusters.yaml`
- Skill alias/governance updates: `config/skill_aliases.yaml` + `skillsfuture_mapping`
- Retrieval tuning: `pipeline_config.yaml` thresholds/model knobs
- Degree scope changes: `degree_mapping/degree_mapping_AY2425.csv`

### 6.5 Dangerous modification areas
- Changing module code normalization can invalidate degree map joins.
- Changing retrieval thresholds can silently alter candidate coverage and downstream score distributions.
- Changing output schema without updating DB persistence/dashboard contracts can break runtime UI.
- Removing/renaming role columns breaks query API and summary layers.

### 6.6 Error handling profile
Observed behavior:
- Most stages fail fast on missing dependencies/tables.
- Parquet writes have CSV fallback helper.
- Report generation is straightforward and assumes core outputs exist.

Limitations:
- No centralized retry policy for DB/network operations in main pipeline.
- Some test paths require external connectivity and may fail in restricted environments.

### 6.7 Testing strategy in repository
Test modules under `tests/` cover:
- ingestion filters and role assignment behavior
- skill taxonomy behavior
- scoring and aggregation behaviors
- retrieval evaluation mechanics
- query API behavior

Observed in restricted environment:
- DB/network dependent tests fail when external connections are blocked.

### 6.8 Known ambiguities and caveats
- In `ingestion/modules.py`, fallback branch references `config.focus_module_overrides`, which is not declared in `PipelineConfig` class.  
  Inferred impact: branch may raise error if reached (for example if module rows are empty), though not triggered in normal populated runs.

### 6.9 Reading Scores as Insights (Plain Language)
Use this as a quick interpretation guide when reading outputs:

- Evidence level (`module_job_evidence.csv`):
  - High `rrf_score` with both `bm25_rank` and `embedding_rank` present:
    - strong lexical + semantic match.
  - High `rrf_score` with one rank missing:
    - strong in one channel, weaker in the other.

- Module-target level (`module_role_scores.csv`, `module_ssoc5_scores.csv`):
  - `raw_role_score` tells you average alignment quality of supporting evidence.
  - `role_score` tells you alignment after support adjustment.
  - If `raw_role_score` is high but `role_score` drops a lot:
    - evidence set is small, so confidence is reduced.

- Degree-target level (`degree_role_scores.csv`, `degree_ssoc5_scores.csv`):
  - `degree_role_score` / `degree_ssoc5_score` combines contributor strength + contributor count.
  - `module_support_share` tells how much of matched curriculum supports that target.
  - High score with low support share:
    - concentrated strength in a small subset of modules.

- Skill gap tables (`module_gap_summary.csv`, `degree_*_skill_gaps.csv`):
  - Positive `gap_score` (`undersupply`):
    - demand share is higher than supply share.
  - Negative `gap_score` (`oversupply`):
    - supply share exceeds observed demand share.
  - These are relative demand-supply signals, not direct causal proof of curriculum quality.

## 7) Key workflows or examples
Example: "Add a new role cluster"
1. Edit `config/role_clusters.yaml`.
2. Run pipeline and inspect:
  - `job_role_map.csv`
  - `module_role_scores.csv`
  - `module_summary.csv`
3. Validate dashboard behavior for degree role views.

Example: "Tune retrieval strictness"
1. Adjust `bm25_*`, `embedding_*`, or `retrieval_top_n` in config.
2. Re-run pipeline.
3. Compare diagnostics and retrieval evaluation outputs.

## 8) Common pitfalls / gotchas
- Mistaking query-time ranking for fresh scoring; query API reads precomputed state.
- Editing rule files without documenting impact on comparability across runs.
- Ignoring consolidated module code effects in degree mapping analyses.

## 9) Troubleshooting or FAQ
Q: Why does quick mode produce different outputs?  
A: Quick mode changes corpus size and retrieval breadth for faster iteration, so results are not equivalent to full runs.

Q: Where do I debug missing DB output tables?  
A: Start in `_persist_output_tables_to_db(...)` inside `orchestration/pipeline.py`.

## 10) Related documents / next reads
- Data schemas: [03-data-contracts.md](./03-data-contracts.md)
- Formula-level math reference: [11-calculation-reference.md](./11-calculation-reference.md)
- Run/ops procedures: [05-operations-runbook.md](./05-operations-runbook.md)
- Debug playbook: [06-troubleshooting.md](./06-troubleshooting.md)

## Explain this system in 5 minutes
The pipeline loads jobs and modules from DB, assigns jobs to curated role clusters, and normalizes skill channels. It consolidates module variants, builds BM25 and embedding artifacts, then ranks module-job relevance with reciprocal rank fusion. It aggregates evidence into support-weighted module role and occupation scores, produces demand-supply skill gap tables, and rolls module outputs up to degree-level summaries via required-module mappings. Finally it writes CSV mirrors and persists all output tables back to DB for dashboard consumption.

## 11) Source basis
- Verified from code/config/tests:
  - `scripts/run_test2_pipeline.py`
  - `src/module_readiness/orchestration/pipeline.py`
  - `src/module_readiness/ingestion/*.py`
  - `src/module_readiness/processing/*.py`
  - `src/module_readiness/retrieval/*.py`
  - `src/module_readiness/analysis/*.py`
  - `src/module_readiness/reporting/reports.py`
  - `tests/test_*.py`
- Inferred from implementation:
  - "Dangerous modification areas" risk prioritization
  - Potential impact of undeclared `focus_module_overrides`
- Not verified / needs confirmation:
  - Any undocumented external orchestration outside this repository

## Confidence rating
High (with one medium-confidence caveat on undeclared fallback config branch)

## Validation checklist
- [x] Stage-by-stage flow is code-traceable
- [x] Extension points and risky areas identified
- [x] Testing strategy and limitations documented
- [x] 5-minute summary included
