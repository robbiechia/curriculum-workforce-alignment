# System Architecture

## 1) Purpose
Provide a code-accurate architecture view of runtime components, boundaries, and data/control flow.

## 2) Intended audience
- Engineers who need to reason about system internals before making changes
- Reviewers validating technical correctness
- Presenters needing an end-to-end architecture narrative

## 3) When to read this
Read after [01-product-overview.md](./01-product-overview.md) and before [04-pipeline-deep-dive.md](./04-pipeline-deep-dive.md).

## 4) Prerequisites / assumed knowledge
- Python package/module layout
- Dataframe-based pipelines
- Basic retrieval/ranking concepts

## 5) High-level summary
The system is a single-repo Python pipeline with a DB-centric operating model:
1. Reads raw/source tables from external PostgreSQL
2. Runs deterministic staged transformations + retrieval scoring
3. Writes output tables to both CSV artifacts and DB tables
4. Serves a read-only dashboard over DB output tables

## 6) Main content sections

### 6.1 System context (external dependencies)
External systems:
- PostgreSQL (configured via `DATABASE_URL`) as canonical data source/sink
- NUSMods API during data setup (not during standard pipeline run if DB already populated)
- HuggingFace model assets for sentence-transformer embeddings (cold-cache dependent)

Local repository components:
- Pipeline scripts in `scripts/`
- Core package in `src/module_readiness/`
- Data setup tooling in `src/data_utils/`
- Streamlit dashboard in `streamlit_app.py`

### 6.2 Major runtime components
Pipeline orchestration:
- `run_pipeline(...)` in `src/module_readiness/orchestration/pipeline.py`
- Shared in-memory state via `ModuleReadinessState`

Ingestion:
- Jobs: `src/module_readiness/ingestion/jobs.py`
- Modules: `src/module_readiness/ingestion/modules.py`

Processing:
- Role clustering: `src/module_readiness/processing/role_families.py`
- Skill taxonomy: `src/module_readiness/processing/skill_taxonomy.py`
- Module variant consolidation: `src/module_readiness/processing/module_variants.py`

Retrieval:
- Artifacts + engine: `src/module_readiness/retrieval/engine.py`
- BM25 + embedding + fusion helpers under `src/module_readiness/retrieval/`

Analysis:
- Scoring: `analysis/scoring.py`
- Indicators + gaps: `analysis/aggregation.py`
- Degree outputs: `analysis/degrees.py`

Reporting:
- Markdown report generation: `reporting/reports.py`

User-facing query API:
- `api/query.py` over precomputed pipeline state

Dashboard:
- `streamlit_app.py` reads persisted DB output tables (read-only behavior)

### 6.3 Data flow vs control flow
Data flow (what data moves):
1. `raw_jobs`, `raw_modules`, taxonomy tables loaded from DB
2. Jobs/modules normalized and enriched
3. Retrieval text/tokens/embeddings built for jobs/modules
4. Module-job ranking evidence produced
5. Aggregated module and degree tables computed
6. Outputs persisted to CSV + DB tables

Control flow (what executes in what order):
1. Load config and rules
2. Ingest jobs
3. Assign role clusters
4. Ingest modules
5. Apply skill taxonomy
6. Consolidate module variants
7. Build retrieval artifacts and engine
8. Compute module scores
9. Build indicators and gaps
10. Build degree outputs
11. Persist outputs + diagnostics + reports
12. Return in-memory state

### 6.4 Synchronous vs asynchronous behavior
Synchronous:
- Main pipeline orchestration and scoring stages
- DB read/write operations during run

Concurrent/asynchronous-like behavior:
- NUSMods data setup script can fetch module details concurrently with thread pool (`scrape_nusmods.py`)

### 6.5 Persistence boundaries
Authoritative persistent boundary:
- External PostgreSQL tables via `data_utils.db_utils`

Secondary artifact boundary:
- CSV files under `outputs/` for inspection/export

Caching boundary:
- Embedding cache under `cache/embeddings/`

### 6.6 Integration boundaries and authority
Authority by concern:
- Raw source authority: DB raw tables (`raw_jobs`, `raw_modules`, etc.)
- Runtime computed authority: pipeline output tables (`module_*`, `degree_*`)
- Presentation/UI authority: Streamlit app reads output DB tables only

### 6.7 Failure boundaries
Potential failure zones:
- DB connectivity/auth failures
- Missing required source tables in DB
- Embedding model fetch/cold-cache network failures
- Schema drift in source tables
- Rule/config inconsistencies

### 6.8 Request lifecycle examples
Pipeline run lifecycle:
1. User runs `scripts/run_test2_pipeline.py`
2. Script calls `run_pipeline(...)`
3. Pipeline reads DB sources, computes outputs, persists tables, writes diagnostics/reports
4. Script prints diagnostics and sample query previews

Dashboard lifecycle:
1. User starts `streamlit run streamlit_app.py`
2. App checks required output tables in DB
3. App loads degree/module/skill-gap tables
4. User explores summaries and target-specific gap views

## 7) Key workflows or examples
Minimal architecture trace for "module recommendation from text query":
1. Query text enters `ModuleReadinessQueryAPI.recommend_relevant_modules(...)`
2. Retrieval engine ranks modules from text against prebuilt corpus artifacts
3. API enriches ranked module rows with module summary fields
4. Dataframe returned for display/consumption

## 8) Common pitfalls / gotchas
- Assuming dashboard triggers pipeline runs; it does not.
- Assuming CSV-only persistence; DB persistence is part of standard run path.
- Missing that retrieval candidates are thresholded before fusion (`bm25_*`, `embedding_*` mins).

## 9) Troubleshooting or FAQ
Q: Where should I start if output tables are missing in dashboard?  
A: Run pipeline first, then verify DB output tables listed in [03-data-contracts.md](./03-data-contracts.md).

Q: Where to modify ranking behavior?  
A: Retrieval thresholds and fusion settings in `config/pipeline_config.yaml` and retrieval engine/scoring modules.

## 10) Related documents / next reads
- Data schemas and contracts: [03-data-contracts.md](./03-data-contracts.md)
- Stage-level implementation: [04-pipeline-deep-dive.md](./04-pipeline-deep-dive.md)
- Formula-level calculations: [11-calculation-reference.md](./11-calculation-reference.md)
- Operating procedures: [05-operations-runbook.md](./05-operations-runbook.md)

## Explain this system in 5 minutes
Architecture is a DB-backed analytics pipeline with a read-only dashboard. The pipeline ingests raw jobs/modules/taxonomy from PostgreSQL, standardizes role and skill representations, then computes module-job relevance via BM25 + embedding similarity fused with reciprocal rank fusion. It aggregates this evidence into module-level and degree-level outputs, writes CSV snapshots for inspection, and persists final outputs back to DB. The dashboard reads those DB outputs directly and never recomputes the pipeline.

## 11) Source basis
- Verified from code/config:
  - `src/module_readiness/orchestration/pipeline.py`
  - `src/module_readiness/retrieval/engine.py`
  - `src/data_utils/db_utils.py`
  - `streamlit_app.py`
  - `config/pipeline_config.yaml`
- Inferred from implementation:
  - Authority/boundary framing of components for architectural communication
- Not verified / needs confirmation:
  - Formal production deployment topology beyond local/dashboard usage in repository

## Confidence rating
High

## Validation checklist
- [x] System context before internals
- [x] Data flow and control flow separated
- [x] Sync vs concurrent behavior identified
- [x] Persistence and failure boundaries documented
- [x] 5-minute summary included
