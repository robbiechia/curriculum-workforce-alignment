# Pipeline Workflow Diagram

This diagram is based on the actual repository flow in:

- `src/data_utils/` for source ingestion and database loading
- `src/module_readiness/orchestration/pipeline.py` for the main runtime pipeline
- `src/module_readiness/reporting/reports.py` and `streamlit_app.py` for downstream outputs

## End-to-End Workflow

```mermaid
flowchart TD
    subgraph Sources["External and local data sources"]
        MSF["MyCareersFuture JSON files"]
        NUS["NUSMods API or cached JSON"]
        SF["SkillsFuture Excel files"]
        SSOC["SSOC 2024 Excel file"]
        DEG["Degree mapping CSV"]
        CFG["Role, cluster, and skill alias configs"]
    end

    subgraph Setup["Data setup layer"]
        NUSSCRAPE["scrape_nusmods.py"]
        SFMAP["generate_skillsfuture_mapping.py"]
        SSOCGEN["generate_ssoc_definitions.py"]
        DBLOAD["data_setup.sh and db_utils.py"]
    end

    subgraph DB["PostgreSQL source tables"]
        RAWJ["raw_jobs"]
        RAWM["raw_modules"]
        SFTBL["skillsfuture_mapping"]
        SSOCTBL["ssoc2024_definitions"]
    end

    subgraph Pipeline["Main pipeline: scripts/run_test2_pipeline.py -> run_pipeline()"]
        LOAD["Load and clean jobs and modules"]
        ROLE["Assign SSOC-backed role clusters"]
        TAX["Apply skill taxonomy and infer module skills"]
        VAR["Consolidate module variants"]
        RET["Build retrieval artifacts: retrieval_text, tokens, BM25, embeddings"]
        SCORE["Hybrid scoring: BM25 + embeddings + RRF"]
        MOD["Module-level outputs: evidence, role scores, SSOC5 scores"]
        GAP["Indicators: module_summary and module_gap_summary"]
        DEGREE["Degree aggregation: degree mapping, scores, gaps, summary"]
        WRITE["Persist files, database tables, reports, diagnostics"]
    end

    subgraph Outputs["Outputs and consumers"]
        OUTDIR["outputs/: CSV, parquet, diagnostics.json, policy_brief.md"]
        RPTDIR["reports/: technical and plain-language markdown reports"]
        DBOUT["App-facing DB tables: degree_*, module_*, jobs_clean, modules_clean, job_role_map"]
        STATE["Returned ModuleReadinessState"]
        API["ModuleReadinessQueryAPI over in-memory state"]
        APP["streamlit_app.py reads persisted DB tables"]
    end

    MSF --> DBLOAD --> RAWJ
    NUS --> NUSSCRAPE --> DBLOAD --> RAWM
    SF --> SFMAP --> DBLOAD --> SFTBL
    SSOC --> SSOCGEN --> DBLOAD --> SSOCTBL

    RAWJ --> LOAD
    RAWM --> LOAD
    CFG --> ROLE
    CFG --> TAX
    SSOCTBL --> ROLE
    SFTBL --> TAX
    DEG --> DEGREE

    LOAD --> ROLE --> TAX --> VAR --> RET --> SCORE --> MOD --> GAP --> DEGREE --> WRITE

    WRITE --> OUTDIR
    WRITE --> RPTDIR
    WRITE --> DBOUT
    WRITE --> STATE
    STATE --> API
    DBOUT --> APP
```

## Runtime Sequence

```mermaid
sequenceDiagram
    autonumber
    actor User as "User or analyst"
    participant Setup as "data_setup.sh + generator scripts"
    participant DB as "PostgreSQL"
    participant Script as "scripts/run_test2_pipeline.py"
    participant Pipe as "run_pipeline()"
    participant Jobs as "load_jobs()"
    participant Roles as "assign_role_families()"
    participant Mods as "load_nus_modules()"
    participant Tax as "apply_skill_taxonomy()"
    participant Var as "consolidate_module_variants()"
    participant Ret as "build_retrieval_artifacts() + HybridRetrievalEngine"
    participant Score as "compute_scores()"
    participant Agg as "build_indicators()"
    participant Deg as "build_degree_outputs()"
    participant Persist as "file, report, and DB writers"
    participant Consumers as "Query API and Streamlit"

    User->>Setup: Prepare and load source data
    Setup->>DB: Write raw_jobs, raw_modules, skillsfuture_mapping, ssoc2024_definitions

    User->>Script: Run the pipeline
    Script->>Pipe: run_pipeline()

    Pipe->>Jobs: Read raw_jobs, clean text, normalize skills, filter scope
    Jobs->>DB: Read raw_jobs
    Jobs-->>Pipe: Cleaned jobs

    Pipe->>Roles: Assign role clusters and SSOC labels
    Roles->>DB: Read ssoc2024_definitions
    Roles-->>Pipe: Jobs with role_family, role_cluster, broad_family, SSOC names

    Pipe->>Mods: Read raw_modules and build module text
    Mods->>DB: Read raw_modules
    Mods-->>Pipe: Undergraduate module catalog

    Pipe->>Tax: Split technical and soft skills; infer module skills
    Tax->>DB: Read skillsfuture_mapping
    Tax-->>Pipe: Taxonomy-enriched jobs and modules

    Pipe->>Var: Merge suffix variants into base module codes
    Var-->>Pipe: Consolidated modules

    Pipe->>Ret: Build retrieval_text, token lists, BM25 indices, embedding cache
    Ret-->>Pipe: Retrieval artifacts and retrieval engine

    loop For each consolidated module
        Pipe->>Score: Rank matching jobs with BM25 and embeddings fused by RRF
        Score-->>Pipe: module_job_evidence rows
    end

    Pipe->>Score: Aggregate evidence into module_role_scores and module_ssoc5_scores
    Score-->>Pipe: Scored module tables

    Pipe->>Agg: Build module_summary and module_gap_summary
    Agg-->>Pipe: Module-level indicators

    Pipe->>Deg: Join degree mapping and aggregate degree outputs
    Deg-->>Pipe: degree_module_map, degree_skill_supply, degree_role_scores, degree_ssoc5_scores, degree gaps, degree_summary

    Pipe->>Persist: Write CSV/parquet snapshots, markdown reports, diagnostics.json
    Persist->>DB: Persist app-facing output tables
    Persist-->>Pipe: Outputs saved

    Pipe-->>Script: Return ModuleReadinessState
    Script->>Consumers: Query API can search jobs and recommend modules from state
    DB->>Consumers: Streamlit app reads persisted degree and module tables
```

## Notes

- Raw data ingestion happens before the main scoring pipeline. The pipeline itself starts from database tables rather than from raw files.
- `load_jobs()` narrows the job corpus to early-career, degree-level postings.
- `load_nus_modules()` narrows the module corpus to undergraduate modules and assembles the retrieval text inputs.
- The retrieval stage uses both BM25 and sentence-transformer embeddings, then combines them with reciprocal rank fusion.
- Degree outputs are built from module-level outputs and the degree mapping file; they do not rerun ingestion or retrieval.
- The Streamlit app is read-only and consumes persisted database tables instead of rerunning the pipeline.
