# Operations Runbook

## 1) Purpose
Provide practical run/verify/debug procedures for operating this project safely in day-to-day development.

## 2) Intended audience
- Engineers running the pipeline and dashboard
- Teammates validating outputs before presentation
- Maintainers triaging runtime issues

## 3) When to read this
Read before running the project, and keep open during execution.

## 4) Prerequisites / assumed knowledge
- Working Python virtual environment
- Access to live PostgreSQL credentials (`DATABASE_URL`)
- Ability to run shell commands in project root

## 5) High-level summary
Canonical runtime model:
- Live external DB is the source of truth
- Pipeline writes output tables back to DB
- Dashboard reads output DB tables (read-only)
- CSV files are mirrors for inspection, not canonical runtime storage

Primary workflow assumes internet access for first-time data/model setup.  
A restricted-network fallback is available if raw/source tables and embedding caches already exist.

## 6) Main content sections

### 6.1 Environment setup
Minimal required env var:
- `DATABASE_URL`

Dependencies:
- `requirements.txt` packages
- `sentence-transformers` model fetch may require internet on cold cache

Recommended setup commands:

PowerShell (Windows):
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Bash:
```bash
bash setup.sh
```

### 6.2 Data readiness checks
Use data setup script:
```bash
bash src/data_utils/data_setup.sh
```

What it does:
- Validates DB connectivity
- Checks presence of source tables
- Loads missing tables when possible

Source tables expected:
- `raw_jobs`
- `raw_modules`
- `skillsfuture_mapping`
- `ssoc2024_definitions`

### 6.3 Full pipeline run
PowerShell:
```powershell
.\.venv\Scripts\python.exe scripts/run_test2_pipeline.py
```

Quick run (faster iteration):
```powershell
.\.venv\Scripts\python.exe scripts/run_test2_pipeline.py --quick
```

Quick mode changes:
- caps modules around 350
- reduces fetch workers
- adjusts retrieval breadth for speed

### 6.4 Post-run validation checklist
1. Check `outputs/diagnostics.json` exists and contains non-zero key counts.
2. Check core output CSVs exist:
   - `module_summary.csv`
   - `module_role_scores.csv`
   - `module_ssoc5_scores.csv`
   - `degree_summary.csv`
3. Verify DB output tables exist (see [03-data-contracts.md](./03-data-contracts.md)).
4. Confirm `db_outputs_persisted` is `"yes"` in diagnostics.

### 6.5 Dashboard operation
Start dashboard:
```powershell
streamlit run streamlit_app.py
```

Behavior:
- Dashboard checks required DB output tables
- It does not rerun pipeline
- If tables are missing, run pipeline first

### 6.6 Query and retrieval evaluation workflows
Query demo:
```powershell
.\.venv\Scripts\python.exe scripts/run_test2_queries.py --query "data analyst python sql" --top-k 5
```

Retrieval evaluation:
```powershell
.\.venv\Scripts\python.exe scripts/evaluate_retrieval.py export-pool --sample-size 100 --output outputs/retrieval_label_pool.csv
.\.venv\Scripts\python.exe scripts/evaluate_retrieval.py evaluate --labels outputs/retrieval_label_pool.csv --k 10
```

### 6.7 Internet-enabled primary workflow
Use this mode when possible:
- DB accessible
- NUSMods/API access available when data setup needs refresh
- Embedding model can download on cold cache

Benefits:
- reliable end-to-end refresh
- latest source data and model cache updates

### 6.8 Restricted-network fallback workflow
Use when network is limited:
- Ensure DB source tables are already populated
- Ensure embedding cache already exists under `cache/embeddings`
- Run pipeline without forcing external refresh

Limitations:
- Cannot refresh missing external data/model artifacts
- Tests requiring live DB or model downloads may fail

### 6.9 Release/readiness checklist for presentations
Before presenting:
1. Re-run full pipeline (not quick mode) if time allows.
2. Record diagnostics snapshot date/time.
3. Validate top findings against current output tables.
4. Confirm dashboard renders selected majors and gap views.
5. Document any unresolved caveats in presentation notes.

### 6.10 Rollback and recovery approach
If a pipeline change breaks outputs:
1. Restore prior code/config state in git.
2. Re-run pipeline.
3. Compare diagnostics and top-level output shapes.
4. Re-validate dashboard table availability.

Note:
- Output tables are `replace`-written; a successful rerun restores consistent table state.

## 7) Key workflows or examples
Operational happy path:
1. `data_setup.sh` passes
2. run pipeline
3. verify diagnostics + DB tables
4. launch dashboard

## 8) Common pitfalls / gotchas
- Running dashboard before pipeline outputs are present.
- Using quick mode results in place of full-run outputs for final reporting.
- Assuming local CSVs alone are enough for dashboard (dashboard reads DB tables).
- Running tests without understanding DB/network dependencies.

## 9) Troubleshooting or FAQ
Q: `Permission denied` connecting to Supabase/Postgres.  
A: Environment/network is blocking DB traffic; use restricted-network fallback expectations or adjust network access.

Q: Embedding model download errors.  
A: Ensure internet access on first run or reuse warm cache.

Q: Why do output row counts vary between runs?  
A: Source data refreshes, rule changes, and mode differences (quick vs full).

## 10) Related documents / next reads
- Deep implementation details: [04-pipeline-deep-dive.md](./04-pipeline-deep-dive.md)
- Debug matrix: [06-troubleshooting.md](./06-troubleshooting.md)
- Presentation prep: [09-presentation-pack.md](./09-presentation-pack.md)

## Explain this system in 5 minutes
Operations are DB-centric: validate source tables, run the pipeline, verify diagnostics and output tables, then use the dashboard for analysis. Internet access is preferred for complete refreshes (data/model cold-start), while restricted-network mode can still work if DB data and embedding caches already exist. For final presentation-quality outputs, use full runs and document diagnostics snapshots.

## 11) Source basis
- Verified from code/scripts:
  - `scripts/run_test2_pipeline.py`
  - `scripts/run_test2_queries.py`
  - `scripts/evaluate_retrieval.py`
  - `src/data_utils/data_setup.sh`
  - `src/data_utils/db_utils.py`
  - `streamlit_app.py`
  - `outputs/diagnostics.json`
- Inferred from implementation:
  - Suggested release/readiness checklist and rollback flow
- Not verified / needs confirmation:
  - CI/CD deployment standards outside local repository workflow

## Confidence rating
High (runtime commands), Medium (release process recommendations)

## Validation checklist
- [x] Setup, run, verify, and UI workflows documented
- [x] Internet-primary and restricted-network modes explained
- [x] Operational caveats and quick/full mode differences included
- [x] 5-minute summary included
