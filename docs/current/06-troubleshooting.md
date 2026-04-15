# Troubleshooting Guide

## 1) Purpose
Provide a practical fault-diagnosis guide for common runtime, data, and interpretation issues.

## 2) Intended audience
- Engineers debugging failed runs
- Teammates diagnosing dashboard or output inconsistencies

## 3) When to read this
Use during incident/debug sessions after reviewing [05-operations-runbook.md](./05-operations-runbook.md).

## 4) Prerequisites / assumed knowledge
- Ability to run shell commands
- Access to `outputs/diagnostics.json`
- Basic familiarity with DB table names

## 5) High-level summary
Most failures cluster into four categories:
1. DB connectivity/config problems
2. Missing source/output tables
3. Network/model-caching issues
4. Interpretation mismatches (reading outdated docs/columns)

## 6) Main content sections

### 6.1 Quick triage checklist
1. Is `DATABASE_URL` set correctly?
2. Can DB be reached from current environment?
3. Do required source tables exist?
4. Did pipeline complete and write `outputs/diagnostics.json`?
5. Are required dashboard tables present?
6. Are you using current docs under `docs/current/`?

### 6.2 Symptom -> cause -> action matrix

#### Symptom: `Permission denied` / DB connection refused
Likely cause:
- Network or firewall blocking external DB endpoint

Actions:
1. Verify `DATABASE_URL`.
2. Confirm network policy allows outbound DB port.
3. If network cannot be opened, use restricted-network fallback expectations.

#### Symptom: Missing source table errors (`raw_jobs`, `raw_modules`, etc.)
Likely cause:
- Data setup not completed in DB

Actions:
1. Run `bash src/data_utils/data_setup.sh`.
2. Re-check table existence.
3. Re-run pipeline.

#### Symptom: Dashboard says missing output tables
Likely cause:
- Pipeline not run successfully, or DB writes failed

Actions:
1. Run pipeline script.
2. Check diagnostics key `db_outputs_persisted`.
3. Confirm required tables listed in [03-data-contracts.md](./03-data-contracts.md).

#### Symptom: Embedding model download failure (HuggingFace/network)
Likely cause:
- Internet blocked during cold cache

Actions:
1. Re-run with internet access to warm cache once.
2. Reuse `cache/embeddings` thereafter in restricted environments.

#### Symptom: Test suite failures in local environment
Likely cause:
- Some tests require DB/model network access

Actions:
1. Separate environment-dependent failures from logic regressions.
2. Run deterministic unit-like tests first.
3. Document environment constraints in review notes.

#### Symptom: Score interpretation confusion (missing expected columns)
Likely cause:
- Using stale docs expecting legacy fields (for example `weighted_fit`, `top_hybrid_score`)

Actions:
1. Use `docs/current/*` only.
2. Inspect actual output headers in `outputs/*.csv`.

### 6.3 Diagnostics-first debugging
Use `outputs/diagnostics.json` to localize stage failures:
- Job counts and filters
- Module selection and detail availability
- Retrieval corpus sizes
- Score table row counts
- Degree mapping coverage
- DB persistence status

If a stage has zeros or sudden drops, debug the corresponding module from [04-pipeline-deep-dive.md](./04-pipeline-deep-dive.md).

### 6.4 Data contract mismatch debugging
Checks:
1. Compare required columns in source tables against ingestion code expectations.
2. Confirm JSON-like fields parse correctly (`skills`, `categories`, `workload`).
3. Validate degree mapping file path and schema.

### 6.5 Ranking/debugging tips
If rankings seem off:
1. Inspect `module_job_evidence.csv` for `bm25_score`, `embedding_score`, `rrf_score`.
2. Check retrieval thresholds in config (`bm25_*`, `embedding_*`).
3. Compare quick vs full mode behavior.
4. Use retrieval evaluation scripts for controlled checks.

### 6.6 Known caveats to keep in mind
- Module variant consolidation can change apparent module identity in outputs.
- Role labels are rule-driven and sensitive to config changes.
- Restricted-network runs may not reflect latest external data/model state.

## 7) Key workflows or examples
Example: Dashboard missing tables
1. Run pipeline.
2. Open `outputs/diagnostics.json` and verify `db_outputs_persisted`.
3. Verify table names in DB.
4. Relaunch Streamlit app.

Example: Unexpected role label for module
1. Check `module_summary.csv`.
2. Inspect `module_role_scores.csv` and `module_ssoc5_scores.csv`.
3. Trace role rules in `config/role_clusters.yaml`.

## 8) Common pitfalls / gotchas
- Treating environment failures as code regressions.
- Editing config without recording expected output shifts.
- Comparing outputs across runs without noting quick/full mode and data snapshot date.

## 9) Troubleshooting or FAQ
Q: Should I trust archived reports when results look inconsistent?  
A: No. Archived docs are for history only. Use `docs/current/*` and code.

Q: How do I know if DB writes happened?  
A: Check `db_outputs_persisted` in diagnostics and validate output tables in DB.

## 10) Related documents / next reads
- Run procedures: [05-operations-runbook.md](./05-operations-runbook.md)
- Contracts: [03-data-contracts.md](./03-data-contracts.md)
- Deep internals: [04-pipeline-deep-dive.md](./04-pipeline-deep-dive.md)

## Explain this system in 5 minutes
When something breaks, first classify the issue as DB connectivity, missing tables, network/model cache, or interpretation mismatch. Use diagnostics to locate failing stages, then inspect the corresponding output and code module. Most runtime issues are environmental (DB/network) rather than algorithmic. For algorithmic issues, trace from evidence-level outputs to aggregation tables before changing rules/config.

## 11) Source basis
- Verified from code/scripts/artifacts:
  - `src/data_utils/db_utils.py`
  - `src/module_readiness/orchestration/pipeline.py`
  - `streamlit_app.py`
  - `outputs/diagnostics.json`
  - `tests/test_*.py` behavior patterns
- Inferred from implementation:
  - Triage prioritization and issue grouping
- Not verified / needs confirmation:
  - Organization-specific incident response process

## Confidence rating
High for technical troubleshooting paths; Medium for process guidance

## Validation checklist
- [x] Symptom/cause/action mappings provided
- [x] Diagnostics-based debugging workflow included
- [x] Current-vs-archived doc guidance explicit
- [x] 5-minute summary included
