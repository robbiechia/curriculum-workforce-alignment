# ADR-0001: Live External DB as Source of Truth

## Status
Accepted

## Date
2026-04-10

## Context
The pipeline depends on multiple source datasets and produces many derived tables consumed by scripts and dashboard workflows.  
A single canonical state is needed to avoid divergence across local files and downstream consumers.

## Decision
Use live external PostgreSQL (configured by `DATABASE_URL`) as the canonical source and sink:
- Read source tables from DB (`raw_jobs`, `raw_modules`, `skillsfuture_mapping`, `ssoc2024_definitions`)
- Persist output tables back to DB for runtime consumption
- Keep CSV files as inspection mirrors, not canonical runtime storage

## Consequences
Positive:
- Single authoritative runtime state across team members/tools
- Dashboard and downstream consumers can use the same table contracts
- Easier reproducibility of output state by table snapshot

Negative:
- Runtime depends on DB connectivity and credentials
- Restricted-network environments cannot fully execute without preloaded DB state

## Alternatives considered
1. File-only local artifacts as canonical
   - Rejected: high drift risk and weak multi-user consistency
2. Hybrid local-first canonical with optional DB push
   - Rejected for current project phase due complexity and duplicate authority risk

## Evidence
- Verified from:
  - `src/module_readiness/orchestration/pipeline.py` DB persistence path
  - `src/data_utils/db_utils.py`
  - `streamlit_app.py` reading DB output tables
- Inferred:
  - Team collaboration consistency rationale

## Review triggers
- If project moves to offline-first constraints
- If deployment architecture introduces a different canonical storage layer
