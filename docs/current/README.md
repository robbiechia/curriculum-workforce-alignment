# Module Readiness Documentation Hub

## 1) Purpose
This is the canonical documentation entry point for the Module Readiness project.  
It is designed to help a new engineer understand the system end-to-end, operate it safely, and present it confidently.

## 2) Intended audience
- New engineers onboarding to this repository
- Current contributors modifying pipeline logic
- Teammates preparing technical or stakeholder presentations
- Reviewers validating method, assumptions, and output interpretation

## 3) When to read this
Read this first before opening other docs in `docs/current/`.

## 4) Prerequisites / assumed knowledge
- Python and pandas basics
- SQL table concepts
- Basic retrieval/ranking concepts (BM25, embeddings, rank fusion)
- Familiarity with repository layout

## 5) High-level summary
The project maps NUS module content to early-career, degree-level job demand using a retrieval-and-aggregation pipeline:
- Ingest jobs/modules/taxonomy from a live external database
- Normalize roles and skills
- Build hybrid retrieval evidence (BM25 + embeddings + reciprocal rank fusion)
- Aggregate evidence into module-level and degree-level outputs
- Persist outputs back to the database
- Expose decision-facing tables and a Streamlit dashboard for analysis

Canonical assumption: data source of truth is a live external PostgreSQL database configured by `DATABASE_URL`.

## 6) Main content sections
- [01-product-overview.md](./01-product-overview.md)
- [02-system-architecture.md](./02-system-architecture.md)
- [03-data-contracts.md](./03-data-contracts.md)
- [04-pipeline-deep-dive.md](./04-pipeline-deep-dive.md)
- [05-operations-runbook.md](./05-operations-runbook.md)
- [06-troubleshooting.md](./06-troubleshooting.md)
- [07-adr](./07-adr/)
- [08-glossary.md](./08-glossary.md)
- [09-presentation-pack.md](./09-presentation-pack.md)
- [10-validation-and-confidence.md](./10-validation-and-confidence.md)
- [11-calculation-reference.md](./11-calculation-reference.md)

## 7) Key workflows or examples
Suggested onboarding sequence (about 90 minutes):
1. Read [01-product-overview.md](./01-product-overview.md) to understand problem framing and outputs.
2. Read [02-system-architecture.md](./02-system-architecture.md) for system context and component boundaries.
3. Read [03-data-contracts.md](./03-data-contracts.md) to understand input/output tables.
4. Read [04-pipeline-deep-dive.md](./04-pipeline-deep-dive.md) and open referenced code modules.
5. Read [11-calculation-reference.md](./11-calculation-reference.md) for formula-level backend calculations.
6. Use [05-operations-runbook.md](./05-operations-runbook.md) to run or validate the pipeline.
7. Use [09-presentation-pack.md](./09-presentation-pack.md) to prepare a 5-10 minute presentation.

## 8) Common pitfalls / gotchas
- Reading archived docs as if they are canonical.
- Assuming outputs are file-only; canonical runtime persists outputs to DB and dashboard reads DB tables.
- Treating role labels as purely SSOC-driven; current implementation uses curated role clusters.
- Missing that some behavior is rule-driven via `config/*.yaml`, not hardcoded in one place.

## 9) Troubleshooting or FAQ
Q: Which docs should I trust if there is a conflict?  
A: `docs/current/*` first, then code under `src/module_readiness/*`, then generated outputs in `outputs/*`.

Q: Where are old docs?  
A: `docs/archive/2026-04-10-stale-docs/`.

## 10) Related documents / next reads
- After this file: [01-product-overview.md](./01-product-overview.md)
- For runtime operations: [05-operations-runbook.md](./05-operations-runbook.md)
- For debugging: [06-troubleshooting.md](./06-troubleshooting.md)

## Explain this system in 5 minutes
This project estimates how well NUS modules align with early-career labor demand. It reads jobs/modules/taxonomy from a live DB, filters jobs to degree-level entry roles, clusters jobs into curated role families, and extracts technical/soft skill signals. It then computes module-job relevance using BM25 and embedding similarity fused by reciprocal rank fusion. Evidence is aggregated into module and degree outputs, including role alignment and skill-gap tables. Results are persisted to DB for consistent downstream use, including a Streamlit dashboard.

## 11) Source basis
- Verified from code:
  - `src/module_readiness/orchestration/pipeline.py`
  - `src/module_readiness/*` package modules
  - `streamlit_app.py`
  - `config/pipeline_config.yaml`
  - output contracts in `outputs/*.csv`, `outputs/diagnostics.json`
- Inferred from implementation:
  - Recommended onboarding order and estimated reading times
- Not verified / needs confirmation:
  - Team ownership assignments and formal documentation ownership process

## Confidence rating
High

## Validation checklist
- [x] Canonical DB-backed architecture stated
- [x] Cross-links to all core docs
- [x] Archive location explained
- [x] 5-minute summary included
