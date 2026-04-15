# Documentation Validation and Confidence

## 1) Purpose
Provide documentation inventory, hierarchy, confidence ratings, and validation checklists for maintainability and auditability.

## 2) Intended audience
- Maintainers reviewing doc completeness/quality
- Reviewers auditing traceability to code
- Teammates planning future documentation updates

## 3) When to read this
Read after the core docs are drafted or updated.

## 4) Prerequisites / assumed knowledge
- Familiarity with `docs/current/*`
- Familiarity with repository code layout

## 5) High-level summary
This document is the meta-layer for documentation quality control.  
It records:
- canonical doc hierarchy
- confidence per document
- known gaps/uncertainties
- what to revalidate when code changes

## 6) Main content sections

### 6.1 Documentation inventory and recommended hierarchy
Canonical set:
1. `docs/current/README.md`
2. `docs/current/01-product-overview.md`
3. `docs/current/02-system-architecture.md`
4. `docs/current/03-data-contracts.md`
5. `docs/current/04-pipeline-deep-dive.md`
6. `docs/current/05-operations-runbook.md`
7. `docs/current/06-troubleshooting.md`
8. `docs/current/07-adr/*`
9. `docs/current/08-glossary.md`
10. `docs/current/09-presentation-pack.md`
11. `docs/current/10-validation-and-confidence.md`

Archive set:
- `docs/archive/2026-04-10-stale-docs/*`

### 6.2 Highest-priority docs
Priority order for onboarding and handover:
1. README
2. Product overview
3. System architecture
4. Data contracts
5. Pipeline deep dive
6. Calculation reference
7. Operations runbook
8. Troubleshooting
9. ADR index + key ADRs
10. Glossary
11. Presentation pack

### 6.3 Missing-docs gap analysis (current state)
Closed in this pass:
- Canonical architecture narrative aligned with current code
- DB-first runtime model documented explicitly
- Data contract tables and dashboard requirements consolidated
- Presentation-ready narrative from current implementation
- Archived stale docs separated from canonical path

Remaining known gaps:
- No formal ownership metadata per doc file (person/team)
- No automated docs drift checks in CI
- No changelog process specific to `docs/current/*`

### 6.4 Confidence ratings by document
- `README.md`: High
- `01-product-overview.md`: High
- `02-system-architecture.md`: High
- `03-data-contracts.md`: High
- `04-pipeline-deep-dive.md`: High (with one medium-confidence caveat on undeclared fallback config branch)
- `05-operations-runbook.md`: High/Medium (high for commands, medium for process recommendations)
- `06-troubleshooting.md`: High/Medium
- `07-adr/*`: High/Medium depending on explicitness of rationale in code
- `08-glossary.md`: High
- `09-presentation-pack.md`: High/Medium
- `11-calculation-reference.md`: High

Confidence scale:
- High: directly traceable to code/config/tests/artifacts
- Medium: mostly traceable with minor inferred rationale/process framing
- Low: significant inference or missing repository evidence

### 6.5 Validation checklist (global)
- [x] All canonical docs are under `docs/current/`
- [x] Stale docs archived under `docs/archive/...`
- [x] Cross-linking across major docs
- [x] Source basis sections present
- [x] Explicit inferred/not-verified statements included where relevant
- [x] "Explain this system in 5 minutes" included in major docs

### 6.6 Anti-rot maintenance plan
On every significant code/config change:
1. Update affected docs in `docs/current/`.
2. Add/adjust ADR if decision-level change.
3. Re-check data contracts if schema/output columns changed.
4. Revalidate runbook/troubleshooting if operational behavior changed.
5. Record confidence impact in this file.

### 6.7 Owner-review suggestions
Suggested review sequence:
1. Method/architecture owner reviews `01`, `02`, `04`, `11`.
2. Data/operations owner reviews `03`, `05`, `06`.
3. Presentation lead reviews `09`.

### 6.8 Last validated-against references
Validated against repository state including:
- `src/module_readiness/*`
- `src/data_utils/*`
- `scripts/*`
- `streamlit_app.py`
- `config/*`
- `outputs/*.csv` and `outputs/diagnostics.json`

## 7) Key workflows or examples
Example doc update workflow:
1. Change retrieval thresholds in config.
2. Update `03-data-contracts.md` (config contract section).
3. Update `04-pipeline-deep-dive.md` (retrieval behavior section).
4. Update runbook/troubleshooting if operator behavior changed.
5. Record validation date and confidence impact in this file.

## 8) Common pitfalls / gotchas
- Updating code without updating docs in same PR.
- Leaving archived docs discoverable without clear canonical pointer.
- Treating confidence ratings as static.

## 9) Troubleshooting or FAQ
Q: What if docs and code conflict?  
A: Code is authoritative. Update docs and note mismatch in PR review.

Q: When should confidence be lowered?  
A: Whenever behavior is inferred or depends on unavailable external context.

## 10) Related documents / next reads
- Start here for canonical docs: [README.md](./README.md)
- Decision history: [07-adr](./07-adr/)
- Archive rationale: [../archive/2026-04-10-stale-docs/ARCHIVE_NOTES.md](../archive/2026-04-10-stale-docs/ARCHIVE_NOTES.md)

## Explain this system in 5 minutes
The documentation system is now split into canonical (`docs/current`) and archived legacy (`docs/archive`) areas. Canonical docs are structured from high-level purpose to architecture, data contracts, implementation, operations, troubleshooting, decision records, glossary, and presentation narrative. Each major doc includes source-basis traceability and confidence markers so newcomers can trust what is code-verified versus inferred.

## 11) Source basis
- Verified from:
  - Files created under `docs/current/*`
  - Archived files under `docs/archive/2026-04-10-stale-docs/*`
  - Repository modules and outputs cited above
- Inferred from implementation:
  - Recommended owner-review flow and anti-rot process
- Not verified / needs confirmation:
  - Formal team role assignment and review governance outside repo

## Confidence rating
High

## Validation checklist
- [x] Includes inventory, hierarchy, gap analysis
- [x] Includes per-document confidence map
- [x] Includes maintenance process recommendations
