# Glossary

## 1) Purpose
Define project-specific terms and acronyms used across code, outputs, and presentations.

## 2) Intended audience
- New engineers and teammates who need fast vocabulary alignment
- Presenters preparing audience-friendly explanations

## 3) When to read this
Use while reading any other document in `docs/current/`.

## 4) Prerequisites / assumed knowledge
None.

## 5) High-level summary
This glossary standardizes language to reduce ambiguity in technical discussion and presentation narration.

## 6) Main content sections

### 6.1 Core system terms
- Module Readiness:
  - Project objective of estimating module alignment to labor demand.
- Evidence row:
  - One module-job ranked match in `module_job_evidence`.
- Role cluster:
  - Curated role family label used for interpretability (for example `Data Science / Analytics`).
- Broad family:
  - Higher-level grouping of role clusters (for example `ICT`, `Finance`).

### 6.2 Data and standards
- SSOC:
  - Singapore Standard Occupational Classification.
- SSOC 4d / SSOC 5d:
  - Four-digit and five-digit SSOC occupation granularity levels.
- SSEC EQA:
  - Education qualification attained code in job data; this project defaults to degree-level code `70`.

### 6.3 Retrieval and scoring
- BM25:
  - Lexical ranking algorithm over tokenized text.
- Embedding similarity:
  - Semantic similarity using sentence-transformer vector representations.
- RRF (Reciprocal Rank Fusion):
  - Rank-based fusion combining multiple retriever rankings.
- Retrieval thresholds:
  - Configured score floors that filter candidates before fusion.
- Support weight:
  - Shrinkage factor based on evidence count used in final aggregated scores.

### 6.4 Pipeline outputs
- `module_summary`:
  - One row per module with top aligned role labels/scores.
- `module_role_scores`:
  - Module-to-role-cluster scored table.
- `module_ssoc5_scores`:
  - Module-to-occupation scored table.
- `module_gap_summary`:
  - Role-family technical skill demand-supply differences.
- `degree_summary`:
  - One row per degree with coverage and top aligned role/occupation indicators.

### 6.5 Operational terms
- Canonical source of truth:
  - Live external DB tables used as authoritative runtime data.
- CSV mirror:
  - File artifacts in `outputs/` for inspection; not canonical runtime store.
- Quick mode:
  - Faster, reduced-scope pipeline run for iteration/debug.

## 7) Key workflows or examples
Presentation tip:
- Use "alignment evidence" instead of "prediction" when describing scores.
- Use "undersupply signal" instead of "definitive curriculum deficiency" for gap tables.

## 8) Common pitfalls / gotchas
- Equating role clusters directly with SSOC codes.
- Treating any single score as causal outcome evidence.
- Using archived doc vocabulary that references removed fields.

## 9) Troubleshooting or FAQ
Q: Is "role family" always SSOC-derived?  
A: Current implementation outputs curated role clusters in `role_family` after assignment logic.

Q: Is RRF score the same as final role score?  
A: No. RRF is evidence-level retrieval score; final role score is support-weighted aggregate.

## 10) Related documents / next reads
- Data contracts: [03-data-contracts.md](./03-data-contracts.md)
- Pipeline internals: [04-pipeline-deep-dive.md](./04-pipeline-deep-dive.md)
- Presentation framing: [09-presentation-pack.md](./09-presentation-pack.md)

## 11) Source basis
- Verified from:
  - `src/module_readiness/*`
  - `config/*.yaml`
  - output table headers in `outputs/*.csv`
- Inferred from implementation:
  - Recommended presentation phrasing for clarity
- Not verified / needs confirmation:
  - External policy definitions beyond repository references

## Confidence rating
High

## Validation checklist
- [x] Acronyms and internal terms normalized
- [x] Output table vocabulary aligned with current schemas
