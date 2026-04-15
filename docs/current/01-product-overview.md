# Product Overview: Module Readiness Mapping

## 1) Purpose
Define what problem this project solves, for whom, and what outputs are produced for decision-making.

## 2) Intended audience
- New engineers onboarding to project scope and purpose
- Teammates preparing project presentations
- Stakeholders reviewing what conclusions can and cannot be drawn

## 3) When to read this
Read after [README.md](./README.md) and before architecture/deep-dive docs.

## 4) Prerequisites / assumed knowledge
- Basic understanding of curriculum modules and job postings
- Basic understanding of ranking outputs (higher score means stronger alignment)

## 5) High-level summary
The project maps university modules to labor-market demand signals using retrieval evidence from job descriptions and normalized skill taxonomies.  
It produces interpretable module-level and degree-level tables that show:
- strongest aligned role clusters / occupations
- supporting evidence strength
- technical skill demand-supply gaps

Primary usage in repository context is policy/curriculum analysis and presentation support.

## 6) Main content sections

### 6.1 Problem statement
Given many modules and many jobs, identify which modules appear most aligned to early-career, degree-level demand and where skill undersupply may exist.

### 6.2 Scope
In scope:
- Entry-level / early-career jobs (`experience_years <= exp_max`, default 2)
- Degree-level jobs (`ssec_eqa == "70"` by default)
- Module/job text and extracted technical/soft skills
- Role-cluster and SSOC occupation reporting layers
- Degree-level aggregation from required module mappings

Out of scope:
- Causal claims about graduate outcomes
- Personalized student recommendations with individual constraints
- Full production deployment and SRE stack (only local runbook exists in repo)

### 6.3 Primary users and decisions supported
- Curriculum teams: identify module-to-demand fit and skill gaps
- Career guidance teams: understand role/occupation alignment patterns
- Student project teams: explain system mechanics and findings in presentations

### 6.4 Primary outputs
Core outputs in `outputs/`:
- Module-level:
  - `module_job_evidence.csv`
  - `module_ssoc5_scores.csv`
  - `module_role_scores.csv`
  - `module_summary.csv`
  - `module_gap_summary.csv`
- Degree-level:
  - `degree_module_map.csv`
  - `degree_skill_supply.csv`
  - `degree_role_scores.csv`
  - `degree_ssoc5_scores.csv`
  - `degree_role_skill_gaps.csv`
  - `degree_ssoc5_skill_gaps.csv`
  - `degree_summary.csv`
- Operational:
  - `jobs_clean.csv`, `modules_clean.csv`, `job_role_map.csv`
  - `diagnostics.json`

DB persistence mirrors these outputs into database tables for dashboard consumption.

### 6.5 Interpretation boundaries
- A high alignment score means stronger retrieval-based evidence alignment, not guaranteed employability outcome.
- Low scores across unrelated roles can reflect specialization, not module quality failure.
- Skill-gap tables represent relative demand-supply signals under current extraction and aggregation rules.

### 6.6 Stability vs volatility
More stable:
- pipeline stage ordering
- output table families
- role/skill rule framework locations

More volatile:
- rule mappings in `config/*.yaml`
- score distributions as source data refreshes
- exact row counts and top-ranked labels across runs

## 7) Key workflows or examples
Example question: "What does module `CS3244` align to?"
1. Check `module_summary.csv` for top role label.
2. Check `module_role_scores.csv` for broader role distribution and evidence strength.
3. Check `module_job_evidence.csv` for concrete job-level evidence and overlap terms.

Example question: "Which skills look undersupplied for a degree?"
1. Check `degree_role_scores.csv` or `degree_ssoc5_scores.csv` to choose target role/occupation.
2. Inspect `degree_role_skill_gaps.csv` or `degree_ssoc5_skill_gaps.csv`.

## 8) Common pitfalls / gotchas
- Using one table in isolation without checking evidence/support columns.
- Ignoring `support_weight` and `evidence_*` fields when interpreting rank order.
- Confusing role-cluster labels with raw SSOC labels.

## 9) Troubleshooting or FAQ
Q: Why do module codes in outputs differ from raw mappings?  
A: Variant consolidation normalizes trailing letter variants to base codes (for example `ACC1701A` -> `ACC1701`).

Q: Why do we see both role-cluster and SSOC outputs?  
A: Role clusters improve interpretability; SSOC outputs preserve occupation-level granularity.

## 10) Related documents / next reads
- Architecture: [02-system-architecture.md](./02-system-architecture.md)
- Data contracts: [03-data-contracts.md](./03-data-contracts.md)
- Pipeline internals: [04-pipeline-deep-dive.md](./04-pipeline-deep-dive.md)

## Explain this system in 5 minutes
This system analyzes how NUS modules align with early-career degree-level job demand. It ingests jobs, modules, and taxonomy data from a live database, normalizes role and skill structures, and ranks module-job relevance using BM25 and sentence embeddings fused by reciprocal rank fusion. It aggregates evidence into module and degree outputs, including top role alignment and skill-gap tables, then persists results back to the database for dashboard use and downstream analysis. It is an evidence-alignment system, not a causal outcome predictor.

## 11) Source basis
- Verified from code/config:
  - `src/module_readiness/orchestration/pipeline.py`
  - `src/module_readiness/analysis/*.py`
  - `config/pipeline_config.yaml`
  - `outputs/*.csv`, `outputs/diagnostics.json`
  - `streamlit_app.py`
- Inferred from implementation:
  - Stakeholder usage framing from report generation and dashboard UI language
- Not verified / needs confirmation:
  - External stakeholder adoption process and decision governance outside repository

## Confidence rating
High

## Validation checklist
- [x] Scope and non-goals are explicit
- [x] Output families and intended interpretation documented
- [x] Interpretation boundaries stated
- [x] 5-minute summary included
