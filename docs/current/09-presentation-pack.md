# Presentation Pack

## 1) Purpose
Provide a presentation-ready narrative and slide structure so a newcomer can explain the system clearly and accurately.

## 2) Intended audience
- Student presenters
- Teammates preparing project demos or defenses

## 3) When to read this
Read after product overview and architecture docs, before building slides.

## 4) Prerequisites / assumed knowledge
- Understanding from:
  - [01-product-overview.md](./01-product-overview.md)
  - [02-system-architecture.md](./02-system-architecture.md)
  - [04-pipeline-deep-dive.md](./04-pipeline-deep-dive.md)

## 5) High-level summary
Use a 10-slide storyline:
1. Problem
2. Scope
3. Data foundations
4. Architecture
5. Method
6. Outputs
7. Findings
8. Reliability/limits
9. Operational workflow
10. Next steps

## 6) Main content sections

### 6.1 Suggested 10-slide structure
1. Title and objective
   - "Module Readiness Mapping for Early-Career Degree-Level Jobs"
2. Why this matters
   - Curriculum-to-market alignment question
3. Scope and assumptions
   - Entry-level filter, degree-level filter, non-goals
4. Data and system context
   - DB source tables and output consumers
5. Method overview
   - Role clustering, skill taxonomy, hybrid retrieval, support-weighted aggregation
6. Architecture flow
   - Ingest -> process -> retrieve -> score -> aggregate -> persist
7. Output products
   - Module and degree tables, diagnostics, dashboard
8. Example findings
   - Top aligned roles and representative gap signals
9. Reliability and limitations
   - What scores mean, what they do not mean
10. Recommendations and roadmap
   - How to use now, what to improve next

### 6.2 Speaker notes by section
Problem framing:
- "We are not predicting employment outcomes; we are quantifying alignment evidence."

Method framing:
- "We combine lexical and semantic retrieval to reduce one-model blind spots."

Interpretation framing:
- "High alignment means stronger evidence from current postings and skills, not causal guarantee."

### 6.3 Slide-ready architecture narrative
Use this one-line flow:
- "Raw jobs/modules/taxonomy from DB are normalized into role and skill structures, matched via hybrid retrieval, aggregated into module and degree score/gap tables, then persisted back to DB for dashboard analysis."

### 6.4 Slide-ready risk/limitations narrative
- Rule-driven labels can shift with config changes.
- External data freshness affects distributions.
- Gap scores are relative demand-supply signals, not direct curriculum quality judgments.
- Restricted-network environments can block full reproducibility.

### 6.5 Live demo recommendation (optional)
Demo sequence:
1. Show `degree_summary` top metrics in dashboard.
2. Drill into role cluster comparison for one degree.
3. Show skill-gap table and covered/uncovered skills.
4. Show one module evidence trail in output CSV if needed.

### 6.6 Questions you should expect (and answer pattern)
Q: "Why trust these rankings?"  
A: Explain evidence transparency (`bm25_score`, `embedding_score`, `rrf_score`) and support-weighted aggregation.

Q: "Is this predictive of graduate outcomes?"  
A: No, it is alignment evidence; outcome prediction is out of scope.

Q: "How sensitive is it to rules?"  
A: Role and skill rules are config-driven; changes are trackable and should be documented.

Q: "How reproducible is this?"  
A: Deterministic pipeline logic with DB-backed source/output contracts; constrained by external data/model availability.

## 7) Key workflows or examples
10-minute talk timing guide:
1. Slides 1-3 (2.5 min): context/scope
2. Slides 4-6 (3.0 min): architecture/method
3. Slides 7-8 (2.0 min): outputs/findings
4. Slides 9-10 (2.5 min): limitations/roadmap/Q&A bridge

## 8) Common pitfalls / gotchas
- Overstating causality.
- Mixing archived terminology with current output schema.
- Showing quick-mode results as final if full run is available.
- Skipping explicit limitation statements.

## 9) Troubleshooting or FAQ
Q: "What if numbers changed since last rehearsal?"  
A: Anchor interpretation to table structure and method, not to exact row counts; include diagnostics snapshot date.

Q: "What if dashboard is unavailable?"  
A: Present from `outputs/*.csv` and explain DB-backed normal workflow.

## 10) Related documents / next reads
- Product story: [01-product-overview.md](./01-product-overview.md)
- Technical backing: [04-pipeline-deep-dive.md](./04-pipeline-deep-dive.md)
- Reliability caveats: [06-troubleshooting.md](./06-troubleshooting.md)
- Terminology: [08-glossary.md](./08-glossary.md)

## Explain this system in 5 minutes
We built a DB-backed pipeline to estimate how well university modules align with early-career, degree-level labor demand. The system ingests jobs and module data, maps jobs into interpretable role clusters, standardizes skill channels, and computes module-job relevance with BM25 plus embedding similarity fused by reciprocal rank fusion. It aggregates evidence into module and degree outputs, including top role/occupation alignment and skill gap signals, then writes these outputs back to DB for a read-only dashboard. The results support curriculum and communication decisions, but they are alignment evidence, not causal employment predictions.

## 11) Source basis
- Verified from:
  - `src/module_readiness/*` implementation and outputs
  - `streamlit_app.py`
  - `outputs/*.csv` and `outputs/diagnostics.json`
- Inferred from implementation:
  - Presentation sequencing and timing guidance
- Not verified / needs confirmation:
  - Specific panel grading criteria or presentation rubric outside repository

## Confidence rating
High for technical narrative, Medium for presentation process guidance

## Validation checklist
- [x] Slide sequence mapped to technical truth
- [x] Q&A framing aligned with scope boundaries
- [x] 5-minute summary included
