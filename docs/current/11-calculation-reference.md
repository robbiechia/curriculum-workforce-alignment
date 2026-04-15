# Calculation Reference

## 1) Purpose
Provide an explicit, formula-level reference for backend calculations used in the pipeline.

## 2) Intended audience
- New engineers who need to understand backend scoring mathematics
- Reviewers validating technical correctness
- Presenters who need defensible formula explanations

## 3) When to read this
Read after:
- [04-pipeline-deep-dive.md](./04-pipeline-deep-dive.md)
- [03-data-contracts.md](./03-data-contracts.md)

Use this doc when you need exact computational definitions.

## 4) Prerequisites / assumed knowledge
- Basic ranking/retrieval concepts
- Mean, quantiles, dense rank
- Familiarity with table names in `outputs/`

## 5) High-level summary
This document explains backend calculations in a way that is precise but still easy to read.

The pipeline does four calculation layers:
1. Build retrieval evidence (`bm25_score`, `embedding_score`, `rrf_score`).
2. Aggregate module-level alignment scores with support adjustment.
3. Compute module-level demand vs supply skill gaps.
4. Roll module outputs up to degree-level scores and degree-level skill gaps.

All definitions below are aligned to current code in `src/module_readiness/*`.

## 6) Main content sections

### 6.1 Quick vocabulary
- Module-job evidence row:
  - one retrieved job supporting one module.
- Role cluster:
  - the curated role label stored in `role_family`.
- Support prior:
  - a smoothing constant that prevents tiny evidence sets from dominating.
- Top-k:
  - how many strongest rows are kept when aggregating scores.

### 6.2 Job corpus filters (what jobs are eligible)
Before scoring, jobs are filtered to:
- `experience_years` is not null
- `experience_years <= exp_max`
- `ssec_eqa == primary_ssec_eqa`

Practical meaning:
- the model intentionally focuses on early-career, degree-level jobs.

### 6.3 Retrieval text construction
For each document (job or module):

`retrieval_text = normalized base text + "technical skills" + normalized technical skills`

where:
- job base text is `job_text`
- module base text is `module_title + module_description`

Tokenization:
- lowercase normalization
- keep alphanumeric plus `+` and `#`
- remove stopwords
- minimum token length 2

### 6.4 Candidate filtering (before fusion)
Each retriever keeps only "strong enough" candidates.

Rule used (same idea for BM25 and embeddings):
1. Find the best score in the candidate pool.
2. Build a threshold as:
   - the larger of:
     - a fixed floor (absolute minimum), and
     - a relative floor (best score multiplied by a percentage).
3. Keep only rows that meet or exceed that threshold.

This is applied separately to:
- BM25 scores (using BM25 thresholds)
- embedding cosine similarities (using embedding thresholds)

Plain-English interpretation:
- A row must be reasonably good globally and relative to the current best row.

### 6.5 Hybrid ranking (BM25 + embedding + RRF)
After filtering, the system fuses BM25 ranking and embedding ranking with Reciprocal Rank Fusion (RRF).

RRF in words:
- A candidate gets points from each ranker.
- Higher-ranked candidates get more points.
- Points decay as rank position gets deeper.
- Final hybrid ranking sorts by total fused points.

In implementation, hybrid scores are normalized to a 0-to-1 scale by dividing by the run's maximum fused score.

Supported ranking modes:
- `hybrid`: rank by `RRF_norm`, candidate set `C_bm25 union C_emb`
- `bm25`: rank by BM25 score, candidate set `C_bm25`
- `embedding`: rank by embedding score, candidate set `C_emb`

### 6.6 Evidence table definition (`module_job_evidence`)
For each module `m`, retrieve top `top_n = max(top_k * 4, retrieval_top_n)` jobs.

Each evidence row includes:
- retrieval fields: `bm25_score`, `embedding_score`, `bm25_rank`, `embedding_rank`, `rrf_score`
- mapped role/SSOC fields from job row
- overlap explanation term string

Important:
- this table is the bridge between retrieval and aggregated scoring.

### 6.7 Module-level scoring (role and occupation)
For each module-target pair (role cluster or occupation), the pipeline:
1. Sorts supporting evidence rows by `rrf_score`.
2. Keeps top `top_k` rows.
3. Computes:
   - `raw_score = average of kept rrf_score values`
   - `evidence_count = number of kept rows`
4. Applies support shrinkage:
   - `support_weight = evidence_count / (evidence_count + role_support_prior)`
5. Produces final:
   - `role_score = raw_score * support_weight`

Why this is done:
- strong but tiny evidence sets should not outrank slightly lower but well-supported targets.

Distribution diagnostics:
- `q1`, `median`, `q3` from kept evidence values
- `iqr = q3 - q1`
- `top5_mean = average of top 5 kept values (or fewer if not available)`
- `topk_mean = average of all kept values`
- `concentration_ratio = top5_mean / topk_mean` if `topk_mean > 0` else 0
- `bimodality_flag = 1` if `(concentration_ratio > 1.5 and iqr > 0.3)`, else 0

### 6.8 Near-tie rule for `Other` role label
When a module's `Other` role score is almost tied with a named role:
- if `Other` is within margin (`0.02`) of the best named role,
- `selection_score` is adjusted so the named role ranks first.

Purpose:
- avoid non-informative labels winning close ties.

### 6.9 Module-level skill gap formulas (`module_gap_summary`)
For each role cluster:
1. Build demand counts from jobs:
   - how often each skill appears in jobs of that role.
2. Build supply counts from modules:
   - assign each module to its top role targets (default top 2),
   - count how often each skill appears in assigned modules.
3. Normalize both sides within that role:
   - `demand_score = skill demand count / total demand counts`
   - `supply_score = skill supply count / total supply counts`
4. Gap:
- `gap_score = demand_score - supply_score`
- `gap_type = undersupply` if `gap_score > 0` else `oversupply`

### 6.10 Degree mapping and supply formulas
Degree mapping workflow:
1. Expand each degree's semicolon-separated required module list.
2. Normalize each required module code to base code (same variant consolidation logic).
3. Join with module outputs to mark matched vs missing required modules.

Degree skill supply:
- For each degree and skill:
  - `module_count = number of matched required modules that contain the skill`
  - `matched_required_module_count = total matched required modules for that degree`
  - `supply_score = module_count / matched_required_module_count` (or 0 if denominator is 0)

### 6.11 Degree-level role and occupation scores
For each degree and each target (role cluster or occupation):
1. Join matched required modules with module target scores.
2. Keep top contributors (`degree_role_top_n`) for that target.
3. Compute:
   - `raw_degree_score = average of contributing module scores`
   - `evidence_module_count = unique contributing modules`
4. Apply degree support shrinkage:
   - `support_weight = evidence_module_count / (evidence_module_count + degree_support_prior)`
   - `degree_score = raw_degree_score * support_weight`
5. Compute coverage:
   - `module_support_share = evidence_module_count / matched_required_module_count`
6. Rank targets within each degree by descending degree score.

### 6.12 Degree skill gap formulas
For each degree-target pair (role or occupation):
1. Build target-side demand from jobs:
   - normalize to `demand_score` shares within that target.
2. Pull degree-side supply from `degree_skill_supply`:
   - `supply_score` for each skill (0 if missing).
3. Compute:
- `gap_score = demand_score - supply_score`
- `gap_type = undersupply` if `gap_score > 0` else `oversupply`

### 6.13 Retrieval evaluation metrics (plain definitions)
Given labeled relevance map `rel(job_id)` for a module:
- `nDCG@k`:
  - ranking quality vs ideal ordering, weighted by rank position.
- `Precision@k`:
  - fraction of top-k retrieved jobs that are labeled relevant.
- `Recall@k`:
  - fraction of all labeled relevant jobs retrieved in top-k.
- `LabelCoverage@k`:
  - fraction of top-k predictions that are present in the labeled set.

### 6.14 Worked mini-example: support weighting
Assume module-role top evidence RRF scores:
- `[0.90, 0.80, 0.70]`, `prior=5`

Then:
- `raw = (0.90 + 0.80 + 0.70)/3 = 0.80`
- `w = 3/(3+5) = 0.375`
- `role_score = 0.80 * 0.375 = 0.30`

Interpretation:
- strong raw evidence but limited support, so final score is appropriately shrunk.

### 6.15 Worked mini-example: gap score
For role `r`, top demand skill `python`:
- `D_r(python)=20`, `D_total(r)=200` -> `demand_score=0.10`
- `S_r(python)=5`, `S_total(r)=100` -> `supply_score=0.05`
- `gap_score=0.10-0.05=0.05` -> `undersupply`

Interpretation:
- demand share exceeds supply share for that skill in this role context.

### 6.16 End-to-end worked example from actual outputs
This example uses real rows from your current output files (`outputs/*.csv`) and walks one path through the pipeline.

Example entities:
- Module: `BSP1702`
- Degree: `BIZ__Business_Administration`
- Degree target role: `Accounting / Audit / Tax`

Step A: retrieval evidence for one module-job row (`module_job_evidence.csv`)
- Row 1 for `BSP1702`:
  - `bm25_rank = 2`
  - `embedding_rank = 7`
  - `rrf_score = 1.0` (normalized)

Important constant used below:
- `60` is **not** the number of jobs.
- `60` is `rrf_k` from `config/pipeline_config.yaml` (the RRF damping constant).

Raw fused score before normalization:
- `raw_rrf = 1/(60+2) + 1/(60+7)`
- `raw_rrf = 1/62 + 1/67`
- `raw_rrf = 0.016129 + 0.014925 = 0.031054`

For this module query, this row is the max raw fused score, so:
- `normalized_rrf = 0.031054 / 0.031054 = 1.0`

Possible insight:
- This row ranks highly in both BM25 and embedding views, so the match is strong in both lexical and semantic terms.

Step B: second row to show normalization effect
- Another `BSP1702` row:
  - `bm25_rank = 3`
  - `embedding_rank = 9`
  - CSV `rrf_score = 0.977824856490175`

Compute:
- `raw_rrf_2 = 1/(60+3) + 1/(60+9) = 1/63 + 1/69`
- `raw_rrf_2 = 0.015873 + 0.014493 = 0.030366`
- `normalized_rrf_2 = 0.030366 / 0.031054 = 0.9778` (matches CSV after rounding)

Possible insight:
- Slightly lower ranks in both channels produce a slightly lower fused score; ranking is smooth, not all-or-nothing.

Step C: row where one ranker is missing after thresholding
- A `BSP1702` row has:
  - `bm25_rank = (blank)`
  - `embedding_rank = 4`
  - CSV `rrf_score = 0.5031492248062015`

Compute:
- `raw_rrf = 1/(60+4) = 1/64 = 0.015625` (BM25 contributes 0 here)
- `normalized_rrf = 0.015625 / 0.031054 = 0.5031` (matches CSV after rounding)

Possible insight:
- The row still survives because embedding evidence is strong enough, but confidence is lower than rows supported by both channels.

Step D: module-level role aggregation (`module_role_scores.csv`)
- For `BSP1702` and role `Accounting / Audit / Tax`:
  - `raw_role_score = 0.37171174130194695`
  - `evidence_job_count = 8`
  - `role_support_prior = 5`

Where `raw_role_score` comes from:
- Take the top `top_k` evidence rows for this module-role pair (here `top_k=50`, but only 8 rows exist, so all 8 are used).
- Average their `rrf_score` values.
- The 8 values are:
  - `0.4182019531`
  - `0.4025193798`
  - `0.3927018340`
  - `0.3879704866`
  - `0.3618151729`
  - `0.3462532300`
  - `0.3389636883`
  - `0.3252681857`
- Mean of these 8 values:
  - `raw_role_score = 0.3717117413`

Support and final score:
- `support_weight = 8 / (8 + 5) = 0.6153846154`
- `role_score = 0.3717117413 * 0.6153846154 = 0.2287456870`

This matches CSV `role_score = 0.2287456869550443` (minor rounding difference).

Possible insight:
- Raw alignment is moderate, but evidence volume is also modest (`8` rows), so support shrinkage lowers the final score to avoid overconfidence.

Step E: degree-level role aggregation (`degree_role_scores.csv`)
- For degree `BIZ__Business_Administration`, role `Accounting / Audit / Tax`:
  - `top_contributing_modules = ACC1701; BSP1702; FIN2704; RE1707; DAO1704`
  - `raw_degree_score = 0.25589171206569455`
  - `evidence_module_count = 5`
  - `degree_support_prior = 3`

Support and final degree score:
- `support_weight = 5 / (5 + 3) = 0.625`
- `degree_role_score = 0.2558917121 * 0.625 = 0.1599323200`

This matches CSV `degree_role_score = 0.1599323200410591`.
This role is ranked `8` within this degree (`role_rank_within_degree = 8`).

Possible insight:
- Accounting alignment exists for this degree, but it is not the dominant direction versus other role targets in the same degree profile.

Step F: degree-role skill gap (`degree_role_skill_gaps.csv`)
- Same degree and role, skill `audit`:
  - `demand_score = 0.05434782608695652`
  - `supply_score = 0.0`
  - `gap_score = 0.05434782608695652`
  - `gap_type = undersupply`

Interpretation of the chain:
- Strong retrieval evidence rows contribute to module role scores.
- Strong module role scores can make that module part of degree top contributors.
- Degree-level role score is support-adjusted again at the degree aggregation layer.
- Gap rows then compare target demand share to degree supply share for each skill.

Possible insight:
- For this degree-role view, `audit` appears as a meaningful employer-demand skill with little to no mapped curriculum supply signal, suggesting a curriculum or mapping review opportunity.

## 7) Key workflows or examples
Formula-debug workflow:
1. Recompute candidate threshold step from retrieval outputs/config.
2. Recompute RRF from rank columns for a sample row.
3. Recompute group-level `raw`, `support_weight`, `role_score`.
4. Recompute gap score from demand/supply normalization.

## 8) Common pitfalls / gotchas
- Treating `rrf_score` as final module or degree score.
- Ignoring support shrinkage when comparing groups with different evidence counts.
- Comparing gaps across roles without noting role-specific normalization denominators.
- Assuming `Other` ranking behavior has no tie-handling logic.

## 9) Troubleshooting or FAQ
Q: Why is a high raw score still low in final ranking?  
A: Support shrinkage can reduce final score when evidence count is small.

Q: Why can `Other` lose to a slightly lower named role?  
A: Near-tie preference margin intentionally favors interpretable named clusters.

Q: Why are degree scores lower than module scores numerically?  
A: Degree scores are separate support-adjusted aggregates with different priors/evidence sets.

## 10) Related documents / next reads
- Stage flow and code mapping: [04-pipeline-deep-dive.md](./04-pipeline-deep-dive.md)
- Table schemas and columns: [03-data-contracts.md](./03-data-contracts.md)
- Troubleshooting interpretation issues: [06-troubleshooting.md](./06-troubleshooting.md)

## Explain this system in 5 minutes
The backend first builds module-job evidence using two retrieval signals (BM25 and embeddings) fused by RRF. It then aggregates evidence into module-level role and occupation scores, while shrinking weakly supported groups so tiny samples do not dominate. After that, it rolls module outputs up to degree-level scores using required module mappings and the same support-adjusted logic. Finally, it computes skill gaps by comparing normalized demand from jobs against normalized supply from module/degree coverage.

## 11) Source basis
- Verified from code:
  - `src/module_readiness/retrieval/text.py`
  - `src/module_readiness/retrieval/fusion.py`
  - `src/module_readiness/retrieval/engine.py`
  - `src/module_readiness/analysis/scoring.py`
  - `src/module_readiness/analysis/aggregation.py`
  - `src/module_readiness/analysis/degrees.py`
  - `src/module_readiness/analysis/retrieval_eval.py`
  - `src/module_readiness/processing/skill_taxonomy.py`
  - `config/pipeline_config.yaml`
- Inferred from implementation:
  - Compact notation and grouped equation presentation
  - Worked toy examples for comprehension
- Not verified / needs confirmation:
  - Whether future model or ranking changes should retain exact same formula semantics

## Confidence rating
High

## Validation checklist
- [x] Retrieval, aggregation, and gap formulas explicitly defined
- [x] Symbol definitions and config dependencies included
- [x] Worked examples included
- [x] Scope of interpretation limits included
- [x] Explanations prioritize readability over dense notation
