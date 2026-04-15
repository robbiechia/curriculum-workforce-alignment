# Technical Deep-Dive Report (Test2): Module-First Job Readiness Mapping

Generated on: 2026-03-20

## 1) Problem formulation and design target

### 1.1 Objective
The technical objective is to estimate, for each National University of Singapore (NUS) module, how strongly it aligns with labor market demand for early-career degree-level jobs.

### 1.2 Why module-first instead of degree-first
The implemented system computes alignment at module level because the project requirement emphasized module-level usefulness and because an official machine-readable degree-to-module requirement graph is not readily available in this codebase.

### 1.3 Scope boundaries
The system enforces two hard filters on job advertisements:
1. `minimumYearsExperience <= 2`
2. `ssecEqa = 70`

`SSEC` means **Singapore Standard Educational Classification**; `EQA` means education qualification attained code. This keeps the target market aligned with early-career university-level jobs.

## 2) Data inputs, schemas, and ingestion logic

### 2.1 Job advertisements (MyCareersFuture raw JavaScript Object Notation (JSON))
Source pattern: `proj data/data/*.json`

Each job row is canonicalized into:
- identifiers: `job_id`, `ssoc_code`, `source_code`
- text fields: `title`, `description_raw`, `description_clean`, `job_text`
- market attributes: `applications`, `views`, `demand_weight`
- role context: `categories`, `primary_category`
- controls: `experience_years`, `ssec_eqa`, `dedupe_weight`

Text normalization steps:
1. remove HyperText Markup Language (HTML) tags
2. Unicode symbol stripping (category `S*` removed)
3. lowercase
4. alphanumeric plus `+` and `#` retention
5. whitespace normalization

### 2.2 Duplicate-template debiasing
**Problem:** Job postings are often repeated (same company, same template, different batch/deadline). Without debiasing, a single role posted 4 times appears 4× as important.

**Solution:** Cluster identical postings by `title_clean || company_clean || md5(description_clean)`. Assign each posting a debias weight:
\[
\text{dedupe\_weight}_j = \frac{1}{\sqrt{\text{cluster\_size}_j}}
\]

**Effect:** A 4-posting cluster → each posting weighted 0.5 → total impact = 2.0 (not 4.0). This preserves that repeated posting signals hot demand, while preventing artificial inflation.

### 2.3 Demand weighting
**Problem:** Jobs with 500 applications indicate hot demand; jobs with 5 applications are niche. Both should not be treated equally.

**Solution:** Weight each job based on market intensity:
\[
\text{demand\_weight}_j = 1 + \ln(1 + \text{applications}_j) + 0.2 \times \ln(1 + \text{views}_j)
\]

**Breakdown:**
- Base multiplier `1`: every job starts at 1× weight
- `ln(applications)`: primary signal (stronger intent)
- `0.2 × ln(views)`: secondary signal (passive interest, weighted 1/5th)
- Logarithm prevents outliers from dominating; 100 vs 500 applications don't differ by 5×, but by ~1.3×

**Example:** A niche job (5 apps, 100 views) → weight ≈ 1.98; a popular job (100 apps, 1000 views) → weight ≈ 5.75. Popular jobs boost module alignment scores 2.9× more than niche ones.



### How They Work Together in Scoring 
When computing final module-job fit:

$$
WeightedFit
(m
,
j)
=
FinalFit
(m
,
j)
×
demand-weight
(j)
×
dedupe-weight
(j)

​$$
 

Example:   

Module CS1010 vs Job "ML Engineer @Google":

FinalFit = 0.75 (moderate relevance) 
demand_weight = 3.2 (popular job: 200 apps, 2000 views)  
dedupe_weight = 0.6 (part of 3-posting cluster)  
WeightedFit = 0.75 × 3.2 × 0.6 = 1.44  

VS 

Module CS1010 vs Job "Obscure startup role":  

FinalFit = 0.75 (same relevance)
demand_weight = 1.1 (niche: 2 apps, 50 views)  
dedupe_weight = 1.0 (unique posting)  
WeightedFit = 0.75 × 1.1 × 1.0 = 0.83  

Result: CS1010 appears more aligned with the popular ML role than the obscure one, reflecting actual market gravity.

### 2.4 NUS modules (NUSMods API)
`NUSMods` means **National University of Singapore Modules API**.

Ingestion strategy:
1. resolve academic year from candidate list (`2025-2026`, then fallback years)
2. fetch `moduleList.json`
3. select undergraduate modules from the catalog
4. fetch `modules/<moduleCode>.json` detail records concurrently
5. cache everything under `test2/cache/nusmods/<year>/...`

If live/API is unavailable, the system degrades to cache; if both unavailable, module ingestion returns no rows.

Module text representation is concatenated from:
- title
- description
- additional information
- prerequisite and preclusion text
- department/faculty metadata
- module credit
- workload tokenization
- profile seed terms

### 2.5 Module variant consolidation
Many NUS modules are offered in multiple variants that share the same base course but differ in implementation. Examples include:

| Base | Variants | Context |
|------|----------|---------|
| CS1010 | A, E, R, S, X | Different programming languages (Python, MATLAB, R, Scheme) |
| ACC1701 | A, B, C, D | Different delivery streams |
| MKT1705 | A, B, C, D, X | Different emphasis or audience |
| FIN3703 | A, B, C | Multiple offerings |

**Consolidation strategy:**
1. Extract numeric base code using regex to remove trailing alphabetic characters
2. Group modules by base code
3. For each variant group, merge:
   - `technical_skills`: union with deduplication
   - `transferable_skills`: union with deduplication
   - `transferable_cues`: union with deduplication
   - Text fields (title, description): keep first variant to preserve consistency

**Result:** 1,200 NUS modules → **1,080 consolidated modules** (120 variant consolidations removed).

**Distribution impact:**
- technical_fit and transfer_fit distributions computed during scoring are based on consolidated modules
- skill vocabulary reflects merged variant skill sets (more comprehensive)
- module-job evidence table contains consolidated module codes
- role_score aggregation uses consolidated module identifiers

**Known limitation:** Module descriptions remain identical across variants even when actual difficulty, pedagogy, or content focus differs. Example: CS1010A teaches Python procedural programming while CS1010S teaches Scheme functional programming, but both share "Introduction to Computer Science" description. When consolidated, the merged module captures all variant skills but single description may underrepresent variant heterogeneity.

**Mitigations for curriculum teams:**
- Check `module_code` field in outputs; consolidated codes have no letter suffix
- Merged skills are comprehensive across variant pedagogy
- For variant-specific interpretation, cross-reference NUSMods pages (e.g., nusmods.com/CS1010A vs CS1010S)
- Gap analysis and role scores reflect demand aggregated across variant offerings

## 3) Role-family and taxonomy construction

### 3.1 Role family assignment
Primary backbone is `SSOC` (Singapore Standard Occupational Classification) 2-digit prefix mapping.

Fallback precedence for unassigned jobs:
1. title+skill keyword rule
2. category rule
3. `Other`

This produces `role_family` and `role_family_source` for auditability.

### 3.2 Skill taxonomy channels
The project separates two channels:
1. technical/domain channel
2. transferable channel

Skills are normalized via alias map first, then mapped through a SkillsFuture Skills Framework-aligned table.

If a skill is not in the mapping table, heuristic channel assignment is applied using lexical markers (for example `communication`, `team`, `leadership`).

### 3.3 Module skill extraction
Module skills are extracted by exact phrase matching over module textual fields against known job-derived skill vocabulary. Transferable cues are extracted from cue list matching.

## 4) Natural Language Processing (NLP) and feature engineering

`NLP` means **Natural Language Processing**.

### 4.1 Lexical representation
`TF-IDF` means **Term Frequency-Inverse Document Frequency**.

Pipeline:
1. tokenize cleaned text
2. remove stopwords
3. add unigrams + bigrams
4. fit sparse TF-IDF vocabulary on combined job/module corpus
5. L2 normalize vectors

### 4.2 Sparse similarity engine
Cosine similarity is computed via sparse dot products using an inverted index implementation.

### 4.3 Entropy-based technical skill weighting
For each technical skill \(s\), compute distribution over role families and normalized entropy:
\[
H_s = -\sum_f p(f\mid s)\ln p(f\mid s),\qquad
\hat{H}_s = \frac{H_s}{\ln(F)}
\]
where \(F\) is number of families.

Weight function:
\[
w_s = \max(0.05,\ 1 - \gamma \hat{H}_s)
\]
with \(\gamma = 0.8\) by default.

Interpretation: generic skills spread across many families receive lower discriminative weight.

### 4.4 Feature blocks
For module \(m\) and job \(j\), the pipeline computes the following components in `src/module_readiness/scoring.py`.

#### 4.4.1 Lexical similarity
$$
\text{LexSim}_{m,j} = \cos\left(\mathbf{v}^{\text{desc}}_m,\ \mathbf{v}^{\text{desc}}_j\right)
$$
Plain-text equivalent: `LexSim_mj = cosine(module_desc_vector_m, job_desc_vector_j)`
- Simple English: This compares module description text with job description text. Similar wording and concepts produce a higher score.
- Derived from: `similarity_matrix(features.module_desc_vectors, features.job_desc_vectors)`.

#### 4.4.2 Technical skill similarity
$$
\text{TechSkillSim}_{m,j} = \cos\left(\mathbf{v}^{\text{tech}}_m,\ \mathbf{v}^{\text{tech}}_j\right)
$$
Plain-text equivalent: `TechSkillSim_mj = cosine(module_tech_skill_vector_m, job_tech_skill_vector_j)`
- Simple English: This compares technical skills covered by the module with technical skills required by the job.
- Derived from: `similarity_matrix(features.module_tech_vectors, features.job_tech_vectors)`.

#### 4.4.3 Transferable skill similarity
$$
\text{TransferSkillSim}_{m,j} = \cos\left(\mathbf{v}^{\text{trans\_skill}}_m,\ \mathbf{v}^{\text{trans\_skill}}_j\right)
$$
Plain-text equivalent: `TransferSkillSim_mj = cosine(module_transfer_skill_vector_m, job_transfer_skill_vector_j)`
- Simple English: This compares transferable skills (for example, communication and teamwork) from the module with transferable skills requested by employers.
- Derived from: `similarity_matrix(features.module_transfer_skill_vectors, features.job_transfer_skill_vectors)`.

#### 4.4.4 Transferable cue similarity
$$
\text{TransferCueSim}_{m,j} = \cos\left(\mathbf{v}^{\text{trans\_cue}}_m,\ \mathbf{v}^{\text{trans\_cue}}_j\right)
$$
Plain-text equivalent: `TransferCueSim_mj = cosine(module_transfer_cue_vector_m, job_transfer_cue_vector_j)`
- Simple English: This compares transferable cue words in module text and job text (for example, leadership, collaboration, and stakeholder communication).
- Derived from: `similarity_matrix(features.module_transfer_cue_vectors, features.job_transfer_cue_vectors)`.

## 5) Scoring model and equations

### 5.1 Technical fit
Configured weights: lexical 0.50, technical skill 0.50 (equally balanced for explainability and performance).

Technical fit combines lexical and skill-based similarity:
$$
\text{TechnicalFit}_{m,j} = 0.50\cdot\text{LexSim}_{m,j} + 0.50\cdot\text{TechSkillSim}_{m,j}
$$
Plain-text equivalent: `TechnicalFit_mj = 0.50*LexSim_mj + 0.50*TechSkillSim_mj`
- Simple English: Technical fit is a balanced average of text similarity and technical-skill similarity.
- Derived from implementation: `technical_fit = w_lex * lex_sim + w_skill * tech_skill_sim` in `src/module_readiness/scoring.py`.

### 5.2 Transferable fit
$$
\text{TransferFit}_{m,j} = 0.70\cdot\text{TransferSkillSim}_{m,j} + 0.30\cdot\text{TransferCueSim}_{m,j}
$$
Plain-text equivalent: `TransferFit_mj = 0.70*TransferSkillSim_mj + 0.30*TransferCueSim_mj`
- Simple English: Transferable fit gives more weight to explicit transferable skills and less weight to cue words.
- Derived from implementation: `transferable_fit = weight_transfer_skill * transfer_skill_sim + weight_transfer_cue * transfer_cue_sim` in `src/module_readiness/scoring.py`.

### 5.3 Final module-job fit
$$
\text{FinalFit}_{m,j} = 0.90\cdot\text{TechnicalFit}_{m,j} + 0.10\cdot\text{TransferFit}_{m,j}
$$
Plain-text equivalent: `FinalFit_mj = 0.90*TechnicalFit_mj + 0.10*TransferFit_mj`
- Simple English: Final fit is mostly technical readiness, with a smaller supporting contribution from transferable readiness.
- Derived from implementation: `final_fit = weight_technical_final * technical_fit + weight_transfer_final * transferable_fit` in `src/module_readiness/scoring.py`.

### 5.4 Market-adjusted fit
$$
\text{WeightedFit}_{m,j} = \text{FinalFit}_{m,j}\cdot\text{demand\_weight}_j\cdot\text{dedupe\_weight}_j
$$
Plain-text equivalent: `WeightedFit_mj = FinalFit_mj * demand_weight_j * dedupe_weight_j`
- Simple English: A module-job match gets boosted when the job is in stronger demand, and controlled when repeated posting templates are detected.
- Derived from implementation: `job_market_weight = jobs["demand_weight"] * jobs["dedupe_weight"]` and `weighted_fit = final_fit * job_market_weight` in `src/module_readiness/scoring.py`.

### 5.5 Module-role aggregation
For each module-role pair, collect top-K jobs by weighted fit and compute:
$$
\text{RoleScore}_{m,r} = \frac{1}{K}\sum_{j\in\text{TopK}(m,r)}\text{WeightedFit}_{m,j}
$$
Plain-text equivalent: `RoleScore_mr = mean(WeightedFit_mj for j in TopK jobs for module m within role r)`
- Simple English: For each module and role family, we keep the strongest evidence jobs, then average their weighted fit scores.
- Derived from implementation: in `_aggregate_role_scores()`, jobs are sorted by `weighted_fit`, truncated to `top_k`, and averaged into `role_score`.

Additional distribution diagnostics:
- quartiles (`q1`, `median`, `q3`)
- interquartile range (`iqr`)
- concentration ratio:
$$
\text{CR} = \frac{\text{mean(top 5 weighted fits)}}{\text{mean(top K weighted fits)}}
$$
  Plain-text equivalent: `CR = mean(top5_weighted_fit) / mean(topK_weighted_fit)`
  Simple English: A high concentration ratio means a score is driven by a few very strong jobs rather than broad support.
- bimodality flag if `CR > 1.5` and `IQR > 0.3`
  - Derived from implementation: `bimodal_flag = int((concentration_ratio > 1.5) and (iqr > 0.3))` in `src/module_readiness/scoring.py`.

`TDAS` (Technical and Domain Alignment Score) is mean technical fit over top-K.
`TSS` (Transferable Skills Signal) is mean transfer fit over top-K.

## 6) Support-adjusted family scoring

### 6.1 Why support adjustment is needed
A plain family mean can let a single high-scoring job dominate the final module-family ranking. That is undesirable when another family has many strong supporting jobs with a slightly lower mean.

### 6.2 Current adjustment
The current pipeline keeps two values:

1. `raw_role_score`: the mean of the top supporting module-job RRF scores within that family.
2. `support_weight`: a shrinkage factor based on evidence count.

The final family score is:

`role_score = raw_role_score * support_weight`

where:

`support_weight = evidence_job_count / (evidence_job_count + role_support_prior)`

This keeps singleton families visible, but prevents them from outranking well-supported families purely because of one extreme match.

### 6.3 Interpretation
- `raw_role_score` tells you how strong the within-family matches are on average.
- `support_weight` tells you how much evidence exists for that family.
- `role_score` is the ranking score used downstream.

## 7) Defaults, assumptions, and rationale

### 7.1 Core defaults
- `exp_max = 2`
- `primary_ssec_eqa = 70`
- `top_k = 50`
- `tfidf_min_df = 3`
- `tfidf_max_features = 20000`
- `entropy_penalty = 0.8`
- `weight_lexical = 0.50`
- `weight_skill = 0.50`
- `weight_transfer_skill = 0.70`
- `weight_transfer_cue = 0.30`
- `weight_technical_final = 0.90`
- `weight_transfer_final = 0.10`
- `nusmods_max_modules = 1200`

### 7.2 Why these defaults are technically defensible
- early-career and degree-level filters align target variable with fresh-graduate labor market.
- entropy weighting reduces high-frequency generic-skill dominance.
- lexical + explicit skills blend improves explainability and mitigates sparse-skill-only failure modes.
- top-K aggregation reduces noise from long-tail low-similarity jobs.

## 8) Decisions with weaker backing (explicitly flagged)

The following choices are implementation-pragmatic and not strongly empirically calibrated in the current build.

1. **Manual weight values** (`0.50/0.50`, `0.90/0.10`, `0.70/0.30`).
- Status: weak empirical backing in this version.
- Why flagged: no supervised tuning against labeled module-job relevance data.
- Further research: Bayesian optimization, cross-validated ranking objective, or pairwise learning-to-rank with human labels.

2. **Role-family mapping tables and keyword rules are handcrafted.**
- Status: partially backed (SSOC backbone is strong; fallback layers are heuristic).
- Further research: estimate confusion matrices against expert-labeled role-family assignments.

3. **Module profile inference by prefix and seed terms.**
- Status: heuristic.
- Further research: program handbook integration and official curriculum requirement parsing.

4. **Gap summary supply proxy uses top-2 role assignment per module.**
- Status: heuristic simplification.
- Further research: soft assignment with probabilistic contribution weights per module-role pair.

5. **Module cap (`nusmods_max_modules`) introduces selection effects.**
- Status: mitigated by round-robin prefix sampling, but still a truncation.
- Further research: full-catalog processing with batch compute scaling or stratified weighting.

## 9) Robustness and validation status

Existing test coverage includes:
- ingestion integrity checks for jobs and Graduate Employment Survey
- skill alias and channel split checks
- scoring formula sanity check
- behavioral sanity checks for non-uniform module family scores

Current gaps in validation:
- no gold-standard labeled relevance set
- no temporal holdout validation by posting week
- no causal validation between module scores and realized graduate outcomes


## 10) Reproducibility and provenance

- Config: `test2/config/pipeline_config.yaml`
- Role rules: `test2/config/role_family_rules.yaml`
- Skill aliases: `test2/config/skill_aliases.yaml`
- Skills mapping: `test2/config/skillsfuture_mapping.csv`
- Main pipeline: `test2/src/module_readiness/pipeline.py`
- Diagnostics artifact: `test2/outputs/diagnostics.json`

This document describes the implemented pipeline as of 2026-03-20 and is intended for technical defense with clear separation between strongly justified components and heuristic components requiring further research.
