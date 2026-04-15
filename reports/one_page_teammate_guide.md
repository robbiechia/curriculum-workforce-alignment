# One-Page Teammate Guide: How to Use Test2 Workflow and Results

## 1) What this project answers
This pipeline measures how well **individual National University of Singapore (NUS) modules** align with early-career university-level jobs.

It does this by filtering job ads to:
- `minimumYearsExperience <= 2`
- `ssecEqa = 70` (Singapore Standard Educational Classification (SSEC) Education Qualification Attained code for degree-level jobs)

Role grouping is led by **Singapore Standard Occupational Classification (SSOC)**, with job categories as supporting labels.

## 2) Quick workflow (what teammates should do)
1. Run pipeline:
```powershell
.\projvenv\Scripts\python.exe test2\scripts\run_test2_pipeline.py --quick
```
2. Open files in this order:
- `test2/outputs/diagnostics.json`
- `test2/outputs/module_summary.csv`
- `test2/outputs/module_role_scores.csv`
- `test2/outputs/module_job_evidence.csv`
- `test2/outputs/module_gap_summary.csv`
- `test2/outputs/policy_brief.md`
3. If needed, run query demo:
```powershell
.\projvenv\Scripts\python.exe test2\scripts\run_test2_queries.py --quick --query "entry level data analyst python sql" --top-k 5
```

## 3) How to read each output file

### A) `module_summary.csv` (best starting table)
Each row = one module.

Key columns:
- `top_role_family`: highest-alignment role family for this module.
- `top_role_score`: strongest module-to-family fit score.
- `top_hybrid_score`: fused BM25 + embedding score for the top family.

### B) `module_role_scores.csv` (full ranking per module across families)
Each row = one module-family pair.

Key columns:
- `role_score`: final weighted fit for this module in this family.
- `hybrid_score`: average fused retrieval score across top supporting jobs.
- `q1`, `median`, `q3`, `iqr`: distribution spread of supporting job-level fits.
- `concentration_ratio`: whether score is broad-based or driven by few jobs.
- `bimodality_flag`: warning that fit distribution is uneven (interpret with care).
- `role_rank_within_module`: rank of this family for the module.

### C) `module_job_evidence.csv` (explainability table)
Each row = one module-job evidence match.

Use this to justify scores with concrete evidence:
- `bm25_score`, `embedding_score`, `rrf_score`, `weighted_fit`
- `job_title`, `company`, `role_family`, `primary_category`

### D) `module_gap_summary.csv` (policy/action table)
Each row = one role-family skill gap line.

Key columns:
- `demand_score`: job market demand signal for a skill.
- `supply_score`: module-side coverage signal for the same skill.
- `gap_score = demand_score - supply_score`
- `gap_type`: `undersupply` or `oversupply`

Use `undersupply` rows to prioritize curriculum updates.

### E) `diagnostics.json` (health check)
Confirms if run was valid and data-rich.

Check these first:
- `jobs_after_filters`
- `modules_rows`
- `modules_detail_available_rows`
- `modules_catalog_source` (`api` or `cache`)
- `scores_module_role_rows`

## 4) Interpretation rules for stakeholder discussions
- High `role_score` for a module-family pair means strong alignment to current employer demand in that family.
- Low score in unrelated families usually indicates specialization, not poor module quality.
- `hybrid_score` is the fused retrieval signal from BM25 and sentence embeddings.
- If `bimodality_flag = 1`, avoid overclaiming from one number; inspect `module_job_evidence.csv`.
- Use `module_gap_summary.csv` to convert findings into practical actions.

## 5) Example talk-track
"`CS3244` scores strongly in data-oriented families. This means its content language and skill signals match what employers ask in entry-level data jobs. It scores lower in communications-heavy families, which is expected specialization rather than failure."

## 6) Common pitfalls to avoid
- Do not use `role_score` alone without checking evidence counts and distribution fields.
- Do not interpret low transferable signal as zero transferable value; module descriptions may under-document soft skills.
- Do not treat this as graduate placement proof by module; this is alignment evidence, not causality proof.
- **Module consolidation caveat:** Some modules are consolidated from multiple variants (e.g., CS1010A, CS1010S, CS1010X merged into CS1010). The merged module shows **combined skills from all variants**, but the single module description may not reflect differences in difficulty or pedagogy. For example, CS1010A (Python) and CS1010S (Scheme) teach fundamentally different programming paradigms despite sharing the same description. If variant differences are critical for your decision, verify on NUSMods directly.
