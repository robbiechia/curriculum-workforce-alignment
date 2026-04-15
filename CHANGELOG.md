# Changelog: Module Readiness Pipeline (Test2)

## 2026-03-25: Removal of Secondary Indicators (TI and EI)

### What Changed

**Secondary indicators (Transferability Index and Employability Index) have been removed from the entire project.**

This simplification focuses the scoring system on primary metrics:
- `role_score`: Final weighted fit score for module-role family
- `raw_role_score`: Mean fused retrieval score before support adjustment
- `support_weight`: Evidence-count shrinkage factor

### Why This Change

The removed indicators (TI and EI) were:
- **TI (Transferability Index)**: Computed entropy across role-family scores - useful but secondary to understanding module specialization
- **EI (Employability Index)**: Contextual signal from GES degree outcomes - informative but not module-specific evidence

Both were post-hoc computations not part of the core scoring formula, and their removal simplifies outputs without affecting the primary alignment measurement.

### Files Modified

#### Code Changes
- `src/module_readiness/aggregation.py`
  - Removed `_transferability_index()` function
  - Removed TI and EI calculation blocks
  - Updated empty DataFrame schemas to exclude ti/ei columns
  
- `src/module_readiness/reporting.py`
  - Removed TI and EI from column rename mappings
  - Removed TI and EI from output table headers

- `scripts/run_test2_pipeline.py`
  - Removed ti and ei from console output display columns

- `tests/test_behavioral_sanity.py`
  - Removed `test_transferability_index_in_valid_range()` test method

#### Documentation Updates
- `reports/one_page_teammate_guide.md`
  - Removed ti/ei from module_summary.csv column descriptions
  - Updated interpretation rules to focus on score and evidence fields

- `reports/technical_report_deep_dive.md`
  - Removed Section 6.1 (Transferability Index definition)
  - Removed Section 6.2 (Employability Index definition)
  - Consolidated role-scoring explanation as Section 6.1
  - Removed EI from assumptions section (Section 8.1)
  - Updated numbering for subsequent sections
  - Removed TI range check from validation test coverage

- `reports/technical_report_test2.md`
  - Updated runtime diagnostics to reflect current pipeline output (1200 modules, expanded metrics)

### Output File Changes

**CSV Outputs** (`module_role_scores.csv`, `module_summary.csv`):
- `ti` column: **REMOVED**
- `ei` column: **REMOVED**
- All other columns unchanged

**Console/Report Outputs**:
- TI and EI metrics no longer displayed
- Primary metrics now center on `role_score`, `raw_role_score`, and `support_weight`

### Validated Working

- ✅ Pipeline execution successful (exit code 0)
- ✅ Output CSV files cleaned and verified
- ✅ Test suite passes (1 test, 143 seconds)
- ✅ All reports updated for consistency

### Backward Compatibility Notes

If you have scripts or analyses relying on the ti/ei columns:
1. Update column selections to remove references to ti and ei
2. Use evidence-aware scoring fields instead of deprecated summary heuristics
3. Refer to statistical distribution metrics (q1, median, q3, iqr, concentration_ratio) for detailed alignment patterns

### Next Steps

1. Regenerate any downstream analyses that depend on TI/EI columns
2. Update stakeholder communication to reference the current scoring fields
3. Consider future enhancements: module-level outcome tracking, supervised weight calibration

---

**For questions or clarification**, refer to:
- `reports/technical_report_deep_dive.md` for technical details
- `reports/one_page_teammate_guide.md` for workflow and output interpretation
- Code comments in `src/module_readiness/aggregation.py` for implementation details
