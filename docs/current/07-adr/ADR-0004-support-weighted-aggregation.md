# ADR-0004: Support-Weighted Aggregation for Role and Degree Scores

## Status
Accepted

## Date
2026-04-10

## Context
Raw mean scores can over-reward targets with very few supporting evidence rows.  
The system needs scores that reflect both alignment strength and evidence support.

## Decision
Use support-weighted aggregation:
- Compute raw score as mean over top supporting evidence rows
- Compute support weight:
  - `support_weight = evidence_count / (evidence_count + prior)`
- Final score:
  - `final_score = raw_score * support_weight`

Apply this pattern in:
- module role/occupation aggregation
- degree role/occupation aggregation

Additional interpretability rule:
- Near-tie suppression for `Other` role label in module role ranking to prefer named clusters when effectively tied.

## Consequences
Positive:
- Reduces singleton/outlier dominance
- Improves ranking stability and interpretability
- Makes support level explicit in outputs

Negative:
- Adds prior hyperparameter sensitivity
- Score interpretation requires understanding support-adjustment behavior

## Alternatives considered
1. Raw mean only
   - Rejected: weak support signals can dominate
2. Hard minimum evidence cutoff
   - Rejected: too aggressive; discards potentially useful sparse evidence

## Evidence
- Verified from:
  - `src/module_readiness/analysis/scoring.py`
  - `src/module_readiness/analysis/degrees.py`
  - Config priors in `config/pipeline_config.yaml`
- Inferred:
  - Rationale for smoother shrinkage vs hard cutoff

## Review triggers
- If evaluation demonstrates better calibration with alternative shrinkage formulations
- If support priors require domain-specific retuning
