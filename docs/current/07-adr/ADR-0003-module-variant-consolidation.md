# ADR-0003: Consolidate Module Variants to Base Codes

## Status
Accepted

## Date
2026-04-10

## Context
Module mappings and retrieval outputs can contain suffix variants (for example `ACC1701A`, `ACC1701B`) that refer to closely related base modules.  
Without consolidation:
- scores fragment across variant rows
- degree mapping joins become noisier
- output interpretation becomes harder for non-specialists

## Decision
Normalize module codes by removing one trailing uppercase letter and consolidate variants:
- Base code examples: `ACC1701A` -> `ACC1701`
- Merge list-like skill fields with deduplication
- Retain representative non-list metadata from first variant row

## Consequences
Positive:
- Cleaner, less fragmented module-level outputs
- Better alignment between degree mapping references and scored module keys
- Simpler presentation narrative

Negative:
- Potential loss of variant-level distinctions in final tables
- Assumes suffix variants are sufficiently similar for current analysis goals

## Alternatives considered
1. Keep all variants separate
   - Rejected: output fragmentation and interpretation burden
2. Manual curated mapping table for variants
   - Rejected for current phase due maintenance overhead

## Evidence
- Verified from:
  - `src/module_readiness/processing/module_variants.py`
  - Orchestrator stage ordering in `orchestration/pipeline.py`
- Inferred:
  - Interpretation/simplicity rationale for presentation and policy usage

## Review triggers
- If variant-level pedagogy differences become first-class analytical requirement
- If official structured variant equivalence data becomes available
