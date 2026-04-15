# ADR-0002: Hybrid Retrieval with Reciprocal Rank Fusion

## Status
Accepted

## Date
2026-04-10

## Context
Module-job similarity based on a single retriever can underperform across mixed text styles:
- BM25 captures lexical overlap strongly
- Embeddings capture semantic similarity better for paraphrase and sparse keyword overlap

The system needs robust evidence ranking with explainable components.

## Decision
Use a hybrid retriever:
- Compute BM25 scores and embedding cosine similarities
- Threshold candidates independently per retriever
- Fuse ranked candidates via reciprocal rank fusion (RRF)
- Expose both component scores and fused score in evidence outputs

## Consequences
Positive:
- Better robustness across lexical and semantic match cases
- Transparent evidence fields (`bm25_score`, `embedding_score`, `rrf_score`)
- Supports comparative evaluation modes (`hybrid`, `bm25`, `embedding`)

Negative:
- More tuning parameters
- Requires embedding model availability/caching

## Alternatives considered
1. BM25-only retrieval
   - Rejected: weaker semantic recall in many cases
2. Embedding-only retrieval
   - Rejected: weaker exact keyword precision and term overlap transparency
3. Weighted linear score fusion
   - Rejected: rank-based fusion selected for stability against score scale mismatch

## Evidence
- Verified from:
  - `src/module_readiness/retrieval/engine.py`
  - `src/module_readiness/retrieval/fusion.py`
  - `config/pipeline_config.yaml` retrieval parameters
- Inferred:
  - Stability rationale for rank-based fusion over direct score fusion

## Review triggers
- If labeled relevance data supports a clearly superior learned ranker
- If operational constraints remove embedding capability
