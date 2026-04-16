from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers.util import cos_sim

from ..config.settings import PipelineConfig
from .embeddings import SentenceEmbeddingService
from .fusion import reciprocal_rank_fusion, scores_to_ranks, top_indices
from .text import build_overlap_terms, build_retrieval_text, tokenize_text


RetrievalMode = Literal["hybrid", "bm25", "embedding"]
VALID_RETRIEVAL_MODES = ("hybrid", "bm25", "embedding")

# Number of overlapping keyword terms kept as human-readable evidence text on
# each retrieved result. Increasing this makes evidence strings verbose without
# adding meaningful signal.
EVIDENCE_TERMS_TOP_N = 6


def _build_bm25_index(
    tokenized_docs: List[List[str]],
    *,
    k1: float,
    b: float,
) -> BM25Okapi | None:
    if not tokenized_docs:
        return None
    return BM25Okapi(tokenized_docs, k1=float(k1), b=float(b))


def _bm25_scores(index: BM25Okapi | None, query_tokens: Sequence[str], doc_count: int) -> np.ndarray:
    if index is None or doc_count == 0 or not query_tokens:
        return np.zeros(doc_count, dtype=float)
    return np.asarray(index.get_scores(list(query_tokens)), dtype=float)


def _candidate_indices(
    scores: np.ndarray,
    absolute_min_score: float,
    relative_min_score: float,
    allowed_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return the indices that pass both score thresholds.

    A candidate must clear an absolute floor (e.g. BM25 score ≥ 0.5) AND a
    score-relative floor (e.g. ≥ 10 % of the highest score in this query).
    Using both prevents noisy low-quality matches from entering fusion while
    still adapting to queries that produce universally weak BM25 scores.

    Args:
        scores: Full score array over the corpus.
        absolute_min_score: Hard lower bound — any score below this is excluded.
        relative_min_score: Fraction of the max score that a candidate must reach.
        allowed_indices: Optional pre-filter; only indices in this set are considered.

    Returns:
        Array of integer indices that satisfy both thresholds.
    """
    if scores.size == 0:
        return np.array([], dtype=int)

    if allowed_indices is None:
        indices = np.arange(len(scores), dtype=int)
    else:
        indices = np.asarray(allowed_indices, dtype=int)
    if indices.size == 0:
        return indices

    max_score = float(scores[indices].max())
    if max_score <= 0:
        return np.array([], dtype=int)

    # A score must clear both a global floor and a score-relative floor to become a
    # fusion candidate.
    threshold = max(float(absolute_min_score), float(relative_min_score) * max_score)
    return indices[scores[indices] >= threshold]


@dataclass
class CorpusArtifacts:
    texts: List[str]
    tokens: List[List[str]]
    embeddings: np.ndarray
    bm25: BM25Okapi | None


@dataclass
class RetrievalArtifacts:
    jobs: CorpusArtifacts
    modules: CorpusArtifacts
    embedding_service: SentenceEmbeddingService
    diagnostics: Dict[str, float | str]


class HybridRetrievalEngine:
    def __init__(self, config: PipelineConfig, artifacts: RetrievalArtifacts):
        self.config = config
        self.artifacts = artifacts
        self.embedding_service = artifacts.embedding_service

    @staticmethod
    def _empty_ranking_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "index",
                "ranking_mode",
                "ranking_score",
                "ranking_score_normalized",
                "bm25_score",
                "embedding_score",
                "bm25_rank",
                "embedding_rank",
                "rrf_score",
                "evidence_terms",
            ]
        )

    def _rank_against_corpus(
        self,
        query_tokens: Sequence[str],
        query_embedding: np.ndarray,
        corpus: CorpusArtifacts,
        top_k: int,
        mode: RetrievalMode = "hybrid",
        allowed_indices: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        This function takes in a query (can be job or module), in two forms: 
        1. Tokenized form (which will be a list of strings)
        2. Embedding form (which will just be a n-dimensional array)
        It also takes in the corpus (which we will compare the query against)

        It then proceeds as follows:
        1. It calculates the BM25 score against that corpus to get an array of scores, and also the cosine similarity score against the embeddings to get another array of scores
        2. Converts both score arrays into ranks
        3. fuses the two rankings with Reciprocal Rank Fusion (RRF)
        4. return a dataframe of the top_k results with scores, and some keyword overlaps between the query and the document in the corpus.
        """
        if mode not in VALID_RETRIEVAL_MODES:
            raise ValueError(
                f"Unsupported retrieval mode '{mode}'. Expected one of {', '.join(VALID_RETRIEVAL_MODES)}."
            )

        bm25_scores = _bm25_scores(corpus.bm25, query_tokens, len(corpus.texts))
        embedding_scores = (
            cos_sim(np.asarray([query_embedding]), corpus.embeddings).cpu().numpy().ravel()
        )
        embedding_scores = np.clip(embedding_scores, 0.0, None)

        # BM25 and embedding candidates are thresholded independently, then fused with RRF.
        bm25_candidates = _candidate_indices(
            bm25_scores,
            absolute_min_score=float(self.config.bm25_min_score),
            relative_min_score=float(self.config.bm25_relative_min),
            allowed_indices=allowed_indices,
        )
        embedding_candidates = _candidate_indices(
            embedding_scores,
            absolute_min_score=float(self.config.embedding_min_similarity),
            relative_min_score=float(self.config.embedding_relative_min),
            allowed_indices=allowed_indices,
        )
        fused_candidates = np.union1d(bm25_candidates, embedding_candidates)

        bm25_ranks = scores_to_ranks(bm25_scores, allowed_indices=bm25_candidates)
        embedding_ranks = scores_to_ranks(embedding_scores, allowed_indices=embedding_candidates)
        rrf_scores = reciprocal_rank_fusion(
            [bm25_ranks, embedding_ranks],
            rrf_k=int(self.config.rrf_k),
        )
        # Normalize the fused scores so downstream tables and dashboards can compare them
        # on a stable [0, 1] scale.
        max_rrf = float(rrf_scores.max()) if rrf_scores.size else 0.0
        if max_rrf > 0:
            rrf_scores = rrf_scores / max_rrf

        if mode == "hybrid":
            ranking_scores = rrf_scores
            selected_candidates = fused_candidates
        elif mode == "bm25":
            ranking_scores = bm25_scores
            selected_candidates = bm25_candidates
        else:
            ranking_scores = embedding_scores
            selected_candidates = embedding_candidates

        if selected_candidates.size == 0:
            return self._empty_ranking_frame()

        max_selected_score = float(ranking_scores[selected_candidates].max())
        top_idx = top_indices(ranking_scores, top_k=top_k, allowed_indices=selected_candidates)
        rows = []
        for idx in top_idx:
            ranking_score = float(ranking_scores[idx])
            normalized_score = 0.0
            if max_selected_score > 0:
                normalized_score = float(ranking_score / max_selected_score)
            rows.append(
                {
                    "index": int(idx),
                    "ranking_mode": mode,
                    "ranking_score": ranking_score,
                    "ranking_score_normalized": normalized_score,
                    "bm25_score": float(bm25_scores[idx]),
                    "embedding_score": float(embedding_scores[idx]),
                    "bm25_rank": int(bm25_ranks[idx]) if bm25_ranks[idx] > 0 else None,
                    "embedding_rank": int(embedding_ranks[idx]) if embedding_ranks[idx] > 0 else None,
                    "rrf_score": float(rrf_scores[idx]),
                    "evidence_terms": " | ".join(
                        build_overlap_terms(query_tokens, corpus.tokens[idx], top_n=EVIDENCE_TERMS_TOP_N)
                    ),
                }
            )
        return pd.DataFrame(rows)

    def rank_jobs_from_module(
        self,
        module_index: int,
        top_k: int,
        mode: RetrievalMode = "hybrid",
        allowed_indices: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Given a module, return the top_k jobs that are most relevant to this module
        """
        return self._rank_against_corpus(
            self.artifacts.modules.tokens[module_index],
            self.artifacts.modules.embeddings[module_index],
            self.artifacts.jobs,
            top_k,
            mode=mode,
            allowed_indices=allowed_indices,
        )

    def rank_modules_from_job_index(
        self,
        job_index: int,
        top_k: int,
        mode: RetrievalMode = "hybrid",
        *,
        allowed_indices: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Given a job index, return the top_k modules most relevant to that job.

        Mirror of rank_jobs_from_module — used during retrieval evaluation when
        we want to invert the query direction (job → modules rather than module → jobs).
        """
        return self._rank_against_corpus(
            self.artifacts.jobs.tokens[job_index],
            self.artifacts.jobs.embeddings[job_index],
            self.artifacts.modules,
            top_k,
            mode=mode,
            allowed_indices=allowed_indices,
        )

    def rank_jobs_from_text(
        self,
        query_text: str,
        top_k: int,
        mode: RetrievalMode = "hybrid",
        *,
        allowed_indices: np.ndarray | None = None,
    ) -> pd.DataFrame:
        query_tokens = tokenize_text(query_text)
        query_embedding = self.embedding_service.encode_one(query_text)
        return self._rank_against_corpus(
            query_tokens,
            query_embedding,
            self.artifacts.jobs,
            top_k,
            mode=mode,
            allowed_indices=allowed_indices,
        )

    def rank_modules_from_text(
        self,
        query_text: str,
        top_k: int,
        mode: RetrievalMode = "hybrid",
        *,
        allowed_indices: np.ndarray | None = None,
    ) -> pd.DataFrame:
        query_tokens = tokenize_text(query_text)
        query_embedding = self.embedding_service.encode_one(query_text)
        return self._rank_against_corpus(
            query_tokens,
            query_embedding,
            self.artifacts.modules,
            top_k,
            mode=mode,
            allowed_indices=allowed_indices,
        )

    def rank_modules_from_job_indices(
        self,
        job_indices: Sequence[int],
        top_k: int,
        *,
        allowed_indices: np.ndarray | None = None,
    ) -> pd.DataFrame:
        if not job_indices:
            return pd.DataFrame()

        module_count = len(self.artifacts.modules.texts)
        total_rrf = np.zeros(module_count, dtype=float)
        total_bm25 = np.zeros(module_count, dtype=float)
        total_embedding = np.zeros(module_count, dtype=float)
        bm25_rank_sum = np.zeros(module_count, dtype=float)
        embedding_rank_sum = np.zeros(module_count, dtype=float)
        bm25_rank_hits = np.zeros(module_count, dtype=float)
        embedding_rank_hits = np.zeros(module_count, dtype=float)

        # Aggregate evidence across several jobs to support workflows where a user selects
        # multiple postings as the target set.
        for job_index in job_indices:
            query_tokens = self.artifacts.jobs.tokens[job_index]
            query_embedding = self.artifacts.jobs.embeddings[job_index]
            bm25_scores = _bm25_scores(
                self.artifacts.modules.bm25,
                query_tokens,
                len(self.artifacts.modules.texts),
            )
            embedding_scores = (
                cos_sim(
                    np.asarray([query_embedding]),
                    self.artifacts.modules.embeddings,
                )
                .cpu()
                .numpy()
                .ravel()
            )
            embedding_scores = np.clip(embedding_scores, 0.0, None)
            bm25_candidates = _candidate_indices(
                bm25_scores,
                absolute_min_score=float(self.config.bm25_min_score),
                relative_min_score=float(self.config.bm25_relative_min),
                allowed_indices=allowed_indices,
            )
            embedding_candidates = _candidate_indices(
                embedding_scores,
                absolute_min_score=float(self.config.embedding_min_similarity),
                relative_min_score=float(self.config.embedding_relative_min),
                allowed_indices=allowed_indices,
            )
            bm25_ranks = scores_to_ranks(bm25_scores, allowed_indices=bm25_candidates)
            embedding_ranks = scores_to_ranks(embedding_scores, allowed_indices=embedding_candidates)
            total_rrf += reciprocal_rank_fusion(
                [bm25_ranks, embedding_ranks],
                rrf_k=int(self.config.rrf_k),
            )
            total_bm25 += bm25_scores
            total_embedding += embedding_scores

            bm25_hit_mask = bm25_ranks > 0
            bm25_rank_sum[bm25_hit_mask] += bm25_ranks[bm25_hit_mask]
            bm25_rank_hits[bm25_hit_mask] += 1

            embedding_hit_mask = embedding_ranks > 0
            embedding_rank_sum[embedding_hit_mask] += embedding_ranks[embedding_hit_mask]
            embedding_rank_hits[embedding_hit_mask] += 1

        max_rrf = float(total_rrf.max()) if total_rrf.size else 0.0
        if max_rrf > 0:
            total_rrf = total_rrf / max_rrf

        if allowed_indices is None:
            fused_candidates = np.flatnonzero(total_rrf > 0)
        else:
            allowed = np.asarray(allowed_indices, dtype=int)
            fused_candidates = allowed[total_rrf[allowed] > 0]
        top_idx = top_indices(total_rrf, top_k=top_k, allowed_indices=fused_candidates)
        merged_tokens = []
        for job_index in job_indices:
            merged_tokens.extend(self.artifacts.jobs.tokens[job_index])

        rows = []
        job_count = float(len(job_indices))
        for idx in top_idx:
            bm25_rank = None
            if bm25_rank_hits[idx] > 0:
                bm25_rank = int(round(bm25_rank_sum[idx] / bm25_rank_hits[idx]))

            embedding_rank = None
            if embedding_rank_hits[idx] > 0:
                embedding_rank = int(round(embedding_rank_sum[idx] / embedding_rank_hits[idx]))

            rows.append(
                {
                    "index": int(idx),
                    "bm25_score": float(total_bm25[idx] / job_count),
                    "embedding_score": float(total_embedding[idx] / job_count),
                    "bm25_rank": bm25_rank,
                    "embedding_rank": embedding_rank,
                    "rrf_score": float(total_rrf[idx]),
                    "evidence_terms": " | ".join(
                        build_overlap_terms(merged_tokens, self.artifacts.modules.tokens[idx], top_n=EVIDENCE_TERMS_TOP_N)
                    ),
                }
            )
        return pd.DataFrame(rows)


def _build_retrieval_text_series(
    frame: pd.DataFrame,
    *,
    base_columns: Sequence[str],
) -> pd.Series:
    """Build a retrieval-ready text string for each row in frame.

    Concatenates the specified base columns then appends normalized technical
    skills so that exact skill tokens boost BM25 recall without duplicating
    the full description text.

    Args:
        frame: DataFrame of jobs or modules.
        base_columns: Column names whose text content forms the base query string.

    Returns:
        A string Series aligned to frame.index.
    """
    values = []
    for _, row in frame.iterrows():
        base_text = " ".join(
            str(row.get(column, "")).strip()
            for column in base_columns
            if str(row.get(column, "")).strip()
        )
        values.append(
            build_retrieval_text(
                base_text,
                row.get("technical_skills", []),
            )
        )
    return pd.Series(values, index=frame.index, dtype="object")


def build_retrieval_artifacts(
    config: PipelineConfig,
    jobs: pd.DataFrame,
    modules: pd.DataFrame,
) -> RetrievalArtifacts:
    """Build all retrieval artifacts needed by HybridRetrievalEngine.

    Constructs BM25 indices and sentence embeddings for both the job and module
    corpora. Results are cached on disk (keyed by a SHA-256 of the text) so
    repeated pipeline runs skip re-encoding unchanged documents.

    Args:
        config: Pipeline configuration (model name, cache paths, BM25 params, etc.).
        jobs: Filtered, enriched jobs DataFrame from the ingestion stage.
        modules: Consolidated modules DataFrame from the processing stage.

    Returns:
        RetrievalArtifacts containing CorpusArtifacts for jobs and modules,
        the shared embedding service, and diagnostics.
    """
    jobs = jobs.copy()
    modules = modules.copy()

    jobs["retrieval_text"] = _build_retrieval_text_series(
        jobs,
        base_columns=("job_text",),
    )
    modules["retrieval_text"] = _build_retrieval_text_series(
        modules,
        base_columns=("module_title", "module_description"),
    )

    job_tokens = [tokenize_text(text) for text in jobs["retrieval_text"].fillna("")]
    module_tokens = [tokenize_text(text) for text in modules["retrieval_text"].fillna("")]

    embedding_service = SentenceEmbeddingService(
        model_name=config.embedding_model_name,
        cache_dir=config.cache_dir / config.embedding_cache_dir,
        batch_size=int(config.embedding_batch_size),
    )

    job_embeddings = embedding_service.encode_many(
        jobs["retrieval_text"].fillna("").tolist(),
        namespace="jobs",
    )
    module_embeddings = embedding_service.encode_many(
        modules["retrieval_text"].fillna("").tolist(),
        namespace="modules",
    )

    diagnostics: Dict[str, float | str] = {
        "retrieval_jobs_rows": float(len(jobs)),
        "retrieval_modules_rows": float(len(modules)),
        "retrieval_embedding_backend": embedding_service.backend,
        "retrieval_embedding_dimension": float(embedding_service.dimension),
        "retrieval_job_token_count_mean": float(np.mean([len(tokens) for tokens in job_tokens])) if job_tokens else 0.0,
        "retrieval_module_token_count_mean": float(np.mean([len(tokens) for tokens in module_tokens])) if module_tokens else 0.0,
    }

    return RetrievalArtifacts(
        jobs=CorpusArtifacts(
            texts=jobs["retrieval_text"].fillna("").tolist(),
            tokens=job_tokens,
            embeddings=job_embeddings,
            bm25=_build_bm25_index(
                job_tokens,
                k1=float(config.bm25_k1),
                b=float(config.bm25_b),
            ),
        ),
        modules=CorpusArtifacts(
            texts=modules["retrieval_text"].fillna("").tolist(),
            tokens=module_tokens,
            embeddings=module_embeddings,
            bm25=_build_bm25_index(
                module_tokens,
                k1=float(config.bm25_k1),
                b=float(config.bm25_b),
            ),
        ),
        embedding_service=embedding_service,
        diagnostics=diagnostics,
    )
