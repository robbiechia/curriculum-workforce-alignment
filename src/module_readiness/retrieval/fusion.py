from __future__ import annotations

from typing import Sequence

import numpy as np


def scores_to_ranks(
    scores: np.ndarray,
    allowed_indices: np.ndarray | None = None,
) -> np.ndarray:
    ranks = np.zeros(len(scores), dtype=int)
    if scores.size == 0:
        return ranks

    if allowed_indices is None:
        indices = np.arange(len(scores), dtype=int)
    else:
        indices = np.asarray(allowed_indices, dtype=int)

    if indices.size == 0:
        return ranks

    subset = scores[indices]
    order = indices[np.argsort(subset)[::-1]]
    # Example: rank[4] = 1 means the 5th corpus item is ranked first among the candidates.
    ranks[order] = np.arange(1, len(order) + 1, dtype=int)
    return ranks


def reciprocal_rank_fusion(
    rank_lists: Sequence[np.ndarray],
    rrf_k: int,
) -> np.ndarray:
    if not rank_lists:
        return np.array([], dtype=float)

    fused = np.zeros(len(rank_lists[0]), dtype=float)
    for ranks in rank_lists:
        valid = ranks > 0
        # RRF rewards agreement across rankers while damping the influence of any
        # single very deep rank position.
        fused[valid] += 1.0 / (float(rrf_k) + ranks[valid])
    return fused


def top_indices(scores: np.ndarray, top_k: int, allowed_indices: np.ndarray | None = None) -> np.ndarray:
    if scores.size == 0 or top_k <= 0:
        return np.array([], dtype=int)

    if allowed_indices is None:
        indices = np.arange(len(scores), dtype=int)
    else:
        indices = np.asarray(allowed_indices, dtype=int)

    if indices.size == 0:
        return np.array([], dtype=int)

    subset = scores[indices]
    n = min(int(top_k), len(indices))
    chosen = np.argpartition(subset, -n)[-n:]
    ordered = chosen[np.argsort(subset[chosen])[::-1]]
    return indices[ordered]
