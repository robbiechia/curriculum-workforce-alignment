from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class SentenceEmbeddingService:
    """Sentence embedding encoder with disk-based caching.

    Wraps a ``sentence-transformers`` model and caches the resulting numpy
    arrays as ``.npz`` files keyed by a SHA-256 hash of the model name,
    namespace, and full text list.  On a cache hit the model is never called,
    which makes subsequent pipeline runs much faster once the corpus is stable.

    The ``namespace`` parameter on ``encode_many`` is what keeps job, module,
    and query embeddings from sharing a cache file even if their raw text
    happens to be identical.
    """

    model_name: str
    cache_dir: Path
    batch_size: int = 32

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.backend = "sentence-transformers"
        self.model = SentenceTransformer(self.model_name)
        self.dimension = 384
        dim = getattr(self.model, "get_sentence_embedding_dimension", None) # retrieve embedding dimension for the SentenceTransformer model
        if callable(dim):
            self.dimension = int(dim())

    def _cache_key(self, texts: Sequence[str], namespace: str) -> str:
        # The namespace keeps cached job/module/query embeddings distinct even if the
        # raw text happens to match.
        payload = {
            "backend": self.backend,
            "model_name": self.model_name,
            "namespace": namespace,
            "texts": list(texts),
        }
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def encode_many(self, texts: Sequence[str], namespace: str) -> np.ndarray:
        """Encode a list of texts, returning a (N, D) float array.

        Checks the disk cache first.  If the cached file exists and loads
        cleanly, the model is skipped entirely.  Otherwise encodes the full
        list and saves the result.

        Args:
            texts: Texts to encode — order matters since the cache key is
                   computed over the full list.
            namespace: A short label like ``"jobs"`` or ``"modules"`` that
                       scopes the cache file and prevents cross-corpus collisions.

        Returns:
            Float array of shape (len(texts), embedding_dimension), L2-normalised.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=float)

        cache_key = self._cache_key(texts, namespace)
        cache_path = self.cache_dir / f"{namespace}_{cache_key}.npz"
        # Encoding dominates runtime, so always prefer a cache hit when available.
        if cache_path.exists():
            try:
                payload = np.load(cache_path)
                return np.asarray(payload["embeddings"], dtype=float)
            except Exception:
                pass

        embeddings = self.model.encode(
            sentences            = texts,
            batch_size           = int(self.batch_size),
            show_progress_bar    = True,
            convert_to_numpy     = True,
            normalize_embeddings = True,
        )
        embeddings = np.asarray(embeddings, dtype=float)
        np.savez_compressed(cache_path, embeddings=embeddings)
        return embeddings

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single string and return a 1-D float array of length ``dimension``.

        Routes through ``encode_many`` under the ``"query"`` namespace so
        ad-hoc query embeddings use the same cache machinery as corpus embeddings.
        """
        # Queries reuse the exact same code path as corpus embeddings for consistency.
        embeddings = self.encode_many(texts = [text], namespace="query")
        if embeddings.size == 0:
            return np.zeros(self.dimension, dtype=float)
        return embeddings[0]
