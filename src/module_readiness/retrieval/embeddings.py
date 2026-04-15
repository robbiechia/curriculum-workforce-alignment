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
        # Queries reuse the exact same code path as corpus embeddings for consistency.
        embeddings = self.encode_many(texts = [text], namespace="query")
        if embeddings.size == 0:
            return np.zeros(self.dimension, dtype=float)
        return embeddings[0]
