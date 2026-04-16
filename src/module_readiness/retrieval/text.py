from __future__ import annotations

import re
from typing import Iterable, List, Sequence


DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "you",
    "your",
    "we",
    "our",
}


def normalize_text(text: str) -> str:
    # Jobs, modules, and ad-hoc queries all pass through the same normalization so the
    # lexical retriever sees a consistent token space.
    cleaned = (text or "").lower()
    cleaned = re.sub(r"[^a-z0-9\s\+\#]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tokenize_text(text: str, min_len: int = 2) -> List[str]:
    """Tokenize and filter a string into a list of index-ready tokens.

    Passes through ``normalize_text`` first, then drops stopwords and tokens
    shorter than ``min_len``.  The same function is called for jobs, modules,
    and live queries so the BM25 index and query token spaces always match.
    """
    tokens = []
    for token in normalize_text(text).split():
        if len(token) < min_len or token in DEFAULT_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def build_overlap_terms(
    query_tokens: Sequence[str],
    doc_tokens: Sequence[str],
    top_n: int = 6,
) -> List[str]:
    """Return the tokens that appear in both query and document, ranked by doc frequency.

    Used to produce human-readable evidence strings on retrieved results —
    the returned tokens explain *why* a job or module ranked highly for a given
    query.  Tokens are ordered by how often they appear in the document (most
    frequent first) so the most distinctive shared terms surface at the top.
    """
    if not query_tokens or not doc_tokens:
        return []

    doc_counts = {}
    for token in doc_tokens:
        doc_counts[token] = doc_counts.get(token, 0) + 1

    seen = set()
    overlap = []
    for token in query_tokens:
        if token in seen:
            continue
        seen.add(token)
        if token in doc_counts:
            overlap.append((token, doc_counts[token]))

    overlap.sort(key=lambda item: (-item[1], item[0]))
    return [token for token, _ in overlap[:top_n]]


def _serialize_terms(prefix: str, values: Iterable[str]) -> str:
    items = [normalize_text(str(value)) for value in values if str(value).strip()]
    items = [item for item in items if item]
    if not items:
        return ""
    return f"{prefix} " + " ".join(items)


def build_retrieval_text(
    base_text: str,
    technical_skills: Iterable[str],
) -> str:
    """Combine free text and skill tokens into a single retrieval string.

    Skills are deliberately appended as a separate "technical skills ..." section
    rather than just included in the base text.  This gives BM25 an extra hit for
    exact skill matches — e.g. a module mentioning "python" gets a double reward
    when the job also lists "python" as a skill — without requiring a separate
    skill-specific index.
    """
    # Skills are appended as an explicit section so BM25 can reward exact skill overlap
    # without needing a separate index.
    parts = [
        normalize_text(base_text),
        _serialize_terms("technical skills", technical_skills),
    ]
    return " ".join(part for part in parts if part).strip()
