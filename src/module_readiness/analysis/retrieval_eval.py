from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers.util import cos_sim

from ..retrieval import HybridRetrievalEngine
from ..retrieval.engine import RetrievalMode, _bm25_scores, _candidate_indices
from ..retrieval.fusion import reciprocal_rank_fusion, scores_to_ranks, top_indices


DEFAULT_RETRIEVAL_EVAL_MODES: tuple[RetrievalMode, ...] = ("hybrid", "bm25", "embedding")
POSITIVE_RELEVANCE_THRESHOLD = 2.0


@dataclass
class _CachedModuleRetrievalEvaluation:
    module_code: str
    module_title: str
    labeled_pairs: int
    positive_labels: int
    label_map: dict[str, float]
    bm25_scores: np.ndarray
    embedding_scores: np.ndarray


def _normalize_codes(values: Iterable[object]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        code = str(value).strip()
        if code:
            normalized.append(code)
    return normalized


def _stringify_list(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return "; ".join(items)
    return str(value).strip()


def _module_faculty_label(row: pd.Series) -> str:
    for column in ("module_faculty", "faculty", "module_profile"):
        value = str(row.get(column, "")).strip()
        if value:
            return value
    return "Unknown"


def _prepare_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the labelled file for validation
    1. strips whitespace of module_code, job_id, relevance columns
    2. converts relevance to numeric
    3. deduplicate similar rows, keeping the maximum relevanc
    """
    required_cols = {"module_code", "job_id", "relevance"}
    missing = sorted(required_cols - set(labels.columns))
    if missing:
        raise ValueError(f"Labels file is missing required columns: {', '.join(missing)}")

    out = labels.copy()
    out["module_code"] = out["module_code"].astype(str).str.strip()
    out["job_id"] = out["job_id"].astype(str).str.strip()
    out["relevance"] = pd.to_numeric(out["relevance"], errors="coerce")
    out = out[
        out["module_code"].ne("")
        & out["job_id"].ne("")
        & out["relevance"].notna()
    ].copy()
    if out.empty:
        raise ValueError("Labels file does not contain any usable module_code, job_id, relevance rows.")

    # If the same pair is labeled multiple times, keep the most relevant judgment.
    out = (
        out.groupby(["module_code", "job_id"], as_index=False)["relevance"]
        .max()
        .sort_values(["module_code", "job_id"])
        .reset_index(drop=True)
    )
    return out


def select_modules_for_evaluation(
    modules: pd.DataFrame,
    *,
    module_codes: Sequence[str] | None = None,
    sample_size: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    if "module_code" not in modules.columns:
        raise ValueError("Modules frame must contain a module_code column.")

    if module_codes:
        requested = _normalize_codes(module_codes)
        if not requested:
            raise ValueError("At least one non-empty module code is required.")

        selected = modules[modules["module_code"].astype(str).isin(requested)].copy()
        present_codes = selected["module_code"].astype(str).tolist()
        missing = [code for code in requested if code not in present_codes]
        if missing:
            raise ValueError(
                "Requested module codes were not found in the module corpus: "
                + ", ".join(missing)
            )

        order = {code: position for position, code in enumerate(requested)}
        selected["__order"] = selected["module_code"].astype(str).map(order)
        return selected.sort_values("__order").drop(columns="__order")

    if sample_size is None or int(sample_size) <= 0:
        raise ValueError("Provide either module_codes or a positive sample_size.")

    sample_n = min(int(sample_size), len(modules))
    working = modules.copy()
    working["__faculty_label"] = working.apply(_module_faculty_label, axis=1)

    # Build a faculty-stratified module order by shuffling within each faculty bucket
    # and then drawing round-robin across faculties. This avoids a purely random slice
    # while still remaining deterministic for a fixed seed.
    faculty_groups: dict[str, list[int]] = {}
    for faculty, group in working.groupby("__faculty_label", sort=True):
        sampled = group.sample(frac=1.0, random_state=int(seed) + len(faculty))
        faculty_groups[str(faculty)] = sampled.index.tolist()

    ordered_indices: list[int] = []
    faculties = sorted(faculty_groups.keys())
    while len(ordered_indices) < len(working):
        progressed = False
        for faculty in faculties:
            bucket = faculty_groups[faculty]
            if not bucket:
                continue
            ordered_indices.append(bucket.pop(0))
            progressed = True
            if len(ordered_indices) >= len(working):
                break
        if not progressed:
            break

    ordered = working.loc[ordered_indices].drop(columns="__faculty_label")
    return ordered.head(sample_n)


def split_selected_modules_for_evaluation(
    selected_modules: pd.DataFrame,
    *,
    test_size: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "module_code" not in selected_modules.columns:
        raise ValueError("Selected modules frame must contain a module_code column.")

    total_modules = len(selected_modules)
    if total_modules < 2:
        raise ValueError("At least two modules are required to form a train/test split.")

    requested_test_size = int(test_size)
    if requested_test_size <= 0 or requested_test_size >= total_modules:
        raise ValueError(
            f"test_size must be between 1 and {total_modules - 1}, got {requested_test_size}."
        )

    working = selected_modules.copy().reset_index(drop=False).rename(
        columns={"index": "__module_index"}
    )
    working["__selection_order"] = np.arange(len(working), dtype=int)
    working["__faculty_label"] = working.apply(_module_faculty_label, axis=1)

    faculty_counts = working["__faculty_label"].value_counts().sort_index()
    ideal_test_counts = faculty_counts.astype(float) * float(requested_test_size) / float(total_modules)
    faculty_test_counts = ideal_test_counts.apply(np.floor).astype(int)

    remaining = int(requested_test_size - faculty_test_counts.sum())
    if remaining > 0:
        faculty_priority = sorted(
            faculty_counts.index.tolist(),
            key=lambda faculty: (
                -(float(ideal_test_counts[faculty]) - float(faculty_test_counts[faculty])),
                -int(faculty_counts[faculty]),
                str(faculty),
            ),
        )
        for faculty in faculty_priority:
            if remaining <= 0:
                break
            if int(faculty_test_counts[faculty]) >= int(faculty_counts[faculty]):
                continue
            faculty_test_counts[faculty] += 1
            remaining -= 1

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for faculty, group in working.groupby("__faculty_label", sort=True):
        shuffled = group.sample(frac=1.0, random_state=int(seed) + len(str(faculty)))
        test_count = int(faculty_test_counts.get(faculty, 0))
        test_parts.append(shuffled.head(test_count))
        train_parts.append(shuffled.iloc[test_count:])

    test_modules = (
        pd.concat(test_parts, ignore_index=False)
        .sort_values("__selection_order")
        .drop(columns=["__module_index", "__selection_order", "__faculty_label"])
    )
    train_modules = (
        pd.concat(train_parts, ignore_index=False)
        .sort_values("__selection_order")
        .drop(columns=["__module_index", "__selection_order", "__faculty_label"])
    )
    return train_modules, test_modules


def split_modules_for_evaluation(
    modules: pd.DataFrame,
    *,
    module_codes: Sequence[str] | None = None,
    sample_size: int | None = None,
    test_size: int | None = None,
    test_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_modules = select_modules_for_evaluation(
        modules,
        module_codes=module_codes,
        sample_size=sample_size,
        seed=seed,
    )
    if selected_modules.empty:
        raise ValueError("No modules were selected for evaluation.")

    derived_test_size = test_size
    if derived_test_size is None:
        derived_test_size = int(round(len(selected_modules) * float(test_fraction)))
    return split_selected_modules_for_evaluation(
        selected_modules,
        test_size=int(derived_test_size),
        seed=seed,
    )


def split_labeled_retrieval_dataset(
    labels: pd.DataFrame,
    *,
    test_size: int | None = None,
    test_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "module_code" not in labels.columns:
        raise ValueError("Labels frame must contain a module_code column.")

    working = labels.copy()
    working["module_code"] = working["module_code"].astype(str).str.strip()
    working = working[working["module_code"].ne("")].copy()
    if working.empty:
        raise ValueError("Labels frame does not contain any usable module_code rows.")

    module_manifest = working.drop_duplicates(subset=["module_code"], keep="first").copy()
    if len(module_manifest) < 2:
        raise ValueError("At least two unique modules are required to form a train/test split.")

    derived_test_size = test_size
    if derived_test_size is None:
        derived_test_size = int(round(len(module_manifest) * float(test_fraction)))

    train_modules, test_modules = split_selected_modules_for_evaluation(
        module_manifest,
        test_size=int(derived_test_size),
        seed=seed,
    )

    train_codes = set(train_modules["module_code"].astype(str))
    test_codes = set(test_modules["module_code"].astype(str))
    train_labels = working[working["module_code"].isin(train_codes)].copy()
    test_labels = working[working["module_code"].isin(test_codes)].copy()

    manifest_frames: list[pd.DataFrame] = []
    for split_name, frame in (("train", train_modules), ("test", test_modules)):
        manifest = frame.copy()
        manifest["split"] = split_name
        manifest_frames.append(manifest)
    split_manifest = pd.concat(manifest_frames, ignore_index=True)
    return train_labels, test_labels, split_manifest


def _mode_rank(row: dict[str, object], mode: RetrievalMode) -> float:
    rank = row.get(f"{mode}_rank")
    return float(rank) if rank is not None else float(10**9)


def _candidate_backfill_sort_key(
    row: dict[str, object],
    modes: Sequence[RetrievalMode],
) -> tuple[float, float, float, float, float, float, str, str]:
    present_modes = sum(1 for mode in modes if bool(row.get(f"{mode}_selected")))
    observed_ranks = [_mode_rank(row, mode) for mode in modes if row.get(f"{mode}_rank") is not None]
    best_rank = min(observed_ranks) if observed_ranks else float(10**9)
    mean_rank = sum(observed_ranks) / len(observed_ranks) if observed_ranks else float(10**9)
    return (
        -float(present_modes),
        best_rank,
        mean_rank,
        _mode_rank(row, "hybrid"),
        _mode_rank(row, "bm25"),
        _mode_rank(row, "embedding"),
        str(row.get("job_title", "")),
        str(row.get("job_id", "")),
    )


def _select_pooled_job_ids(
    # Pooled IR Evaluation
    candidates       : dict[str, dict[str, object]],
    per_mode_job_ids : dict[RetrievalMode, list[str]],
    final_top_k      : int,
    modes            : Sequence[RetrievalMode],
) -> list[str]:
    if final_top_k <= 0:
        target_count = len(candidates)
    else:
        target_count = min(int(final_top_k), len(candidates))

    if target_count <= 0:
        return []

    selected_job_ids = []
    selected_set     = set()
    max_depth        = max((len(job_ids) for job_ids in per_mode_job_ids.values()), default = 0)  # for each mode (hybrid, embedding only, bm25 only), they will have their own selected jobs, pick the one with the most selected jobs

    # Walk the ranked lists round-robin so each retrieval mode contributes to the
    # labeling pool before we backfill with the strongest shared candidates.
    for rank_idx in range(max_depth):
        for mode in modes:
            mode_job_ids = per_mode_job_ids.get(mode, [])
            if rank_idx >= len(mode_job_ids):
                continue
            job_id = mode_job_ids[rank_idx]
            if job_id in selected_set:
                continue
            selected_job_ids.append(job_id)
            selected_set.add(job_id)
            if len(selected_job_ids) >= target_count:
                return selected_job_ids
    # If have not collected enough jobs, look at all remaining candidates not selected
    # jobs sorted firstly by those that appear in more modes, then jobs with better best/average ranks, then the individual mode ranks as tie-breakers
    remaining_rows = sorted(
        (row for job_id, row in candidates.items() if job_id not in selected_set),
        key=lambda row: _candidate_backfill_sort_key(row, modes),
    )
    # For each of those remaining rows, it adds the job until we hit the target_count
    for row in remaining_rows:
        job_id = str(row.get("job_id", "")).strip()
        if not job_id or job_id in selected_set:
            continue
        selected_job_ids.append(job_id)
        selected_set.add(job_id)
        if len(selected_job_ids) >= target_count:
            break

    return selected_job_ids


def _build_module_candidate_rows(
    *,
    jobs: pd.DataFrame,
    module_row: pd.Series,
    module_index: int,
    retrieval: HybridRetrievalEngine,
    per_mode_top_k: int,
    modes: Sequence[RetrievalMode],
    final_top_k: int,
) -> list[dict[str, object]]:
    candidates: dict[str, dict[str, object]] = {}
    per_mode_job_ids: dict[RetrievalMode, list[str]] = {mode: [] for mode in modes}
    for mode in modes:
        ranking = retrieval.rank_jobs_from_module(
            int(module_index),
            top_k=int(per_mode_top_k),
            mode=mode,
        )
        for rank_position, (_, match) in enumerate(ranking.iterrows(), start=1):
            job_row = jobs.iloc[int(match["index"])]
            job_id = str(job_row.get("job_id", "")).strip()
            if not job_id:
                continue
            if job_id not in per_mode_job_ids[mode]:
                per_mode_job_ids[mode].append(job_id)

            candidate = candidates.setdefault(
                job_id,
                {
                    "module_code": str(module_row.get("module_code", "")).strip(),
                    "module_title": str(module_row.get("module_title", "")).strip(),
                    "module_faculty": _module_faculty_label(module_row),
                    "module_description": str(module_row.get("module_description", "")).strip(),
                    "module_technical_skills": _stringify_list(module_row.get("technical_skills")),
                    "module_soft_skills": _stringify_list(module_row.get("soft_skills")),
                    "job_id": job_id,
                    "job_title": str(job_row.get("title", "")).strip(),
                    "company": str(job_row.get("company", "")).strip(),
                    "role_family": str(job_row.get("role_family", "")).strip(),
                    "role_family_name": str(job_row.get("role_family_name", "")).strip(),
                    "broad_family": str(job_row.get("broad_family", "")).strip(),
                    "ssoc_4d_name": str(job_row.get("ssoc_4d_name", "")).strip(),
                    "ssoc_5d_name": str(job_row.get("ssoc_5d_name", "")).strip(),
                    "job_text": str(job_row.get("job_text", "")).strip(),
                    "job_technical_skills": _stringify_list(job_row.get("technical_skills")),
                    "hybrid_selected": False,
                    "bm25_selected": False,
                    "embedding_selected": False,
                    "hybrid_rank": None,
                    "bm25_rank": None,
                    "embedding_rank": None,
                    "hybrid_rrf_score": 0.0,
                    "bm25_score": 0.0,
                    "embedding_score": 0.0,
                    "evidence_terms": "",
                    "relevance": "",
                },
            )

            candidate[f"{mode}_selected"] = True
            candidate[f"{mode}_rank"] = int(rank_position)
            candidate["hybrid_rrf_score"] = max(
                float(candidate["hybrid_rrf_score"]),
                float(match.get("rrf_score", 0.0)),
            )
            candidate["bm25_score"] = float(match.get("bm25_score", 0.0))
            candidate["embedding_score"] = float(match.get("embedding_score", 0.0))
            evidence_terms = str(match.get("evidence_terms", "")).strip()
            if evidence_terms and not str(candidate["evidence_terms"]).strip():
                candidate["evidence_terms"] = evidence_terms

    selected_job_ids = _select_pooled_job_ids(
        candidates=candidates,
        per_mode_job_ids=per_mode_job_ids,
        final_top_k=int(final_top_k),
        modes=modes,
    )
    trimmed = [candidates[job_id] for job_id in selected_job_ids]
    for pool_rank, row in enumerate(trimmed, start=1):
        row["pool_rank"] = int(pool_rank)
    return trimmed


def build_retrieval_candidate_pool(
    *,
    jobs: pd.DataFrame,
    modules: pd.DataFrame,
    retrieval: HybridRetrievalEngine,
    module_codes: Sequence[str] | None = None,
    sample_size: int | None = None,
    seed: int = 42,
    per_mode_top_k: int = 10,
    final_top_k: int = 10,
    modes: Sequence[RetrievalMode] = DEFAULT_RETRIEVAL_EVAL_MODES,
) -> pd.DataFrame:
    final_top_k = int(final_top_k)
    per_mode_top_k = max(int(per_mode_top_k), final_top_k)

    explicit_cap = final_top_k > 0

    selected_modules = select_modules_for_evaluation(
        modules,
        module_codes=module_codes,
        sample_size=sample_size if module_codes else len(modules),
        seed=seed,
    )

    rows: list[dict[str, object]] = []
    selected_module_codes: list[str] = []
    requested_modules = len(_normalize_codes(module_codes)) if module_codes else int(sample_size or 0)
    for module_index, module_row in selected_modules.iterrows():
        candidate_rows = _build_module_candidate_rows(
            jobs=jobs,
            module_row=module_row,
            module_index=int(module_index),
            retrieval=retrieval,
            per_mode_top_k=per_mode_top_k,
            modes=modes,
            final_top_k=final_top_k,
        )
        if not candidate_rows:
            if module_codes:
                raise ValueError(
                    f"Module {module_row.get('module_code', '')} did not produce any pooled jobs."
                )
            continue

        if explicit_cap and len(candidate_rows) < final_top_k:
            if module_codes:
                raise ValueError(
                    f"Module {module_row.get('module_code', '')} only produced {len(candidate_rows)} pooled jobs; "
                    f"at least {final_top_k} are required."
                )
            continue

        rows.extend(candidate_rows)
        selected_module_codes.append(str(module_row.get("module_code", "")).strip())
        if not module_codes and len(selected_module_codes) >= requested_modules:
            break

    if not module_codes and len(selected_module_codes) < requested_modules:
        requirement = f"at least {final_top_k} pooled jobs" if explicit_cap else "at least 1 pooled job"
        raise ValueError(
            f"Only {len(selected_module_codes)} modules produced {requirement}, "
            f"so the exporter could not satisfy sample_size={requested_modules}."
        )

    return pd.DataFrame(
        rows,
        columns=[
            "module_code",
            "module_title",
            "module_faculty",
            "module_description",
            "module_technical_skills",
            "module_soft_skills",
            "job_id",
            "job_title",
            "company",
            "role_family",
            "role_family_name",
            "broad_family",
            "ssoc_4d_name",
            "ssoc_5d_name",
            "job_text",
            "job_technical_skills",
            "hybrid_selected",
            "bm25_selected",
            "embedding_selected",
            "pool_rank",
            "hybrid_rank",
            "bm25_rank",
            "embedding_rank",
            "hybrid_rrf_score",
            "bm25_score",
            "embedding_score",
            "evidence_terms",
            "relevance",
        ],
    )


def _dcg_at_k(relevances: Sequence[float], k: int) -> float:
    scores = np.asarray(list(relevances)[: int(k)], dtype=float)
    if scores.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, scores.size + 2))
    gains = np.power(2.0, scores) - 1.0
    return float(np.sum(gains * discounts))


def _ndcg_at_k(predicted_job_ids: Sequence[str], label_map: dict[str, float], k: int) -> float:
    ideal_relevances = sorted(label_map.values(), reverse=True)
    ideal_dcg = _dcg_at_k(ideal_relevances, k)
    if ideal_dcg <= 0:
        return float("nan")
    predicted_relevances = [float(label_map.get(job_id, 0.0)) for job_id in list(predicted_job_ids)[: int(k)]]
    return float(_dcg_at_k(predicted_relevances, k) / ideal_dcg)


def _precision_at_k(
    predicted_job_ids: Sequence[str],
    label_map: dict[str, float],
    k: int,
) -> float:
    predicted = list(predicted_job_ids)[: int(k)]
    if not predicted:
        return 0.0
    hits = sum(
        1
        for job_id in predicted
        if float(label_map.get(job_id, 0.0)) >= POSITIVE_RELEVANCE_THRESHOLD
    )
    return float(hits / len(predicted))


def _recall_at_k(
    predicted_job_ids: Sequence[str],
    label_map: dict[str, float],
    k: int,
) -> float:
    positive_total = sum(
        1
        for relevance in label_map.values()
        if float(relevance) >= POSITIVE_RELEVANCE_THRESHOLD
    )
    if positive_total == 0:
        return float("nan")
    predicted = list(predicted_job_ids)[: int(k)]
    hits = sum(
        1
        for job_id in predicted
        if float(label_map.get(job_id, 0.0)) >= POSITIVE_RELEVANCE_THRESHOLD
    )
    return float(hits / positive_total)


def _label_coverage_at_k(predicted_job_ids: Sequence[str], label_map: dict[str, float], k: int) -> float:
    predicted = list(predicted_job_ids)[: int(k)]
    if not predicted:
        return 0.0
    covered = sum(1 for job_id in predicted if job_id in label_map)
    return float(covered / len(predicted))


def _build_module_lookup(modules: pd.DataFrame) -> dict[str, int]:
    if "module_code" not in modules.columns:
        raise ValueError("Modules frame must contain a module_code column.")

    return {
        str(module_code).strip(): int(module_index)
        for module_index, module_code in modules["module_code"].items()
        if str(module_code).strip()
    }


def _build_cached_retrieval_eval_contexts(
    *,
    labels_prepared: pd.DataFrame,
    jobs: pd.DataFrame,
    modules: pd.DataFrame,
    retrieval: HybridRetrievalEngine,
) -> tuple[list[_CachedModuleRetrievalEvaluation], np.ndarray]:
    module_lookup = _build_module_lookup(modules)
    missing_modules = sorted(set(labels_prepared["module_code"]) - set(module_lookup))
    if missing_modules:
        raise ValueError(
            "Labels contain module codes that are not present in the current module corpus: "
            + ", ".join(missing_modules)
        )

    if "job_id" not in jobs.columns:
        raise ValueError("Jobs frame must contain a job_id column.")

    job_ids = jobs["job_id"].astype(str).str.strip().to_numpy()
    job_count = len(job_ids)
    job_artifacts = retrieval.artifacts.jobs
    module_artifacts = retrieval.artifacts.modules

    contexts: list[_CachedModuleRetrievalEvaluation] = []
    for module_code, group in labels_prepared.groupby("module_code", sort=True):
        module_index = module_lookup[str(module_code)]
        module_row = modules.loc[module_index]
        label_map = {
            str(job_id): float(relevance)
            for job_id, relevance in zip(group["job_id"], group["relevance"], strict=False)
        }

        bm25_scores = _bm25_scores(
            job_artifacts.bm25,
            module_artifacts.tokens[module_index],
            job_count,
        )
        embedding_scores = (
            cos_sim(
                np.asarray([module_artifacts.embeddings[module_index]]),
                job_artifacts.embeddings,
            )
            .cpu()
            .numpy()
            .ravel()
        )
        embedding_scores = np.clip(embedding_scores, 0.0, None)

        contexts.append(
            _CachedModuleRetrievalEvaluation(
                module_code=str(module_code),
                module_title=str(module_row.get("module_title", "")).strip(),
                labeled_pairs=int(len(group)),
                positive_labels=int((group["relevance"] >= POSITIVE_RELEVANCE_THRESHOLD).sum()),
                label_map=label_map,
                bm25_scores=bm25_scores,
                embedding_scores=embedding_scores,
            )
        )

    if not contexts:
        raise ValueError("No evaluation rows were produced from the provided labels.")

    return contexts, job_ids


def _predict_job_ids_from_cached_scores(
    *,
    context: _CachedModuleRetrievalEvaluation,
    job_ids: np.ndarray,
    retrieval: HybridRetrievalEngine,
    k: int,
    modes: Sequence[RetrievalMode],
) -> dict[RetrievalMode, list[str]]:
    config = retrieval.config
    bm25_candidates = _candidate_indices(
        context.bm25_scores,
        absolute_min_score=float(config.bm25_min_score),
        relative_min_score=float(config.bm25_relative_min),
    )
    embedding_candidates = _candidate_indices(
        context.embedding_scores,
        absolute_min_score=float(config.embedding_min_similarity),
        relative_min_score=float(config.embedding_relative_min),
    )
    fused_candidates = np.union1d(bm25_candidates, embedding_candidates)

    bm25_ranks = scores_to_ranks(context.bm25_scores, allowed_indices=bm25_candidates)
    embedding_ranks = scores_to_ranks(context.embedding_scores, allowed_indices=embedding_candidates)
    rrf_scores = reciprocal_rank_fusion(
        [bm25_ranks, embedding_ranks],
        rrf_k=int(config.rrf_k),
    )
    max_rrf = float(rrf_scores.max()) if rrf_scores.size else 0.0
    if max_rrf > 0:
        rrf_scores = rrf_scores / max_rrf

    mode_predictions: dict[RetrievalMode, list[str]] = {}
    for mode in modes:
        if mode == "hybrid":
            ranking_scores = rrf_scores
            selected_candidates = fused_candidates
        elif mode == "bm25":
            ranking_scores = context.bm25_scores
            selected_candidates = bm25_candidates
        else:
            ranking_scores = context.embedding_scores
            selected_candidates = embedding_candidates

        top_idx = top_indices(
            ranking_scores,
            top_k=int(k),
            allowed_indices=selected_candidates,
        )
        mode_predictions[mode] = [str(job_ids[int(job_index)]).strip() for job_index in top_idx]

    return mode_predictions


def _build_retrieval_evaluation_frames(
    *,
    cached_contexts: Sequence[_CachedModuleRetrievalEvaluation],
    job_ids: np.ndarray,
    retrieval: HybridRetrievalEngine,
    k: int,
    modes: Sequence[RetrievalMode],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, object]] = []
    for context in cached_contexts:
        mode_predictions = _predict_job_ids_from_cached_scores(
            context=context,
            job_ids=job_ids,
            retrieval=retrieval,
            k=int(k),
            modes=modes,
        )
        for mode in modes:
            predicted_job_ids = mode_predictions[mode]
            detail_rows.append(
                {
                    "mode": mode,
                    "module_code": context.module_code,
                    "module_title": context.module_title,
                    "k": int(k),
                    "labeled_pairs": int(context.labeled_pairs),
                    "positive_labels": int(context.positive_labels),
                    "predicted_count": int(len(predicted_job_ids)),
                    "retrieved_relevant_count": int(
                        sum(
                            1
                            for job_id in predicted_job_ids[: int(k)]
                            if float(context.label_map.get(job_id, 0.0))
                            >= POSITIVE_RELEVANCE_THRESHOLD
                        )
                    ),
                    "label_coverage_at_k": _label_coverage_at_k(predicted_job_ids, context.label_map, int(k)),
                    "ndcg_at_k": _ndcg_at_k(predicted_job_ids, context.label_map, int(k)),
                    "precision_at_k": _precision_at_k(
                        predicted_job_ids,
                        context.label_map,
                        int(k),
                    ),
                    "recall_at_k": _recall_at_k(
                        predicted_job_ids,
                        context.label_map,
                        int(k),
                    ),
                    "predicted_job_ids": "; ".join(predicted_job_ids[: int(k)]),
                }
            )

    details = pd.DataFrame(detail_rows)
    if details.empty:
        raise ValueError("No evaluation rows were produced from the provided labels.")

    mode_order = {mode: position for position, mode in enumerate(DEFAULT_RETRIEVAL_EVAL_MODES)}
    details["__mode_order"] = details["mode"].map(mode_order).fillna(len(mode_order))
    details = details.sort_values(["__mode_order", "module_code"]).drop(columns="__mode_order")

    summary = (
        details.groupby("mode", dropna=False)
        .agg(
            modules_evaluated=("module_code", "nunique"),
            mean_ndcg_at_k=("ndcg_at_k", "mean"),
            median_ndcg_at_k=("ndcg_at_k", "median"),
            mean_precision_at_k=("precision_at_k", "mean"),
            mean_recall_at_k=("recall_at_k", "mean"),
            mean_label_coverage_at_k=("label_coverage_at_k", "mean"),
            mean_retrieved_relevant_at_k=("retrieved_relevant_count", "mean"),
        )
        .reset_index()
    )
    summary["k"] = int(k)
    summary["__mode_order"] = summary["mode"].map(mode_order).fillna(len(mode_order))
    summary = summary.sort_values("__mode_order").drop(columns="__mode_order").reset_index(drop=True)
    return summary, details


def evaluate_retrieval_labels(
    *,
    labels: pd.DataFrame,
    jobs: pd.DataFrame,
    modules: pd.DataFrame,
    retrieval: HybridRetrievalEngine,
    k: int = 10,
    modes: Sequence[RetrievalMode] = DEFAULT_RETRIEVAL_EVAL_MODES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels_prepared = _prepare_labels(labels)
    cached_contexts, job_ids = _build_cached_retrieval_eval_contexts(
        labels_prepared=labels_prepared,
        jobs=jobs,
        modules=modules,
        retrieval=retrieval,
    )
    return _build_retrieval_evaluation_frames(
        cached_contexts=cached_contexts,
        job_ids=job_ids,
        retrieval=retrieval,
        k=int(k),
        modes=modes,
    )


def grid_search_retrieval_thresholds(
    *,
    labels: pd.DataFrame,
    jobs: pd.DataFrame,
    modules: pd.DataFrame,
    retrieval: HybridRetrievalEngine,
    k: int = 10,
    bm25_min_scores: Sequence[float],
    bm25_relative_mins: Sequence[float],
    embedding_min_similarities: Sequence[float],
    embedding_relative_mins: Sequence[float],
    modes: Sequence[RetrievalMode] = DEFAULT_RETRIEVAL_EVAL_MODES,
) -> pd.DataFrame:
    config = getattr(retrieval, "config", None)
    if config is None:
        raise ValueError("Retrieval engine must expose a config object for threshold grid search.")

    grid_axes = {
        "bm25_min_scores": [float(value) for value in bm25_min_scores],
        "bm25_relative_mins": [float(value) for value in bm25_relative_mins],
        "embedding_min_similarities": [float(value) for value in embedding_min_similarities],
        "embedding_relative_mins": [float(value) for value in embedding_relative_mins],
    }
    empty_axes = [name for name, values in grid_axes.items() if not values]
    if empty_axes:
        raise ValueError(
            "Each threshold grid must contain at least one value. Missing: " + ", ".join(empty_axes)
        )

    original_thresholds = {
        "bm25_min_score": float(config.bm25_min_score),
        "bm25_relative_min": float(config.bm25_relative_min),
        "embedding_min_similarity": float(config.embedding_min_similarity),
        "embedding_relative_min": float(config.embedding_relative_min),
    }

    labels_prepared = _prepare_labels(labels)
    total_combinations = (
        len(grid_axes["bm25_min_scores"])
        * len(grid_axes["bm25_relative_mins"])
        * len(grid_axes["embedding_min_similarities"])
        * len(grid_axes["embedding_relative_mins"])
    )
    logger.info(
        "Preparing cached retrieval scores for {} labeled modules before grid search.",
        labels_prepared["module_code"].nunique(),
    )
    cache_start = perf_counter()
    cached_contexts, job_ids = _build_cached_retrieval_eval_contexts(
        labels_prepared=labels_prepared,
        jobs=jobs,
        modules=modules,
        retrieval=retrieval,
    )
    logger.info(
        "Prepared cached retrieval scores for {} labeled modules in {:.2f}s.",
        len(cached_contexts),
        perf_counter() - cache_start,
    )

    summary_frames: list[pd.DataFrame] = []
    try:
        for combo_id, (
            bm25_min_score,
            bm25_relative_min,
            embedding_min_similarity,
            embedding_relative_min,
        ) in enumerate(
            product(
                grid_axes["bm25_min_scores"],
                grid_axes["bm25_relative_mins"],
                grid_axes["embedding_min_similarities"],
                grid_axes["embedding_relative_mins"],
            ),
            start=1,
        ):
            combo_start = perf_counter()
            logger.info(
                "Grid search combo {}/{}: bm25_min_score={}, bm25_relative_min={}, embedding_min_similarity={}, embedding_relative_min={}",
                combo_id,
                total_combinations,
                float(bm25_min_score),
                float(bm25_relative_min),
                float(embedding_min_similarity),
                float(embedding_relative_min),
            )
            config.bm25_min_score = float(bm25_min_score)
            config.bm25_relative_min = float(bm25_relative_min)
            config.embedding_min_similarity = float(embedding_min_similarity)
            config.embedding_relative_min = float(embedding_relative_min)

            summary, _ = _build_retrieval_evaluation_frames(
                cached_contexts=cached_contexts,
                job_ids=job_ids,
                retrieval=retrieval,
                k=int(k),
                modes=modes,
            )
            summary = summary.copy()
            summary.insert(0, "combo_id", int(combo_id))
            summary.insert(1, "bm25_min_score", float(bm25_min_score))
            summary.insert(2, "bm25_relative_min", float(bm25_relative_min))
            summary.insert(3, "embedding_min_similarity", float(embedding_min_similarity))
            summary.insert(4, "embedding_relative_min", float(embedding_relative_min))
            summary_frames.append(summary)
            logger.info(
                "Completed combo {}/{} in {:.2f}s.",
                combo_id,
                total_combinations,
                perf_counter() - combo_start,
            )
    finally:
        config.bm25_min_score = float(original_thresholds["bm25_min_score"])
        config.bm25_relative_min = float(original_thresholds["bm25_relative_min"])
        config.embedding_min_similarity = float(original_thresholds["embedding_min_similarity"])
        config.embedding_relative_min = float(original_thresholds["embedding_relative_min"])

    if not summary_frames:
        raise ValueError("Threshold grid search did not produce any evaluation rows.")

    results = pd.concat(summary_frames, ignore_index=True)
    mode_order = {mode: position for position, mode in enumerate(DEFAULT_RETRIEVAL_EVAL_MODES)}
    results["__mode_order"] = results["mode"].map(mode_order).fillna(len(mode_order))
    results = results.sort_values(
        [
            "__mode_order",
            "mean_ndcg_at_k",
            "mean_recall_at_k",
            "mean_precision_at_k",
            "combo_id",
        ],
        ascending=[True, False, False, False, True],
    ).drop(columns="__mode_order").reset_index(drop=True)
    return results


def write_evaluation_outputs(
    *,
    summary: pd.DataFrame,
    details: pd.DataFrame,
    summary_output: Path | None = None,
    details_output: Path | None = None,
) -> None:
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_output, index=False)
    if details_output is not None:
        details_output.parent.mkdir(parents=True, exist_ok=True)
        details.to_csv(details_output, index=False)
