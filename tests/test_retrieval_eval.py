from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from module_readiness.analysis.retrieval_eval import (  # noqa: E402
    build_retrieval_candidate_pool,
    evaluate_retrieval_labels,
    grid_search_retrieval_thresholds,
    split_labeled_retrieval_dataset,
    split_modules_for_evaluation,
)
from module_readiness.config import PipelineConfig  # noqa: E402
from module_readiness.retrieval.engine import (  # noqa: E402
    CorpusArtifacts,
    HybridRetrievalEngine,
    RetrievalArtifacts,
)


class _DummyEmbeddingService:
    backend = "dummy"
    dimension = 2


class _FakeRetrievalEngine:
    def __init__(self, rankings: dict[tuple[int, str], list[dict[str, object]]]):
        self.rankings = rankings

    def rank_jobs_from_module(self, module_index: int, top_k: int, mode: str = "hybrid") -> pd.DataFrame:
        rows = self.rankings.get((module_index, mode), [])
        return pd.DataFrame(rows[:top_k])


class TestRetrievalEvaluation(unittest.TestCase):
    def _build_fixture(self) -> tuple[pd.DataFrame, pd.DataFrame, HybridRetrievalEngine]:
        config = PipelineConfig.from_file()
        config.bm25_min_score = 0.0
        config.bm25_relative_min = 0.0
        config.embedding_min_similarity = 0.0
        config.embedding_relative_min = 0.0

        jobs = pd.DataFrame(
            {
                "job_id": ["J1", "J2", "J3"],
                "title": ["Data Analyst", "ML Engineer", "Marketing Exec"],
                "company": ["A", "B", "C"],
                "job_text": [
                    "python sql analytics",
                    "python machine learning",
                    "campaign branding",
                ],
                "role_family": ["Data Science / Analytics", "AI / ML", "Sales and Marketing"],
                "role_family_name": ["Data Science / Analytics", "AI / ML", "Sales and Marketing"],
                "broad_family": ["ICT", "ICT", "Business"],
                "ssoc_4d_name": [
                    "Statisticians and Data Scientists",
                    "Software and Applications Developers and Analysts",
                    "Sales and Marketing Managers and Professionals",
                ],
                "ssoc_5d_name": ["Data analyst", "Machine learning engineer", "Marketing executive"],
                "technical_skills": [["python", "sql"], ["python", "machine learning"], ["branding"]],
            }
        )

        modules = pd.DataFrame(
            {
                "module_code": ["CS3244"],
                "module_title": ["Machine Learning"],
                "module_description": ["python sql machine learning"],
                "technical_skills": [["python", "sql", "machine learning"]],
                "soft_skills": [["analytical skills"]],
            }
        )

        job_tokens = [
            ["python", "sql", "analytics"],
            ["python", "machine", "learning"],
            ["campaign", "branding"],
        ]
        module_tokens = [["python", "sql"]]

        artifacts = RetrievalArtifacts(
            jobs=CorpusArtifacts(
                texts=jobs["job_text"].tolist(),
                tokens=job_tokens,
                embeddings=np.asarray(
                    [
                        [0.2, 0.8],
                        [0.0, 1.0],
                        [1.0, 0.0],
                    ],
                    dtype=float,
                ),
                bm25=BM25Okapi(job_tokens, k1=float(config.bm25_k1), b=float(config.bm25_b)),
            ),
            modules=CorpusArtifacts(
                texts=modules["module_description"].tolist(),
                tokens=module_tokens,
                embeddings=np.asarray([[0.0, 1.0]], dtype=float),
                bm25=BM25Okapi(module_tokens, k1=float(config.bm25_k1), b=float(config.bm25_b)),
            ),
            embedding_service=_DummyEmbeddingService(),
            diagnostics={},
        )
        retrieval = HybridRetrievalEngine(config, artifacts)
        return jobs, modules, retrieval

    def test_rank_jobs_from_module_supports_bm25_and_embedding_modes(self) -> None:
        jobs, _, retrieval = self._build_fixture()

        bm25_results = retrieval.rank_jobs_from_module(0, top_k=2, mode="bm25")
        embedding_results = retrieval.rank_jobs_from_module(0, top_k=2, mode="embedding")

        self.assertEqual(jobs.iloc[int(bm25_results.iloc[0]["index"])]["job_id"], "J1")
        self.assertEqual(jobs.iloc[int(embedding_results.iloc[0]["index"])]["job_id"], "J2")
        self.assertTrue((bm25_results["ranking_mode"] == "bm25").all())
        self.assertTrue((embedding_results["ranking_mode"] == "embedding").all())

    def test_candidate_pool_and_metrics_compare_bm25_and_embedding(self) -> None:
        jobs, modules, retrieval = self._build_fixture()

        pool = build_retrieval_candidate_pool(
            jobs=jobs,
            modules=modules,
            retrieval=retrieval,
            module_codes=["CS3244"],
            per_mode_top_k=1,
            final_top_k=2,
            modes=("bm25", "embedding"),
        )
        self.assertEqual(set(pool["job_id"]), {"J1", "J2"})

        labels = pd.DataFrame(
            {
                "module_code": ["CS3244", "CS3244"],
                "job_id": ["J1", "J2"],
                "relevance": [3, 2],
            }
        )
        summary, details = evaluate_retrieval_labels(
            labels=labels,
            jobs=jobs,
            modules=modules,
            retrieval=retrieval,
            k=2,
            modes=("bm25", "embedding"),
        )

        self.assertEqual(len(details), 2)
        bm25_ndcg = float(summary.loc[summary["mode"] == "bm25", "mean_ndcg_at_k"].iloc[0])
        embedding_ndcg = float(summary.loc[summary["mode"] == "embedding", "mean_ndcg_at_k"].iloc[0])
        self.assertGreater(bm25_ndcg, embedding_ndcg)

    def test_evaluate_retrieval_labels_uses_cached_scores_instead_of_reranking(self) -> None:
        jobs, modules, retrieval = self._build_fixture()

        def _unexpected_rerank(*args: object, **kwargs: object) -> pd.DataFrame:
            raise AssertionError("evaluate_retrieval_labels should reuse cached raw scores.")

        retrieval.rank_jobs_from_module = _unexpected_rerank  # type: ignore[method-assign]

        labels = pd.DataFrame(
            {
                "module_code": ["CS3244", "CS3244"],
                "job_id": ["J1", "J2"],
                "relevance": [3, 2],
            }
        )

        summary, details = evaluate_retrieval_labels(
            labels=labels,
            jobs=jobs,
            modules=modules,
            retrieval=retrieval,
            k=2,
            modes=("bm25", "embedding"),
        )

        self.assertEqual(len(details), 2)
        self.assertEqual(set(summary["mode"]), {"bm25", "embedding"})

    def test_precision_and_recall_use_hardcoded_positive_relevance_threshold(self) -> None:
        jobs, modules, retrieval = self._build_fixture()

        labels = pd.DataFrame(
            {
                "module_code": ["CS3244", "CS3244"],
                "job_id": ["J1", "J2"],
                "relevance": [1, 2],
            }
        )

        summary, details = evaluate_retrieval_labels(
            labels=labels,
            jobs=jobs,
            modules=modules,
            retrieval=retrieval,
            k=2,
            modes=("bm25",),
        )

        self.assertEqual(float(details.iloc[0]["positive_labels"]), 1.0)
        self.assertAlmostEqual(float(details.iloc[0]["precision_at_k"]), 0.5)
        self.assertAlmostEqual(float(details.iloc[0]["recall_at_k"]), 1.0)
        self.assertNotIn("positive_relevance_threshold", summary.columns)

    def test_candidate_pool_keeps_full_union_by_default_instead_of_hybrid_only(self) -> None:
        jobs = pd.DataFrame(
            {
                "job_id": ["J1", "J2", "J3", "J4"],
                "title": ["Job 1", "Job 2", "Job 3", "Job 4"],
                "company": ["Org"] * 4,
                "role_family": ["Role"] * 4,
                "role_family_name": ["Role"] * 4,
                "broad_family": ["Broad"] * 4,
                "ssoc_4d_name": ["Group"] * 4,
                "ssoc_5d_name": ["Occupation 1", "Occupation 2", "Occupation 3", "Occupation 4"],
                "job_text": ["text 1", "text 2", "text 3", "text 4"],
                "technical_skills": [["skill"]] * 4,
            }
        )
        modules = pd.DataFrame(
            {
                "module_code": ["CS1"],
                "module_title": ["Module 1"],
                "module_description": ["desc"],
                "module_faculty": ["Computing"],
                "technical_skills": [["skill"]],
                "soft_skills": [["communication"]],
            }
        )
        retrieval = _FakeRetrievalEngine(
            {
                (0, "hybrid"): [
                    {"index": 0, "bm25_score": 1.0, "embedding_score": 0.9, "rrf_score": 1.0, "evidence_terms": "x"},
                    {"index": 1, "bm25_score": 0.8, "embedding_score": 0.7, "rrf_score": 0.9, "evidence_terms": "y"},
                ],
                (0, "bm25"): [
                    {"index": 2, "bm25_score": 1.0, "embedding_score": 0.1, "rrf_score": 0.7, "evidence_terms": "b"},
                    {"index": 0, "bm25_score": 0.9, "embedding_score": 0.2, "rrf_score": 0.6, "evidence_terms": "x"},
                ],
                (0, "embedding"): [
                    {"index": 3, "bm25_score": 0.2, "embedding_score": 1.0, "rrf_score": 0.8, "evidence_terms": "e"},
                    {"index": 0, "bm25_score": 0.1, "embedding_score": 0.9, "rrf_score": 0.5, "evidence_terms": "x"},
                ],
            }
        )

        pool = build_retrieval_candidate_pool(
            jobs=jobs,
            modules=modules,
            retrieval=retrieval,
            module_codes=["CS1"],
            per_mode_top_k=2,
            final_top_k=0,
        )

        self.assertEqual(list(pool["job_id"]), ["J1", "J3", "J4", "J2"])
        self.assertFalse(pool["hybrid_selected"].all())
        self.assertIn("J3", set(pool.loc[pool["bm25_selected"], "job_id"]))
        self.assertIn("J4", set(pool.loc[pool["embedding_selected"], "job_id"]))

    def test_grid_search_retrieval_thresholds_compares_combinations_and_restores_config(self) -> None:
        jobs, modules, retrieval = self._build_fixture()

        labels = pd.DataFrame(
            {
                "module_code": ["CS3244", "CS3244"],
                "job_id": ["J1", "J2"],
                "relevance": [3, 2],
            }
        )
        original_bm25_min_score = float(retrieval.config.bm25_min_score)

        results = grid_search_retrieval_thresholds(
            labels=labels,
            jobs=jobs,
            modules=modules,
            retrieval=retrieval,
            k=2,
            bm25_min_scores=[0.0, 999.0],
            bm25_relative_mins=[0.0],
            embedding_min_similarities=[0.0],
            embedding_relative_mins=[0.0],
            modes=("bm25", "embedding"),
        )

        self.assertEqual(len(results), 4)
        self.assertEqual(set(results["combo_id"]), {1, 2})
        self.assertEqual(float(retrieval.config.bm25_min_score), original_bm25_min_score)

        permissive_bm25 = float(
            results[
                (results["mode"] == "bm25")
                & (results["bm25_min_score"] == 0.0)
            ]["mean_ndcg_at_k"].iloc[0]
        )
        strict_bm25 = float(
            results[
                (results["mode"] == "bm25")
                & (results["bm25_min_score"] == 999.0)
            ]["mean_ndcg_at_k"].iloc[0]
        )
        self.assertGreater(permissive_bm25, strict_bm25)

    def test_split_modules_for_evaluation_creates_disjoint_train_and_test_sets(self) -> None:
        modules = pd.DataFrame(
            {
                "module_code": [f"A{i}" for i in range(1, 7)] + [f"B{i}" for i in range(1, 5)] + [f"C{i}" for i in range(1, 3)],
                "module_title": [f"Module {i}" for i in range(1, 13)],
                "module_description": ["desc"] * 12,
                "module_faculty": ["Faculty A"] * 6 + ["Faculty B"] * 4 + ["Faculty C"] * 2,
                "technical_skills": [["skill"]] * 12,
                "soft_skills": [["communication"]] * 12,
            }
        )

        train_modules, test_modules = split_modules_for_evaluation(
            modules,
            module_codes=modules["module_code"].tolist(),
            test_size=4,
            seed=11,
        )

        self.assertEqual(len(train_modules), 8)
        self.assertEqual(len(test_modules), 4)
        self.assertEqual(
            set(train_modules["module_code"]) | set(test_modules["module_code"]),
            set(modules["module_code"]),
        )
        self.assertFalse(set(train_modules["module_code"]) & set(test_modules["module_code"]))
        self.assertTrue({"Faculty A", "Faculty B", "Faculty C"}.issubset(set(train_modules["module_faculty"])))
        self.assertTrue({"Faculty A", "Faculty B"}.issubset(set(test_modules["module_faculty"])))

    def test_split_labeled_retrieval_dataset_keeps_module_rows_together(self) -> None:
        labels = pd.DataFrame(
            {
                "module_code": ["A1", "A1", "A2", "A2", "B1", "B1", "B2", "B2", "C1", "C1"],
                "module_title": ["A1", "A1", "A2", "A2", "B1", "B1", "B2", "B2", "C1", "C1"],
                "module_faculty": [
                    "Faculty A", "Faculty A",
                    "Faculty A", "Faculty A",
                    "Faculty B", "Faculty B",
                    "Faculty B", "Faculty B",
                    "Faculty C", "Faculty C",
                ],
                "job_id": ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10"],
                "relevance": [1, 2, 1, 3, 2, 2, 0, 1, 3, 2],
            }
        )

        train_labels, test_labels, split_manifest = split_labeled_retrieval_dataset(
            labels,
            test_size=2,
            seed=7,
        )

        self.assertEqual(train_labels["module_code"].nunique(), 3)
        self.assertEqual(test_labels["module_code"].nunique(), 2)
        self.assertFalse(
            set(train_labels["module_code"].unique()) & set(test_labels["module_code"].unique())
        )
        self.assertEqual(len(split_manifest), 5)
        self.assertEqual(
            set(split_manifest["module_code"]),
            set(labels["module_code"]),
        )

    def test_candidate_pool_backfills_to_requested_module_count_with_top_k_rows(self) -> None:
        jobs = pd.DataFrame(
            {
                "job_id": [f"J{i}" for i in range(1, 13)],
                "title": [f"Job {i}" for i in range(1, 13)],
                "company": ["Org"] * 12,
                "role_family": ["Role"] * 12,
                "role_family_name": ["Role"] * 12,
                "broad_family": ["Broad"] * 12,
                "ssoc_4d_name": ["Group"] * 12,
                "ssoc_5d_name": [f"Occupation {i}" for i in range(1, 13)],
                "job_text": [f"text {i}" for i in range(1, 13)],
                "technical_skills": [["skill"]] * 12,
            }
        )
        modules = pd.DataFrame(
            {
                "module_code": ["A1", "A2", "A3", "B1", "B2"],
                "module_title": ["A1", "A2", "A3", "B1", "B2"],
                "module_description": ["desc"] * 5,
                "module_faculty": ["Faculty A", "Faculty A", "Faculty A", "Faculty B", "Faculty B"],
                "technical_skills": [["skill"]] * 5,
                "soft_skills": [["communication"]] * 5,
            }
        )

        rankings = {
            (0, "hybrid"): [{"index": 0, "bm25_score": 1.0, "embedding_score": 0.9, "bm25_rank": 1, "embedding_rank": 1, "rrf_score": 1.0, "evidence_terms": "x"}],
            (0, "bm25"): [{"index": 0, "bm25_score": 1.0, "embedding_score": 0.9, "bm25_rank": 1, "embedding_rank": 1, "rrf_score": 1.0, "evidence_terms": "x"}],
            (0, "embedding"): [{"index": 0, "bm25_score": 1.0, "embedding_score": 0.9, "bm25_rank": 1, "embedding_rank": 1, "rrf_score": 1.0, "evidence_terms": "x"}],
            (1, "hybrid"): [{"index": 1, "bm25_score": 1.0, "embedding_score": 0.9, "bm25_rank": 1, "embedding_rank": 1, "rrf_score": 1.0, "evidence_terms": "x"}, {"index": 2, "bm25_score": 0.9, "embedding_score": 0.8, "bm25_rank": 2, "embedding_rank": 2, "rrf_score": 0.9, "evidence_terms": "y"}],
            (1, "bm25"): [{"index": 1, "bm25_score": 1.0, "embedding_score": 0.9, "bm25_rank": 1, "embedding_rank": 1, "rrf_score": 1.0, "evidence_terms": "x"}, {"index": 2, "bm25_score": 0.9, "embedding_score": 0.8, "bm25_rank": 2, "embedding_rank": 2, "rrf_score": 0.9, "evidence_terms": "y"}],
            (1, "embedding"): [{"index": 1, "bm25_score": 1.0, "embedding_score": 0.9, "bm25_rank": 1, "embedding_rank": 1, "rrf_score": 1.0, "evidence_terms": "x"}, {"index": 2, "bm25_score": 0.9, "embedding_score": 0.8, "bm25_rank": 2, "embedding_rank": 2, "rrf_score": 0.9, "evidence_terms": "y"}],
            (2, "hybrid"): [{"index": 3, "bm25_score": 1.0, "embedding_score": 0.9, "bm25_rank": 1, "embedding_rank": 1, "rrf_score": 1.0, "evidence_terms": "x"}, {"index": 4, "bm25_score": 0.9, "embedding_score": 0.8, "bm25_rank": 2, "embedding_rank": 2, "rrf_score": 0.9, "evidence_terms": "y"}],
            (2, "bm25"): [{"index": 3, "bm25_score": 1.0, "embedding_score": 0.9, "bm25_rank": 1, "embedding_rank": 1, "rrf_score": 1.0, "evidence_terms": "x"}, {"index": 4, "bm25_score": 0.9, "embedding_score": 0.8, "bm25_rank": 2, "embedding_rank": 2, "rrf_score": 0.9, "evidence_terms": "y"}],
            (2, "embedding"): [{"index": 3, "bm25_score": 1.0, "embedding_score": 0.9, "bm25_rank": 1, "embedding_rank": 1, "rrf_score": 1.0, "evidence_terms": "x"}, {"index": 4, "bm25_score": 0.9, "embedding_score": 0.8, "bm25_rank": 2, "embedding_rank": 2, "rrf_score": 0.9, "evidence_terms": "y"}],
            (3, "hybrid"): [{"index": 5, "bm25_score": 1.0, "embedding_score": 0.9, "bm25_rank": 1, "embedding_rank": 1, "rrf_score": 1.0, "evidence_terms": "x"}, {"index": 6, "bm25_score": 0.9, "embedding_score": 0.8, "bm25_rank": 2, "embedding_rank": 2, "rrf_score": 0.9, "evidence_terms": "y"}],
            (3, "bm25"): [{"index": 5, "bm25_score": 1.0, "embedding_score": 0.9, "bm25_rank": 1, "embedding_rank": 1, "rrf_score": 1.0, "evidence_terms": "x"}, {"index": 6, "bm25_score": 0.9, "embedding_score": 0.8, "bm25_rank": 2, "embedding_rank": 2, "rrf_score": 0.9, "evidence_terms": "y"}],
            (3, "embedding"): [{"index": 5, "bm25_score": 1.0, "embedding_score": 0.9, "bm25_rank": 1, "embedding_rank": 1, "rrf_score": 1.0, "evidence_terms": "x"}, {"index": 6, "bm25_score": 0.9, "embedding_score": 0.8, "bm25_rank": 2, "embedding_rank": 2, "rrf_score": 0.9, "evidence_terms": "y"}],
            (4, "hybrid"): [{"index": 7, "bm25_score": 1.0, "embedding_score": 0.9, "bm25_rank": 1, "embedding_rank": 1, "rrf_score": 1.0, "evidence_terms": "x"}, {"index": 8, "bm25_score": 0.9, "embedding_score": 0.8, "bm25_rank": 2, "embedding_rank": 2, "rrf_score": 0.9, "evidence_terms": "y"}],
            (4, "bm25"): [{"index": 7, "bm25_score": 1.0, "embedding_score": 0.9, "bm25_rank": 1, "embedding_rank": 1, "rrf_score": 1.0, "evidence_terms": "x"}, {"index": 8, "bm25_score": 0.9, "embedding_score": 0.8, "bm25_rank": 2, "embedding_rank": 2, "rrf_score": 0.9, "evidence_terms": "y"}],
            (4, "embedding"): [{"index": 7, "bm25_score": 1.0, "embedding_score": 0.9, "bm25_rank": 1, "embedding_rank": 1, "rrf_score": 1.0, "evidence_terms": "x"}, {"index": 8, "bm25_score": 0.9, "embedding_score": 0.8, "bm25_rank": 2, "embedding_rank": 2, "rrf_score": 0.9, "evidence_terms": "y"}],
        }
        retrieval = _FakeRetrievalEngine(rankings)

        pool = build_retrieval_candidate_pool(
            jobs=jobs,
            modules=modules,
            retrieval=retrieval,
            sample_size=4,
            seed=7,
            per_mode_top_k=2,
            final_top_k=2,
        )

        self.assertEqual(pool["module_code"].nunique(), 4)
        self.assertTrue((pool.groupby("module_code").size() == 2).all())
        self.assertEqual(set(pool["module_faculty"]), {"Faculty A", "Faculty B"})
        self.assertTrue((pool["pool_rank"].isin([1, 2])).all())


if __name__ == "__main__":
    unittest.main()
