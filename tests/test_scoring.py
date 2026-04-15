from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from module_readiness.retrieval import HybridRetrievalEngine, build_retrieval_artifacts
from module_readiness.analysis.scoring import (
    _aggregate_group_scores,
    _apply_other_near_tie_preference,
    compute_scores,
)
from module_readiness.config import PipelineConfig


class TestScoring(unittest.TestCase):
    def test_support_adjustment_downweights_singleton_role_families(self) -> None:
        # A single very strong match should not dominate a cluster that has almost no support.
        module_job_scores = pd.DataFrame(
            {
                "module_code": ["DSA3101"] * 6,
                "module_title": ["Data Science in Practice"] * 6,
                "ssoc_4d": ["2423"] + ["2122"] * 5,
                "ssoc_4d_name": ["Human Resource Professionals"] + ["Statisticians and Data Scientists"] * 5,
                "ssoc_5d": ["24231"] + ["21222"] * 5,
                "ssoc_5d_name": ["HR consultant"] + ["Data scientist"] * 5,
                "rrf_score": [0.88, 0.70, 0.68, 0.66, 0.64, 0.62],
            }
        )

        result = _aggregate_group_scores(
            module_job_scores=module_job_scores,
            top_k=50,
            group_cols=["ssoc_4d", "ssoc_4d_name", "ssoc_5d", "ssoc_5d_name"],
            support_prior=5.0,
        )

        hr = result[result["ssoc_4d"] == "2423"].iloc[0]
        ds = result[result["ssoc_4d"] == "2122"].iloc[0]

        self.assertGreater(float(hr["raw_role_score"]), float(ds["raw_role_score"]))
        self.assertLess(float(hr["role_score"]), float(ds["role_score"]))

    def test_scoring_returns_rrf_scores(self) -> None:
        # End-to-end scoring should expose RRF-based evidence rather than the older weighted-fit output.
        config = PipelineConfig.from_file()
        config.top_k = 1
        config.retrieval_top_n = 2

        jobs = pd.DataFrame(
            {
                "job_id": ["J1", "J2"],
                "title": ["Data Analyst", "Marketing Exec"],
                "company": ["A", "B"],
                "job_text": [
                    "data analyst python sql dashboards analytics",
                    "marketing campaign branding communications",
                ],
                "role_family": ["Data and Artificial Intelligence", "Sales and Marketing"],
                "primary_category": ["Information Technology", "Marketing / Public Relations"],
                "technical_skills": [["python", "sql"], ["marketing strategy"]],
                "soft_skills": [["communication skills"], ["communication skills"]],
            }
        )

        modules = pd.DataFrame(
            {
                "module_code": ["CS3244"],
                "module_title": ["Machine Learning"],
                "module_description": ["machine learning python sql data analysis modelling"],
                "module_text": ["machine learning python sql data analysis modelling"],
                "technical_skills": [["python", "sql", "machine learning"]],
                "soft_skills": [["communication skills", "analytical skills"]],
            }
        )

        artifacts = build_retrieval_artifacts(config, jobs, modules)
        self.assertIn("technical skills python sql machine learning", artifacts.modules.texts[0])
        self.assertNotIn("communication skills", artifacts.modules.texts[0])
        retrieval = HybridRetrievalEngine(config, artifacts)

        result = compute_scores(config, jobs, modules, retrieval)
        self.assertFalse(result.module_job_scores.empty)

        top_row = result.module_job_scores.sort_values("rrf_score", ascending=False).iloc[0]
        self.assertEqual(top_row["job_id"], "J1")
        self.assertGreater(float(top_row["rrf_score"]), 0.5)
        self.assertNotIn("weighted_fit", result.module_job_scores.columns)

    def test_other_does_not_win_when_nearly_tied_with_named_cluster(self) -> None:
        # The display-friendly near-tie rule should prevent "Other" from taking the top slot.
        module_role_scores = pd.DataFrame(
            {
                "module_code": ["ACC1701", "ACC1701"],
                "module_title": ["Accounting for Decision Makers", "Accounting for Decision Makers"],
                "role_family": ["Other", "Accounting / Audit / Tax"],
                "role_family_name": ["Other", "Accounting / Audit / Tax"],
                "broad_family": ["Other", "Finance"],
                "raw_role_score": [0.761706, 0.662728],
                "support_weight": [0.761905, 0.875000],
                "role_score": [0.580347, 0.579887],
                "evidence_job_count": [16.0, 35.0],
            }
        )

        adjusted = _apply_other_near_tie_preference(module_role_scores)
        top = adjusted.sort_values(["selection_score", "role_score"], ascending=[False, False]).iloc[0]

        self.assertEqual(top["role_family"], "Accounting / Audit / Tax")
        other_row = adjusted[adjusted["role_family"] == "Other"].iloc[0]
        self.assertLessEqual(float(other_row["selection_score"]), float(top["selection_score"]))


if __name__ == "__main__":
    unittest.main()
