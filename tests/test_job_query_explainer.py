from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from module_readiness.llm import (
    build_fallback_job_query_explanation,
    build_job_query_prompt_context,
)


class TestJobQueryExplainer(unittest.TestCase):
    def test_prompt_context_contains_jobs_modules_and_degree_context(self) -> None:
        jobs = pd.DataFrame(
            {
                "job_id": ["J1"],
                "title": ["Data Scientist"],
                "company": ["Example Co"],
                "role_family_name": ["Data Science / Analytics"],
                "primary_category": ["Information Technology"],
                "score": [0.91],
                "technical_skills": [["python", "sql", "machine learning"]],
                "soft_skills": [["communication skills"]],
                "job_summary": ["build predictive models and analytics pipelines"],
                "evidence_terms": ["python | sql"],
            }
        )
        modules = pd.DataFrame(
            {
                "module_code": ["CS3244"],
                "module_title": ["Machine Learning"],
                "top_role_family_name": ["Data Science / Analytics"],
                "similarity_score": [0.88],
                "matched_job_count": [1],
                "matched_jobs": ["Data Scientist (Example Co)"],
                "technical_skill_overlap": [["python", "machine learning"]],
                "soft_skill_overlap": [["communication skills"]],
                "technical_skills": [["python", "machine learning"]],
                "module_summary": ["covers supervised and unsupervised learning"],
                "job_evidence_terms": ["python | machine learning"],
            }
        )

        context = build_job_query_prompt_context(
            natural_language_query="data scientist jobs",
            jobs=jobs,
            modules=modules,
            degree_label="Data Science and Economics",
        )

        self.assertIn('"query": "data scientist jobs"', context)
        self.assertIn('"degree_context": "Data Science and Economics"', context)
        self.assertIn("Data Scientist", context)
        self.assertIn("CS3244", context)

    def test_fallback_explanation_mentions_jobs_and_modules(self) -> None:
        jobs = pd.DataFrame(
            {
                "role_family_name": ["Data Science / Analytics", "Data Science / Analytics"],
                "technical_skills": [["python", "sql"], ["python", "statistics"]],
            }
        )
        modules = pd.DataFrame(
            {
                "module_code": ["CS3244", "ST2334"],
                "module_title": ["Machine Learning", "Probability and Statistics"],
                "technical_skill_overlap": [["python"], ["statistics"]],
            }
        )

        summary = build_fallback_job_query_explanation(
            natural_language_query="data scientist jobs",
            jobs=jobs,
            modules=modules,
            degree_label="Business Analytics",
        )

        self.assertIn("data scientist jobs", summary.lower())
        self.assertIn("Business Analytics", summary)
        self.assertIn("CS3244", summary)
        self.assertIn("ST2334", summary)


if __name__ == "__main__":
    unittest.main()
