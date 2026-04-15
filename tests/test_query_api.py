from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from module_readiness.api import ModuleReadinessQueryAPI
from module_readiness.config import PipelineConfig
from module_readiness.orchestration import ModuleReadinessState
from module_readiness.retrieval import HybridRetrievalEngine, build_retrieval_artifacts


class TestQueryAPI(unittest.TestCase):
    def test_query_api_uses_hybrid_retrieval_outputs(self) -> None:
        # This fixture bypasses the full pipeline and checks that the query API presents
        # hybrid-retrieval results correctly.
        config = PipelineConfig.from_file()
        config.retrieval_top_n = 5

        jobs = pd.DataFrame(
            {
                "job_id": ["J1", "J2"],
                "title": ["Data Analyst", "Content Strategist"],
                "company": ["A", "B"],
                "job_text": [
                    "data analyst python sql dashboards analytics reporting",
                    "content strategy storytelling communications campaigns",
                ],
                "experience_years": [1, 1],
                "role_family": ["Data and Artificial Intelligence", "Communications and Media"],
                "role_family_name": ["Data and Artificial Intelligence", "Communications and Media"],
                "role_family_source": ["test", "test"],
                "primary_category": ["Information Technology", "Marketing / Public Relations"],
                "ssoc_code": ["25211", "24310"],
                "technical_skills": [["python", "sql"], ["content strategy"]],
                "soft_skills": [["communication skills"], ["communication skills"]],
            }
        )

        modules = pd.DataFrame(
            {
                "module_code": ["CS3244", "NM2207"],
                "module_title": ["Machine Learning", "Digital Media Strategy"],
                "module_description": [
                    "machine learning python sql data analysis modelling",
                    "digital communication campaigns content strategy media planning",
                ],
                "module_text": [
                    "machine learning python sql data analysis modelling",
                    "digital communication campaigns content strategy media planning",
                ],
                "technical_skills": [
                    ["python", "sql", "machine learning"],
                    ["content strategy"],
                ],
                "soft_skills": [
                    ["communication skills", "analytical skills"],
                    ["communication skills"],
                ],
                "module_profile": ["computing", "communications"],
                "module_source": ["cache", "cache"],
            }
        )

        artifacts = build_retrieval_artifacts(config, jobs, modules)
        retrieval = HybridRetrievalEngine(config, artifacts)

        module_role_scores = pd.DataFrame(
            {
                "module_code": ["CS3244", "NM2207"],
                "module_title": ["Machine Learning", "Digital Media Strategy"],
                "role_family": ["Data and Artificial Intelligence", "Communications and Media"],
                "role_family_name": ["Data and Artificial Intelligence", "Communications and Media"],
                "role_score": [1.5, 1.2],
            }
        )
        module_summary = pd.DataFrame(
            {
                "module_code": ["CS3244", "NM2207"],
                "module_title": ["Machine Learning", "Digital Media Strategy"],
                "module_profile": ["computing", "communications"],
                "module_source": ["cache", "cache"],
                "top_role_family": ["Data and Artificial Intelligence", "Communications and Media"],
                "top_role_family_name": ["Data and Artificial Intelligence", "Communications and Media"],
                "top_role_score": [1.5, 1.2],
            }
        )

        state = ModuleReadinessState(
            config=config,
            role_rules={},
            jobs=jobs,
            modules=modules,
            module_job_scores=pd.DataFrame(),
            module_ssoc5_scores=pd.DataFrame(),
            module_role_scores=module_role_scores,
            module_summary=module_summary,
            module_gap_summary=pd.DataFrame(),
            retrieval_artifacts=artifacts,
            retrieval=retrieval,
            skill_channel_map={},
            known_skills=["python", "sql", "content strategy"],
            diagnostics={},
        )

        api = ModuleReadinessQueryAPI(state)

        job_results = api.search_jobs("python sql analyst", exp_max=2, top_k=1)
        self.assertEqual(job_results.iloc[0]["job_id"], "J1")

        module_results = api.recommend_relevant_modules("python sql analyst", top_k=1)
        self.assertEqual(module_results.iloc[0]["module_code"], "CS3244")


if __name__ == "__main__":
    unittest.main()
