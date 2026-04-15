from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from module_readiness.config import PipelineConfig, read_yaml_json
from module_readiness.ingestion.jobs import _normalize_skill, load_jobs
from module_readiness.processing import assign_role_families, load_skill_aliases


class TestIngestJobs(unittest.TestCase):
    def test_skill_normalizer_preserves_programming_symbols(self) -> None:
        # Programming-language tokens such as C++ and C# must survive normalization.
        config = PipelineConfig.from_file()
        aliases = load_skill_aliases(config.skill_aliases_file)

        self.assertEqual(_normalize_skill("C++", aliases), "c++")
        self.assertEqual(_normalize_skill("C#", aliases), "c#")
        self.assertEqual(_normalize_skill("A+", aliases), "a+")

    def test_job_filters(self) -> None:
        # The runtime job corpus should already be filtered to the intended policy scope.
        config = PipelineConfig.from_file()
        aliases = load_skill_aliases(config.skill_aliases_file)

        result = load_jobs(config, aliases)
        self.assertFalse(result.jobs.empty)
        self.assertTrue((result.jobs["experience_years"] <= 2).all())
        self.assertTrue((result.jobs["ssec_eqa"] == "70").all())

    def test_role_family_assignment_populates_column(self) -> None:
        # A quick integration check that ingestion + clustering produces the expected
        # downstream label columns.
        config = PipelineConfig.from_file()
        aliases = load_skill_aliases(config.skill_aliases_file)
        jobs = load_jobs(config, aliases).jobs.head(200).copy()
        rules = read_yaml_json(config.role_rules_file)
        cluster_rules = read_yaml_json(config.role_clusters_file)

        out = assign_role_families(jobs, rules, cluster_rules=cluster_rules).jobs
        self.assertIn("role_family", out.columns)
        self.assertGreater(out["role_family"].nunique(), 0)
        self.assertIn("role_cluster", out.columns)
        self.assertIn("broad_family", out.columns)

    def test_role_cluster_assignment_splits_software_and_ai(self) -> None:
        # The curated cluster rules should be able to split broad SSOC groups into more
        # useful labels such as Software Engineering vs AI / ML.
        rules = {}
        cluster_rules = {
            "cluster_broad_family_map": {
                "Software Engineering": "ICT",
                "AI / ML": "ICT",
                "Other": "Other",
            },
            "ssoc4_exact_map": {
                "2512": "Software Engineering",
            },
            "split_rules": [
                {
                    "cluster": "AI / ML",
                    "ssoc4": ["2519"],
                    "keywords": ["ai", "machine learning", "llm"],
                }
            ],
        }
        jobs = pd.DataFrame(
            {
                "job_id": ["J1", "J2"],
                "title": ["Software Engineer", "AI/ML Engineer"],
                "title_clean": ["software engineer", "ai ml engineer"],
                "description_raw": ["", ""],
                "description_clean": [
                    "build backend api services in java",
                    "build llm and machine learning systems in python",
                ],
                "job_text": [
                    "software engineer build backend api services in java",
                    "ai ml engineer build llm and machine learning systems in python",
                ],
                "experience_years": [1, 1],
                "ssec_eqa": ["70", "70"],
                "ssoc_code": ["25121", "25191"],
                "skills": [
                    ["Java", "API", "Software Development"],
                    ["Python", "Machine Learning", "LLM"],
                ],
                "categories": [[], []],
                "primary_category": ["Information Technology", "Information Technology"],
            }
        )

        out = assign_role_families(jobs, rules, cluster_rules=cluster_rules).jobs
        self.assertEqual(out.loc[0, "role_family"], "Software Engineering")
        self.assertEqual(out.loc[1, "role_family"], "AI / ML")
        self.assertEqual(out.loc[1, "broad_family"], "ICT")


if __name__ == "__main__":
    unittest.main()
