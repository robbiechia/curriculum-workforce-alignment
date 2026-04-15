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
from module_readiness.processing import apply_skill_taxonomy, load_skill_aliases


class TestSkillTaxonomy(unittest.TestCase):
    def test_alias_and_channel_split(self) -> None:
        # Verify the job-side explicit skills and module-side inferred skills both end up
        # in the expected channels.
        config = PipelineConfig.from_file()
        rules = read_yaml_json(config.role_rules_file)

        jobs = pd.DataFrame(
            {
                "job_id": ["J1"],
                "title": ["Data Analyst"],
                "description_clean": ["need communication and teamwork with python sql"],
                "skills": [["Python", "SQL", "Teamwork", "Communication skill"]],
                "role_family": ["Data and Artificial Intelligence"],
            }
        )
        modules = pd.DataFrame(
            {
                "module_code": ["CS3244"],
                "module_title": ["Machine Learning"],
                "module_description": ["Machine learning with python and data analysis"],
                "module_workload": ["lecture_1 tutorial_2 project_5 preparation_2"],
                "module_text": ["machine learning python data analysis project presentation report"],
            }
        )

        result = apply_skill_taxonomy(config, rules, jobs, modules)

        self.assertIn("technical_skills", result.jobs.columns)
        self.assertIn("soft_skills", result.jobs.columns)
        self.assertIn("python", result.jobs.iloc[0]["technical_skills"])
        self.assertIn("team player", result.jobs.iloc[0]["soft_skills"])
        self.assertIn("communication skills", result.jobs.iloc[0]["soft_skills"])
        self.assertIn("team player", result.modules.iloc[0]["soft_skills"])
        self.assertIn("communication skills", result.modules.iloc[0]["soft_skills"])

    def test_filters_noisy_single_character_skills_from_modules(self) -> None:
        # Module extraction should keep meaningful short skills like C++ but drop noise.
        config = PipelineConfig.from_file()
        rules = read_yaml_json(config.role_rules_file)

        jobs = pd.DataFrame(
            {
                "job_id": ["J1", "J2"],
                "title": ["Software Engineer", "IT Support Officer"],
                "description_clean": [
                    "python c++ machine learning communication",
                    "a+ troubleshooting support",
                ],
                "skills": [["Python", "C++", "S"], ["A+", "Communication skill"]],
                "role_family": ["Software Engineering", "Information Technology"],
            }
        )
        modules = pd.DataFrame(
            {
                "module_code": ["CS3244"],
                "module_title": ["Machine Learning"],
                "module_description": ["Machine learning with python and c++"],
                "module_workload": ["lecture_2 tutorial_1 lab_1 project_4 preparation_2"],
                "module_text": ["machine learning with python and c++ a s project report"],
            }
        )

        result = apply_skill_taxonomy(config, rules, jobs, modules)
        module_skills = result.modules.iloc[0]["technical_skills"]

        self.assertIn("python", module_skills)
        self.assertIn("c++", module_skills)
        self.assertNotIn("a", module_skills)
        self.assertNotIn("s", module_skills)


if __name__ == "__main__":
    unittest.main()
