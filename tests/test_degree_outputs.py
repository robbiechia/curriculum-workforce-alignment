from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from module_readiness.analysis import build_degree_outputs
from module_readiness.analysis.degrees import _expand_token
from module_readiness.config import PipelineConfig


class TestDegreeOutputs(unittest.TestCase):
    def _base_config(self, tmpdir: str) -> PipelineConfig:
        config = PipelineConfig.from_file()
        config.degree_plan_file = Path(tmpdir) / "nus_degree_plan.csv"
        config.degree_mapping_file = Path(tmpdir) / "degree_mapping.csv"
        config.degree_demand_skill_top_n = 5
        return config

    def _fixtures(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        jobs = pd.DataFrame(
            {
                "role_family": ["Data Science / Analytics", "Software Engineering"],
                "role_family_name": ["Data Science / Analytics", "Software Engineering"],
                "ssoc_4d": ["2122", "2512"],
                "ssoc_4d_name": ["Statisticians and Data Scientists", "Software Developers"],
                "ssoc_5d": ["21222", "25121"],
                "ssoc_5d_name": ["Data scientist", "Software developer"],
                "technical_skills": [["python", "sql", "statistics"], ["python", "software engineering"]],
            }
        )
        modules = pd.DataFrame(
            {
                "module_code": ["CS1010", "BT2101", "ST2334", "CS2100", "CS2101", "ACC1701"],
                "module_title": [
                    "Programming Methodology",
                    "Decision Making Methods",
                    "Probability and Statistics",
                    "Computer Organisation",
                    "Software Engineering",
                    "Accounting",
                ],
                "module_profile": ["computing", "business", "statistics", "computing", "computing", "business"],
                "module_source": ["db"] * 6,
                "module_faculty": ["SOC", "SOC", "Science", "SOC", "SOC", "BIZ"],
                "module_department": ["CS", "IS", "Stats", "CS", "CS", "Accounting"],
                "module_credit": ["4", "4", "4", "4", "4", "4"],
                "technical_skills": [
                    ["python"],
                    ["sql", "dashboarding"],
                    ["statistics"],
                    ["computer architecture"],
                    ["software engineering"],
                    ["accounting"],
                ],
                "soft_skills": [["problem solving"]] * 6,
            }
        )
        raw_modules = modules.copy()
        raw_modules["module_code"] = ["CS1010A", "BT2101", "ST2334", "CS2100", "CS2101", "ACC1701A"]
        module_summary = pd.DataFrame(
            {
                "module_code": ["CS1010", "BT2101", "ST2334", "CS2100", "CS2101", "ACC1701"],
                "top_role_cluster": [
                    "Software Engineering",
                    "Data Science / Analytics",
                    "Data Science / Analytics",
                    "Software Engineering",
                    "Software Engineering",
                    "Accounting",
                ],
                "top_role_family_name": [
                    "Software Engineering",
                    "Data Science / Analytics",
                    "Data Science / Analytics",
                    "Software Engineering",
                    "Software Engineering",
                    "Accounting",
                ],
                "top_role_family_name_source": ["role_cluster"] * 6,
                "top_role_score": [0.8, 0.9, 0.85, 0.7, 0.95, 0.6],
                "top_broad_family": ["ICT", "ICT", "ICT", "ICT", "ICT", "Finance"],
            }
        )
        module_role_scores = pd.DataFrame(
            {
                "module_code": ["CS1010", "CS1010", "BT2101", "ST2334", "CS2100", "CS2101", "ACC1701"],
                "role_family": [
                    "Software Engineering",
                    "Data Science / Analytics",
                    "Data Science / Analytics",
                    "Data Science / Analytics",
                    "Software Engineering",
                    "Software Engineering",
                    "Accounting",
                ],
                "role_family_name": [
                    "Software Engineering",
                    "Data Science / Analytics",
                    "Data Science / Analytics",
                    "Data Science / Analytics",
                    "Software Engineering",
                    "Software Engineering",
                    "Accounting",
                ],
                "broad_family": ["ICT", "ICT", "ICT", "ICT", "ICT", "ICT", "Finance"],
                "role_score": [0.8, 0.3, 0.9, 0.85, 0.7, 0.95, 0.6],
            }
        )
        module_ssoc5_scores = pd.DataFrame(
            {
                "module_code": ["CS1010", "BT2101", "ST2334", "CS2101"],
                "ssoc_4d": ["2512", "2122", "2122", "2512"],
                "ssoc_4d_name": ["Software Developers", "Statisticians and Data Scientists", "Statisticians and Data Scientists", "Software Developers"],
                "ssoc_5d": ["25121", "21222", "21222", "25121"],
                "ssoc_5d_name": ["Software developer", "Data scientist", "Data scientist", "Software developer"],
                "role_score": [0.8, 0.9, 0.85, 0.95],
                "evidence_job_count": [2, 3, 3, 2],
            }
        )
        return jobs, modules, raw_modules, module_summary, module_role_scores, module_ssoc5_scores

    def test_token_expansion_handles_exact_percent_and_numeric_x(self) -> None:
        raw_codes = {"CS1010A", "CS2100", "CS2101", "ACC1701A", "MXX1000"}
        consolidated_codes = {"CS1010", "CS2100", "CS2101", "ACC1701", "MXX1000"}

        exact, exact_rule, exact_error = _expand_token("CS1010A", raw_codes=raw_codes, consolidated_codes=consolidated_codes)
        self.assertEqual(exact, [("CS1010A", "CS1010")])
        self.assertEqual(exact_rule, "exact")
        self.assertIsNone(exact_error)

        percent, percent_rule, _ = _expand_token("ACC1701%", raw_codes=raw_codes, consolidated_codes=consolidated_codes)
        self.assertEqual(percent, [("ACC1701%", "ACC1701")])
        self.assertEqual(percent_rule, "wildcard")

        wildcard, wildcard_rule, _ = _expand_token("CS21XX", raw_codes=raw_codes, consolidated_codes=consolidated_codes)
        self.assertEqual([code for _, code in wildcard], ["CS2100", "CS2101"])
        self.assertEqual(wildcard_rule, "wildcard")

        literal_x, _, _ = _expand_token("MXX1000", raw_codes=raw_codes, consolidated_codes=consolidated_codes)
        self.assertEqual(literal_x, [("MXX1000", "MXX1000")])

        malformed, malformed_rule, malformed_error = _expand_token("3N4XXX", raw_codes=raw_codes, consolidated_codes=consolidated_codes)
        self.assertEqual(malformed, [])
        self.assertEqual(malformed_rule, "malformed")
        self.assertIsNotNone(malformed_error)

    def test_degree_outputs_use_plan_buckets_and_two_score_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._base_config(tmpdir)
            pd.DataFrame(
                {
                    "faculty": ["School of Computing", "School of Computing", "School of Computing", "School of Computing"],
                    "faculty_code": ["SOC", "SOC", "SOC", "SOC"],
                    "degree": ["Bachelor of Test"] * 4,
                    "primary_major": ["Business Analytics"] * 4,
                    "curriculum_type": ["Primary Major"] * 3 + ["Unrestricted Electives"],
                    "curriculum_credits": ["16", "16", "16", "4"],
                    "module_type": ["Foundation", "Analytics Basket", "Computing Basket", "Unrestricted Electives"],
                    "module_credits": ["4", "4", "4", "4"],
                    "modules": ["['CS1010A']", "['BT2101', 'ST2334']", "['CS21XX']", ""],
                    "curriculum_website": ["https://example.com"] * 4,
                }
            ).to_csv(config.degree_plan_file, index=False)

            jobs, modules, raw_modules, module_summary, module_role_scores, module_ssoc5_scores = self._fixtures()
            result = build_degree_outputs(
                config=config,
                jobs=jobs,
                modules=modules,
                raw_modules=raw_modules,
                module_summary=module_summary,
                module_role_scores=module_role_scores,
                module_ssoc5_scores=module_ssoc5_scores,
            )

            self.assertEqual(len(result.degree_requirement_buckets), 4)
            self.assertIn("best_case", set(result.degree_role_scores["score_mode"]))
            self.assertIn("expected", set(result.degree_role_scores["score_mode"]))
            self.assertTrue(result.degree_module_map["is_unrestricted_elective"].astype(bool).any())

            analytics_expected = result.degree_role_scores[
                (result.degree_role_scores["score_mode"] == "expected")
                & (result.degree_role_scores["role_family"] == "Data Science / Analytics")
            ].iloc[0]
            self.assertGreater(float(analytics_expected["degree_role_score"]), 0.0)
            self.assertGreater(float(analytics_expected["elective_half_potential_score"]), 0.0)
            self.assertGreater(float(analytics_expected["score_with_full_electives"]), 0.0)

            summary = result.degree_summary.iloc[0]
            self.assertEqual(summary["major"], "Business Analytics")
            self.assertEqual(int(summary["required_bucket_count"]), 4)
            self.assertGreater(int(summary["matched_required_module_count"]), 0)

            supply = result.degree_skill_supply
            self.assertIn("python", set(supply["skill"]))
            self.assertTrue(result.degree_plan_expansion_audit.empty)

    def test_legacy_mapping_still_deduplicates_variant_modules_for_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._base_config(tmpdir)
            config.degree_plan_file = Path(tmpdir) / "missing_plan.csv"
            pd.DataFrame(
                {
                    "faculty": ["School of Computing"],
                    "faculty_code": ["SOC"],
                    "major": ["Variant Test Major"],
                    "required_modules": ["CS1010A; CS1010B; BT2101"],
                    "curriculum_link": [""],
                    "notes": [""],
                }
            ).to_csv(config.degree_mapping_file, index=False)

            jobs, modules, raw_modules, module_summary, _, _ = self._fixtures()
            result = build_degree_outputs(
                config=config,
                jobs=jobs,
                modules=modules,
                raw_modules=raw_modules,
                module_summary=module_summary,
                module_role_scores=pd.DataFrame(),
                module_ssoc5_scores=pd.DataFrame(),
            )

            summary = result.degree_summary.iloc[0]
            self.assertEqual(int(summary["required_module_count"]), 2)
            self.assertEqual(int(summary["matched_required_module_count"]), 2)
            self.assertEqual(float(summary["matched_required_module_share"]), 1.0)


if __name__ == "__main__":
    unittest.main()
