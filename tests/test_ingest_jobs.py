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
        self.assertIn("broad_family", out.columns)

    def test_role_family_assignment_splits_software_and_ai(self) -> None:
        # The curated role-family rules should be able to split broad SSOC groups into
        # more useful labels such as Software Engineering vs AI / ML.
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
                    "keywords": ["machine learning", "llm"],
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
        self.assertEqual(out.loc[1, "role_family_match_detail"], "split_rule[1]")
        self.assertEqual(out.loc[1, "role_family_matched_keyword"], "machine learning")

    def test_role_family_assignment_avoids_short_keyword_substrings(self) -> None:
        # Short keywords such as "ai" must match as standalone tokens, not inside
        # unrelated words like "detailed" or "liaise".
        rules = {}
        cluster_rules = {
            "cluster_broad_family_map": {
                "AI / ML": "ICT",
                "Other": "Other",
            },
            "split_rules": [
                {
                    "cluster": "AI / ML",
                    "ssoc4": ["2519"],
                    "keywords": ["ai"],
                }
            ],
        }
        jobs = pd.DataFrame(
            {
                "job_id": ["J1", "J2"],
                "title": ["Project Engineer", "AI/ML Engineer"],
                "title_clean": ["project engineer", "ai ml engineer"],
                "description_raw": ["", ""],
                "description_clean": [
                    "develop detailed project schedule and liaise with contractors",
                    "build ai systems for production use",
                ],
                "job_text": [
                    "project engineer develop detailed project schedule and liaise with contractors",
                    "ai ml engineer build ai systems for production use",
                ],
                "experience_years": [1, 1],
                "ssec_eqa": ["70", "70"],
                "ssoc_code": ["25191", "25191"],
                "skills": [
                    ["Planning", "Stakeholder Management"],
                    ["Python", "AI"],
                ],
                "categories": [[], []],
                "primary_category": ["Engineering", "Information Technology"],
            }
        )

        out = assign_role_families(jobs, rules, cluster_rules=cluster_rules).jobs
        self.assertEqual(out.loc[0, "role_family"], "Other")
        self.assertEqual(out.loc[0, "role_family_matched_keyword"], "")
        self.assertEqual(out.loc[1, "role_family"], "AI / ML")
        self.assertEqual(out.loc[1, "role_family_matched_keyword"], "ai")

    def test_role_family_assignment_supports_title_keywords(self) -> None:
        # Ambiguous engineering buckets should only flow into AI / ML when the title
        # itself carries explicit AI wording rather than a generic description mention.
        rules = {}
        cluster_rules = {
            "cluster_broad_family_map": {
                "AI / ML": "ICT",
                "Engineering R&D / Technical Services": "Engineering",
                "Other": "Other",
            },
            "split_rules": [
                {
                    "cluster": "AI / ML",
                    "ssoc4": ["2149"],
                    "title_keywords": ["ai"],
                    "keywords": ["machine learning"],
                },
                {
                    "cluster": "Engineering R&D / Technical Services",
                    "ssoc4": ["2149"],
                },
            ],
        }
        jobs = pd.DataFrame(
            {
                "job_id": ["J1", "J2"],
                "title": ["Project Engineer", "AI Engineer"],
                "title_clean": ["project engineer", "ai engineer"],
                "description_raw": ["", ""],
                "description_clean": [
                    "deliver machine learning enabled projects for clients",
                    "deliver machine learning systems for clients",
                ],
                "job_text": [
                    "project engineer deliver machine learning enabled projects for clients",
                    "ai engineer deliver machine learning systems for clients",
                ],
                "experience_years": [1, 1],
                "ssec_eqa": ["70", "70"],
                "ssoc_code": ["21499", "21499"],
                "skills": [["Project Management"], ["Python", "Machine Learning"]],
                "categories": [[], []],
                "primary_category": ["Engineering", "Engineering"],
            }
        )

        out = assign_role_families(jobs, rules, cluster_rules=cluster_rules).jobs
        self.assertEqual(out.loc[0, "role_family"], "Engineering R&D / Technical Services")
        self.assertEqual(out.loc[1, "role_family"], "AI / ML")
        self.assertEqual(out.loc[1, "role_family_matched_keyword"], "ai")

    def test_role_family_assignment_splits_engineering_disciplines_from_ssoc(self) -> None:
        # Stable engineering SSOC groups should map directly into more specific
        # engineering families instead of a single broad catch-all bucket.
        rules = {}
        cluster_rules = {
            "cluster_broad_family_map": {
                "Civil / Construction / Built Environment": "Engineering",
                "Mechanical / M&E / Transport": "Engineering",
                "Process / Chemical / Environmental": "Engineering",
                "Electrical / Electronics / Embedded / Semiconductor": "Engineering",
                "Industrial / Manufacturing / Quality / Automation": "Engineering",
                "Other": "Other",
            },
            "ssoc4_exact_map": {
                "1321": "Industrial / Manufacturing / Quality / Automation",
                "2142": "Civil / Construction / Built Environment",
                "2144": "Mechanical / M&E / Transport",
                "2145": "Process / Chemical / Environmental",
                "2151": "Electrical / Electronics / Embedded / Semiconductor",
            },
        }
        jobs = pd.DataFrame(
            {
                "job_id": ["J1", "J2", "J3", "J4", "J5"],
                "title": [
                    "Civil Engineer",
                    "Mechanical Engineer",
                    "Chemical Engineer",
                    "Electrical Engineer",
                    "Production Manager",
                ],
                "title_clean": [
                    "civil engineer",
                    "mechanical engineer",
                    "chemical engineer",
                    "electrical engineer",
                    "production manager",
                ],
                "description_raw": ["", "", "", "", ""],
                "description_clean": ["", "", "", "", ""],
                "job_text": [
                    "civil engineer",
                    "mechanical engineer",
                    "chemical engineer",
                    "electrical engineer",
                    "production manager",
                ],
                "experience_years": [1, 1, 1, 1, 1],
                "ssec_eqa": ["70"] * 5,
                "ssoc_code": ["21421", "21441", "21451", "21511", "13210"],
                "skills": [[], [], [], [], []],
                "categories": [[], [], [], [], []],
                "primary_category": ["Engineering"] * 5,
            }
        )

        out = assign_role_families(jobs, rules, cluster_rules=cluster_rules).jobs
        self.assertEqual(out.loc[0, "role_family"], "Civil / Construction / Built Environment")
        self.assertEqual(out.loc[1, "role_family"], "Mechanical / M&E / Transport")
        self.assertEqual(out.loc[2, "role_family"], "Process / Chemical / Environmental")
        self.assertEqual(out.loc[3, "role_family"], "Electrical / Electronics / Embedded / Semiconductor")
        self.assertEqual(out.loc[4, "role_family"], "Industrial / Manufacturing / Quality / Automation")

    def test_role_family_assignment_splits_ambiguous_2149_engineering_roles(self) -> None:
        # SSOC-2149 is too broad to map directly; split rules should carve it into
        # domain-specific engineering families before a final technical-services fallback.
        rules = {}
        cluster_rules = {
            "cluster_broad_family_map": {
                "AI / ML": "ICT",
                "Civil / Construction / Built Environment": "Engineering",
                "Engineering R&D / Technical Services": "Engineering",
                "Other": "Other",
            },
            "split_rules": [
                {
                    "cluster": "AI / ML",
                    "ssoc4": ["2149"],
                    "title_keywords": ["ai"],
                    "keywords": ["machine learning"],
                },
                {
                    "cluster": "Civil / Construction / Built Environment",
                    "ssoc4": ["2149"],
                    "title_keywords": ["project engineer", "quantity surveyor"],
                    "keywords": ["construction", "maincon", "quantity surveyor"],
                },
                {
                    "cluster": "Engineering R&D / Technical Services",
                    "ssoc4": ["2149"],
                },
            ],
        }
        jobs = pd.DataFrame(
            {
                "job_id": ["J1", "J2", "J3"],
                "title": ["Project Engineer", "AI Engineer", "Application Engineer"],
                "title_clean": ["project engineer", "ai engineer", "application engineer"],
                "description_raw": ["", "", ""],
                "description_clean": [
                    "maincon construction coordination and tender support",
                    "machine learning systems for production use",
                    "technical support for scientific analyzers",
                ],
                "job_text": [
                    "project engineer maincon construction coordination and tender support",
                    "ai engineer machine learning systems for production use",
                    "application engineer technical support for scientific analyzers",
                ],
                "experience_years": [1, 1, 1],
                "ssec_eqa": ["70", "70", "70"],
                "ssoc_code": ["21499", "21499", "21499"],
                "skills": [
                    ["Tendering"],
                    ["Python", "Machine Learning"],
                    ["Support"],
                ],
                "categories": [[], [], []],
                "primary_category": ["Engineering", "Engineering", "Engineering"],
            }
        )

        out = assign_role_families(jobs, rules, cluster_rules=cluster_rules).jobs
        self.assertEqual(out.loc[0, "role_family"], "Civil / Construction / Built Environment")
        self.assertEqual(out.loc[1, "role_family"], "AI / ML")
        self.assertEqual(out.loc[2, "role_family"], "Engineering R&D / Technical Services")

    def test_role_family_assignment_prioritizes_support_and_sales_before_industrial(self) -> None:
        # Ambiguous 2149 engineering titles should resolve to support/sales specialist
        # families before a broad manufacturing catch-all consumes them.
        rules = {}
        cluster_rules = {
            "cluster_broad_family_map": {
                "Sales / Business Development": "Business",
                "Engineering R&D / Technical Services": "Engineering",
                "Industrial / Manufacturing / Quality / Automation": "Engineering",
                "Other": "Other",
            },
            "split_rules": [
                {
                    "cluster": "Sales / Business Development",
                    "ssoc4": ["2149"],
                    "title_keywords": ["sales engineer"],
                    "keywords": ["sales", "customer", "quotation"],
                },
                {
                    "cluster": "Engineering R&D / Technical Services",
                    "ssoc4": ["2149"],
                    "title_keywords": ["application engineer", "technical support engineer"],
                    "keywords": ["technical support", "solution", "application engineer"],
                },
                {
                    "cluster": "Industrial / Manufacturing / Quality / Automation",
                    "ssoc4": ["2149"],
                    "keywords": ["manufacturing", "production", "automation"],
                },
            ],
        }
        jobs = pd.DataFrame(
            {
                "job_id": ["J1", "J2", "J3"],
                "title": ["Application Engineer", "Sales Engineer", "Manufacturing Engineer"],
                "title_clean": ["application engineer", "sales engineer", "manufacturing engineer"],
                "description_raw": ["", "", ""],
                "description_clean": [
                    "provide technical support for automation products and recommend solutions",
                    "handle customer quotation follow-ups and technical sales support",
                    "improve manufacturing automation and production throughput",
                ],
                "job_text": [
                    "application engineer provide technical support for automation products and recommend solutions",
                    "sales engineer handle customer quotation follow-ups and technical sales support",
                    "manufacturing engineer improve manufacturing automation and production throughput",
                ],
                "experience_years": [1, 1, 1],
                "ssec_eqa": ["70", "70", "70"],
                "ssoc_code": ["21499", "21499", "21499"],
                "skills": [["Support"], ["Sales"], ["Manufacturing"]],
                "categories": [[], [], []],
                "primary_category": ["Engineering", "Engineering", "Engineering"],
            }
        )

        out = assign_role_families(jobs, rules, cluster_rules=cluster_rules).jobs
        self.assertEqual(out.loc[0, "role_family"], "Engineering R&D / Technical Services")
        self.assertEqual(out.loc[1, "role_family"], "Sales / Business Development")
        self.assertEqual(out.loc[2, "role_family"], "Industrial / Manufacturing / Quality / Automation")

    def test_role_family_assignment_prefers_mechanical_rule_before_civil_keyword_fallback(self) -> None:
        # Generic design/service titles should first use stronger domain-specific rules
        # such as mechanical M&E before a later civil keyword-only fallback.
        rules = {}
        cluster_rules = {
            "cluster_broad_family_map": {
                "Civil / Construction / Built Environment": "Engineering",
                "Mechanical / M&E / Transport": "Engineering",
                "Other": "Other",
            },
            "split_rules": [
                {
                    "cluster": "Mechanical / M&E / Transport",
                    "ssoc4": ["2149"],
                    "title_keywords": ["mechanical design engineer", "service engineer"],
                    "keywords": ["mechanical", "hvac", "building services", "chiller"],
                },
                {
                    "cluster": "Civil / Construction / Built Environment",
                    "ssoc4": ["2149"],
                    "keywords": ["civil", "construction", "c&s", "roadwork"],
                },
            ],
        }
        jobs = pd.DataFrame(
            {
                "job_id": ["J1", "J2"],
                "title": ["Mechanical Design Engineer", "Design Engineer (C&S)"],
                "title_clean": ["mechanical design engineer", "design engineer c&s"],
                "description_raw": ["", ""],
                "description_clean": [
                    "deliver mechanical hvac building services design for chiller systems",
                    "support civil roadwork and c&s coordination for construction projects",
                ],
                "job_text": [
                    "mechanical design engineer deliver mechanical hvac building services design for chiller systems",
                    "design engineer c&s support civil roadwork and c&s coordination for construction projects",
                ],
                "experience_years": [1, 1],
                "ssec_eqa": ["70", "70"],
                "ssoc_code": ["21499", "21499"],
                "skills": [["HVAC"], ["C&S"]],
                "categories": [[], []],
                "primary_category": ["Engineering", "Engineering"],
            }
        )

        out = assign_role_families(jobs, rules, cluster_rules=cluster_rules).jobs
        self.assertEqual(out.loc[0, "role_family"], "Mechanical / M&E / Transport")
        self.assertEqual(out.loc[1, "role_family"], "Civil / Construction / Built Environment")

    def test_role_family_assignment_splits_noisy_13291_manager_rows(self) -> None:
        # SSOC-13291 is too noisy for exact mapping; split it by domain evidence while
        # keeping the more specific 13292 quality-manager code exact.
        rules = {}
        cluster_rules = {
            "cluster_broad_family_map": {
                "Civil / Construction / Built Environment": "Engineering",
                "Process / Chemical / Environmental": "Engineering",
                "Engineering R&D / Technical Services": "Engineering",
                "Industrial / Manufacturing / Quality / Automation": "Engineering",
                "Other": "Other",
            },
            "ssoc5_exact_map": {
                "13292": "Industrial / Manufacturing / Quality / Automation",
            },
            "split_rules": [
                {
                    "cluster": "Civil / Construction / Built Environment",
                    "ssoc5": ["13291"],
                    "keywords": ["construction", "roadwork", "civil", "worksite"],
                },
                {
                    "cluster": "Process / Chemical / Environmental",
                    "ssoc5": ["13291"],
                    "keywords": ["chemical", "material planning"],
                },
                {
                    "cluster": "Engineering R&D / Technical Services",
                    "ssoc5": ["13291"],
                },
            ],
        }
        jobs = pd.DataFrame(
            {
                "job_id": ["J1", "J2", "J3"],
                "title": [
                    "Project Engineer (Civil / Roadwork)",
                    "Chemical Engineer (Material Planning)",
                    "QA Assistant Manager",
                ],
                "title_clean": [
                    "project engineer civil roadwork",
                    "chemical engineer material planning",
                    "qa assistant manager",
                ],
                "description_raw": ["", "", ""],
                "description_clean": [
                    "manage worksite construction coordination and roadwork planning",
                    "chemical material planning for plant operations",
                    "lead quality assurance team for factory operations",
                ],
                "job_text": [
                    "project engineer civil roadwork manage worksite construction coordination and roadwork planning",
                    "chemical engineer material planning for plant operations",
                    "qa assistant manager lead quality assurance team for factory operations",
                ],
                "experience_years": [1, 1, 1],
                "ssec_eqa": ["70", "70", "70"],
                "ssoc_code": ["13291", "13291", "13292"],
                "skills": [["Construction"], ["Chemical"], ["Quality Assurance"]],
                "categories": [[], [], []],
                "primary_category": ["Engineering", "Engineering", "Engineering"],
            }
        )

        out = assign_role_families(jobs, rules, cluster_rules=cluster_rules).jobs
        self.assertEqual(out.loc[0, "role_family"], "Civil / Construction / Built Environment")
        self.assertEqual(out.loc[1, "role_family"], "Process / Chemical / Environmental")
        self.assertEqual(out.loc[2, "role_family"], "Industrial / Manufacturing / Quality / Automation")
        self.assertEqual(out.loc[2, "role_family_source"], "ssoc5_exact")


if __name__ == "__main__":
    unittest.main()
