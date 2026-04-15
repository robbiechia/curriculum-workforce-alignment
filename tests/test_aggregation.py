from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from module_readiness.analysis import build_indicators


class TestAggregation(unittest.TestCase):
    def test_module_summary_uses_ssoc4_fallback_label_when_top_role_is_other(self) -> None:
        # The summary should stay analytically honest (`top_role_family = Other`) while
        # still exposing a more useful human-facing fallback label.
        jobs = pd.DataFrame(
            {
                "role_family": ["Other"],
                "role_family_name": ["Other"],
                "technical_skills": [["visual design"]],
            }
        )
        modules = pd.DataFrame(
            {
                "module_code": ["AR1102"],
                "module_title": ["Design 2"],
                "module_profile": ["general"],
                "module_source": ["db"],
                "technical_skills": [["visual design"]],
            }
        )
        module_role_scores = pd.DataFrame(
            {
                "module_code": ["AR1102"],
                "module_title": ["Design 2"],
                "role_family": ["Other"],
                "role_family_name": ["Other"],
                "broad_family": ["Other"],
                "role_score": [0.60],
            }
        )
        module_ssoc5_scores = pd.DataFrame(
            {
                "module_code": ["AR1102", "AR1102"],
                "module_title": ["Design 2", "Design 2"],
                "ssoc_4d": ["3432", "2161"],
                "ssoc_4d_name": [
                    "Interior Designers and Decorators",
                    "Building Architects",
                ],
                "ssoc_5d": ["34321", "21610"],
                "ssoc_5d_name": ["Interior designer", "Building architect"],
                "role_score": [0.50, 0.32],
                "evidence_job_count": [6.0, 3.0],
            }
        )

        result = build_indicators(
            jobs=jobs,
            modules=modules,
            module_role_scores=module_role_scores,
            role_rules={},
            module_ssoc5_scores=module_ssoc5_scores,
        )

        summary = result.module_summary.iloc[0]
        self.assertEqual(summary["top_role_family"], "Other")
        self.assertEqual(summary["top_role_family_name"], "Interior Designers and Decorators")
        self.assertEqual(summary["top_role_family_name_source"], "ssoc4_fallback")


if __name__ == "__main__":
    unittest.main()
