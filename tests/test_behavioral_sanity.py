from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from module_readiness.orchestration import run_pipeline


class TestBehavioralSanity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Use a quick pipeline run for a coarse regression check over real-ish data.
        cls.state = run_pipeline(quick=True)

    def test_cs3244_is_not_uniformly_high_everywhere(self) -> None:
        # A specialization-heavy module should show variation across role families.
        profile = self.state.module_role_scores
        cs = profile[profile["module_code"] == "CS3244"]
        if cs.empty:
            self.skipTest("CS3244 not available in module set")

        top = cs.sort_values("role_score", ascending=False).iloc[0]
        bottom = cs.sort_values("role_score", ascending=True).iloc[0]
        self.assertGreater(float(top["role_score"]), float(bottom["role_score"]))


if __name__ == "__main__":
    unittest.main()
