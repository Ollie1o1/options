"""Tests for the short-squeeze setup detector."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.squeeze import detector as D


def nbis_fields():
    """NBIS as scanned 2026-07-16 — the live case that motivated the feature."""
    return {
        "short_interest": 0.2797,
        "short_interest_dtc": 3.5,
        "short_interest_trend": "rising",
        "iv_skew": -0.089,
        "ret_5d": -18.2,
        "rvol": 0.56,
        "gex_flip_price": 155.0,
        "spot": 176.88,
    }


class TestAssessSqueeze(unittest.TestCase):
    def test_nbis_replay_is_setup(self):
        setup = D.assess_squeeze(nbis_fields())
        self.assertEqual(setup.grade, D.SETUP)
        # SI 2 + dtc 1 + rising 1 + call-skew 1 + late-shorts 1 = 6
        self.assertEqual(setup.points, 6)
        self.assertTrue(setup.evidence)

    def test_low_si_is_none_even_with_flow(self):
        # MU 2026-07-16: 2.8% float short, covering — a gamma story, not SI squeeze
        setup = D.assess_squeeze({
            "short_interest": 0.028,
            "short_interest_dtc": 0.55,
            "short_interest_trend": "falling",
            "iv_skew": -0.05,
            "ret_5d": 4.0,
            "rvol": 2.5,
        })
        self.assertEqual(setup.grade, D.NONE)

    def test_watch_band(self):
        # 16% SI + rising: 1 + 1 = 2 points → WATCH (SI < 20% blocks SETUP)
        setup = D.assess_squeeze({
            "short_interest": 0.16,
            "short_interest_trend": "rising",
        })
        self.assertEqual(setup.grade, D.WATCH)

    def test_high_si_but_thin_evidence_is_watch(self):
        # 22% SI alone = 2 points: SETUP needs >= 4, SI >= 15% keeps WATCH
        setup = D.assess_squeeze({"short_interest": 0.22})
        self.assertEqual(setup.grade, D.WATCH)
        self.assertEqual(setup.points, 2)

    def test_setup_threshold_edge(self):
        # exactly 20% SI, dtc 5, no more: 2 + 2 = 4 points → SETUP
        setup = D.assess_squeeze({
            "short_interest": 0.20,
            "short_interest_dtc": 5.0,
        })
        self.assertEqual(setup.grade, D.SETUP)

    def test_all_missing_is_none(self):
        setup = D.assess_squeeze({})
        self.assertEqual(setup.grade, D.NONE)
        self.assertEqual(setup.points, 0)

    def test_nan_tolerance(self):
        setup = D.assess_squeeze({
            "short_interest": float("nan"),
            "short_interest_dtc": float("nan"),
            "iv_skew": float("nan"),
            "ret_5d": float("nan"),
            "rvol": float("nan"),
        })
        self.assertEqual(setup.grade, D.NONE)

    def test_si_accepts_percent_scale(self):
        # defensive: a 0-100-scaled SI (27.97) must not read as 2797% float
        setup = D.assess_squeeze({"short_interest": 27.97,
                                  "short_interest_trend": "rising"})
        self.assertEqual(setup.si_pct, 27.97)
        self.assertIn(setup.grade, (D.WATCH, D.SETUP))

    def test_gex_context_line_present_when_available(self):
        setup = D.assess_squeeze(nbis_fields())
        joined = " ".join(setup.evidence).lower()
        self.assertIn("gamma", joined)

    def test_evidence_mentions_core_facts(self):
        setup = D.assess_squeeze(nbis_fields())
        joined = " ".join(setup.evidence)
        self.assertIn("28.0%", joined)   # SI pct rendered
        self.assertIn("3.5", joined)     # days to cover
        self.assertIn("rising", joined)


class TestAssessSqueezeRow(unittest.TestCase):
    def test_row_adapter_maps_underlying_price(self):
        row = dict(nbis_fields())
        del row["spot"]
        row["underlying_price"] = 176.88
        setup = D.assess_squeeze_row(row)
        self.assertEqual(setup.grade, D.SETUP)

    def test_row_adapter_handles_missing_keys(self):
        setup = D.assess_squeeze_row({"strike": 100.0, "type": "call"})
        self.assertEqual(setup.grade, D.NONE)


if __name__ == "__main__":
    unittest.main()
