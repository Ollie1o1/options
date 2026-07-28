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


class TestEvidenceBasedScoring(unittest.TestCase):
    """Scoring follows docs/SQUEEZE_BACKTEST.md (480,744 graded observations).

    Only effects whose bootstrapped 95% CI excludes zero are scored:
      days-to-cover >= 5   -2.38pp [-4.82, -0.75]  -> NOT scored (was +2)
      5d return >= +10%    +3.31pp [+1.31, +5.77]  -> scored +2 (was absent)
      5d return <= -10%    -1.96pp [-8.53, +2.30]  -> NOT scored (was +1)
      RVOL > 1.5           -1.39pp [-5.46, +1.77]  -> NOT scored (was +1)
      SI rising MoM        +1.30pp [-0.01, +2.70]  -> kept at +1 (borderline)
    Unscored factors remain visible as evidence lines.
    """

    def test_days_to_cover_is_not_scored(self):
        # dtc was the grader's largest bonus and measured significantly harmful.
        high = D.assess_squeeze({"short_interest": 0.22, "short_interest_dtc": 12.0})
        none = D.assess_squeeze({"short_interest": 0.22})
        self.assertEqual(high.points, none.points)

    def test_days_to_cover_still_shown_as_evidence(self):
        setup = D.assess_squeeze({"short_interest": 0.22, "short_interest_dtc": 12.0})
        self.assertTrue(any("days to cover" in e for e in setup.evidence))

    def test_upward_momentum_is_scored(self):
        hot = D.assess_squeeze({"short_interest": 0.22, "ret_5d": 12.0})
        flat = D.assess_squeeze({"short_interest": 0.22, "ret_5d": 1.0})
        self.assertEqual(hot.points - flat.points, 2)

    def test_sharp_drop_is_not_scored(self):
        # "late shorts pressing" pointed the wrong way: squeezes follow strength.
        dropped = D.assess_squeeze({"short_interest": 0.22, "ret_5d": -18.2})
        flat = D.assess_squeeze({"short_interest": 0.22, "ret_5d": 1.0})
        self.assertEqual(dropped.points, flat.points)

    def test_rvol_is_not_scored(self):
        hot = D.assess_squeeze({"short_interest": 0.22, "rvol": 3.0})
        cold = D.assess_squeeze({"short_interest": 0.22, "rvol": 0.5})
        self.assertEqual(hot.points, cold.points)

    def test_short_interest_still_drives_the_score(self):
        heavy = D.assess_squeeze({"short_interest": 0.30})
        light = D.assess_squeeze({"short_interest": 0.16})
        self.assertGreater(heavy.points, light.points)


class TestRet5dUnits(unittest.TestCase):
    """The scan pipeline stores ret_5d as a FRACTION.

    data_fetching.calculate_momentum_indicators returns
    ``close[-1]/close[-6] - 1.0`` (so +12% arrives as 0.12), but assess_squeeze
    compares against percent thresholds. That mismatch is why the old
    "late shorts" rule (<= -10.0) could never fire in production, documented in
    docs/SQUEEZE_BACKTEST.md as "the ret_5d rule is dead". The momentum rule
    would inherit exactly the same defect, so the adapter converts.
    """

    def test_row_adapter_converts_fraction_to_percent(self):
        scored = D.assess_squeeze_row({"short_interest": 0.22, "ret_5d": 0.12})
        flat = D.assess_squeeze_row({"short_interest": 0.22, "ret_5d": 0.01})
        self.assertEqual(scored.points - flat.points, 2)

    def test_row_adapter_handles_missing_ret_5d(self):
        setup = D.assess_squeeze_row({"short_interest": 0.22})
        self.assertIsNotNone(setup.grade)

    def test_direct_percent_call_is_unchanged(self):
        # assess_squeeze itself keeps its percent contract.
        setup = D.assess_squeeze({"short_interest": 0.22, "ret_5d": 12.0})
        flat = D.assess_squeeze({"short_interest": 0.22, "ret_5d": 1.0})
        self.assertEqual(setup.points - flat.points, 2)


class TestAssessSqueeze(unittest.TestCase):
    def test_nbis_replay_still_grades_setup(self):
        # NBIS motivated the feature. It scored 6 under the old rules, two of
        # which (dtc, -18.2% 5d return) are now known to be backwards. It keeps
        # SETUP on SI 2 + rising 1 + call-skew 1 = 4.
        setup = D.assess_squeeze(nbis_fields())
        self.assertEqual(setup.grade, D.SETUP)
        self.assertEqual(setup.points, 4)
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
        # exactly 20% SI + 5d momentum: 2 + 2 = 4 points → SETUP. Momentum
        # replaces days-to-cover as the second scored leg (dtc measured -2.38pp).
        setup = D.assess_squeeze({
            "short_interest": 0.20,
            "ret_5d": 11.0,
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
