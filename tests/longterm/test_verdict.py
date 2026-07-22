"""Tests for the DISCOVER entry-verdict engine (src/longterm/verdict.py)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.longterm import verdict as V
from src.longterm.discover import CandidateRead, DeepRead
from src.longterm.plan import Tranche


def _candidate(spot=100.0, ladder_target=99.0, support_label="50d MA",
               support_level=None, ann_vol_pct=15.8745):
    """ann_vol_pct=15.8745 recovers to daily_vol_frac=0.01 exactly
    (15.8745 / 100 / sqrt(252) == 0.01), so the 2% floor dominates the
    widening rule unless a test overrides ann_vol_pct."""
    level = support_level if support_level is not None else ladder_target
    return CandidateRead(
        ticker="TST", spot=spot, drawdown_pct=-10.0,
        ma200_distance_pct=-5.0, momentum_12_1=-0.05,
        supports=[{"label": support_label, "level": level, "pct": (level / spot - 1)}],
        bounce={}, ann_vol_pct=ann_vol_pct,
        suggested_ladder=[Tranche(spot, 0.5), Tranche(ladder_target, 0.5)],
    )


class TestVerdictFor(unittest.TestCase):
    def test_buy_now_when_within_two_percent(self):
        c = _candidate(spot=100.0, ladder_target=99.0)  # 1.01% away
        v = V.verdict_for(c)
        self.assertEqual(v.state, V.BUY_NOW)
        self.assertIsNone(v.target)
        self.assertIn("50d MA", v.reason)

    def test_wait_when_beyond_two_percent(self):
        c = _candidate(spot=100.0, ladder_target=85.0)  # 17.6% away
        v = V.verdict_for(c)
        self.assertEqual(v.state, V.WAIT)
        self.assertEqual(v.target, 85.0)

    def test_wait_reason_names_the_support_label(self):
        c = _candidate(spot=100.0, ladder_target=85.0, support_label="200d MA")
        v = V.verdict_for(c)
        self.assertIn("200d MA", v.reason)

    def test_no_real_support_uses_fallback_label(self):
        c = _candidate(spot=100.0, ladder_target=90.0)
        c.supports = []  # no real support — ladder's -10% synthetic step
        v = V.verdict_for(c)
        self.assertEqual(v.state, V.WAIT)
        self.assertIn("fallback", v.reason.lower())

    def test_malformed_support_above_spot_treated_as_no_real_support(self):
        # support_resistance_levels never returns this, but guard the same
        # way suggest_ladder itself guards against bad data.
        c = _candidate(spot=100.0, ladder_target=90.0, support_level=110.0,
                       support_label="bad data")
        v = V.verdict_for(c)
        self.assertIn("fallback", v.reason.lower())

    def test_high_volatility_widens_the_buy_now_band(self):
        # 4.17% away — outside the 2% floor, but ann_vol_pct=80 gives a
        # daily vol of ~5.04%, wide enough to make this a BUY NOW.
        c = _candidate(spot=100.0, ladder_target=96.0, ann_vol_pct=80.0)
        v = V.verdict_for(c)
        self.assertEqual(v.state, V.BUY_NOW)

    def test_zero_ann_vol_still_uses_the_two_percent_floor(self):
        c = _candidate(spot=100.0, ladder_target=99.0, ann_vol_pct=None)
        v = V.verdict_for(c)
        self.assertEqual(v.state, V.BUY_NOW)  # 1.01% < 2% floor


class TestApplyCaution(unittest.TestCase):
    def _buy_now(self):
        return V.Verdict(state=V.BUY_NOW, reason="within 1.0% of 50d MA")

    def _wait(self):
        return V.Verdict(state=V.WAIT, target=85.0, reason="17.6% below 200d MA")

    def test_earnings_within_window_adds_caution(self):
        deep = DeepRead(ticker="TST", earnings_days=6)
        v = V.apply_caution(self._buy_now(), deep)
        self.assertEqual(v.caution, "earnings in 6 days")
        self.assertEqual(v.state, V.BUY_NOW)  # state itself unchanged

    def test_earnings_at_exact_boundary_adds_caution(self):
        deep = DeepRead(ticker="TST", earnings_days=14)
        v = V.apply_caution(self._buy_now(), deep)
        self.assertIsNotNone(v.caution)

    def test_earnings_beyond_window_no_caution(self):
        deep = DeepRead(ticker="TST", earnings_days=15)
        v = V.apply_caution(self._buy_now(), deep)
        self.assertIsNone(v.caution)

    def test_wait_verdict_never_captioned(self):
        deep = DeepRead(ticker="TST", earnings_days=1)
        v = V.apply_caution(self._wait(), deep)
        self.assertIsNone(v.caution)

    def test_no_deep_read_no_caution(self):
        v = V.apply_caution(self._buy_now(), None)
        self.assertIsNone(v.caution)

    def test_earnings_days_none_no_caution(self):
        deep = DeepRead(ticker="TST", earnings_days=None)
        v = V.apply_caution(self._buy_now(), deep)
        self.assertIsNone(v.caution)


if __name__ == "__main__":
    unittest.main()
