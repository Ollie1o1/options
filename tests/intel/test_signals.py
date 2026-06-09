"""Tests for src/intel/signals.py — pure, offline."""
from __future__ import annotations

import unittest

from src.intel import signals as S


class TrendTests(unittest.TestCase):
    def test_uptrend_positive(self):
        sig = S.trend_signal(price=110, ma50=105, ma200=100)
        self.assertGreater(sig.value, 0)
        self.assertEqual(sig.label, "UP")

    def test_downtrend_negative(self):
        sig = S.trend_signal(price=90, ma50=95, ma200=100)
        self.assertLess(sig.value, 0)
        self.assertEqual(sig.label, "DOWN")

    def test_missing_data_is_zero(self):
        self.assertEqual(S.trend_signal(None, None, None).value, 0.0)

    def test_limited_history_no_200d(self):
        sig = S.trend_signal(price=110, ma50=100, ma200=None)
        self.assertGreater(sig.value, 0)
        self.assertIn("limited", sig.detail)


class MomentumTests(unittest.TestCase):
    def test_falling(self):
        sig = S.momentum_signal(ret_5d=-0.04, ret_20d=-0.05)
        self.assertLess(sig.value, 0)
        self.assertEqual(sig.label, "falling")

    def test_rising(self):
        sig = S.momentum_signal(ret_5d=0.04, ret_20d=0.05)
        self.assertGreater(sig.value, 0)

    def test_clamped(self):
        sig = S.momentum_signal(ret_5d=0.5, ret_20d=0.5)
        self.assertLessEqual(sig.value, 1.0)


class BounceTests(unittest.TestCase):
    def test_high_rate_bullish(self):
        sig = S.bounce_signal(0.88, 26)
        self.assertGreater(sig.value, 0)

    def test_thin_sample_discounted(self):
        thick = S.bounce_signal(0.88, 26).value
        thin = S.bounce_signal(0.88, 5).value
        self.assertLess(thin, thick)

    def test_none_is_zero(self):
        self.assertEqual(S.bounce_signal(None, None).value, 0.0)


class RsiTests(unittest.TestCase):
    def test_oversold_bullish(self):
        sig = S.rsi_signal(20)
        self.assertGreater(sig.value, 0)
        self.assertEqual(sig.label, "oversold")

    def test_overbought_bearish(self):
        sig = S.rsi_signal(80)
        self.assertLess(sig.value, 0)

    def test_neutral_zero(self):
        self.assertEqual(S.rsi_signal(50).value, 0.0)


class SupportTests(unittest.TestCase):
    def test_at_support_strong(self):
        self.assertGreater(S.support_proximity_signal(-0.005).value, 0.5)

    def test_far_weak(self):
        self.assertEqual(S.support_proximity_signal(-0.10).value, 0.0)


class AnalystTests(unittest.TestCase):
    def test_upgrades_positive(self):
        self.assertGreater(S.analyst_signal(3, 0).value, 0)

    def test_downgrades_negative(self):
        self.assertLess(S.analyst_signal(0, 3).value, 0)

    def test_none_zero(self):
        self.assertEqual(S.analyst_signal(0, 0).value, 0.0)


class EarningsTests(unittest.TestCase):
    def test_imminent_is_risk_and_nondirectional(self):
        sig = S.earnings_signal(2)
        self.assertEqual(sig.value, -1.0)
        self.assertFalse(sig.directional)

    def test_far_small(self):
        self.assertGreaterEqual(S.earnings_signal(40).value, -0.1)

    def test_none_neutral(self):
        self.assertEqual(S.earnings_signal(None).value, 0.0)


class BuildSignalsTests(unittest.TestCase):
    def test_full_state(self):
        state = {
            "price": 200, "ma50": 205, "ma200": 188,
            "ret_5d": -0.10, "ret_20d": -0.08,
            "bounce_rate": 0.88, "bounce_n": 26, "rsi": 36,
            "pct_to_support": -0.058, "news_sentiment": -0.2,
            "analyst_raises": 0, "analyst_cuts": 2,
            "iv_rank": 0.78, "skew": 0.1, "days_to_earnings": 14,
        }
        sigs = S.build_signals(state)
        self.assertEqual(set(sigs), {
            "trend", "momentum", "bounce", "rsi", "support", "news",
            "analyst", "options", "earnings"})
        self.assertGreater(sigs["trend"].value, 0)       # above 200d
        self.assertLess(sigs["momentum"].value, 0)       # falling
        self.assertLess(sigs["analyst"].value, 0)        # cuts

    def test_empty_state_no_crash(self):
        sigs = S.build_signals({})
        self.assertTrue(all(s.value == 0.0 for s in sigs.values()))


if __name__ == "__main__":
    unittest.main()
