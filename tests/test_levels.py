"""Tests for src/levels.py — support/resistance levels and empirical bounce odds.

All pure-function, no network.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_levels -v
"""
from __future__ import annotations

import math
import unittest

from src.levels import (
    bounce_stats,
    rsi,
    support_resistance_levels,
)

# Rally to a peak, then pull back off it: leaves the prior peak as resistance
# above and the moving averages / swing lows as support below.
_PULLBACK = (
    [100.0 - i for i in range(30)]          # 100 -> 71  (down)
    + [71.0 + i for i in range(30)]         # 71  -> 100 (up to a peak)
    + [100.0 - i * 0.4 for i in range(20)]  # 100 -> 92.4 (pull back off peak)
)


class SupportResistanceTests(unittest.TestCase):

    def test_levels_split_above_and_below_price(self):
        """A pullback off a peak has both support below and resistance above."""
        res = support_resistance_levels(_PULLBACK)
        self.assertTrue(res["supports"], "expected at least one support below price")
        self.assertTrue(res["resistances"], "expected at least one resistance above price")
        for s in res["supports"]:
            self.assertLess(s["level"], res["price"])
            self.assertLess(s["pct"], 0)
        for r in res["resistances"]:
            self.assertGreater(r["level"], res["price"])
            self.assertGreater(r["pct"], 0)

    def test_supports_sorted_nearest_first(self):
        res = support_resistance_levels(_PULLBACK)
        levels = [s["level"] for s in res["supports"]]
        self.assertEqual(levels, sorted(levels, reverse=True))

    def test_current_price_override(self):
        closes = [100.0 + i * 0.1 for i in range(250)]
        res = support_resistance_levels(closes, current=999.0)
        self.assertEqual(res["price"], 999.0)
        # Everything is far below an inflated current price → all support.
        self.assertEqual(res["resistances"], [])

    def test_empty_series_is_safe(self):
        res = support_resistance_levels([])
        self.assertIsNone(res["price"])
        self.assertEqual(res["supports"], [])
        self.assertEqual(res["resistances"], [])


class BounceStatsTests(unittest.TestCase):

    def test_uptrend_has_high_bounce_rate(self):
        """In a noisy uptrend, forward returns after pullbacks are usually positive."""
        # Wiggle creates real 5-day dips to condition on; drift keeps fwd up.
        closes = [100.0 + i * 0.3 + 5.0 * math.sin(i / 3.0) for i in range(400)]
        b = bounce_stats(closes, lookback_drop_days=5, horizons=(10,))
        st = b["by_horizon"][10]
        self.assertGreater(st["n"], 0)
        self.assertGreater(st["bounce_rate"], 0.6)
        self.assertGreater(st["median"], 0)

    def test_downtrend_has_low_bounce_rate(self):
        closes = [400.0 - i * 0.3 + 5.0 * math.sin(i / 3.0) for i in range(400)]
        b = bounce_stats(closes, lookback_drop_days=5, horizons=(10,))
        st = b["by_horizon"][10]
        self.assertGreater(st["n"], 0)
        self.assertLess(st["bounce_rate"], 0.4)

    def test_short_series_degrades_gracefully(self):
        b = bounce_stats([100.0, 101.0, 102.0], lookback_drop_days=5)
        self.assertIsNone(b["trailing_return"])
        self.assertEqual(b["by_horizon"], {})

    def test_trailing_return_is_reported(self):
        closes = [100.0 + i * 0.3 + 5.0 * math.sin(i / 3.0) for i in range(400)]
        b = bounce_stats(closes, lookback_drop_days=5, horizons=(10,))
        self.assertIsNotNone(b["trailing_return"])
        self.assertEqual(b["lookback_days"], 5)


class RsiTests(unittest.TestCase):

    def test_pure_uptrend_is_overbought(self):
        self.assertEqual(rsi([100.0 + i for i in range(30)]), 100.0)

    def test_pure_downtrend_is_oversold(self):
        val = rsi([200.0 - i for i in range(30)])
        self.assertIsNotNone(val)
        self.assertLess(val, 30)

    def test_short_series_returns_none(self):
        self.assertIsNone(rsi([1.0, 2.0, 3.0], period=14))


if __name__ == "__main__":
    unittest.main()
