"""Tests for the market-direction classifier in src/regime_dashboard.py.

The classifier is a pure function: given an index's price, 50/200-day moving
averages, and short-term returns, it returns an UP / NEUTRAL / DOWN verdict
plus a human-readable reason. No network — these run offline.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_market_direction -v
"""
from __future__ import annotations

import unittest

from src.regime_dashboard import classify_index_direction, direction_from_closes


class ClassifyIndexDirectionTests(unittest.TestCase):

    def test_strong_uptrend_returns_up(self):
        """Above 200d, golden cross, positive momentum → UP."""
        res = classify_index_direction(
            price=510.0, ma50=495.0, ma200=470.0, ret_5d=0.012, ret_20d=0.038
        )
        self.assertEqual(res["verdict"], "UP")

    def test_strong_downtrend_returns_down(self):
        """Below 200d, death cross, negative momentum → DOWN."""
        res = classify_index_direction(
            price=430.0, ma50=445.0, ma200=470.0, ret_5d=-0.018, ret_20d=-0.052
        )
        self.assertEqual(res["verdict"], "DOWN")

    def test_conflicting_signals_returns_neutral(self):
        """Above 200d but death cross and falling momentum → NEUTRAL (mixed)."""
        res = classify_index_direction(
            price=472.0, ma50=468.0, ma200=470.0, ret_5d=-0.004, ret_20d=0.006
        )
        self.assertEqual(res["verdict"], "NEUTRAL")

    def test_reason_names_the_drivers(self):
        """Reason should reference the 200d trend and momentum so it's explainable."""
        res = classify_index_direction(
            price=510.0, ma50=495.0, ma200=470.0, ret_5d=0.012, ret_20d=0.038
        )
        self.assertIn("200d", res["reason"])
        self.assertTrue(any(c.isdigit() for c in res["reason"]))

    def test_insufficient_history_degrades_gracefully(self):
        """Missing 200d MA (short history) must not crash and must flag the gap."""
        res = classify_index_direction(
            price=510.0, ma50=495.0, ma200=None, ret_5d=0.012, ret_20d=0.038
        )
        self.assertIn(res["verdict"], {"UP", "NEUTRAL", "DOWN"})
        self.assertIn("limited", res["reason"].lower())


class DirectionFromClosesTests(unittest.TestCase):

    def test_rising_series_is_up(self):
        """A long, steadily rising close series → UP."""
        closes = [100.0 + i * 0.5 for i in range(260)]  # ~1y, strong uptrend
        res = direction_from_closes(closes)
        self.assertEqual(res["verdict"], "UP")

    def test_falling_series_is_down(self):
        """A long, steadily falling close series → DOWN."""
        closes = [300.0 - i * 0.5 for i in range(260)]
        res = direction_from_closes(closes)
        self.assertEqual(res["verdict"], "DOWN")

    def test_short_series_uses_limited_history_fallback(self):
        """Fewer than 200 closes → no 200d MA → limited-history fallback, no crash."""
        closes = [100.0 + i * 0.5 for i in range(60)]
        res = direction_from_closes(closes)
        self.assertIn("limited", res["reason"].lower())
        self.assertIn(res["verdict"], {"UP", "NEUTRAL", "DOWN"})

    def test_empty_series_is_neutral_not_crash(self):
        res = direction_from_closes([])
        self.assertEqual(res["verdict"], "NEUTRAL")


if __name__ == "__main__":
    unittest.main()
