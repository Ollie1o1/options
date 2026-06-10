"""Tests for the outlook factor library — pure, offline.

Each factor maps a price series (and index) to a forward-looking directional
signal: positive = bullish tilt, negative = bearish. The factors chosen all
have academic support at the 1-3 month horizon (12-1 momentum, trend, 1-month
reversal, relative strength).

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.outlook.test_factors -v
"""
from __future__ import annotations

import math
import unittest

from src.outlook.factors import (
    mom_12_1, trend_score, reversal_1m, relative_strength,
)


def _rising(n=300, drift=0.002):
    return [100.0 * math.exp(drift * i) for i in range(n)]


def _falling(n=300, drift=0.002):
    return [100.0 * math.exp(-drift * i) for i in range(n)]


class MomentumTests(unittest.TestCase):
    def test_uptrend_positive_downtrend_negative(self):
        self.assertGreater(mom_12_1(_rising()), 0)
        self.assertLess(mom_12_1(_falling()), 0)

    def test_insufficient_history_none(self):
        self.assertIsNone(mom_12_1([100.0] * 50))


class TrendTests(unittest.TestCase):
    def test_above_200d_positive(self):
        self.assertGreater(trend_score(_rising()), 0)
        self.assertLess(trend_score(_falling()), 0)


class ReversalTests(unittest.TestCase):
    def test_recent_spike_gives_negative_contrarian_signal(self):
        # Flat then a sharp 1-month rip up → reversal factor should be negative
        closes = [100.0] * 280 + [100.0 * (1 + 0.01 * i) for i in range(1, 21)]
        self.assertLess(reversal_1m(closes), 0)

    def test_recent_drop_gives_positive_contrarian_signal(self):
        closes = [100.0] * 280 + [100.0 * (1 - 0.01 * i) for i in range(1, 21)]
        self.assertGreater(reversal_1m(closes), 0)


class RelativeStrengthTests(unittest.TestCase):
    def test_outperformer_positive(self):
        strong = _rising(drift=0.004)
        bench = _rising(drift=0.001)
        self.assertGreater(relative_strength(strong, bench), 0)

    def test_underperformer_negative(self):
        weak = _rising(drift=0.0005)
        bench = _rising(drift=0.003)
        self.assertLess(relative_strength(weak, bench), 0)


if __name__ == "__main__":
    unittest.main()
