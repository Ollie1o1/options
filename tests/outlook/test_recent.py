"""Tests for recent-action context — pure, offline.

These values are DESCRIPTIVE. They never enter the outlook composite; see the
module docstring for why they live outside factors.py.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.outlook.test_recent -v
"""
from __future__ import annotations

import unittest

from src.outlook.recent import (
    trailing_return, recent_context, DEFAULT_LAG_THRESHOLD_PP,
)


def _flat(n=300, level=100.0):
    return [level] * n


def _drop_at_end(n=300, level=100.0, drop=0.13, days=21):
    """Flat, then a linear slide of `drop` over the final `days` bars."""
    s = [level] * (n - days)
    for i in range(1, days + 1):
        s.append(level * (1.0 - drop * i / days))
    return s


class TrailingReturnTests(unittest.TestCase):
    def test_flat_series_returns_zero(self):
        self.assertAlmostEqual(trailing_return(_flat(), lookback=21), 0.0)

    def test_measures_the_requested_window(self):
        s = _drop_at_end(drop=0.13, days=21)
        self.assertAlmostEqual(trailing_return(s, lookback=21), -0.13, places=6)

    def test_none_when_history_too_short(self):
        self.assertIsNone(trailing_return([100.0] * 10, lookback=21))


class RecentContextTests(unittest.TestCase):
    def test_excess_is_in_percentage_points(self):
        inst = _drop_at_end(drop=0.13, days=21)   # -13% over the month
        bench = _flat()                            # 0% over the month
        ctx = recent_context(inst, bench)
        self.assertAlmostEqual(ctx["ret_21d"], -0.13, places=6)
        self.assertAlmostEqual(ctx["excess_21d"], -13.0, places=4)

    def test_flags_a_genuine_laggard(self):
        inst = _drop_at_end(drop=0.13, days=21)
        ctx = recent_context(inst, _flat(), lag_threshold_pp=-5.0)
        self.assertTrue(ctx["lagging"])

    def test_quiet_when_the_whole_market_falls_together(self):
        # Instrument and benchmark both -13%: excess ~0, so no flag. This is
        # the broad-selloff property that an absolute trigger would violate.
        both = _drop_at_end(drop=0.13, days=21)
        ctx = recent_context(both, both, lag_threshold_pp=-5.0)
        self.assertAlmostEqual(ctx["excess_21d"], 0.0, places=6)
        self.assertFalse(ctx["lagging"])

    def test_outperformer_not_flagged(self):
        bench = _drop_at_end(drop=0.13, days=21)
        ctx = recent_context(_flat(), bench, lag_threshold_pp=-5.0)
        self.assertGreater(ctx["excess_21d"], 0.0)
        self.assertFalse(ctx["lagging"])

    def test_short_history_yields_none_and_no_flag(self):
        ctx = recent_context([100.0] * 10, [100.0] * 10)
        self.assertIsNone(ctx["ret_21d"])
        self.assertIsNone(ctx["excess_21d"])
        self.assertFalse(ctx["lagging"])

    def test_default_threshold_is_negative(self):
        self.assertLess(DEFAULT_LAG_THRESHOLD_PP, 0.0)
