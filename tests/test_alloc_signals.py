"""Signals must be computed from the past only.

A signal that peeks even one day ahead manufactures an edge out of nothing, and
it is the easiest way to fool a backtest. These tests exist mostly to make that
impossible rather than to check arithmetic.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_alloc_signals -v
"""
from __future__ import annotations

import unittest

from src.alloc.signals import (Snapshot, SignalHistory, atm_iv, passes,
                               snapshot)

EXP = "2024-03-15"


def _row(strike, typ, bid, ask, iv, expiration=EXP):
    return {"expiration": expiration, "strike": float(strike), "type": typ,
            "bid": bid, "ask": ask, "iv": iv}


class AtmIvTest(unittest.TestCase):
    def test_picks_the_strike_nearest_spot(self):
        chain = [_row(100, "call", 2.9, 3.1, 0.20),
                 _row(100, "put", 2.9, 3.1, 0.22),
                 _row(150, "call", 0.1, 0.2, 0.90)]
        self.assertAlmostEqual(atm_iv(chain, 100.0), 0.21, places=3)

    def test_ignores_missing_or_zero_iv(self):
        chain = [_row(100, "call", 2.9, 3.1, None),
                 _row(100, "put", 2.9, 3.1, 0.0),
                 _row(105, "call", 1.0, 1.2, 0.30)]
        self.assertAlmostEqual(atm_iv(chain, 100.0), 0.30, places=3)

    def test_no_usable_iv_returns_none(self):
        self.assertIsNone(atm_iv([_row(100, "call", 1, 2, None)], 100.0))

    def test_empty_chain_returns_none(self):
        self.assertIsNone(atm_iv([], 100.0))

    def test_no_spot_returns_none(self):
        self.assertIsNone(atm_iv([_row(100, "call", 1, 2, 0.2)], None))


class SnapshotTest(unittest.TestCase):
    def test_snapshot_recovers_spot_and_iv(self):
        chain = [_row(100, "call", 2.9, 3.1, 0.20),
                 _row(100, "put", 2.9, 3.1, 0.20)]
        s = snapshot(chain, "2024-01-05")
        self.assertAlmostEqual(s.spot, 100.0, places=2)
        self.assertAlmostEqual(s.atm_iv, 0.20, places=3)

    def test_empty_chain_snapshot_is_all_none(self):
        s = snapshot([], "2024-01-05")
        self.assertIsNone(s.spot)
        self.assertIsNone(s.atm_iv)


class CausalityTest(unittest.TestCase):
    """THE tests. Nothing may be known before it happens."""

    def _feed(self, ivs, start_day=1):
        h = SignalHistory(lookback=52)
        for i, iv in enumerate(ivs):
            h.update("AAA", Snapshot(f"2024-01-{start_day+i:02d}", 100.0, iv))
        return h

    def test_features_reflect_only_what_was_fed(self):
        h = self._feed([0.10] * 20)
        first = h.features("AAA")["iv_rank"]
        h.update("AAA", Snapshot("2024-02-01", 100.0, 0.99))   # a later spike
        self.assertNotEqual(first, h.features("AAA")["iv_rank"],
                            "adding a later day must change nothing retroactively")

    def test_an_out_of_order_update_is_ignored(self):
        h = self._feed([0.10] * 20)
        before = h.features("AAA")
        h.update("AAA", Snapshot("2023-01-01", 999.0, 999.0))  # older date
        self.assertEqual(h.features("AAA"), before)

    def test_a_repeated_date_is_ignored(self):
        h = self._feed([0.10] * 20)
        before = h.features("AAA")
        h.update("AAA", Snapshot("2024-01-20", 500.0, 500.0))  # same last date
        self.assertEqual(h.features("AAA"), before)

    def test_no_history_gives_no_features(self):
        self.assertEqual(SignalHistory().features("NOPE"), {})

    def test_symbols_do_not_share_history(self):
        h = SignalHistory()
        for i in range(15):
            h.update("AAA", Snapshot(f"2024-01-{i+1:02d}", 100.0, 0.10))
        self.assertEqual(h.features("BBB"), {})


class IvRankTest(unittest.TestCase):
    def test_highest_iv_in_the_window_ranks_at_the_top(self):
        h = SignalHistory()
        for i, iv in enumerate([0.10] * 19 + [0.50]):
            h.update("AAA", Snapshot(f"2024-01-{i+1:02d}", 100.0, iv))
        self.assertGreater(h.features("AAA")["iv_rank"], 95)

    def test_lowest_iv_in_the_window_ranks_at_the_bottom(self):
        h = SignalHistory()
        for i, iv in enumerate([0.50] * 19 + [0.10]):
            h.update("AAA", Snapshot(f"2024-01-{i+1:02d}", 100.0, iv))
        self.assertLess(h.features("AAA")["iv_rank"], 5)

    def test_too_little_history_gives_no_rank(self):
        h = SignalHistory()
        for i in range(5):
            h.update("AAA", Snapshot(f"2024-01-{i+1:02d}", 100.0, 0.2))
        self.assertIsNone(h.features("AAA")["iv_rank"])


class TrendTest(unittest.TestCase):
    def test_price_above_its_average_is_a_positive_trend(self):
        h = SignalHistory()
        for i in range(15):
            h.update("AAA", Snapshot(f"2024-01-{i+1:02d}", 100.0 + i, 0.2))
        self.assertGreater(h.features("AAA")["trend"], 0)

    def test_price_below_its_average_is_a_negative_trend(self):
        h = SignalHistory()
        for i in range(15):
            h.update("AAA", Snapshot(f"2024-01-{i+1:02d}", 100.0 - i, 0.2))
        self.assertLess(h.features("AAA")["trend"], 0)

    def test_four_week_return_uses_the_fourth_prior_observation(self):
        h = SignalHistory()
        for i, spot in enumerate([100.0, 100.0, 100.0, 100.0, 110.0]):
            h.update("AAA", Snapshot(f"2024-01-{i+1:02d}", spot, 0.2))
        self.assertAlmostEqual(h.features("AAA")["ret_4w"], 10.0, places=6)


class PassesTest(unittest.TestCase):
    def test_no_conditions_always_passes(self):
        self.assertTrue(passes({"iv_rank": 50.0}, {}))

    def test_threshold_met_passes(self):
        self.assertTrue(passes({"iv_rank": 60.0}, {"iv_rank_min": 50}))

    def test_threshold_missed_fails(self):
        self.assertFalse(passes({"iv_rank": 40.0}, {"iv_rank_min": 50}))

    def test_an_uncomputable_feature_FAILS_rather_than_passing(self):
        """Treating unknown as a pass would silently turn a signalled strategy
        back into the unconditional one and make the comparison meaningless."""
        self.assertFalse(passes({"iv_rank": None}, {"iv_rank_min": 50}))
        self.assertFalse(passes({}, {"iv_rank_min": 50}))

    def test_every_condition_must_hold(self):
        f = {"iv_rank": 60.0, "trend": -5.0}
        self.assertFalse(passes(f, {"iv_rank_min": 50, "trend_min": 0}))
        self.assertTrue(passes(f, {"iv_rank_min": 50, "trend_max": 0}))

    def test_max_conditions_work(self):
        self.assertTrue(passes({"iv_rank": 20.0}, {"iv_rank_max": 30}))
        self.assertFalse(passes({"iv_rank": 40.0}, {"iv_rank_max": 30}))


if __name__ == "__main__":
    unittest.main()
