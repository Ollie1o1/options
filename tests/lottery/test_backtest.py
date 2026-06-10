"""Tests for the lottery-ticket backtest harness (pure pieces).

The harness pulls real underlying price history, but its math is pure and
tested here offline: feature construction, single-ticket outcome simulation,
and sleeve summary statistics (the win-rate / breakeven-rate the experiment
is really about).

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.lottery.test_backtest -v
"""
from __future__ import annotations

import math
import unittest

from src.lottery.selector import DEFAULT_LOTTERY_CONFIG
from src.lottery.backtest import build_candidate, simulate_outcome, summarize


class SimulateOutcomeTests(unittest.TestCase):

    def test_otm_call_expires_worthless_is_total_loss(self):
        # Bought a $1.00 call, qty 100, stock finishes below strike.
        res = simulate_outcome("call", strike=120.0, premium=1.0, s_exp=110.0,
                               qty=100.0, cost_mult=1.0)
        self.assertFalse(res["win"])
        self.assertAlmostEqual(res["pnl_usd"], -100.0, places=6)

    def test_itm_call_pays_intrinsic_minus_cost(self):
        # $1.00 call, qty 100, finishes $15 ITM → payoff 1500, cost 100, pnl 1400.
        res = simulate_outcome("call", strike=120.0, premium=1.0, s_exp=135.0,
                               qty=100.0, cost_mult=1.0)
        self.assertTrue(res["win"])
        self.assertAlmostEqual(res["pnl_usd"], 1400.0, places=6)

    def test_put_intrinsic(self):
        res = simulate_outcome("put", strike=80.0, premium=1.0, s_exp=60.0,
                               qty=100.0, cost_mult=1.0)
        self.assertAlmostEqual(res["payoff_usd"], 2000.0, places=6)

    def test_entry_slippage_increases_cost(self):
        base = simulate_outcome("call", 120.0, 1.0, 110.0, 100.0, cost_mult=1.0)
        slip = simulate_outcome("call", 120.0, 1.0, 110.0, 100.0, cost_mult=1.10)
        self.assertLess(slip["pnl_usd"], base["pnl_usd"])


class SummarizeTests(unittest.TestCase):

    def test_one_big_win_can_beat_nine_losses(self):
        """The core hypothesis: 1 win / 9 losses can still be net positive."""
        trades = [{"pnl_usd": -100.0, "cost_usd": 100.0, "payoff_usd": 0.0, "win": False}] * 9
        trades = list(trades) + [
            {"pnl_usd": 1500.0, "cost_usd": 100.0, "payoff_usd": 1600.0, "win": True}
        ]
        s = summarize(trades)
        self.assertEqual(s["n"], 10)
        self.assertEqual(s["wins"], 1)
        self.assertAlmostEqual(s["win_rate"], 0.10, places=6)
        self.assertGreater(s["net_pnl"], 0.0)  # 1500 - 900 = +600

    def test_breakeven_win_rate_is_inverse_of_avg_win_multiple(self):
        # One win returning 5x cost (payoff/cost = 16x? no) -> avg_win_mult=16,
        # breakeven = 1/16.
        trades = [
            {"pnl_usd": -100.0, "cost_usd": 100.0, "payoff_usd": 0.0, "win": False},
            {"pnl_usd": 1500.0, "cost_usd": 100.0, "payoff_usd": 1600.0, "win": True},
        ]
        s = summarize(trades)
        self.assertAlmostEqual(s["avg_win_multiple"], 16.0, places=6)
        self.assertAlmostEqual(s["breakeven_win_rate"], 1.0 / 16.0, places=6)

    def test_empty_is_safe(self):
        s = summarize([])
        self.assertEqual(s["n"], 0)
        self.assertEqual(s["net_pnl"], 0.0)


class BuildCandidateTests(unittest.TestCase):

    def test_uptrend_builds_a_call(self):
        # Uptrend with realistic daily wobble (non-zero vol).
        closes = [100.0 * (1.003 ** i) * (1.0 + 0.01 * math.sin(i)) for i in range(300)]
        cand = build_candidate("UP", closes, t_idx=290, cfg=DEFAULT_LOTTERY_CONFIG)
        self.assertIsNotNone(cand)
        self.assertEqual(cand["direction"], "call")
        self.assertGreater(cand["strike"], cand["spot"])  # OTM call
        self.assertGreater(cand["premium"], 0.0)

    def test_downtrend_builds_a_put(self):
        closes = [300.0 * (0.997 ** i) * (1.0 + 0.01 * math.sin(i)) for i in range(300)]
        cand = build_candidate("DN", closes, t_idx=290, cfg=DEFAULT_LOTTERY_CONFIG)
        self.assertIsNotNone(cand)
        self.assertEqual(cand["direction"], "put")
        self.assertLess(cand["strike"], cand["spot"])  # OTM put

    def test_insufficient_history_returns_none(self):
        closes = [100.0] * 20
        self.assertIsNone(build_candidate("X", closes, t_idx=10, cfg=DEFAULT_LOTTERY_CONFIG))


if __name__ == "__main__":
    unittest.main()
