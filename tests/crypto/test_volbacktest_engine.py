import math
import unittest

import numpy as np

from src.crypto.volbacktest.engine import simulate_trade, run_backtest, TradeResult
from src.crypto.volbacktest.costs import CostModel

ZERO = CostModel(option_spread_frac=0.0, hedge_slippage_bps=0.0)


class TestEngine(unittest.TestCase):
    def test_flat_path_zero_cost_keeps_full_premium(self):
        spot = [100.0] * 15
        dvol = [50.0] * 15
        r = simulate_trade(spot=spot, dvol_pct=dvol, dte=14, r=0.0,
                           premium_notional=1000.0, cost=ZERO, hedge_step=1)
        self.assertIsInstance(r, TradeResult)
        self.assertAlmostEqual(r.terminal_payout, 0.0, places=6)
        self.assertAlmostEqual(r.net_pnl, r.premium, places=4)
        self.assertAlmostEqual(r.net_pnl, 1000.0, places=2)

    @staticmethod
    def _mc_mean_pnl(impl_pct, real_vol, days, n_paths=400, seed=0):
        """Mean delta-hedged net P&L over many GBM paths. A single path has huge
        realized-vol sampling error (short gamma), so the iv=rv -> 0 and
        iv>rv -> +ve properties only hold in expectation."""
        rng = np.random.default_rng(seed)
        dt = 1 / 365
        out = []
        for _ in range(n_paths):
            S = [100.0]
            for _ in range(days):
                S.append(S[-1] * math.exp(-0.5 * real_vol * real_vol * dt
                                          + real_vol * math.sqrt(dt) * rng.standard_normal()))
            out.append(simulate_trade(S, [impl_pct] * len(S), days, 0.0,
                                      1000.0, ZERO, 1).net_pnl)
        return float(np.mean(out))

    def test_iv_equals_rv_zero_cost_mean_pnl_near_zero(self):
        # Theory: continuously delta-hedged short straddle earns 0 when implied
        # == realized, in expectation. Measured mean ~ +7 on $1000 premium, SE ~11.
        mean = self._mc_mean_pnl(impl_pct=50.0, real_vol=0.50, days=14, seed=7)
        self.assertLess(abs(mean), 0.08 * 1000.0)   # < $80 (well inside a few SE of 0)

    def test_iv_above_rv_is_profitable_in_expectation(self):
        # Sell 60% implied, realize 30% -> short vol wins. Measured mean ~ +496.
        mean = self._mc_mean_pnl(impl_pct=60.0, real_vol=0.30, days=30, seed=3)
        self.assertGreater(mean, 0.2 * 1000.0)       # > $200

    def test_costs_reduce_pnl(self):
        spot = [100.0] * 15
        dvol = [50.0] * 15
        free = simulate_trade(spot, dvol, 14, 0.0, 1000.0, ZERO, 1).net_pnl
        paid = simulate_trade(spot, dvol, 14, 0.0, 1000.0, CostModel(0.04, 5.0), 1).net_pnl
        self.assertLess(paid, free)

    def test_attribution_identity_holds(self):
        np.random.seed(11)
        days = 21
        dt = 1 / 365
        S = [100.0]
        for _ in range(days):
            S.append(S[-1] * math.exp(0.4 * math.sqrt(dt) * np.random.randn()))
        dvol = [55.0] * len(S)
        r = simulate_trade(S, dvol, days, 0.0, 1000.0, CostModel(0.03, 3.0), 1)
        recon = (r.premium - r.terminal_payout + r.hedge_pnl
                 - r.option_cost - r.hedge_cost)
        self.assertAlmostEqual(recon, r.net_pnl, places=6)


class TestDefinedRisk(unittest.TestCase):
    def test_wings_cap_the_tail_on_a_large_move(self):
        # A +60% move: naked short straddle takes a big loss; the capped
        # (wing_pct=10%) variant's loss is bounded by the wing width.
        days = 14
        spot = [100.0] + [160.0] * days        # gap up and stay
        dvol = [50.0] * (days + 1)
        naked = simulate_trade(spot, dvol, days, 0.0, 1000.0, ZERO, 1, wing_pct=0.0)
        capped = simulate_trade(spot, dvol, days, 0.0, 1000.0, ZERO, 1, wing_pct=0.10)
        self.assertLess(naked.net_pnl, capped.net_pnl)      # capped loses less
        self.assertLess(capped.net_pnl, 0.0)                # still a loss
        # capped terminal payout bounded by qty * wing width (K*wing_pct)
        self.assertLessEqual(capped.terminal_payout, capped.qty * 100.0 * 0.10 + 1e-6)

    def test_wings_reduce_collected_premium(self):
        spot = [100.0] * 15
        dvol = [50.0] * 15
        naked = simulate_trade(spot, dvol, 14, 0.0, 1000.0, ZERO, 1, wing_pct=0.0)
        capped = simulate_trade(spot, dvol, 14, 0.0, 1000.0, ZERO, 1, wing_pct=0.10)
        self.assertLess(capped.premium, naked.premium)      # paid for protection
        self.assertGreater(capped.premium, 0.0)


class TestBacktestLoop(unittest.TestCase):
    def test_run_backtest_produces_one_trade_per_entry(self):
        spot = [100.0 for _ in range(120)]
        dvol = [50.0] * 120
        res = run_backtest(spot=spot, dvol_pct=dvol, dte=14, freq=7,
                           r=0.0, premium_notional=1000.0, cost=ZERO, hedge_step=1)
        self.assertEqual(len(res.trades), (120 - 14) // 7 + 1)
        self.assertTrue(all(abs(t.net_pnl - 1000.0) < 1.0 for t in res.trades))


if __name__ == "__main__":
    unittest.main()
