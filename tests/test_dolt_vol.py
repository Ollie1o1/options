"""Tests for src/dolt_vol.py — delta-hedged short-straddle (pure vol) backtest.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_dolt_vol -v
"""
import unittest
from unittest import mock

from src import dolt_vol as dv


def _c(opt_type, strike, bid, ask, delta, expiration="2024-06-21"):
    return {"type": opt_type, "strike": strike, "bid": bid, "ask": ask,
            "delta": delta, "expiration": expiration}


class AtmStraddleTest(unittest.TestCase):
    def test_picks_strike_nearest_spot(self):
        chain = [_c("call", 100, 3, 3.2, 0.52), _c("put", 100, 3, 3.2, -0.48),
                 _c("call", 110, 1, 1.2, 0.30), _c("put", 110, 6, 6.2, -0.70)]
        call, put = dv._atm_straddle(chain, "2024-05-01", 101.0, 30, 70)
        self.assertEqual(call["strike"], 100)
        self.assertEqual(put["strike"], 100)

    def test_requires_common_strike(self):
        chain = [_c("call", 100, 3, 3.2, 0.5)]   # no put at 100
        self.assertIsNone(dv._atm_straddle(chain, "2024-05-01", 100.0, 30, 70))


class SimulateTest(unittest.TestCase):
    def setUp(self):
        self.entry = [_c("call", 100, 3.0, 3.2, 0.50), _c("put", 100, 3.0, 3.2, -0.50)]
        # later day: vol crushed, straddle cheaper to buy back
        self.later = [_c("call", 100, 1.0, 1.1, 0.50), _c("put", 100, 1.0, 1.1, -0.50)]
        self.spots = {f"2024-05-{d:02d}": 100.0 for d in range(1, 28)}
        self.sdates = sorted(self.spots)

    def test_flat_spot_profits_from_decay(self):
        # spot pinned at 100 (zero realized vol) -> short straddle keeps most premium
        with mock.patch("src.dolt_options.get_chain_near") as gcn:
            # entry call returns (date, entry chain); subsequent calls return later chain
            gcn.side_effect = [("2024-05-01", self.entry)] + \
                              [("2024-05-%02d" % d, self.later) for d in range(2, 28)]
            t = dv.simulate_delta_hedged_straddle("AAPL", "2024-05-01", self.spots,
                                                  self.sdates, db_path="x", target_days=21)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t["hedge_pnl"], 0.0, places=6)   # flat spot -> no hedge P&L
        self.assertGreater(t["net_pnl"], 0)                     # collected 6, bought back ~2
        self.assertAlmostEqual(t["credit"], 6.0, places=6)

    def test_split_day_jump_excluded_from_hedge(self):
        # a 10:1 split mid-hold: raw spot drops 100 -> 10 on one day; must NOT
        # register as a giant hedge move
        spots = {f"2024-05-{d:02d}": (100.0 if d < 10 else 10.0) for d in range(1, 28)}
        sdates = sorted(spots)
        with mock.patch("src.dolt_options.get_chain_near") as gcn:
            gcn.side_effect = [("2024-05-01", self.entry)] + \
                              [("2024-05-%02d" % d, self.later) for d in range(2, 28)]
            t = dv.simulate_delta_hedged_straddle("AAPL", "2024-05-01", spots,
                                                  sdates, db_path="x", target_days=21)
        self.assertIsNotNone(t)
        # hedge P&L must stay bounded (no ~$1000s artifact from the 90% drop)
        self.assertLess(abs(t["hedge_pnl"]), 500.0)

    def test_none_when_no_straddle(self):
        with mock.patch("src.dolt_options.get_chain_near",
                        return_value=("2024-05-01", [_c("call", 100, 3, 3.2, 0.5)])):
            t = dv.simulate_delta_hedged_straddle("AAPL", "2024-05-01", self.spots,
                                                  self.sdates, db_path="x")
        self.assertIsNone(t)


class SummarizeTest(unittest.TestCase):
    def test_summary_keys(self):
        trades = [{"ret": 0.2, "hedge_pnl": 5.0, "straddle_pnl": 10.0, "net_pnl": 15.0},
                  {"ret": -0.1, "hedge_pnl": -3.0, "straddle_pnl": -2.0, "net_pnl": -5.0}]
        out = dv._summarize(trades, partial=False)
        self.assertEqual(out["n"], 2)
        self.assertIn("mean_hedge_pnl", out)
        self.assertIn("mean_straddle_pnl", out)


if __name__ == "__main__":
    unittest.main()
