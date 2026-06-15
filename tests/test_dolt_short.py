"""Tests for src/dolt_short.py — short-premium backtest on real marks (mocked).

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_dolt_short -v
"""
import unittest
from unittest import mock

from src import dolt_short as sh
from src.paper_manager import _normalize_exit_rules

RULES = _normalize_exit_rules({})   # canonical defaults (tp_ge_21=0.5, sl_prem_mult=2.0, ...)


def _put(bid, ask, strike=90.0, expiration="2024-04-19", delta=-0.25, date="2024-03-01"):
    return {"symbol": "X", "date": date, "expiration": expiration, "strike": strike,
            "type": "put", "bid": bid, "ask": ask, "mid": (bid + ask) / 2, "iv": 0.3,
            "delta": delta, "gamma": 0.01, "theta": -0.03, "vega": 0.1, "rho": -0.02}


class PickShortTest(unittest.TestCase):
    def test_picks_nearest_target_delta(self):
        chain = [_put(1.0, 1.2, strike=85, delta=-0.15),
                 _put(2.0, 2.2, strike=90, delta=-0.25),
                 _put(3.0, 3.3, strike=95, delta=-0.40)]
        c = sh._pick_short(chain, "put", 0.25, "2024-03-01", min_dte=7)
        self.assertEqual(c["strike"], 90)

    def test_respects_min_dte(self):
        chain = [_put(2.0, 2.2, expiration="2024-03-05", delta=-0.25)]  # ~4 DTE
        self.assertIsNone(sh._pick_short(chain, "put", 0.25, "2024-03-01", min_dte=28))


class SimulateShortTest(unittest.TestCase):
    def setUp(self):
        self.sdates = [f"2024-03-{d:02d}" for d in range(1, 13)]
        self.spots = {d: 100.0 for d in self.sdates}   # flat spot, no strike breach

    def _run(self, chains, opt_type="put"):
        def fake(sym, date, db_path=None):
            return date, chains.get(date, [])
        with mock.patch("src.dolt_options.get_chain_near", side_effect=fake):
            return sh.simulate_short_trade("X", "2024-03-01", 100.0, self.sdates,
                                           self.spots, RULES, opt_type=opt_type, target_delta=0.25)

    def test_take_profit_when_premium_decays(self):
        # sell put at bid 2.0; ask decays so buy-back is cheap → TP (>=50%)
        chains = {"2024-03-01": [_put(2.0, 2.2)]}
        for i, d in enumerate(self.sdates[1:], start=1):
            ask = 0.8 if i >= 4 else 2.0
            chains[d] = [_put(ask - 0.1, ask, date=d)]
        res = self._run(chains)
        self.assertEqual(res["exit_reason"], "take_profit")
        self.assertGreater(res["gross_ret"], 0.0)

    def test_stop_loss_on_premium_blowout(self):
        # ask balloons → premium-multiple stop (pnl_raw <= -(2-1) = -1.0)
        chains = {"2024-03-01": [_put(2.0, 2.2)]}
        for d in self.sdates[1:]:
            chains[d] = [_put(4.0, 4.5, date=d)]   # buy back at 4.5 vs sold 2.0 → -125%
        res = self._run(chains)
        self.assertEqual(res["exit_reason"], "stop_loss")
        self.assertLess(res["gross_ret"], 0.0)

    def test_none_when_unfillable(self):
        with mock.patch("src.dolt_options.get_chain_near", return_value=("2024-03-01", [])):
            self.assertIsNone(sh.simulate_short_trade("X", "2024-03-01", 100.0,
                              self.sdates, self.spots, RULES))


class SummarizeTest(unittest.TestCase):
    def test_summary(self):
        trades = [{"ret": 0.4, "exit_reason": "take_profit"},
                  {"ret": -1.0, "exit_reason": "stop_loss"},
                  {"ret": 0.2, "exit_reason": "time_exit"}]
        out = sh._summarize(trades, partial=False)
        self.assertEqual(out["n"], 3)
        self.assertEqual(out["exit_mix"]["take_profit"], 1)


if __name__ == "__main__":
    unittest.main()
