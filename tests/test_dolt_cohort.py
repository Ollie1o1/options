"""Tests for src/dolt_cohort.py — long-call cohort backtest on real marks.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_dolt_cohort -v
"""
import unittest
from unittest import mock

from src import dolt_cohort as dc


# Normalized exit-rules shape (matches paper_manager._normalize_exit_rules output).
RULES = {"time_exit_dte": 21, "min_days_held": 3,
         "long": {"tp": 1.0, "tp_delta": 0.80, "sl": -0.5}}


def _contract(bid, ask, strike=103.0, expiration="2024-04-19", date="2024-03-01"):
    return {"symbol": "X", "date": date, "expiration": expiration, "strike": strike,
            "type": "call", "bid": bid, "ask": ask, "mid": (bid + ask) / 2, "iv": 0.3,
            "delta": 0.32, "gamma": 0.01, "theta": -0.03, "vega": 0.1, "rho": 0.02}


class SimulateTradeTest(unittest.TestCase):
    def setUp(self):
        # 12 consecutive trading dates
        self.sdates = [f"2024-03-{d:02d}" for d in range(1, 13)]

    def _run(self, chain_by_date):
        def fake_near(sym, date, db_path=None):
            return date, chain_by_date.get(date, [])
        with mock.patch("src.dolt_options.get_chain_near", side_effect=fake_near):
            return dc.simulate_trade("X", "2024-03-01", spot=100.0,
                                     sdates=self.sdates, rules=RULES, target_dte=35)

    def test_take_profit_after_min_hold(self):
        # entry ask 2.0; bid doubles by day 4 (held>=3) → TP at +100%+
        chains = {"2024-03-01": [_contract(1.8, 2.0)]}
        for i, d in enumerate(self.sdates[1:], start=1):
            bid = 4.5 if i >= 4 else 2.0
            chains[d] = [_contract(bid, bid + 0.2, date=d)]
        res = self._run(chains)
        self.assertEqual(res["exit_reason"], "take_profit")
        self.assertGreaterEqual(res["ret"], 1.0)
        self.assertGreaterEqual(res["days_held"], 3)

    def test_stop_loss(self):
        chains = {"2024-03-01": [_contract(1.8, 2.0)]}
        for i, d in enumerate(self.sdates[1:], start=1):
            bid = 0.8 if i >= 4 else 1.9   # drops to -60% after min hold
            chains[d] = [_contract(bid, bid + 0.2, date=d)]
        res = self._run(chains)
        self.assertEqual(res["exit_reason"], "stop_loss")
        self.assertLessEqual(res["ret"], -0.5)

    def test_time_exit_when_dte_hits_floor(self):
        # entry needs DTE >= time_exit+7 (28). Use a 29-DTE contract (exp 2024-03-30)
        # that crosses the 21-DTE floor on 2024-03-09; bids stay flat (no TP/SL).
        chains = {"2024-03-01": [_contract(1.8, 2.0, expiration="2024-03-30")]}
        for d in self.sdates[1:]:
            chains[d] = [_contract(2.1, 2.3, expiration="2024-03-30", date=d)]
        res = self._run(chains)
        self.assertEqual(res["exit_reason"], "time_exit")

    def test_none_when_unfillable(self):
        with mock.patch("src.dolt_options.get_chain_near", return_value=("2024-03-01", [])):
            res = dc.simulate_trade("X", "2024-03-01", 100.0, self.sdates, RULES)
        self.assertIsNone(res)


class SummarizeTest(unittest.TestCase):
    def test_summary_stats(self):
        trades = [{"ret": 1.2, "exit_reason": "take_profit", "days_held": 5},
                  {"ret": -0.5, "exit_reason": "stop_loss", "days_held": 4},
                  {"ret": 0.1, "exit_reason": "time_exit", "days_held": 10}]
        out = dc._summarize(trades, RULES, partial=False)
        self.assertEqual(out["n"], 3)
        self.assertAlmostEqual(out["win_rate"], 2 / 3, places=2)
        self.assertEqual(out["exit_mix"]["take_profit"], 1)
        self.assertAlmostEqual(out["profit_factor"], 1.3 / 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
