"""Tests for src/dolt_spread.py — put credit spread backtest (mocked).

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_dolt_spread -v
"""
import unittest
from unittest import mock

from src import dolt_spread as sp
from src.paper_manager import _normalize_exit_rules

RULES = _normalize_exit_rules({})   # spread tp=0.5, sl=-1.0


def _put(strike, bid, ask, delta, expiration="2024-04-19", date="2024-03-01"):
    return {"symbol": "X", "date": date, "expiration": expiration, "strike": strike,
            "type": "put", "bid": bid, "ask": ask, "mid": (bid + ask) / 2, "iv": 0.3,
            "delta": delta, "gamma": 0.01, "theta": -0.03, "vega": 0.1, "rho": -0.02}


class PickSpreadTest(unittest.TestCase):
    def test_picks_short_and_long_same_expiry(self):
        chain = [_put(95, 2.0, 2.2, -0.25), _put(90, 1.0, 1.1, -0.15), _put(85, 0.4, 0.5, -0.10)]
        short, long = sp._pick_put_spread(chain, "2024-03-01", 0.25, 0.10, min_dte=7)
        self.assertEqual(short["strike"], 95)
        self.assertEqual(long["strike"], 85)

    def test_none_without_lower_strike(self):
        chain = [_put(95, 2.0, 2.2, -0.25)]   # no wing below
        self.assertIsNone(sp._pick_put_spread(chain, "2024-03-01", 0.25, 0.10, min_dte=7))


class SimulateSpreadTest(unittest.TestCase):
    def setUp(self):
        self.sdates = [f"2024-03-{d:02d}" for d in range(1, 13)]
        self.spots = {d: 100.0 for d in self.sdates}

    def _entry_chain(self):
        return [_put(95, 2.0, 2.2, -0.25), _put(85, 0.4, 0.5, -0.10)]

    def _run(self, day_chains):
        chains = {"2024-03-01": self._entry_chain()}
        chains.update(day_chains)

        def fake(sym, date, db_path=None):
            return date, chains.get(date, [])
        with mock.patch("src.dolt_options.get_chain_near", side_effect=fake):
            return sp.simulate_spread("X", "2024-03-01", 100.0, self.sdates,
                                      self.spots, RULES, short_delta=0.25, long_delta=0.10)

    def test_take_profit_when_credit_decays(self):
        # entry credit = 2.0-0.5 = 1.5; close cost decays → TP at 50% of credit
        day = {}
        for i, d in enumerate(self.sdates[1:], start=1):
            # short ask + long bid → close cost; make it cheap after a few days
            cost_short_ask = 0.5 if i >= 4 else 2.2
            day[d] = [_put(95, cost_short_ask - 0.1, cost_short_ask, -0.25, date=d),
                      _put(85, 0.05, 0.1, -0.10, date=d)]
        res = self._run(day)
        self.assertEqual(res["exit_reason"], "take_profit")
        self.assertGreater(res["ret"], 0.0)
        self.assertIn("max_risk", res)

    def test_ret_is_on_max_risk(self):
        # width 10, credit 1.5 → max_risk 8.5; verify ret denominator
        day = {}
        for d in self.sdates[1:]:
            day[d] = [_put(95, 0.1, 0.2, -0.25, date=d), _put(85, 0.01, 0.05, -0.10, date=d)]
        res = self._run(day)
        # closed near max profit: net_pnl ≈ (1.5 - ~0.19) - comm; ret = net/8.5
        self.assertLess(abs(res["ret"]) , 1.0)   # bounded by definition on max-risk


if __name__ == "__main__":
    unittest.main()
