"""Tests for src/dolt_earnings_sell.py — earnings IV-crush selling backtest.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_dolt_earnings_sell -v
"""
import unittest
from unittest import mock

from src import dolt_earnings_sell as es


def _c(opt_type, strike, bid, ask, delta, iv=0.5, expiration="2024-05-17"):
    return {"type": opt_type, "strike": strike, "bid": bid, "ask": ask,
            "delta": delta, "iv": iv, "expiration": expiration}


class PickStrangleTest(unittest.TestCase):
    def test_picks_otm_call_and_put(self):
        spot = 100.0
        chain = [_c("call", 105, 1.0, 1.1, 0.20), _c("call", 110, 0.4, 0.5, 0.10),
                 _c("put", 95, 1.0, 1.1, -0.20), _c("put", 90, 0.4, 0.5, -0.10)]
        sc, sp = es._pick_strangle(chain, "2024-05-01", spot, 0.20, 0.20, 2, 45)
        self.assertEqual(sc["strike"], 105)
        self.assertEqual(sp["strike"], 95)

    def test_requires_both_sides(self):
        chain = [_c("call", 105, 1.0, 1.1, 0.20)]   # no puts
        self.assertIsNone(es._pick_strangle(chain, "2024-05-01", 100.0, 0.20, 0.20, 2, 45))


class SimulateTest(unittest.TestCase):
    def setUp(self):
        self.entry_chain = [_c("call", 105, 2.0, 2.1, 0.20), _c("put", 95, 2.0, 2.1, -0.20)]
        # post-earnings: IV crushed, options much cheaper to buy back
        self.exit_chain = [_c("call", 105, 0.3, 0.4, 0.10, iv=0.25),
                           _c("put", 95, 0.3, 0.4, -0.10, iv=0.25)]

    def test_profitable_crush(self):
        with mock.patch("src.dolt_validate._spot_history",
                        return_value={"2024-05-01": 100.0, "2024-05-06": 101.0}), \
             mock.patch("src.dolt_options.get_chain_near") as gcn:
            # first call = entry (direction -1), second = exit (direction +1)
            gcn.side_effect = [("2024-05-01", self.entry_chain),
                               ("2024-05-06", self.exit_chain)]
            t = es.simulate_earnings_strangle("AAPL", "2024-05-03", db_path="x")
        self.assertIsNotNone(t)
        # credit 4.0 collected, bought back for 0.8 -> net positive
        self.assertGreater(t["net_pnl"], 0)
        self.assertAlmostEqual(t["credit"], 4.0, places=4)
        self.assertGreater(t["ret"], 0)

    def test_none_when_entry_after_earnings(self):
        with mock.patch("src.dolt_validate._spot_history", return_value={"2024-05-04": 100.0}), \
             mock.patch("src.dolt_options.get_chain_near",
                        return_value=("2024-05-04", self.entry_chain)):
            # entry date 05-04 is AFTER earnings 05-03 -> reject
            t = es.simulate_earnings_strangle("AAPL", "2024-05-03", db_path="x")
        self.assertIsNone(t)


class SummarizeTest(unittest.TestCase):
    def test_summary_keys(self):
        trades = [{"ret": 0.5, "credit": 4.0, "net_pnl": 2.0, "realized_move": 0.03},
                  {"ret": -0.2, "credit": 3.0, "net_pnl": -0.6, "realized_move": -0.08}]
        out = es._summarize(trades, partial=False)
        self.assertEqual(out["n"], 2)
        self.assertIn("mean_credit", out)
        self.assertIn("mean_abs_move", out)


if __name__ == "__main__":
    unittest.main()
