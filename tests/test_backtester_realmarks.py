"""Tests for the DoltHub real-marks fill helper in src/backtester.py.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_backtester_realmarks -v
"""
import unittest
from unittest import mock

from src import backtester as bt


class RealMarksFillTest(unittest.TestCase):
    def _chain(self, date):
        return [
            {"symbol": "AAPL", "date": date, "expiration": "2024-04-19", "strike": 180.0,
             "type": "call", "bid": 3.0, "ask": 3.4, "mid": 3.2, "iv": 0.28,
             "delta": 0.3, "gamma": 0.01, "theta": -0.03, "vega": 0.1, "rho": 0.02},
        ]

    def test_real_marks_fill_uses_bid_entry_ask_exit(self):
        def fake_get_chain(sym, date, db_path=None):
            return self._chain(date)
        with mock.patch("src.dolt_options.get_chain", side_effect=fake_get_chain):
            fill = bt._real_marks_fill("AAPL", "call", target_strike=180.0,
                                       entry_date="2024-03-15", exit_date="2024-04-05",
                                       target_dte=30)
        self.assertIsNotNone(fill)
        self.assertEqual(fill["entry"], 3.0)   # sell-to-open at bid
        self.assertEqual(fill["exit"], 3.4)    # buy-to-close at ask
        self.assertEqual(fill["entry_iv"], 0.28)

    def test_real_marks_fill_none_when_missing(self):
        with mock.patch("src.dolt_options.get_chain", return_value=[]):
            self.assertIsNone(bt._real_marks_fill("AAPL", "call", 180.0,
                              "2024-03-15", "2024-04-05", 30))


if __name__ == "__main__":
    unittest.main()
