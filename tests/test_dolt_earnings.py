"""Tests for src/dolt_earnings.py — earnings dates + IV-crush (DoltHub mocked).

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_dolt_earnings -v
"""
import os
import tempfile
import unittest
from unittest import mock

from src import dolt_earnings as de


class EarningsDatesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "c.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _rows(self):
        return [{"act_symbol": "AAPL", "date": "2023-02-02", "when": "After market close"},
                {"act_symbol": "AAPL", "date": "2023-05-04", "when": "After market close"},
                {"act_symbol": "AAPL", "date": "2023-08-03", "when": "After market close"}]

    def test_dates_fetch_then_cache(self):
        with mock.patch("src.dolt_earnings._fetch_live", return_value=self._rows()) as f:
            d1 = de.earnings_dates("AAPL", db_path=self.db)
            d2 = de.earnings_dates("AAPL", db_path=self.db)
        self.assertEqual(f.call_count, 1, "second call must hit cache")
        self.assertEqual(d1, ["2023-02-02", "2023-05-04", "2023-08-03"])
        self.assertEqual(d2, d1)

    def test_in_window(self):
        with mock.patch("src.dolt_earnings._fetch_live", return_value=self._rows()):
            w = de.earnings_in_window("AAPL", "2023-03-01", "2023-06-01", db_path=self.db)
        self.assertEqual(w, ["2023-05-04"])

    def test_holds_through_earnings(self):
        with mock.patch("src.dolt_earnings._fetch_live", return_value=self._rows()):
            self.assertTrue(de.holds_through_earnings("AAPL", "2023-04-25", "2023-05-16", db_path=self.db))
            self.assertFalse(de.holds_through_earnings("AAPL", "2023-05-10", "2023-05-30", db_path=self.db))
            # boundary: earnings exactly on entry_date is NOT "through" (strict >)
            self.assertFalse(de.holds_through_earnings("AAPL", "2023-05-04", "2023-05-20", db_path=self.db))


class AtmIvTest(unittest.TestCase):
    def test_atm_iv_picks_nearest_strike(self):
        chain = [{"type": "call", "strike": 95.0, "iv": 0.40},
                 {"type": "call", "strike": 100.0, "iv": 0.30},
                 {"type": "call", "strike": 110.0, "iv": 0.35}]
        self.assertEqual(de._atm_iv(chain, 101.0), 0.30)

    def test_atm_iv_none_when_empty(self):
        self.assertIsNone(de._atm_iv([], 100.0))


if __name__ == "__main__":
    unittest.main()
