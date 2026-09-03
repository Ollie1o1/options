"""tests/test_per_leg_fill_entry.py

log_spread/log_iron_condor forward per-leg entry quotes into the new v23
columns; absent keys (a caller that predates this feature, or a genuine
missing quote) stay NULL, never zero.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from src import paper_manager as pm


class TestSpreadEntryLegRecording(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "t.db")
        self.pm = pm.PaperManager(db_path=self.db)

    def test_bull_put_records_short_and_long_entry_quotes(self):
        self.pm.log_spread({
            "ticker": "AAPL", "type": "Bull Put", "expiration": "2026-10-15",
            "short_strike": 150.0, "long_strike": 145.0, "net_credit": 1.20,
            "max_profit": 120.0, "max_loss": 380.0,
            "short_bid": 1.45, "short_ask": 1.55,
            "long_bid": 0.20, "long_ask": 0.30,
        })
        conn = sqlite3.connect(self.db)
        row = conn.execute(
            "SELECT short_bid_entry, short_ask_entry, long_bid_entry, "
            "long_ask_entry FROM trades WHERE ticker='AAPL'"
        ).fetchone()
        conn.close()
        self.assertEqual(tuple(row), (1.45, 1.55, 0.20, 0.30))

    def test_missing_leg_quotes_stay_null_not_zero(self):
        self.pm.log_spread({
            "ticker": "MSFT", "type": "Bear Call", "expiration": "2026-10-15",
            "short_strike": 400.0, "long_strike": 410.0, "net_credit": 1.00,
            "max_profit": 100.0, "max_loss": 900.0,
        })
        conn = sqlite3.connect(self.db)
        row = conn.execute(
            "SELECT short_bid_entry, short_ask_entry FROM trades "
            "WHERE ticker='MSFT'"
        ).fetchone()
        conn.close()
        self.assertEqual(tuple(row), (None, None))


class TestCondorEntryLegRecording(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "t.db")
        self.pm = pm.PaperManager(db_path=self.db)

    def test_iron_condor_records_all_four_legs(self):
        self.pm.log_iron_condor({
            "ticker": "SPY", "expiration": "2026-10-15",
            "short_put_strike": 440.0, "long_put_strike": 435.0,
            "short_call_strike": 460.0, "long_call_strike": 465.0,
            "total_credit": 2.00, "max_risk": 600.0, "max_profit": 200.0,
            "short_put_bid": 0.78, "short_put_ask": 0.82,
            "long_put_bid": 0.18, "long_put_ask": 0.22,
            "short_call_bid": 0.73, "short_call_ask": 0.77,
            "long_call_bid": 0.13, "long_call_ask": 0.17,
        })
        conn = sqlite3.connect(self.db)
        row = conn.execute(
            "SELECT short_put_bid_entry, short_put_ask_entry, "
            "long_put_bid_entry, long_put_ask_entry, "
            "short_call_bid_entry, short_call_ask_entry, "
            "long_call_bid_entry, long_call_ask_entry FROM trades "
            "WHERE ticker='SPY'"
        ).fetchone()
        conn.close()
        self.assertEqual(tuple(row),
                         (0.78, 0.82, 0.18, 0.22, 0.73, 0.77, 0.13, 0.17))

    def test_condor_spread_columns_stay_null(self):
        """A condor never populates the 2-leg short/long columns — those are
        for Bull Put/Bear Call only."""
        self.pm.log_iron_condor({
            "ticker": "QQQ", "expiration": "2026-10-15",
            "short_put_strike": 440.0, "long_put_strike": 435.0,
            "short_call_strike": 460.0, "long_call_strike": 465.0,
            "total_credit": 2.00, "max_risk": 600.0, "max_profit": 200.0,
            "short_put_bid": 0.78, "short_put_ask": 0.82,
            "long_put_bid": 0.18, "long_put_ask": 0.22,
            "short_call_bid": 0.73, "short_call_ask": 0.77,
            "long_call_bid": 0.13, "long_call_ask": 0.17,
        })
        conn = sqlite3.connect(self.db)
        row = conn.execute(
            "SELECT short_bid_entry, long_bid_entry FROM trades "
            "WHERE ticker='QQQ'"
        ).fetchone()
        conn.close()
        self.assertEqual(tuple(row), (None, None))


if __name__ == "__main__":
    unittest.main()
