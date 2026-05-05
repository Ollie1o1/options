"""Unit tests for src.crypto.auto_logger.

Run directly:  venv/bin/python tests/crypto/test_auto_logger.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.crypto import auto_logger


def _fresh_crypto_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE trades (
            entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            ticker TEXT,
            status TEXT,
            weight_profile TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _insert(path: Path, **cols) -> None:
    keys = ",".join(cols.keys())
    placeholders = ",".join("?" for _ in cols)
    conn = sqlite3.connect(path)
    conn.execute(f"INSERT INTO trades ({keys}) VALUES ({placeholders})", tuple(cols.values()))
    conn.commit()
    conn.close()


class TestCountOpenPositions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "x.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_zero_for_empty_db(self):
        _fresh_crypto_db(self.db)
        self.assertEqual(auto_logger.count_open_positions(str(self.db), "BTC"), 0)

    def test_counts_only_matching_currency_and_open(self):
        _fresh_crypto_db(self.db)
        _insert(self.db, date="2026-05-04 12:00:00", ticker="BTC", status="OPEN",   weight_profile="crypto_baseline")
        _insert(self.db, date="2026-05-04 12:00:00", ticker="BTC", status="CLOSED", weight_profile="crypto_baseline")
        _insert(self.db, date="2026-05-04 12:00:00", ticker="ETH", status="OPEN",   weight_profile="crypto_baseline")
        self.assertEqual(auto_logger.count_open_positions(str(self.db), "BTC"), 1)
        self.assertEqual(auto_logger.count_open_positions(str(self.db), "ETH"), 1)


class TestCountTodayAutoLogs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "x.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_filters_by_weight_profile_and_date(self):
        _fresh_crypto_db(self.db)
        _insert(self.db, date="2026-05-04 12:00:00", ticker="BTC", status="OPEN", weight_profile="crypto_auto_v1")
        _insert(self.db, date="2026-05-04 12:00:00", ticker="BTC", status="OPEN", weight_profile="crypto_baseline")
        _insert(self.db, date="2026-05-03 23:59:59", ticker="BTC", status="OPEN", weight_profile="crypto_auto_v1")
        _insert(self.db, date="2026-05-04 18:00:00", ticker="ETH", status="OPEN", weight_profile="crypto_auto_v1")
        self.assertEqual(auto_logger.count_today_auto_logs(str(self.db), "BTC", today_utc="2026-05-04"), 1)
        self.assertEqual(auto_logger.count_today_auto_logs(str(self.db), "ETH", today_utc="2026-05-04"), 1)


class TestPickWinner(unittest.TestCase):
    def test_returns_highest_score_across_buckets(self):
        # _scan_currency hands these pre-sorted descending; mimic that.
        long_df = pd.DataFrame([
            {"strategy_score": 0.61, "type": "put", "expiration": "2026-06-15"},
            {"strategy_score": 0.42, "type": "put", "expiration": "2026-06-15"},
        ])
        spread_df = pd.DataFrame([{"score": 0.55, "expiration": "2026-06-15"}])
        calendar_df = pd.DataFrame([
            {"score": 0.49, "front_expiration": "2026-05-15", "back_expiration": "2026-06-15"},
        ])
        picks = {
            "Long Put":      long_df,
            "Bear Call":     spread_df,
            "Calendar Put":  calendar_df,
        }
        winner_strategy, winner_row, winner_score = auto_logger.pick_winner(picks)
        self.assertEqual(winner_strategy, "Long Put")
        self.assertAlmostEqual(winner_score, 0.61)
        self.assertEqual(winner_row["expiration"], "2026-06-15")

    def test_returns_none_when_buckets_empty(self):
        self.assertIsNone(auto_logger.pick_winner({}))

    def test_returns_none_when_all_buckets_empty_dfs(self):
        picks = {"Long Put": pd.DataFrame(), "Bear Call": pd.DataFrame()}
        self.assertIsNone(auto_logger.pick_winner(picks))


class TestScoreColumn(unittest.TestCase):
    def test_handles_long_premium_and_others(self):
        self.assertEqual(auto_logger._score_column("Long Put"), "strategy_score")
        self.assertEqual(auto_logger._score_column("Long Call"), "strategy_score")
        self.assertEqual(auto_logger._score_column("Bear Call"), "score")
        self.assertEqual(auto_logger._score_column("Calendar Put"), "score")
        self.assertEqual(auto_logger._score_column("Iron Condor"), "score")


if __name__ == "__main__":
    unittest.main(verbosity=2)
