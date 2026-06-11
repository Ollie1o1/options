"""Tests for src/chain_archive.py — daily option-chain snapshots (offline)."""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from src import chain_archive as ca


def _row(symbol="SPY", type_="call", strike=600.0, expiration="2026-07-17",
         dte_spot=600.0, iv=0.2, oi=100):
    return {
        "symbol": symbol, "contract": f"{symbol}X{strike}", "type": type_,
        "strike": strike, "expiration": expiration, "bid": 1.0, "ask": 1.1,
        "bid_size": 5, "ask_size": 5, "iv": iv, "delta": 0.3, "gamma": 0.01,
        "theta": -0.02, "vega": 0.1, "rho": 0.05, "open_interest": oi,
        "volume": 10, "last_trade_time": None, "spot": dte_spot,
        "snapshot_ts": "2026-06-10 14:05:00", "source": "cboe",
    }


class FilterTest(unittest.TestCase):
    def test_keeps_sane_contracts(self):
        rows = [_row(strike=600.0)]
        kept = ca.filter_rows(rows, today="2026-06-10", max_dte=120,
                              moneyness_band=0.15)
        self.assertEqual(len(kept), 1)

    def test_drops_far_dte(self):
        rows = [_row(expiration="2027-12-17")]
        self.assertEqual(ca.filter_rows(rows, today="2026-06-10",
                                        max_dte=120, moneyness_band=0.15), [])

    def test_drops_extreme_moneyness(self):
        rows = [_row(strike=750.0, dte_spot=600.0)]  # 25% OTM
        self.assertEqual(ca.filter_rows(rows, today="2026-06-10",
                                        max_dte=120, moneyness_band=0.15), [])

    def test_drops_expired(self):
        rows = [_row(expiration="2026-06-09")]
        self.assertEqual(ca.filter_rows(rows, today="2026-06-10",
                                        max_dte=120, moneyness_band=0.15), [])

    def test_drops_zero_open_interest(self):
        rows = [_row(oi=0)]
        self.assertEqual(ca.filter_rows(rows, today="2026-06-10",
                                        max_dte=120, moneyness_band=0.15), [])


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "chains.db")

    def test_archive_writes_rows(self):
        n = ca.archive_symbols(["SPY"], db_path=self.db, today="2026-06-10",
                               fetcher=lambda s: [_row(symbol=s)])
        self.assertEqual(n, 1)
        with sqlite3.connect(self.db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM chain_snapshots").fetchone()[0]
            self.assertEqual(count, 1)

    def test_archive_is_idempotent_per_day(self):
        kw = dict(db_path=self.db, today="2026-06-10",
                  fetcher=lambda s: [_row(symbol=s)])
        ca.archive_symbols(["SPY"], **kw)
        n2 = ca.archive_symbols(["SPY"], **kw)   # same day: skipped
        self.assertEqual(n2, 0)
        with sqlite3.connect(self.db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM chain_snapshots").fetchone()[0]
            self.assertEqual(count, 1)

    def test_fetcher_failure_is_isolated(self):
        def boom(sym):
            if sym == "BAD":
                raise OSError("down")
            return [_row(symbol=sym)]
        n = ca.archive_symbols(["BAD", "SPY"], db_path=self.db,
                               today="2026-06-10", fetcher=boom)
        self.assertEqual(n, 1)  # SPY still archived


class ThrottleTest(unittest.TestCase):
    def test_due_on_weekday_afternoon(self):
        self.assertTrue(ca.due_chain_archive({}, "2026-06-10", weekday=3, hhmm=1405))

    def test_not_due_twice_same_day(self):
        st = {"last_chain_archive": "2026-06-10"}
        self.assertFalse(ca.due_chain_archive(st, "2026-06-10", weekday=3, hhmm=1500))

    def test_not_due_morning_or_weekend(self):
        self.assertFalse(ca.due_chain_archive({}, "2026-06-10", weekday=3, hhmm=1000))
        self.assertFalse(ca.due_chain_archive({}, "2026-06-13", weekday=6, hhmm=1405))


if __name__ == "__main__":
    unittest.main()
