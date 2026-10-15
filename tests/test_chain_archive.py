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


def _ledger(path, rows):
    """A throwaway ledger with just the columns `archive_universe` reads."""
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE trades (ticker TEXT, date TEXT, "
                     "status TEXT, exit_date TEXT)")
        conn.executemany("INSERT INTO trades VALUES (?,?,?,?)", rows)


class UniverseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ledger = os.path.join(self.tmp, "ledger.db")

    def test_adds_open_recent_and_recently_exited_tickers(self):
        _ledger(self.ledger, [
            ("ABT", "2026-10-13 10:00:00", "CLOSED", "2026-10-15 10:34:20"),
            ("unh", "2026-10-01 10:00:00", "CLOSED", "2026-10-03 10:00:00"),
            ("MU", "2026-06-01 10:00:00", "OPEN", None),
            ("OLD", "2026-06-01 10:00:00", "CLOSED", "2026-06-20 10:00:00"),
            ("SPY", "2026-10-14 10:00:00", "OPEN", None),
        ])
        out = ca.archive_universe(["SPY", "qqq"], self.ledger,
                                  lookback_days=30, today="2026-10-15")
        # Configured symbols first and in order; no duplicates; case folded.
        self.assertEqual(out, ["SPY", "QQQ", "ABT", "MU", "UNH"])

    def test_skips_crypto_pairs(self):
        _ledger(self.ledger, [("BTC-USD", "2026-10-14", "OPEN", None)])
        self.assertEqual(ca.archive_universe(["SPY"], self.ledger,
                                             today="2026-10-15"), ["SPY"])

    def test_zero_lookback_or_no_ledger_is_the_configured_list(self):
        _ledger(self.ledger, [("ABT", "2026-10-14", "OPEN", None)])
        self.assertEqual(ca.archive_universe(["SPY"], self.ledger,
                                             lookback_days=0), ["SPY"])
        self.assertEqual(ca.archive_universe(
            ["SPY"], os.path.join(self.tmp, "missing.db")), ["SPY"])

    def test_a_broken_ledger_degrades_to_the_configured_list(self):
        with sqlite3.connect(self.ledger) as conn:
            conn.execute("CREATE TABLE other (x)")
        self.assertEqual(ca.archive_universe(["SPY"], self.ledger,
                                             today="2026-10-15"), ["SPY"])


class GapHealthTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "chains.db")

    def _snap(self, *days):
        for d in days:
            ca.archive_symbols(["SPY"], db_path=self.db, today=d,
                               fetcher=lambda s, d=d: [_row(
                                   symbol=s, expiration="2026-11-20")])

    def test_a_missed_trading_day_warns_and_names_it(self):
        # 2026-10-06 and 10-09 are the real lost days.
        self._snap("2026-10-02", "2026-10-05", "2026-10-07", "2026-10-08",
                   "2026-10-12")
        text = " ".join(ca.health_lines(self.db, days=14, today="2026-10-13"))
        self.assertIn("[WARN]", text)
        self.assertIn("2026-10-06", text)
        self.assertIn("2026-10-09", text)

    def test_an_unbroken_run_is_ok_and_today_is_not_a_gap(self):
        # 10-13 (today) has no snapshot yet: the run is due in the afternoon.
        self._snap("2026-10-08", "2026-10-09", "2026-10-12")
        lines = ca.health_lines(self.db, days=14, today="2026-10-13")
        self.assertEqual(len(lines), 1)
        self.assertIn("[OK]", lines[0])

    def test_no_archive_prints_nothing(self):
        self.assertEqual(ca.health_lines(
            os.path.join(self.tmp, "none.db"), today="2026-10-13"), [])


class ThrottleTest(unittest.TestCase):
    def test_due_on_weekday_afternoon(self):
        self.assertTrue(ca.due_chain_archive({}, "2026-06-10", weekday=3, hhmm=1405))

    def test_not_due_twice_same_day(self):
        st = {"last_chain_archive": "2026-06-10"}
        self.assertFalse(ca.due_chain_archive(st, "2026-06-10", weekday=3, hhmm=1500))

    def test_not_due_morning_or_weekend(self):
        self.assertFalse(ca.due_chain_archive({}, "2026-06-10", weekday=3, hhmm=1000))
        self.assertFalse(ca.due_chain_archive({}, "2026-06-13", weekday=6, hhmm=1405))

    def test_not_due_on_a_market_holiday(self):
        # Labor Day 2026 was a Monday: the archive ran and stored closed-market
        # quotes as a session (528 of 13,549 contracts matched Friday's).
        self.assertFalse(ca.due_chain_archive({}, "2026-09-07", weekday=1, hhmm=1405))


if __name__ == "__main__":
    unittest.main()
