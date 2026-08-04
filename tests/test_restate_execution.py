"""Backfill of the v18 execution columns over the existing ledger.

The ledger is the record of what happened. This backfill adds a second reading
of it and must never alter the first: `entry_price`, `net_credit` and `pnl_usd`
are read-only to it, and every write is reversible.
"""
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.restate_execution import backfill, report

TRADES_DDL = (
    "CREATE TABLE trades ("
    "entry_id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, ticker TEXT,"
    " expiration TEXT, strike REAL, type TEXT, entry_price REAL,"
    " strategy_name TEXT, status TEXT, pnl_usd REAL, long_strike REAL,"
    " spread_width REAL, net_credit REAL, short_call_strike REAL,"
    " long_call_strike REAL, short_put_strike REAL, long_put_strike REAL,"
    " duplicate_of INTEGER,"
    " entry_price_mid REAL, entry_price_fill REAL, entry_price_cross REAL,"
    " fill_policy TEXT, fill_source TEXT)"
)

ARCHIVE_DDL = (
    "CREATE TABLE chain_snapshots ("
    " symbol TEXT, snap_date TEXT, expiration TEXT, strike REAL, type TEXT,"
    " bid REAL, ask REAL)"
)


class BackfillTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "trades.db")
        self.archive = os.path.join(self.tmp.name, "archive.db")

        conn = sqlite3.connect(self.db)
        conn.execute(TRADES_DDL)
        # One Bull Put the archive can price, one it cannot.
        conn.execute(
            "INSERT INTO trades (date, ticker, expiration, strike, type, entry_price,"
            " strategy_name, status, pnl_usd, long_strike, spread_width, net_credit)"
            " VALUES ('2026-06-10','ORCL','2026-07-17',80.0,'put',1.00,"
            "'Bull Put','CLOSED',44.0,79.0,1.0,1.00)")
        conn.execute(
            "INSERT INTO trades (date, ticker, expiration, strike, type, entry_price,"
            " strategy_name, status, pnl_usd, long_strike, spread_width, net_credit)"
            " VALUES ('2026-06-10','ZZZZ','2026-07-17',50.0,'put',0.40,"
            "'Bull Put','CLOSED',-10.0,49.0,1.0,0.40)")
        conn.commit()
        conn.close()

        a = sqlite3.connect(self.archive)
        a.execute(ARCHIVE_DDL)
        a.executemany(
            "INSERT INTO chain_snapshots (symbol, snap_date, expiration, strike, type,"
            " bid, ask) VALUES (?,?,?,?,?,?,?)",
            [("ORCL", "2026-06-10", "2026-07-17", 80.0, "put", 1.40, 1.60),
             ("ORCL", "2026-06-10", "2026-07-17", 79.0, "put", 0.40, 0.60)])
        a.commit()
        a.close()

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        out = [dict(r) for r in conn.execute("SELECT * FROM trades ORDER BY entry_id")]
        conn.close()
        return out

    def test_prices_the_row_the_archive_covers(self):
        backfill(self.db, self.archive)
        r = self.rows()[0]
        self.assertEqual(r["fill_source"], "live_quote")
        self.assertAlmostEqual(r["entry_price_mid"], 1.00)
        self.assertAlmostEqual(r["entry_price_cross"], 0.80)
        self.assertEqual(r["fill_policy"], "limit")

    def test_marks_the_uncovered_row_unknown_rather_than_modelling_it(self):
        backfill(self.db, self.archive)
        r = self.rows()[1]
        self.assertEqual(r["fill_source"], "unknown")
        self.assertIsNone(r["entry_price_fill"])

    def test_never_touches_the_original_record(self):
        before = [(r["entry_price"], r["net_credit"], r["pnl_usd"]) for r in self.rows()]
        backfill(self.db, self.archive)
        after = [(r["entry_price"], r["net_credit"], r["pnl_usd"]) for r in self.rows()]
        self.assertEqual(before, after)

    def test_dry_run_writes_nothing(self):
        backfill(self.db, self.archive, dry_run=True)
        for r in self.rows():
            self.assertIsNone(r["fill_source"])
            self.assertIsNone(r["entry_price_fill"])

    def test_is_idempotent(self):
        first = backfill(self.db, self.archive)
        snapshot = self.rows()
        second = backfill(self.db, self.archive)
        self.assertEqual(self.rows(), snapshot)
        self.assertEqual(first["priced"], second["priced"])

    def test_undo_clears_only_the_new_columns(self):
        backfill(self.db, self.archive)
        backfill(self.db, self.archive, undo=True)
        for r in self.rows():
            self.assertIsNone(r["entry_price_mid"])
            self.assertIsNone(r["entry_price_fill"])
            self.assertIsNone(r["entry_price_cross"])
            self.assertIsNone(r["fill_policy"])
            self.assertIsNone(r["fill_source"])
        # ...and the original record is still intact after a round trip.
        self.assertAlmostEqual(self.rows()[0]["entry_price"], 1.00)
        self.assertAlmostEqual(self.rows()[0]["pnl_usd"], 44.0)

    def test_runs_against_a_ledger_still_on_v17(self):
        """The live ledger is only migrated when a PaperManager is constructed.
        A backfill invoked from the CLI must bring the schema up itself rather
        than dying on `no such column: entry_price_mid`."""
        db = os.path.join(self.tmp.name, "v17.db")
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA user_version = 17")
        conn.execute(
            "CREATE TABLE trades ("
            "entry_id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, ticker TEXT,"
            " expiration TEXT, strike REAL, type TEXT, entry_price REAL,"
            " strategy_name TEXT, status TEXT, pnl_usd REAL, long_strike REAL,"
            " spread_width REAL, net_credit REAL, duplicate_of INTEGER)")
        conn.execute(
            "INSERT INTO trades (date, ticker, expiration, strike, type, entry_price,"
            " strategy_name, status, pnl_usd, long_strike, spread_width, net_credit)"
            " VALUES ('2026-06-10','ORCL','2026-07-17',80.0,'put',1.00,"
            "'Bull Put','CLOSED',44.0,79.0,1.0,1.00)")
        conn.commit()
        conn.close()

        out = backfill(db, self.archive)

        self.assertEqual(out["priced"], 1)
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        r = dict(conn.execute("SELECT * FROM trades").fetchone())
        conn.close()
        self.assertEqual(r["fill_source"], "live_quote")
        self.assertAlmostEqual(r["entry_price"], 1.00)   # original untouched

    def test_report_takes_the_median_of_per_trade_breakevens(self):
        """Not the breakeven of the median credit over the median width.

        The two differ whenever widths are mixed, and mixed widths are the norm
        here — the restated iron condors span $10 to $29. Ratio-of-medians put
        their p* at 56.4%; the median of each trade's own p* is 63.4%, which is
        the difference between clearing a 63.2% win rate and missing it.

        Fixture: p* of 50% and 90% -> median 70%. Ratio-of-medians would give
        1 - 2.55/5.5 = 53.6%."""
        db = os.path.join(self.tmp.name, "widths.db")
        conn = sqlite3.connect(db)
        conn.execute(TRADES_DDL)
        for width, credit, pnl in ((10.0, 5.0, 10.0), (1.0, 0.10, 5.0)):
            conn.execute(
                "INSERT INTO trades (date, ticker, strategy_name, status, pnl_usd,"
                " spread_width, entry_price_mid, entry_price_fill, entry_price_cross,"
                " fill_source) VALUES ('2026-06-10','ORCL','Bull Put','CLOSED',?,?,?,?,?,"
                "'live_quote')", (pnl, width, credit, credit, credit))
        conn.commit()
        conn.close()

        text = report(db)
        self.assertIn("70.0%", text)
        self.assertNotIn("53.6%", text)

    def test_reports_counts_split_by_source(self):
        out = backfill(self.db, self.archive)
        self.assertEqual(out["priced"], 1)
        self.assertEqual(out["unknown"], 1)
        self.assertEqual(out["scanned"], 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
