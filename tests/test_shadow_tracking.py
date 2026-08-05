"""Shadow-tracking: what a stopped-out trade WOULD have done.

The stop fires on 40 of 82 single-leg long trades, realising -60.3% from an
average peak of +16.6%. Whether the stop helps or hurts is currently
unanswerable: max_price_seen stops updating the moment the position closes, so
there is no post-stop path to compare against. Tracking continues after the
exit, into separate columns, so the counterfactual becomes data.
"""
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paper_manager import _SCHEMA_VERSION, PaperManager

SHADOW_COLUMNS = ("shadow_until", "post_exit_max_price", "post_exit_max_date",
                  "post_exit_last_price", "post_exit_last_date")


class SchemaTest(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "pm.db")

    def test_schema_is_at_least_nineteen(self):
        self.assertGreaterEqual(_SCHEMA_VERSION, 19)

    def test_the_shadow_columns_exist(self):
        PaperManager(db_path=self.db)
        conn = sqlite3.connect(self.db)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(trades)")]
        conn.close()
        for c in SHADOW_COLUMNS:
            self.assertIn(c, cols)


class ShadowMarkTest(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "pm.db")
        self.pm = PaperManager(db_path=self.db)
        conn = sqlite3.connect(self.db)
        conn.execute(
            "INSERT INTO trades (entry_id, ticker, expiration, strike, type,"
            " entry_price, strategy_name, status, exit_price, exit_reason,"
            " shadow_until) VALUES (1,'AAPL','2026-12-18',200.0,'call',"
            "5.00,'Long Call','CLOSED',2.50,'Stop Loss (-50%)','2026-12-18')")
        conn.commit(); conn.close()

    def _row(self):
        conn = sqlite3.connect(self.db); conn.row_factory = sqlite3.Row
        r = dict(conn.execute("SELECT * FROM trades WHERE entry_id=1").fetchone())
        conn.close(); return r

    def test_a_shadow_mark_records_the_post_exit_high(self):
        self.pm.shadow_mark(1, 9.00, "2026-09-01")
        r = self._row()
        self.assertAlmostEqual(r["post_exit_max_price"], 9.00)
        self.assertEqual(r["post_exit_max_date"], "2026-09-01")

    def test_the_high_only_ratchets_upward(self):
        self.pm.shadow_mark(1, 9.00, "2026-09-01")
        self.pm.shadow_mark(1, 4.00, "2026-09-02")
        r = self._row()
        self.assertAlmostEqual(r["post_exit_max_price"], 9.00)
        self.assertEqual(r["post_exit_max_date"], "2026-09-01")

    def test_the_last_mark_always_updates(self):
        """The high answers 'could it have recovered'; the last answers 'where
        did it actually end up'. Both are needed to judge the stop."""
        self.pm.shadow_mark(1, 9.00, "2026-09-01")
        self.pm.shadow_mark(1, 4.00, "2026-09-02")
        r = self._row()
        self.assertAlmostEqual(r["post_exit_last_price"], 4.00)
        self.assertEqual(r["post_exit_last_date"], "2026-09-02")

    def test_shadow_marking_never_alters_the_realised_result(self):
        before = self._row()
        self.pm.shadow_mark(1, 99.00, "2026-09-01")
        after = self._row()
        for k in ("status", "exit_price", "exit_reason", "entry_price", "pnl_usd"):
            self.assertEqual(before[k], after[k], f"{k} must not move")

    def test_a_trade_with_no_shadow_window_is_not_marked(self):
        conn = sqlite3.connect(self.db)
        conn.execute("UPDATE trades SET shadow_until=NULL WHERE entry_id=1")
        conn.commit(); conn.close()
        self.pm.shadow_mark(1, 9.00, "2026-09-01")
        self.assertIsNone(self._row()["post_exit_max_price"])

    def test_marks_after_the_window_closes_are_ignored(self):
        self.pm.shadow_mark(1, 9.00, "2027-01-05")
        self.assertIsNone(self._row()["post_exit_max_price"])


class ShadowSelectionTest(unittest.TestCase):
    """Which trades the updater should still be marking."""

    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "pm.db")
        self.pm = PaperManager(db_path=self.db)
        conn = sqlite3.connect(self.db)
        conn.executemany(
            "INSERT INTO trades (ticker, expiration, strike, type, entry_price,"
            " strategy_name, status, exit_reason, shadow_until) VALUES (?,?,?,?,?,?,?,?,?)",
            [("AAPL", "2026-12-18", 200.0, "call", 5.0, "Long Call", "CLOSED",
              "Stop Loss (-50%)", "2026-12-18"),
             ("MSFT", "2026-09-18", 500.0, "call", 5.0, "Long Call", "CLOSED",
              "Take Profit (100%)", None),
             ("NVDA", "2026-12-18", 100.0, "call", 5.0, "Long Call", "CLOSED",
              "Stop Loss (-50%)", "2026-01-01")])
        conn.commit(); conn.close()

    def test_only_open_shadow_windows_are_returned(self):
        got = {r["ticker"] for r in self.pm.shadowed_positions(today="2026-08-04")}
        self.assertEqual(got, {"AAPL"})

    def test_an_expired_window_drops_out(self):
        self.assertEqual(self.pm.shadowed_positions(today="2027-01-01"), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class ShadowWindowOpensOnStopTest(unittest.TestCase):
    """A stop-out must open its own shadow window, or the counterfactual never
    accrues and this whole mechanism sits idle."""

    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "pm.db")
        self.pm = PaperManager(db_path=self.db)

    def _close(self, reason, expiration="2026-12-18"):
        conn = sqlite3.connect(self.db)
        conn.execute(
            "INSERT INTO trades (entry_id, ticker, expiration, strike, type,"
            " entry_price, strategy_name, status) VALUES (1,'AAPL',?,200.0,"
            "'call',5.0,'Long Call','OPEN')", (expiration,))
        conn.commit(); conn.close()
        self.pm.open_shadow_window(1, reason)
        conn = sqlite3.connect(self.db); conn.row_factory = sqlite3.Row
        r = dict(conn.execute("SELECT * FROM trades WHERE entry_id=1").fetchone())
        conn.close(); return r

    def test_a_stop_loss_opens_a_window_to_the_original_expiry(self):
        self.assertEqual(self._close("Stop Loss (-50%)")["shadow_until"], "2026-12-18")

    def test_a_time_exit_opens_one_too(self):
        """Time exits leave 52 points on the table on average — the second
        exit rule worth a counterfactual."""
        self.assertEqual(self._close("Time Exit (21d to expiry)")["shadow_until"],
                         "2026-12-18")

    def test_a_take_profit_does_not(self):
        """Nothing to learn: every trade that peaked at +100% was exited there."""
        self.assertIsNone(self._close("Take Profit (100%)")["shadow_until"])

    def test_an_unknown_reason_does_not_open_a_window(self):
        self.assertIsNone(self._close("Manual close")["shadow_until"])


class ShadowedMarksAccrueTest(unittest.TestCase):
    """The window is useless unless something writes into it. A shadowed trade
    must be quoted alongside the open book on every update run."""

    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "pm.db")
        self.pm = PaperManager(db_path=self.db)
        conn = sqlite3.connect(self.db)
        conn.execute(
            "INSERT INTO trades (entry_id, ticker, expiration, strike, type,"
            " entry_price, strategy_name, status, exit_price, exit_reason,"
            " shadow_until) VALUES (1,'AAPL','2026-12-18',200.0,'call',5.0,"
            "'Long Call','CLOSED',2.5,'Stop Loss (-50%)','2026-12-18')")
        conn.commit(); conn.close()

    def test_a_shadowed_trade_is_offered_for_quoting(self):
        rows = self.pm.shadowed_positions(today="2026-08-04")
        self.assertEqual(len(rows), 1)
        r = rows[0]
        # Everything a quote lookup needs must be present.
        for k in ("ticker", "expiration", "strike", "type", "entry_id"):
            self.assertIsNotNone(r[k], f"{k} needed to quote the contract")

    def test_marking_it_records_against_the_original_entry_price(self):
        self.pm.shadow_mark(1, 12.50, "2026-09-01")
        conn = sqlite3.connect(self.db); conn.row_factory = sqlite3.Row
        r = dict(conn.execute("SELECT * FROM trades WHERE entry_id=1").fetchone())
        conn.close()
        # Stopped out at 2.50 from a 5.00 entry (-50%); later worth 12.50 (+150%).
        self.assertAlmostEqual(r["exit_price"], 2.50)
        self.assertAlmostEqual(r["post_exit_max_price"] / r["entry_price"] - 1, 1.50)
