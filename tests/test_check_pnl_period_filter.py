"""Portfolio viewer period filter: current book vs closed history.

The split is a frozen date, not a rolling one, and CURRENT deliberately carries
every still-open position regardless of entry date — 83 open positions predated
the restart, and routing them into history would hide live exposure from the
view where the book actually gets looked at.
"""
import io
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from src import check_pnl

CUT = check_pnl.BOOK_RESTART_DATE


def _rows():
    return [
        # closed, well before the restart → history
        {"entry_id": 1, "date": "2026-04-18", "status": "CLOSED"},
        # open, before the restart → still current, because it's live
        {"entry_id": 2, "date": "2026-06-20", "status": "OPEN"},
        # closed, on the restart date itself → current (boundary is inclusive)
        {"entry_id": 3, "date": CUT, "status": "CLOSED"},
        # open, logged on the restart date → current
        {"entry_id": 4, "date": CUT, "status": "OPEN"},
        # closed, after the restart → current
        {"entry_id": 5, "date": "2026-08-06", "status": "CLOSED"},
        # closed with no date at all → history, never silently current
        {"entry_id": 6, "date": None, "status": "CLOSED"},
    ]


def _ids(rows):
    return sorted(r["entry_id"] for r in rows)


class PeriodFilterTest(unittest.TestCase):
    def test_current_is_restart_onward_plus_all_open(self):
        out = check_pnl._filter_by_period(_rows(), "current")
        self.assertEqual(_ids(out), [2, 3, 4, 5])

    def test_before_is_closed_history_only(self):
        out = check_pnl._filter_by_period(_rows(), "before")
        self.assertEqual(_ids(out), [1, 6])

    def test_open_position_predating_restart_is_never_hidden(self):
        """The whole point of the OPEN rule — regression guard."""
        out = check_pnl._filter_by_period(_rows(), "current")
        self.assertIn(2, _ids(out), "a live position fell out of the current book")
        self.assertNotIn(2, _ids(check_pnl._filter_by_period(_rows(), "before")))

    def test_boundary_date_is_inclusive(self):
        out = check_pnl._filter_by_period(
            [{"entry_id": 9, "date": CUT, "status": "CLOSED"}], "current")
        self.assertEqual(_ids(out), [9])

    def test_blank_date_sorts_as_history(self):
        rows = [{"entry_id": 9, "date": "", "status": "CLOSED"}]
        self.assertEqual(_ids(check_pnl._filter_by_period(rows, "before")), [9])
        self.assertEqual(check_pnl._filter_by_period(rows, "current"), [])

    def test_status_case_is_ignored(self):
        rows = [{"entry_id": 9, "date": "2026-01-01", "status": "open"}]
        self.assertEqual(_ids(check_pnl._filter_by_period(rows, "current")), [9])

    def test_none_returns_all(self):
        self.assertEqual(len(check_pnl._filter_by_period(_rows(), None)), 6)

    def test_unknown_period_returns_all(self):
        self.assertEqual(len(check_pnl._filter_by_period(_rows(), "bogus")), 6)

    def test_every_row_lands_in_exactly_one_bucket(self):
        rows = _rows()
        cur = _ids(check_pnl._filter_by_period(rows, "current"))
        old = _ids(check_pnl._filter_by_period(rows, "before"))
        self.assertEqual(sorted(cur + old), _ids(rows))
        self.assertEqual(set(cur) & set(old), set())


class FrozenCutoffTest(unittest.TestCase):
    def test_cutoff_is_a_frozen_date_string(self):
        """A rolling date.today() would empty the current book every morning."""
        self.assertIsInstance(check_pnl.BOOK_RESTART_DATE, str)
        self.assertRegex(check_pnl.BOOK_RESTART_DATE, r"^\d{4}-\d{2}-\d{2}$")

    def test_retired_era_and_cohort_filters_are_gone(self):
        for attr in ("_filter_by_era", "_filter_by_cohort", "CALIBRATION_CUTOFFS"):
            self.assertFalse(hasattr(check_pnl, attr),
                             f"{attr} should have been retired with the era/cohort split")


class EmptyResultPathTest(unittest.TestCase):
    """Drives view_portfolio far enough to execute the no-rows branch.

    That branch reported which filter was active and kept referring to the
    retired `cohort`/`era` locals after the split was rewritten — a NameError
    that no unit test reached, because nothing ran view_portfolio end to end.
    Only mypy caught it. These run the real function against a temp ledger.
    """

    # The renderer reads a wide slice of columns once a row survives the filter,
    # so build the table from the live schema instead of guessing at a stub —
    # a stub fails on whichever column the renderer happens to touch next, for
    # reasons that have nothing to do with the split under test. Only the columns
    # the split and the renderer's header path need are populated; the rest stay
    # NULL, which is what a real sparse legacy row looks like anyway.
    COLUMNS = ("entry_id INTEGER PRIMARY KEY, date TEXT, ticker TEXT, expiration TEXT, "
               "strike REAL, type TEXT, entry_price REAL, quality_score REAL, "
               "strategy_name TEXT, status TEXT, exit_price REAL, exit_date TEXT, "
               "pnl_pct REAL, pnl_usd REAL, weight_profile TEXT, exit_reason TEXT, "
               "long_strike REAL, spread_width REAL, net_credit REAL, "
               "max_profit_usd REAL, max_loss_usd REAL, net_delta REAL, "
               "quantity REAL, paper_only INTEGER, era TEXT, capital_at_risk REAL, "
               "duplicate_of INTEGER, entry_price_mid REAL, entry_price_fill REAL, "
               "entry_price_cross REAL, fill_policy TEXT, fill_source TEXT, "
               "shadow_until TEXT")

    def _db(self, rows):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        with sqlite3.connect(path) as conn:
            conn.execute(f"CREATE TABLE trades ({self.COLUMNS})")
            conn.executemany(
                "INSERT INTO trades (entry_id, date, ticker, expiration, strike, "
                "type, entry_price, quantity, status, strategy_name) "
                "VALUES (?,?,'SPY','2026-12-18',500.0,'CALL',1.0,1,?,'Long Call')",
                rows)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        return path

    def _run(self, path, period):
        """Render with exit enforcement stubbed out — it would hit the network."""
        out = io.StringIO()
        with mock.patch.object(check_pnl, "DB_PATH", path), \
                mock.patch("src.paper_manager.PaperManager"), \
                redirect_stdout(out):
            check_pnl.view_portfolio(period=period)
        return out.getvalue()

    def test_filtered_to_nothing_reports_the_filter(self):
        # One closed trade from well before the restart, asking for current.
        path = self._db([(1, "2026-01-01", "CLOSED")])
        self.assertIn("No trades match this filter", self._run(path, "current"))

    def test_empty_ledger_reports_no_trades_at_all(self):
        self.assertIn("No trades logged yet", self._run(self._db([]), None))

    def test_both_periods_render_without_raising(self):
        path = self._db([(1, "2026-01-01", "CLOSED"), (2, CUT, "OPEN")])
        for period in ("current", "before", None):
            with self.subTest(period=period):
                self._run(path, period)  # must not raise

    def test_header_names_the_active_period(self):
        path = self._db([(1, "2026-01-01", "CLOSED"), (2, CUT, "OPEN")])
        self.assertIn("CURRENT BOOK", self._run(path, "current"))
        self.assertIn(f"BEFORE {CUT}", self._run(path, "before"))


if __name__ == "__main__":
    unittest.main()
