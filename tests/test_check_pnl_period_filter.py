"""Portfolio viewer period filter: current book vs closed history.

The split is a frozen date, not a rolling one, and CURRENT deliberately carries
every still-open position regardless of entry date — 83 open positions predated
the restart, and routing them into history would hide live exposure from the
view where the book actually gets looked at.
"""
import unittest

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


if __name__ == "__main__":
    unittest.main()
