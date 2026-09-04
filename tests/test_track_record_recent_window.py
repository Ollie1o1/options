"""The recent-window section of the published track record.

The all-time headline is a single number over the book's whole life — 900+
trades back to April. A bad two-week stretch is invisible inside it, and the
only way to see one has been an ad hoc SQL query against paper_trades.db.
This section is that query, published automatically: the same book, cut to
just the trades entered in the last `days`, using the SAME statistics
(`summarize_book`, `equal_weighted`) as the all-time sections so the numbers
are never computed two different ways.

Anchored to the most recent trade DATE in the rows, not wall-clock time — a
regenerated report is reproducible from its own data, and a book that has
gone quiet is not silently read as "no recent trades" just because nobody
regenerated the file today.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.publish_track_record import (  # noqa: E402
    recent_window,
    render_track_record,
)

_EVIDENCE = {
    "pooled_ic": 0.10, "p_value": 0.48, "n_oos": 94,
    "cohort_n": 2, "gate_decision": "GATHERING", "as_of": "2026-06-07",
}


def _row(pnl, risk, date, strategy="Bull Put"):
    return {"strategy_name": strategy, "status": "CLOSED", "pnl_usd": pnl,
            "capital_at_risk": risk, "pnl_pct": (pnl / risk if risk else None),
            "entry_price": 1.0, "net_credit": 1.0, "quantity": 1.0,
            "date": date, "exit_date": date, "ticker": "AAA"}


class RecentWindow(unittest.TestCase):

    def test_only_rows_inside_the_window_survive(self):
        rows = [_row(1, 100, "2026-08-01"), _row(1, 100, "2026-08-20")]
        out = recent_window(rows, days=7, as_of="2026-08-20")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["date"], "2026-08-20")

    def test_the_boundary_day_is_included(self):
        rows = [_row(1, 100, "2026-08-14")]
        out = recent_window(rows, days=7, as_of="2026-08-21")
        self.assertEqual(len(out), 1)

    def test_a_day_outside_the_window_is_excluded(self):
        rows = [_row(1, 100, "2026-08-13")]
        out = recent_window(rows, days=7, as_of="2026-08-21")
        self.assertEqual(len(out), 0)

    def test_without_as_of_the_window_anchors_to_the_latest_row(self):
        rows = [_row(1, 100, "2026-01-01"), _row(1, 100, "2026-08-20")]
        out = recent_window(rows, days=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["date"], "2026-08-20")

    def test_an_empty_book_returns_empty(self):
        self.assertEqual(recent_window([], days=14), [])

    def test_rows_with_no_date_are_excluded_not_kept(self):
        rows = [_row(1, 100, "2026-08-20"), {"strategy_name": "Bull Put"}]
        out = recent_window(rows, days=1)
        self.assertEqual(len(out), 1)


class Rendered(unittest.TestCase):

    def _doc(self, rows):
        return render_track_record(rows, _EVIDENCE)

    def test_the_section_is_published(self):
        rows = [_row(50.0, 500.0, "2026-08-25")] * 20
        self.assertIn("## Recent", self._doc(rows))

    def test_it_counts_only_the_window_not_the_whole_book(self):
        old = [_row(-500.0, 500.0, "2026-04-18")] * 50   # a bad old stretch
        new = [_row(100.0, 500.0, "2026-08-25")] * 5      # a clean recent one
        doc = self._doc(old + new)
        # The all-time headline still reflects all 55; the recent section
        # must not silently inherit the old losses.
        recent_section = doc.split("## Recent", 1)[1].split("## Equal-weighted", 1)[0]
        self.assertIn("5 closed", recent_section)
        self.assertIn("+$500.00", recent_section)
        self.assertNotIn("-$", recent_section)

    def test_a_quiet_recent_window_says_so_rather_than_going_silent(self):
        # No row carries a usable date, so the window has nothing to anchor
        # to — the empty-book case, not a stale-report case (the window
        # always anchors to the latest date IN the data, so a valid old date
        # can never land outside its own window by default).
        rows = [{"strategy_name": "Bull Put", "status": "CLOSED",
                "pnl_usd": 50.0, "capital_at_risk": 500.0} for _ in range(10)]
        doc = self._doc(rows)
        recent_section = doc.split("## Recent", 1)[1].split("## Equal-weighted", 1)[0]
        self.assertIn("No closed trade", recent_section)

    def test_a_small_window_is_flagged_rather_than_read_as_a_verdict(self):
        rows = [_row(50.0, 500.0, "2026-08-25"), _row(-30.0, 500.0, "2026-08-26")]
        doc = self._doc(rows)
        recent_section = doc.split("## Recent", 1)[1].split("## Equal-weighted", 1)[0]
        self.assertIn("too few", recent_section.lower())

    def test_the_all_time_headline_is_unaffected(self):
        rows = [_row(50.0, 500.0, "2026-04-18")] * 30
        doc = self._doc(rows)
        self.assertIn("## Headline", doc)
        self.assertIn("30 closed trades", doc)


if __name__ == "__main__":
    unittest.main()
