"""The portfolio opens on the current book, not on nine months of dead history.

The viewer had `--current`, `--before` and a menu that defaulted to current —
but running it with no arguments showed the WHOLE book: 866 closed trades from
before the 2026-08-05 restart, almost all of them strategies since switched off
(Long Call, Iron Condor, Bear Call, Long Put), chosen by a ranker measured at
OOS IC -0.12 and scored by EV estimates later found to be the short leg's.

Nothing is deleted. Those rows are what establish Bull Put's 2.023 profit factor
(131 of its 134 closes predate the restart), the 50.9% required win rate over
415 credit trades, and the pre-registered ranker test frozen to 2026-11-19.
They are evidence; they are just not the operator's daily view.

Open positions are ALWAYS shown whatever the filter, because they are live money
regardless of which regime opened them — 42 of them predate the restart.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.check_pnl import (BOOK_RESTART_DATE, _filter_by_period,
                           _period_for_row, resolve_period)


def _row(date, status="CLOSED"):
    return {"date": date, "status": status}


_OLD = "2026-07-01"
_NEW = "2026-08-15"


class DefaultView(unittest.TestCase):

    def test_no_argument_means_the_current_book(self):
        # The change: absent an explicit choice the viewer shows the book the
        # operator is actually running, not its archaeology.
        self.assertEqual(resolve_period(None), "current")

    def test_all_is_how_you_ask_for_everything(self):
        self.assertIsNone(resolve_period("all"))

    def test_an_explicit_choice_is_honoured(self):
        self.assertEqual(resolve_period("before"), "before")
        self.assertEqual(resolve_period("current"), "current")


class WhatEachFilterShows(unittest.TestCase):

    def setUp(self):
        self.rows = [_row(_OLD), _row(_NEW), _row(_OLD, "OPEN"),
                     _row(_NEW, "OPEN")]

    def test_current_hides_closed_history(self):
        out = _filter_by_period(self.rows, "current")
        self.assertNotIn(_row(_OLD), out)
        self.assertIn(_row(_NEW), out)

    def test_current_keeps_every_open_position(self):
        # 42 open positions predate the restart. Hiding live risk to tidy a
        # view would be the worst possible trade-off.
        out = _filter_by_period(self.rows, "current")
        self.assertIn(_row(_OLD, "OPEN"), out)
        self.assertIn(_row(_NEW, "OPEN"), out)

    def test_all_shows_the_whole_book(self):
        self.assertEqual(len(_filter_by_period(self.rows, None)), 4)

    def test_before_shows_only_the_closed_history(self):
        out = _filter_by_period(self.rows, "before")
        self.assertEqual(out, [_row(_OLD)])

    def test_the_restart_date_itself_is_current(self):
        self.assertEqual(_period_for_row(_row(BOOK_RESTART_DATE)), "current")

    def test_a_row_with_no_date_is_history_never_silently_current(self):
        self.assertEqual(_period_for_row(_row("")), "before")


if __name__ == "__main__":
    unittest.main()
