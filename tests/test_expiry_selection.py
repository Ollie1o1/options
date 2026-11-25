"""Expiration budget is spread across the DTE window, not spent on its front."""
from __future__ import annotations

import datetime as dt
import unittest

from src.expiry_selection import is_standard_monthly, select_expirations

TODAY = dt.date(2026, 11, 25)  # a Wednesday


def _dailies(days: int) -> list:
    """SPY-style listing: every weekday for `days` calendar days."""
    out = []
    for k in range(1, days + 1):
        d = TODAY + dt.timedelta(days=k)
        if d.weekday() < 5:
            out.append(d.isoformat())
    return out


def _weeklies(weeks: int) -> list:
    """Single-name style listing: every Friday."""
    first = TODAY + dt.timedelta(days=(4 - TODAY.weekday()) % 7 or 7)
    return [(first + dt.timedelta(days=7 * k)).isoformat() for k in range(weeks)]


def _dte(e: str) -> int:
    return (dt.date.fromisoformat(e) - TODAY).days


class SpreadSelectionTests(unittest.TestCase):
    def test_daily_listing_reaches_back_of_window(self) -> None:
        exps = _dailies(70)
        legacy = select_expirations(exps, n=4, today=TODAY, min_dte=7, max_dte=45, method="nearest")
        spread = select_expirations(exps, n=4, today=TODAY, min_dte=7, max_dte=45)
        self.assertLessEqual(max(map(_dte, legacy)), 12)  # the defect: all at the front
        self.assertEqual(len(spread), 4)
        dtes = sorted(map(_dte, spread))
        self.assertLessEqual(dtes[0], 8)
        self.assertGreaterEqual(dtes[-1], 43)
        gaps = [b - a for a, b in zip(dtes, dtes[1:])]
        self.assertGreater(min(gaps), 5)

    def test_weekly_listing_covers_whole_window(self) -> None:
        exps = _weeklies(12)
        spread = select_expirations(exps, n=4, today=TODAY, min_dte=7, max_dte=45)
        in_window = [e for e in exps if 7 <= _dte(e) <= 45]
        self.assertEqual(spread[0], in_window[0])
        self.assertEqual(spread[-1], in_window[-1])

    def test_interior_targets_prefer_a_nearby_monthly(self) -> None:
        exps = _dailies(70)
        spread = select_expirations(exps, n=4, today=TODAY, min_dte=7, max_dte=45)
        monthly = [e for e in spread if is_standard_monthly(dt.date.fromisoformat(e))]
        self.assertEqual(monthly, ["2026-12-18"])  # Dec third Friday, 23 DTE
        no_pref = select_expirations(exps, n=4, today=TODAY, min_dte=7, max_dte=45, prefer_monthly=False)
        self.assertEqual(len(no_pref), 4)

    def test_budget_at_least_listing_returns_everything_in_window(self) -> None:
        exps = _weeklies(10)
        in_window = [e for e in exps if 7 <= _dte(e) <= 45]
        self.assertEqual(select_expirations(exps, n=10, today=TODAY, min_dte=7, max_dte=45), in_window)

    def test_empty_window_falls_back_to_listing(self) -> None:
        exps = _weeklies(3)  # nothing beyond ~23 DTE
        got = select_expirations(exps, n=2, today=TODAY, min_dte=60, max_dte=120)
        self.assertEqual(got, exps[:2])

    def test_no_window_and_n_one(self) -> None:
        exps = _weeklies(6)
        self.assertEqual(select_expirations(exps, n=1, today=TODAY), [exps[0]])
        self.assertEqual(select_expirations(exps, n=0, today=TODAY), [])

    def test_unique_sorted_and_deterministic(self) -> None:
        exps = _dailies(200)
        a = select_expirations(exps, n=12, today=TODAY, min_dte=7, max_dte=180)
        b = select_expirations(list(reversed(exps)), n=12, today=TODAY, min_dte=7, max_dte=180)
        self.assertEqual(a, b)
        self.assertEqual(a, sorted(set(a)))
        self.assertEqual(len(a), 12)

    def test_unknown_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            select_expirations(_weeklies(8), n=2, today=TODAY, method="random")

    def test_third_friday(self) -> None:
        self.assertTrue(is_standard_monthly(dt.date(2026, 12, 18)))
        self.assertFalse(is_standard_monthly(dt.date(2026, 12, 11)))
        self.assertFalse(is_standard_monthly(dt.date(2026, 12, 17)))


class FetchWiringTests(unittest.TestCase):
    def test_setter_validates_and_defaults(self) -> None:
        from src import data_fetching as dfm

        before = dfm.expiry_selection()
        try:
            self.assertEqual(dfm.set_expiry_selection("nearest"), "nearest")
            self.assertEqual(dfm.set_expiry_selection("bogus"), "spread")
            self.assertEqual(dfm.set_expiry_selection(None), "spread")
        finally:
            dfm.set_expiry_selection(before)


if __name__ == "__main__":
    unittest.main()
