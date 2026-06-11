"""Pin the verified 2026 macro-event calendar in src/macro_analyzer.py.

The FOMC dates were verified against federalreserve.gov/monetarypolicy/
fomccalendars.htm and CPI against the BLS release schedule on 2026-06-11.
The previous hardcoded list had four wrong forward dates (FOMC 05-07/11-04/
12-16, CPI 08-13/10-13) — a macro-penalty firing on the wrong day is worse
than no penalty, so this test fails loudly if anyone regresses the list.
"""
from __future__ import annotations

import unittest

from src.macro_analyzer import _DEFAULT_EVENTS


def _dates(name: str, year: str):
    return sorted(e["date"] for e in _DEFAULT_EVENTS
                  if e["name"] == name and e["date"].startswith(year))


class VerifiedCalendar2026Test(unittest.TestCase):
    def test_fomc_2026_decision_days(self):
        self.assertEqual(_dates("FOMC", "2026"), [
            "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
            "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
        ])

    def test_cpi_2026_release_days(self):
        self.assertEqual(_dates("CPI", "2026"), [
            "2026-01-15", "2026-02-12", "2026-03-12", "2026-04-10",
            "2026-05-13", "2026-06-11", "2026-07-14", "2026-08-12",
            "2026-09-11", "2026-10-14", "2026-11-10", "2026-12-10",
        ])

    def test_wrong_dates_are_gone(self):
        all_dates = {e["date"] for e in _DEFAULT_EVENTS}
        for wrong in ("2026-05-07", "2026-11-04", "2026-12-16",
                      "2026-08-13", "2026-10-13"):
            self.assertNotIn(wrong, all_dates)


if __name__ == "__main__":
    unittest.main()
