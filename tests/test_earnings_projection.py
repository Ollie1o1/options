"""Projecting the next earnings date from a symbol's own cadence.

The gate can only act on a symbol whose next report is CACHED, and DoltHub's
calendar is historical with a thin forward edge: 9 of 125 universe symbols
carried a future date on 2026-08-21. Refetching cannot fix that — the dates do
not exist upstream until the company announces.

But earnings are quarterly and most companies are extremely regular. Validated
against the 16 symbols whose next report WAS known, projecting from the median
of the last seven gaps and hiding the answer:

    regular cadence   n=10  median |error| 1 day   worst 8   9/10 within a week
    irregular/stale   n= 6  median |error| 19 days worst 77  1/6 within a week

Nine of ten regular symbols landed within a day; ADBE, ADSK, AVGO, COST and CRM
were exact. The failures are separable IN ADVANCE by two properties of the
symbol's own history — how much its gaps vary, and how long since it last
reported — which is what makes this usable rather than a guess.

Applied to the universe: coverage goes from 9/125 to 75/125.

A projection is an ESTIMATE and is reported as its own verdict, never merged
into the announced one. `earnings_projection: "report"` counts and prints it;
only `"refuse"` acts on it.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.earnings_gate import (CLEAR, PROJECTED_CLEAR, PROJECTED_THROUGH,
                               THROUGH, UNKNOWN, classify_with_projection,
                               load_earnings_gate_config,
                               project_next_earnings)


def _quarterly(start: str, n: int, jitter=None):
    """n reports about 91 days apart, optionally perturbed day by day."""
    d = dt.date.fromisoformat(start)
    out = []
    for i in range(n):
        step = 91 + ((jitter[i] if jitter and i < len(jitter) else 0))
        out.append(d.isoformat())
        d = d + dt.timedelta(days=step)
    return out


class Projection(unittest.TestCase):
    """`project_next_earnings` — pure, and refuses when it should."""

    def test_a_regular_reporter_projects_one_quarter_on(self):
        hist = _quarterly("2024-08-20", 9)          # last is 2026-08-16
        today = dt.date.fromisoformat(hist[-1]) + dt.timedelta(days=5)
        self.assertEqual(
            project_next_earnings(hist, today=today),
            (dt.date.fromisoformat(hist[-1]) + dt.timedelta(days=91)).isoformat())

    def test_too_little_history_projects_nothing(self):
        # Four quarters is not a cadence, it is a coincidence.
        hist = _quarterly("2025-08-20", 4)
        self.assertIsNone(project_next_earnings(hist, today=dt.date(2026, 8, 21)))

    def test_an_irregular_reporter_projects_nothing(self):
        # GME's gaps ranged 91 days wide and its projection missed by 77.
        hist = _quarterly("2024-01-10", 9, jitter=[0, 40, -35, 50, -40, 45, -30, 35])
        today = dt.date.fromisoformat(hist[-1]) + dt.timedelta(days=5)
        self.assertIsNone(project_next_earnings(hist, today=today))

    def test_a_stale_calendar_projects_nothing(self):
        # More than a quarter since the last known report means at least one
        # report has already happened unrecorded — the anchor is wrong.
        hist = _quarterly("2024-01-10", 9)
        today = dt.date.fromisoformat(hist[-1]) + dt.timedelta(days=200)
        self.assertIsNone(project_next_earnings(hist, today=today))

    def test_it_projects_forward_past_today_not_into_the_past(self):
        # If the anchor plus one quarter is already behind us, step on until
        # the projection is in the future.
        hist = _quarterly("2024-01-10", 9)
        today = dt.date.fromisoformat(hist[-1]) + dt.timedelta(days=100)
        out = project_next_earnings(hist, today=today, max_stale_days=120)
        self.assertIsNotNone(out)
        self.assertGreater(out, today.isoformat())

    def test_an_empty_history_is_none_not_a_crash(self):
        self.assertIsNone(project_next_earnings([], today=dt.date(2026, 8, 21)))


class ClassifyWithProjection(unittest.TestCase):
    """Announced dates always win; a projection only fills the silence."""

    TODAY = dt.date(2026, 8, 21)

    def test_an_announced_event_beats_any_projection(self):
        hist = _quarterly("2024-08-20", 9) + ["2026-09-01"]
        self.assertEqual(
            classify_with_projection(hist, "2026-08-21", "2026-09-18",
                                     today=self.TODAY), THROUGH)

    def test_a_projection_inside_the_window_is_its_own_verdict(self):
        # No announced future date, but this symbol reports like clockwork and
        # the next one lands inside the holding period.
        hist = _quarterly("2024-06-01", 9)
        anchor = dt.date.fromisoformat(hist[-1])
        today = anchor + dt.timedelta(days=10)
        end = (anchor + dt.timedelta(days=95)).isoformat()
        out = classify_with_projection(hist, today.isoformat(), end, today=today)
        self.assertEqual(out, PROJECTED_THROUGH)

    def test_a_projection_outside_the_window_is_projected_clear(self):
        hist = _quarterly("2024-06-01", 9)
        anchor = dt.date.fromisoformat(hist[-1])
        today = anchor + dt.timedelta(days=10)
        end = (anchor + dt.timedelta(days=30)).isoformat()
        self.assertEqual(
            classify_with_projection(hist, today.isoformat(), end, today=today),
            PROJECTED_CLEAR)

    def test_the_buffer_widens_the_window_around_the_estimate(self):
        # 9 of 10 regular symbols landed within 7 days, so a projection 5 days
        # past the window still counts as exposure.
        hist = _quarterly("2024-06-01", 9)
        anchor = dt.date.fromisoformat(hist[-1])
        today = anchor + dt.timedelta(days=10)
        end = (anchor + dt.timedelta(days=86)).isoformat()   # 5 days short
        self.assertEqual(
            classify_with_projection(hist, today.isoformat(), end, today=today,
                                     buffer_days=7), PROJECTED_THROUGH)
        self.assertEqual(
            classify_with_projection(hist, today.isoformat(), end, today=today,
                                     buffer_days=0), PROJECTED_CLEAR)

    def test_an_unprojectable_symbol_stays_unknown(self):
        self.assertEqual(
            classify_with_projection([], "2026-08-21", "2026-09-18",
                                     today=self.TODAY), UNKNOWN)

    def test_projection_off_falls_back_to_the_announced_verdict(self):
        hist = _quarterly("2024-06-01", 9)
        anchor = dt.date.fromisoformat(hist[-1])
        today = anchor + dt.timedelta(days=10)
        end = (anchor + dt.timedelta(days=95)).isoformat()
        self.assertEqual(
            classify_with_projection(hist, today.isoformat(), end, today=today,
                                     enabled=False), UNKNOWN)

    def test_a_next_quarter_landing_in_a_long_window_is_caught(self):
        # The case a two-state gate misses entirely: the cache's last date is
        # just BEHIND the entry, so the announced check says "reaches past
        # entry? no" and the trade is long enough to span the next report.
        hist = _quarterly("2024-06-01", 9)
        anchor = dt.date.fromisoformat(hist[-1])
        today = anchor + dt.timedelta(days=3)
        end = (anchor + dt.timedelta(days=100)).isoformat()
        self.assertEqual(
            classify_with_projection(hist, today.isoformat(), end, today=today),
            PROJECTED_THROUGH)

    def test_an_announced_clear_is_not_overridden_by_a_projection(self):
        # An announced future date outside the window is authoritative for the
        # next event; a projection must not second-guess it.
        hist = _quarterly("2024-08-20", 9) + ["2026-12-01"]
        self.assertEqual(
            classify_with_projection(hist, "2026-08-21", "2026-09-18",
                                     today=self.TODAY), CLEAR)


class Config(unittest.TestCase):

    def test_absent_means_off(self):
        self.assertEqual(load_earnings_gate_config({})["projection"], "off")

    def test_the_real_config_reports_before_it_refuses(self):
        import json
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "config.json")) as f:
            cfg = load_earnings_gate_config(json.load(f))
        self.assertEqual(cfg["projection"], "report")

    def test_an_unrecognised_mode_is_off_not_refuse(self):
        # A typo must never silently start refusing trades.
        cfg = load_earnings_gate_config(
            {"auto_log": {"earnings_projection": "banana"}})
        self.assertEqual(cfg["projection"], "off")


if __name__ == "__main__":
    unittest.main()
