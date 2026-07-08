"""Unit tests for the automation staleness guard.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_maintenance_health -v
"""
from __future__ import annotations

import unittest
from datetime import date

from src import maintenance_health as H


def _state(autolog=None, checkpoint=None, track=None, archive=None):
    st = {}
    if autolog is not None:
        st["last_autolog"] = autolog
    if checkpoint is not None:
        st["last_checkpoint"] = checkpoint
    if track is not None:
        st["last_track_record"] = track
    if archive is not None:
        st["last_chain_archive"] = archive
    return st


ALL_WINDOWS = {"ds": "2026-07-07", "sps": "2026-07-07",
               "ss": "2026-07-07", "ics": "2026-07-07"}


class BusinessDaysTest(unittest.TestCase):
    def test_same_day_is_zero(self):
        self.assertEqual(H.business_days_between(date(2026, 7, 7), date(2026, 7, 7)), 0)

    def test_weekend_not_counted(self):
        # Fri 2026-06-26 -> Mon 2026-06-29 is 1 business day (Mon only)
        self.assertEqual(H.business_days_between(date(2026, 6, 26), date(2026, 6, 29)), 1)

    def test_known_gap(self):
        # Fri 2026-06-26 -> Tue 2026-07-07 = 7 weekdays after the 26th
        self.assertEqual(H.business_days_between(date(2026, 6, 26), date(2026, 7, 7)), 7)


class ComputeHealthTest(unittest.TestCase):
    def test_all_fresh_is_ok_and_silent(self):
        now = date(2026, 7, 7)
        rep = H.compute_health(_state(autolog=ALL_WINDOWS, checkpoint="2026-07-06",
                                      track="2026-07-06", archive="2026-07-07"), now)
        self.assertEqual(rep.worst, "OK")
        self.assertEqual(H.health_banner(rep), "")

    def test_stale_autolog_is_critical_and_loud(self):
        now = date(2026, 7, 7)
        stale = {k: "2026-06-26" for k in ("ds", "sps", "ss", "ics")}
        rep = H.compute_health(_state(autolog=stale), now)
        self.assertEqual(rep.worst, "CRITICAL")
        self.assertEqual(rep.autolog_missed_days, 7)
        banner = H.health_banner(rep)
        self.assertNotEqual(banner, "")
        self.assertIn("auto-log", banner.lower())

    def test_weekend_launch_not_stale(self):
        # ran Friday, launching Monday -> 1 business day -> OK, silent
        now = date(2026, 6, 29)  # Monday
        fri = {k: "2026-06-26" for k in ("ds", "sps", "ss", "ics")}
        rep = H.compute_health(_state(autolog=fri), now)
        autolog = [j for j in rep.jobs if j.name == "auto-log"][0]
        self.assertEqual(autolog.severity, "OK")

    def test_missing_window_treated_as_never_run(self):
        now = date(2026, 7, 7)
        partial = {"ds": "2026-07-07"}  # sps/ss/ics missing entirely
        rep = H.compute_health(_state(autolog=partial), now)
        autolog = [j for j in rep.jobs if j.name == "auto-log"][0]
        self.assertIsNone(autolog.last_run)
        self.assertIn(rep.worst, ("STALE", "CRITICAL"))

    def test_autolog_fresh_but_background_stale_does_not_claim_cohort_loss(self):
        # auto-log current today, but checkpoint/archive long behind: the banner
        # must NOT claim cohort days were missed, and must say auto-log is current.
        now = date(2026, 7, 7)
        rep = H.compute_health(_state(autolog=ALL_WINDOWS, checkpoint="2026-06-22",
                                      track="2026-06-24", archive="2026-06-25"), now)
        autolog = [j for j in rep.jobs if j.name == "auto-log"][0]
        self.assertEqual(autolog.severity, "OK")
        self.assertNotEqual(rep.worst, "OK")  # background jobs still flag
        banner = H.health_banner(rep)
        self.assertIn("cohort auto-log is current", banner)
        self.assertNotIn("cohort filling missed", banner)

    def test_background_jobs_capped_below_critical(self):
        # archive 8 business days stale would be CRITICAL on raw daily cadence,
        # but is capped at STALE so it never shouts as loud as a starving cohort.
        now = date(2026, 7, 7)
        rep = H.compute_health(_state(autolog=ALL_WINDOWS, archive="2026-06-25"), now)
        archive = [j for j in rep.jobs if j.name == "chain-archive"][0]
        self.assertEqual(archive.severity, "STALE")

    def test_missing_state_file_is_critical(self):
        rep = H.compute_health({}, date(2026, 7, 7))
        self.assertIn(rep.worst, ("STALE", "CRITICAL"))
        self.assertNotEqual(H.health_banner(rep), "")


class HealthLinesTest(unittest.TestCase):
    def test_lines_cover_every_job(self):
        rep = H.compute_health(_state(autolog=ALL_WINDOWS), date(2026, 7, 7))
        text = "\n".join(H.health_lines(rep))
        for name in ("auto-log", "checkpoint", "track-record", "chain-archive"):
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
