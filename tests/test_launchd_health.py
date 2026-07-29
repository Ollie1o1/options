"""Detect a dead scheduler, not just a stale state file.

All three LaunchAgents have been exiting 78 (EX_CONFIG) since ~2026-06-15, so
nothing scheduled has run for six weeks. maintenance_health did not notice
because it reads the state file, which the interactive path also stamps — so a
dead scheduler looked like mild staleness whenever the user happened to open the
screener by hand.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.maintenance_health import (
    launchd_failure_message,
    parse_launchctl_list,
)

_HEALTHY = """PID\tStatus\tLabel
-\t0\tcom.ollie.options.maintenance
1234\t0\tcom.ollie.options.crypto-auto-log
-\t0\tcom.apple.something
"""

_BROKEN = """PID\tStatus\tLabel
-\t78\tcom.ollie.options.crypto-enforce-exits
-\t78\tcom.ollie.options.maintenance
-\t78\tcom.ollie.options.crypto-auto-log
-\t0\tcom.apple.something
"""


class TestParsing(unittest.TestCase):
    def test_finds_our_jobs_and_ignores_everything_else(self):
        jobs = parse_launchctl_list(_HEALTHY, prefix="com.ollie.options")
        self.assertEqual(len(jobs), 2)
        self.assertTrue(all(j.label.startswith("com.ollie.options") for j in jobs))

    def test_reads_the_last_exit_status(self):
        jobs = {j.label: j for j in parse_launchctl_list(_BROKEN, "com.ollie.options")}
        self.assertEqual(jobs["com.ollie.options.maintenance"].last_exit_status, 78)

    def test_a_zero_status_is_not_a_failure(self):
        jobs = parse_launchctl_list(_HEALTHY, "com.ollie.options")
        self.assertEqual([j for j in jobs if j.failed], [])

    def test_a_nonzero_status_is_a_failure(self):
        jobs = parse_launchctl_list(_BROKEN, "com.ollie.options")
        self.assertEqual(len([j for j in jobs if j.failed]), 3)

    def test_header_and_blank_lines_are_skipped(self):
        self.assertEqual(parse_launchctl_list("PID\tStatus\tLabel\n\n", "com.ollie"), [])

    def test_unparseable_output_yields_nothing_rather_than_raising(self):
        self.assertEqual(parse_launchctl_list("something unexpected", "com.ollie"), [])

    def test_a_running_job_with_a_pid_is_not_a_failure(self):
        jobs = parse_launchctl_list("PID\tStatus\tLabel\n900\t0\tcom.ollie.options.x\n",
                                    "com.ollie.options")
        self.assertFalse(jobs[0].failed)


class TestMessage(unittest.TestCase):
    def test_no_message_when_everything_is_healthy(self):
        self.assertIsNone(
            launchd_failure_message(parse_launchctl_list(_HEALTHY, "com.ollie.options"))
        )

    def test_names_the_failing_jobs(self):
        msg = launchd_failure_message(parse_launchctl_list(_BROKEN, "com.ollie.options"))
        self.assertIn("3", msg)
        self.assertIn("maintenance", msg)

    def test_exit_78_explains_the_actual_fix(self):
        # EX_CONFIG here means macOS is refusing to run the agent, and the fix is
        # a Login Items toggle the user must click. Saying "job failed" sends the
        # reader looking for a bug in the script instead.
        msg = launchd_failure_message(parse_launchctl_list(_BROKEN, "com.ollie.options"))
        self.assertIn("Login Items", msg)

    def test_other_exit_codes_do_not_claim_it_is_a_permissions_problem(self):
        jobs = parse_launchctl_list("PID\tStatus\tLabel\n-\t1\tcom.ollie.options.x\n",
                                    "com.ollie.options")
        self.assertNotIn("Login Items", launchd_failure_message(jobs))


class TestBannerSurfacing(unittest.TestCase):
    """A dead scheduler must show even when the state file looks fresh — that
    combination is the whole bug, since opening the screener by hand stamps the
    state file and hides six weeks of nothing running."""

    def _fresh_report(self):
        from datetime import date

        from src.maintenance_health import WORKING_WINDOWS, compute_health

        today = date(2026, 7, 29).isoformat()
        state = {
            "last_autolog": {w: today for w in WORKING_WINDOWS},
            "last_checkpoint": today,
            "last_track_record": today,
            "last_chain_archive": today,
            "last_morning_briefing": today,
        }
        return compute_health(state, date(2026, 7, 29))

    def test_a_fresh_report_alone_produces_no_banner(self):
        from src.maintenance_health import health_banner

        self.assertEqual(health_banner(self._fresh_report()), "")

    def test_a_dead_scheduler_forces_a_banner_despite_fresh_state(self):
        from src.maintenance_health import health_banner

        jobs = parse_launchctl_list(_BROKEN, "com.ollie.options")
        banner = health_banner(self._fresh_report(), launchd_jobs=jobs)
        self.assertNotEqual(banner, "")
        self.assertIn("Login Items", banner)

    def test_healthy_schedulers_do_not_add_noise(self):
        from src.maintenance_health import health_banner

        jobs = parse_launchctl_list(_HEALTHY, "com.ollie.options")
        self.assertEqual(health_banner(self._fresh_report(), launchd_jobs=jobs), "")


if __name__ == "__main__":
    unittest.main()
