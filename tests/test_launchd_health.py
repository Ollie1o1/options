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
-\t0\tcom.options-screener.maintenance
1234\t0\tcom.options-screener.crypto-auto-log
-\t0\tcom.apple.something
"""

_BROKEN = """PID\tStatus\tLabel
-\t78\tcom.options-screener.crypto-enforce-exits
-\t78\tcom.options-screener.maintenance
-\t78\tcom.options-screener.crypto-auto-log
-\t0\tcom.apple.something
"""


class TestParsing(unittest.TestCase):
    def test_finds_our_jobs_and_ignores_everything_else(self):
        jobs = parse_launchctl_list(_HEALTHY, prefix="com.options-screener")
        self.assertEqual(len(jobs), 2)
        self.assertTrue(all(j.label.startswith("com.options-screener") for j in jobs))

    def test_reads_the_last_exit_status(self):
        jobs = {j.label: j for j in parse_launchctl_list(_BROKEN, "com.options-screener")}
        self.assertEqual(jobs["com.options-screener.maintenance"].last_exit_status, 78)

    def test_a_zero_status_is_not_a_failure(self):
        jobs = parse_launchctl_list(_HEALTHY, "com.options-screener")
        self.assertEqual([j for j in jobs if j.failed], [])

    def test_a_nonzero_status_is_a_failure(self):
        jobs = parse_launchctl_list(_BROKEN, "com.options-screener")
        self.assertEqual(len([j for j in jobs if j.failed]), 3)

    def test_header_and_blank_lines_are_skipped(self):
        self.assertEqual(parse_launchctl_list("PID\tStatus\tLabel\n\n", "com.options-screener"), [])

    def test_unparseable_output_yields_nothing_rather_than_raising(self):
        self.assertEqual(parse_launchctl_list("something unexpected", "com.options-screener"), [])

    def test_a_running_job_with_a_pid_is_not_a_failure(self):
        jobs = parse_launchctl_list("PID\tStatus\tLabel\n900\t0\tcom.options-screener.x\n",
                                    "com.options-screener")
        self.assertFalse(jobs[0].failed)


class TestMessage(unittest.TestCase):
    def test_no_message_when_everything_is_healthy(self):
        self.assertIsNone(
            launchd_failure_message(parse_launchctl_list(_HEALTHY, "com.options-screener"))
        )

    def test_names_the_failing_jobs(self):
        msg = launchd_failure_message(parse_launchctl_list(_BROKEN, "com.options-screener"))
        self.assertIn("3", msg)
        self.assertIn("maintenance", msg)

    def test_exit_78_explains_the_actual_fix(self):
        # EX_CONFIG here means macOS is refusing to run the agent, and the fix is
        # a Login Items toggle the user must click. Saying "job failed" sends the
        # reader looking for a bug in the script instead.
        msg = launchd_failure_message(parse_launchctl_list(_BROKEN, "com.options-screener"))
        self.assertIn("Login Items", msg)

    def test_other_exit_codes_do_not_claim_it_is_a_permissions_problem(self):
        jobs = parse_launchctl_list("PID\tStatus\tLabel\n-\t1\tcom.options-screener.x\n",
                                    "com.options-screener")
        self.assertNotIn("Login Items", launchd_failure_message(jobs))


def _fresh_report():
    """A report where every job is current — nothing to warn about on its own."""
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


class TestBannerSurfacing(unittest.TestCase):
    """A dead scheduler must show even when the state file looks fresh — that
    combination is the whole bug, since opening the screener by hand stamps the
    state file and hides six weeks of nothing running."""

    def test_a_fresh_report_alone_produces_no_banner(self):
        from src.maintenance_health import health_banner

        self.assertEqual(health_banner(_fresh_report()), "")

    def test_a_dead_scheduler_forces_a_banner_despite_fresh_state(self):
        from src.maintenance_health import health_banner

        jobs = parse_launchctl_list(_BROKEN, "com.options-screener")
        banner = health_banner(_fresh_report(), launchd_jobs=jobs)
        self.assertNotEqual(banner, "")
        self.assertIn("Login Items", banner)

    def test_healthy_schedulers_do_not_add_noise(self):
        from src.maintenance_health import health_banner

        jobs = parse_launchctl_list(_HEALTHY, "com.options-screener")
        self.assertEqual(health_banner(_fresh_report(), launchd_jobs=jobs), "")


class TestBannerFitsItsBox(unittest.TestCase):
    """The launchd line is the longest string the banner can emit — three job
    names plus the exit-78 remedy runs ~280 chars. ui.card pads but never wraps,
    so an unwrapped line pushes the right border off screen on every startup."""

    def setUp(self):
        from src import formatting as fmt

        self._saved = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = False  # plain text, so len() is the visible width

    def tearDown(self):
        from src import formatting as fmt

        fmt._COLOR_ENABLED = self._saved

    def _widths(self, banner):
        return {len(ln) for ln in banner.splitlines()}

    def test_every_line_of_a_dead_scheduler_banner_is_one_width(self):
        from src.maintenance_health import health_banner

        jobs = parse_launchctl_list(_BROKEN, "com.options-screener")
        banner = health_banner(_fresh_report(), width=100, launchd_jobs=jobs)
        self.assertEqual(self._widths(banner), {100})

    def test_it_holds_when_stale_jobs_and_a_dead_scheduler_stack(self):
        # Both branches of health_banner emit launchd_msg; this is the other one.
        from datetime import date

        from src.maintenance_health import WORKING_WINDOWS, compute_health, health_banner

        stale = "2026-06-01"
        state = {
            "last_autolog": {w: stale for w in WORKING_WINDOWS},
            "last_checkpoint": stale,
            "last_track_record": stale,
            "last_chain_archive": stale,
            "last_morning_briefing": stale,
        }
        report = compute_health(state, date(2026, 7, 29))
        jobs = parse_launchctl_list(_BROKEN, "com.options-screener")
        banner = health_banner(report, width=100, launchd_jobs=jobs)
        self.assertEqual(self._widths(banner), {100})

    def test_the_remedy_survives_wrapping(self):
        from src.maintenance_health import health_banner

        jobs = parse_launchctl_list(_BROKEN, "com.options-screener")
        banner = health_banner(_fresh_report(), width=100, launchd_jobs=jobs)
        # Wrapping must not shred the one instruction that fixes it.
        flat = " ".join(ln.strip("│ ") for ln in banner.splitlines())
        self.assertIn("Login Items", flat)
        self.assertIn("Allow in the Background", flat)


if __name__ == "__main__":
    unittest.main()
