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
    LAUNCHD_LABEL_PREFIXES,
    launchd_failure_message,
    parse_launchctl_list,
)

# The labels actually installed on the operator's machine, which predate the
# repo's own `com.options-screener.*` plist. Detection that matches only the
# shipped prefix returns [] here, and an empty list reads as "healthy".
_BROKEN_LEGACY_LABELS = """PID\tStatus\tLabel
-\t78\tcom.ollie.options.crypto-enforce-exits
-\t78\tcom.ollie.options.maintenance
-\t78\tcom.ollie.options.crypto-auto-log
-\t0\tcom.apple.something
"""

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

    def test_the_default_prefixes_match_the_labels_actually_installed(self):
        # The regression this guards: with a single shipped prefix, the three
        # agents that have been dead since 2026-06-15 parse to [], and every
        # consumer downstream reads [] as "nothing failing".
        jobs = parse_launchctl_list(_BROKEN_LEGACY_LABELS, LAUNCHD_LABEL_PREFIXES)
        self.assertEqual(len(jobs), 3)
        self.assertTrue(all(j.failed for j in jobs))

    def test_the_default_prefixes_still_match_the_shipped_plist(self):
        jobs = parse_launchctl_list(_BROKEN, LAUNCHD_LABEL_PREFIXES)
        self.assertEqual(len(jobs), 3)

    def test_a_single_prefix_string_is_still_accepted(self):
        jobs = parse_launchctl_list(_HEALTHY, "com.options-screener")
        self.assertEqual(len(jobs), 2)

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


class DeadSchedulerDurationTest(unittest.TestCase):
    """`launchctl list` carries no timestamps, so "how long has the scheduler
    been dead" is tracked by stamping a marker into the maintenance state the
    first time it's observed dead, then diffing later reads against it."""

    def setUp(self):
        from datetime import date

        self.today = date(2026, 7, 31)
        self.dead_jobs = parse_launchctl_list(_BROKEN, "com.options-screener")
        self.healthy_jobs = parse_launchctl_list(_HEALTHY, "com.options-screener")

    def test_healthy_scheduler_has_no_dead_days(self):
        from src.maintenance_health import launchd_dead_days

        self.assertIsNone(launchd_dead_days(self.healthy_jobs, {}, self.today))

    def test_no_jobs_at_all_has_no_dead_days(self):
        # launchctl unavailable, or nothing matched the prefix — must fail
        # open (never the reason a run blocks), same as read_launchd_status.
        from src.maintenance_health import launchd_dead_days

        self.assertIsNone(launchd_dead_days([], {}, self.today))

    def test_first_observation_is_zero_days(self):
        from src.maintenance_health import launchd_dead_days

        self.assertEqual(launchd_dead_days(self.dead_jobs, {}, self.today), 0)

    def test_days_dead_is_measured_from_the_stamped_marker(self):
        from datetime import date

        from src.maintenance_health import launchd_dead_days

        state = {"launchd_dead_since": "2026-07-20"}
        self.assertEqual(
            launchd_dead_days(self.dead_jobs, state, date(2026, 7, 31)), 11)

    def test_next_state_stamps_the_marker_on_first_dead_observation(self):
        from src.maintenance_health import next_launchd_dead_state

        new_state = next_launchd_dead_state(self.dead_jobs, {}, self.today)
        self.assertEqual(new_state["launchd_dead_since"], "2026-07-31")

    def test_next_state_does_not_restamp_an_existing_marker(self):
        from src.maintenance_health import next_launchd_dead_state

        state = {"launchd_dead_since": "2026-07-20"}
        new_state = next_launchd_dead_state(self.dead_jobs, state, self.today)
        self.assertEqual(new_state["launchd_dead_since"], "2026-07-20")

    def test_next_state_clears_the_marker_once_healthy_again(self):
        from src.maintenance_health import next_launchd_dead_state

        state = {"launchd_dead_since": "2026-07-20"}
        new_state = next_launchd_dead_state(self.healthy_jobs, state, self.today)
        self.assertNotIn("launchd_dead_since", new_state)

    def test_next_state_never_mutates_the_input_dict(self):
        from src.maintenance_health import next_launchd_dead_state

        state = {"launchd_dead_since": "2026-07-20"}
        next_launchd_dead_state(self.healthy_jobs, state, self.today)
        self.assertEqual(state, {"launchd_dead_since": "2026-07-20"})


class SeedDeadSinceFromLogTest(unittest.TestCase):
    """The marker must not be stamped from "today" when better evidence
    exists: `logs/launchagent.log` is written only by the LaunchAgents, so
    its last entry is a sound lower bound on when the scheduler was last
    genuinely alive — unlike the maintenance state file, which the
    interactive path also stamps."""

    def setUp(self):
        import tempfile
        from datetime import date

        self.today = date(2026, 7, 31)
        self.dead_jobs = parse_launchctl_list(_BROKEN, "com.options-screener")
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def _log_path(self, name="launchagent.log"):
        return os.path.join(self._tmpdir.name, name)

    def test_seeds_from_the_last_parsed_log_line(self):
        from src.maintenance_health import seed_dead_since_date

        path = self._log_path()
        with open(path, "w") as f:
            f.write("[2026-06-10 20:07:36] maintenance: auto-log:ics\n")
            f.write("  Forward cohort: 3/50 closed clean\n")
            f.write("[2026-06-15 12:24:10] maintenance: auto-log:sps\n")
            f.write("  Forward cohort: 4/50 closed clean\n")
        self.assertEqual(seed_dead_since_date(path), "2026-06-15")

    def test_falls_back_to_mtime_when_no_line_parses(self):
        from datetime import datetime

        from src.maintenance_health import seed_dead_since_date

        path = self._log_path()
        with open(path, "w") as f:
            f.write("not a timestamped line\n")
        # Noon UTC-ish on a fixed day, away from any local-timezone midnight
        # boundary, so the assertion doesn't depend on the test machine's tz.
        fixed_ts = 1750000000  # 2025-06-15 ~08:26 UTC
        os.utime(path, (fixed_ts, fixed_ts))
        expected = datetime.fromtimestamp(fixed_ts).date().isoformat()
        self.assertEqual(seed_dead_since_date(path), expected)

    def test_returns_none_when_the_log_does_not_exist(self):
        from src.maintenance_health import seed_dead_since_date

        self.assertIsNone(seed_dead_since_date(self._log_path("nope.log")))

    def test_dead_no_marker_and_a_45_day_old_log_seeds_to_that_date_and_fires_immediately(self):
        # Acceptance case 1: on the machine this was built for, the scheduler
        # has genuinely been dead for weeks before this code ever ran. The
        # very first observation must reflect that, not reset the clock to
        # "just noticed today".
        from datetime import date, timedelta

        from src.maintenance_health import (launchd_dead_days, next_launchd_dead_state,
                                            seed_dead_since_date)

        old_date = (self.today - timedelta(days=45)).isoformat()
        path = self._log_path()
        with open(path, "w") as f:
            f.write(f"[{old_date} 12:00:00] maintenance: auto-log:ds\n")

        seed = seed_dead_since_date(path)
        self.assertEqual(seed, old_date)

        new_state = next_launchd_dead_state(self.dead_jobs, {}, self.today, seed_date=seed)
        self.assertEqual(new_state["launchd_dead_since"], old_date)

        dead_days = launchd_dead_days(self.dead_jobs, new_state, self.today)
        self.assertEqual(dead_days, 45)
        self.assertGreater(dead_days, 7)  # the ack threshold — must fire now

    def test_dead_no_marker_and_no_log_seeds_to_today_and_does_not_fire(self):
        # Acceptance case 2: a fresh clone with no LaunchAgent log at all has
        # no evidence to seed from — must fall back to today, not block on
        # its very first run.
        from src.maintenance_health import (launchd_dead_days, next_launchd_dead_state,
                                            seed_dead_since_date)

        seed = seed_dead_since_date(self._log_path("does-not-exist.log"))
        self.assertIsNone(seed)

        new_state = next_launchd_dead_state(self.dead_jobs, {}, self.today, seed_date=seed)
        self.assertEqual(new_state["launchd_dead_since"], self.today.isoformat())

        dead_days = launchd_dead_days(self.dead_jobs, new_state, self.today)
        self.assertEqual(dead_days, 0)
        self.assertLessEqual(dead_days, 7)  # must not fire yet

    def test_existing_marker_is_never_overwritten_by_seeding(self):
        # Acceptance case 3: seeding only applies to a *first* observation.
        # A marker already on record — however it got there — must win over
        # whatever the log says, or a healthy log entry could rewind an
        # already-correct, already-longer-running marker.
        from datetime import timedelta

        from src.maintenance_health import next_launchd_dead_state, seed_dead_since_date

        existing = (self.today - timedelta(days=20)).isoformat()
        state = {"launchd_dead_since": existing}
        log_says = (self.today - timedelta(days=45)).isoformat()
        path = self._log_path()
        with open(path, "w") as f:
            f.write(f"[{log_says} 12:00:00] maintenance: auto-log:ds\n")

        new_state = next_launchd_dead_state(
            self.dead_jobs, state, self.today, seed_date=seed_dead_since_date(path))
        self.assertEqual(new_state["launchd_dead_since"], existing)


class DeadSchedulerAckBannerTest(unittest.TestCase):
    def setUp(self):
        from src import formatting as fmt

        self._saved = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = False

    def tearDown(self):
        from src import formatting as fmt

        fmt._COLOR_ENABLED = self._saved

    def test_states_what_is_degrading_in_plain_terms(self):
        from src.maintenance_health import dead_scheduler_ack_banner

        banner = dead_scheduler_ack_banner(11, width=100)
        flat = " ".join(ln.strip("│ ") for ln in banner.splitlines())
        self.assertIn("manual cadence", flat)
        self.assertIn("manual-only", flat)
        self.assertIn("11", flat)

    def test_every_line_fits_the_box(self):
        from src.maintenance_health import dead_scheduler_ack_banner

        banner = dead_scheduler_ack_banner(30, width=100)
        widths = {len(ln) for ln in banner.splitlines()}
        self.assertEqual(widths, {100})


if __name__ == "__main__":
    unittest.main()
