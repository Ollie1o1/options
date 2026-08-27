"""Point-in-time trial reconstruction. The lookahead guard is the point."""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst import pit, pit_cache

FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures", "catalyst", "pit")


def versions():
    with open(os.path.join(FIX, "versions.json")) as f:
        return json.load(f)["changes"]


def study_v3():
    with open(os.path.join(FIX, "study_v3.json")) as f:
        return json.load(f)


class TestVersionAt(unittest.TestCase):
    def test_picks_the_latest_version_at_or_before_as_of(self):
        self.assertEqual(pit.version_at(versions(), "2024-12-15"), 3)

    def test_boundary_date_is_included(self):
        self.assertEqual(pit.version_at(versions(), "2024-12-02"), 3)

    def test_day_before_a_version_takes_the_previous_one(self):
        self.assertEqual(pit.version_at(versions(), "2024-12-01"), 2)

    def test_before_the_first_version_is_none(self):
        # The trial did not exist yet. None, never version 0.
        self.assertIsNone(pit.version_at(versions(), "2024-01-01"))

    def test_after_the_last_version_takes_the_last(self):
        self.assertEqual(pit.version_at(versions(), "2030-01-01"), 4)

    def test_empty_versions_is_none(self):
        self.assertIsNone(pit.version_at([], "2024-12-15"))


class TrialCase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.conn = pit_cache.connect(os.path.join(self._dir.name, "pit.db"))
        self.addCleanup(self.conn.close)
        pit_cache.put_versions(self.conn, "NCT06510816", versions())
        pit_cache.put_study(self.conn, "NCT06510816", 3, study_v3())


class TestTrialAsOf(TrialCase):
    def test_reconstructs_from_the_cached_version(self):
        t = pit.trial_as_of("NCT06510816", "2024-12-15", self.conn)
        self.assertIsNotNone(t)
        self.assertEqual(t.nct_id, "NCT06510816")
        self.assertEqual(t.event_date, "2026-10-31")
        self.assertEqual(t.phase, "PHASE3")
        self.assertEqual(t.sponsor_name, "Annexon, Inc.")

    def test_never_fetches_a_version_dated_after_as_of(self):
        # THE lookahead guard. v4 exists (2025-02-07) but must be untouchable
        # from a 2024-12-15 vantage point.
        with mock.patch.object(pit, "_fetch_study") as f:
            pit.trial_as_of("NCT06510816", "2024-12-15", self.conn)
        for call in f.call_args_list:
            self.assertNotEqual(call.args[1], 4)

    def test_returns_none_before_the_trial_existed(self):
        self.assertIsNone(pit.trial_as_of("NCT06510816", "2024-01-01", self.conn))

    def test_uses_the_cache_and_does_not_hit_the_network(self):
        with mock.patch.object(pit, "_fetch_study",
                               side_effect=AssertionError("network!")):
            t = pit.trial_as_of("NCT06510816", "2024-12-15", self.conn)
        self.assertIsNotNone(t)

    def test_fetch_failure_yields_none_not_raise(self):
        with mock.patch.object(pit, "_fetch_versions", return_value=None):
            self.assertIsNone(pit.trial_as_of("NCT_UNKNOWN", "2024-12-15",
                                              self.conn))

    def test_a_missing_study_payload_is_none(self):
        pit_cache.put_versions(self.conn, "NCT_X", versions())
        with mock.patch.object(pit, "_fetch_study", return_value=None):
            self.assertIsNone(pit.trial_as_of("NCT_X", "2024-12-15", self.conn))


if __name__ == "__main__":
    unittest.main()


class TestAmendmentsAsOf(unittest.TestCase):
    """Amendment history as it stood on a date, not as it stands today.

    `panel._amendments` called the LIVE `design.amendments_for`, which counts
    every change ever recorded. An endpoint amended in 2025 therefore marked a
    row "amended" at the 2023 vintage — lookahead, in the one feature H2 is
    about. The dated version lists were already in the cache the whole time.

    The outcome-edit definition is validated, not invented: counting versions
    whose moduleLabels contain "Outcome Measures" reproduced the live
    `outcomesUpdateCount` on 12 of 12 trials checked 2026-08-27, and
    "Outcome Measures (Results)" is excluded there too — posting results is
    not amending an endpoint.
    """

    VERSIONS = [
        {"version": 0, "date": "2023-01-10", "status": "RECRUITING",
         "moduleLabels": ["Study Status"]},
        {"version": 1, "date": "2023-06-01", "status": "RECRUITING",
         "moduleLabels": ["Outcome Measures", "Study Design"]},
        {"version": 2, "date": "2024-03-01", "status": "RECRUITING",
         "moduleLabels": ["Outcome Measures"]},
        {"version": 3, "date": "2025-02-01", "status": "COMPLETED",
         "moduleLabels": ["Outcome Measures (Results)"]},
    ]

    def test_it_counts_only_versions_on_or_before_the_date(self):
        a = pit.amendments_as_of(self.VERSIONS, "2023-12-31")
        self.assertTrue(a.available)
        self.assertEqual(a.versions, 2)

    def test_a_later_endpoint_edit_is_invisible_at_an_earlier_vintage(self):
        # THE bug: the 2024 outcome edit must not exist on 2023-12-31.
        early = pit.amendments_as_of(self.VERSIONS, "2023-12-31")
        late = pit.amendments_as_of(self.VERSIONS, "2024-12-31")
        self.assertEqual(early.outcomes_updated, 1)
        self.assertEqual(late.outcomes_updated, 2)

    def test_results_section_updates_are_not_endpoint_amendments(self):
        # "Outcome Measures (Results)" is posting results, not amending.
        a = pit.amendments_as_of(self.VERSIONS, "2025-12-31")
        self.assertEqual(a.outcomes_updated, 2)

    def test_a_trial_not_yet_registered_is_unavailable_not_zero(self):
        # "we could not look" and "nothing changed" are different answers.
        a = pit.amendments_as_of(self.VERSIONS, "2022-01-01")
        self.assertFalse(a.available)
        self.assertEqual(a.versions, 0)

    def test_no_version_list_is_unavailable(self):
        self.assertFalse(pit.amendments_as_of(None, "2024-01-01").available)
        self.assertFalse(pit.amendments_as_of([], "2024-01-01").available)

    def test_status_is_the_status_at_the_date_not_today(self):
        a = pit.amendments_as_of(self.VERSIONS, "2024-06-01")
        self.assertEqual(a.status_now, "RECRUITING")
        later = pit.amendments_as_of(self.VERSIONS, "2025-06-01")
        self.assertEqual(later.status_now, "COMPLETED")

    def test_the_flag_threshold_uses_the_as_of_count(self):
        from src.catalyst.design import OUTCOME_EDIT_FLAG_THRESHOLD
        self.assertEqual(OUTCOME_EDIT_FLAG_THRESHOLD, 2)
        self.assertEqual(pit.amendments_as_of(self.VERSIONS, "2023-12-31").flags, ())
        self.assertTrue(pit.amendments_as_of(self.VERSIONS, "2024-12-31").flags)

    def test_at_a_far_future_date_it_matches_the_live_parser(self):
        # Same payload, same totals — the point-in-time version is a
        # restriction of the live one, not a different statistic.
        from src.catalyst.design import parse_history
        live = parse_history({"changes": self.VERSIONS, "outcomesUpdateCount": 2})
        pit_view = pit.amendments_as_of(self.VERSIONS, "2099-01-01")
        self.assertEqual(pit_view.versions, live.versions)
        self.assertEqual(pit_view.outcomes_updated, live.outcomes_updated)
