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
