"""Amendment-history parsing against a real recorded payload."""
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst import design

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "catalyst",
                       "ctgov_history.json")


def payload():
    with open(FIXTURE) as f:
        return json.load(f)


class TestParseHistory(unittest.TestCase):
    def setUp(self):
        self.a = design.parse_history(payload())

    def test_counts_versions(self):
        self.assertEqual(self.a.versions, 11)

    def test_reads_outcomes_update_count(self):
        self.assertEqual(self.a.outcomes_updated, 3)

    def test_latest_status(self):
        self.assertEqual(self.a.status_now, "ACTIVE_NOT_RECRUITING")

    def test_available_when_parsed(self):
        self.assertTrue(self.a.available)

    def test_flags_repeated_outcome_edits(self):
        self.assertIn("outcome measures edited 3x", self.a.flags)

    def test_does_not_flag_a_single_outcome_edit(self):
        p = payload()
        p["outcomesUpdateCount"] = 1
        self.assertEqual(
            [f for f in design.parse_history(p).flags if "outcome" in f], [])

    def test_empty_payload_is_unavailable_not_zero(self):
        a = design.parse_history({})
        self.assertFalse(a.available)
        self.assertEqual(a.versions, 0)


class TestAmendmentsFor(unittest.TestCase):
    def test_network_failure_is_unavailable_not_raise(self):
        with mock.patch.object(design, "fetch_history", return_value=None):
            a = design.amendments_for("NCT06510816")
        self.assertFalse(a.available)
        self.assertEqual(a.flags, ())

    def test_uses_fetched_payload(self):
        with mock.patch.object(design, "fetch_history", return_value=payload()):
            self.assertEqual(design.amendments_for("NCT06510816").versions, 11)


class TestFetchHistory(unittest.TestCase):
    def test_returns_none_on_error(self):
        with mock.patch.object(design, "_get_json", side_effect=OSError("404")):
            self.assertIsNone(design.fetch_history("NCT06510816"))


if __name__ == "__main__":
    unittest.main()
