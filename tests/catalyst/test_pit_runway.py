"""Point-in-time financials. A figure must be invisible before it was filed."""
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst import pit, pit_cache

# Real ANNX shape: the period ending 2025-12-31 was not filed until 2026-08-12.
FACTS = {"facts": {"us-gaap": {
    "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [
        {"end": "2025-06-30", "filed": "2025-08-10", "val": 150_000_000.0},
        {"end": "2025-12-31", "filed": "2026-08-12", "val": 400_000_000.0},
    ]}},
    "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
        {"start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-20",
         "val": -120_000_000.0},
        {"start": "2025-01-01", "end": "2025-12-31", "filed": "2026-08-12",
         "val": -200_000_000.0},
    ]}},
}}}


class TestFactsAsOf(unittest.TestCase):
    def test_drops_points_filed_after_as_of(self):
        got = pit.facts_as_of(FACTS, "2026-01-01")
        cash = got["facts"]["us-gaap"][
            "CashAndCashEquivalentsAtCarryingValue"]["units"]["USD"]
        self.assertEqual([p["end"] for p in cash], ["2025-06-30"])

    def test_keeps_points_filed_on_the_boundary_date(self):
        got = pit.facts_as_of(FACTS, "2025-08-10")
        cash = got["facts"]["us-gaap"][
            "CashAndCashEquivalentsAtCarryingValue"]["units"]["USD"]
        self.assertEqual(len(cash), 1)

    def test_a_point_with_no_filed_date_is_dropped(self):
        facts = {"facts": {"us-gaap": {"X": {"units": {"USD": [
            {"end": "2025-06-30", "val": 1.0}]}}}}}
        got = pit.facts_as_of(facts, "2026-01-01")
        self.assertEqual(got["facts"]["us-gaap"]["X"]["units"]["USD"], [])

    def test_original_payload_is_not_mutated(self):
        before = len(FACTS["facts"]["us-gaap"][
            "CashAndCashEquivalentsAtCarryingValue"]["units"]["USD"])
        pit.facts_as_of(FACTS, "2026-01-01")
        after = len(FACTS["facts"]["us-gaap"][
            "CashAndCashEquivalentsAtCarryingValue"]["units"]["USD"])
        self.assertEqual(before, after)


class TestRunwayAsOf(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.conn = pit_cache.connect(os.path.join(self._dir.name, "pit.db"))
        self.addCleanup(self.conn.close)
        pit_cache.put_facts(self.conn, 111, FACTS)

    def test_uses_only_what_was_filed(self):
        r = pit.runway_as_of(111, "2026-01-01", "2026-06-30", self.conn)
        self.assertEqual(r.cash, 150_000_000.0)   # not the 400m filed later

    def test_a_later_vantage_sees_the_newer_filing(self):
        r = pit.runway_as_of(111, "2026-08-20", "2026-06-30", self.conn)
        self.assertEqual(r.cash, 400_000_000.0)

    def test_no_filed_data_yet_yields_an_empty_runway(self):
        r = pit.runway_as_of(111, "2024-01-01", "2026-06-30", self.conn)
        self.assertIsNone(r.cash)

    def test_missing_cik_is_empty_not_raise(self):
        with mock.patch.object(pit, "_fetch_facts", return_value=None):
            self.assertIsNone(
                pit.runway_as_of(999, "2026-01-01", "2026-06-30", self.conn).cash)


if __name__ == "__main__":
    unittest.main()
