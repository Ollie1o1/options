"""Content cache for point-in-time reconstruction. Temp DB only."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst import pit_cache


class CacheCase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.conn = pit_cache.connect(os.path.join(self._dir.name, "pit.db"))
        self.addCleanup(self.conn.close)


class TestVersions(CacheCase):
    def test_miss_returns_none_not_empty_list(self):
        # None means "never fetched"; [] would mean "fetched, no versions".
        self.assertIsNone(pit_cache.get_versions(self.conn, "NCT1"))

    def test_roundtrip(self):
        v = [{"version": 0, "date": "2024-07-15"}]
        pit_cache.put_versions(self.conn, "NCT1", v)
        self.assertEqual(pit_cache.get_versions(self.conn, "NCT1"), v)

    def test_empty_list_is_distinguishable_from_a_miss(self):
        pit_cache.put_versions(self.conn, "NCT1", [])
        self.assertEqual(pit_cache.get_versions(self.conn, "NCT1"), [])


class TestStudy(CacheCase):
    def test_keyed_by_nct_and_version(self):
        pit_cache.put_study(self.conn, "NCT1", 3, {"a": 1})
        pit_cache.put_study(self.conn, "NCT1", 4, {"a": 2})
        self.assertEqual(pit_cache.get_study(self.conn, "NCT1", 3), {"a": 1})
        self.assertEqual(pit_cache.get_study(self.conn, "NCT1", 4), {"a": 2})

    def test_miss_is_none(self):
        self.assertIsNone(pit_cache.get_study(self.conn, "NCT1", 3))


class TestFacts(CacheCase):
    def test_roundtrip_and_overwrite(self):
        pit_cache.put_facts(self.conn, 123, {"facts": {"us-gaap": {}}})
        pit_cache.put_facts(self.conn, 123, {"facts": {"us-gaap": {"x": 1}}})
        got = pit_cache.get_facts(self.conn, 123)
        self.assertEqual(got["facts"]["us-gaap"], {"x": 1})

    def test_facts_are_append_safe_not_immutable(self):
        # companyfacts GROWS as filings arrive; overwriting must be allowed.
        pit_cache.put_facts(self.conn, 123, {"n": 1})
        pit_cache.put_facts(self.conn, 123, {"n": 2})
        n = self.conn.execute(
            "SELECT COUNT(*) FROM pit_facts WHERE cik=123").fetchone()[0]
        self.assertEqual(n, 1)


if __name__ == "__main__":
    unittest.main()
