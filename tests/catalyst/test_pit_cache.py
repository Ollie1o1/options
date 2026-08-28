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


class TestUniverse(CacheCase):
    """The swept trial list, pinned so two runs share one universe.

    The sweep is the only remaining live input to the study's population:
    ClinicalTrials.gov gains and edits trials, and `universe.market_caps` reads
    TODAY'S cap to apply the band. Between the 2026-08-26 and 2026-08-27 runs
    that moved H3's arms (755 -> 736) with no code change. A study whose
    population shifts under it cannot be compared with itself.

    Unlike `pit_versions` this is NOT immutable and NOT append-safe — it is a
    deliberate freeze, refreshed only when asked.
    """

    def test_miss_returns_none_not_empty_list(self):
        # None is "never swept"; [] would be "swept, matched nothing".
        self.assertIsNone(pit_cache.get_universe(self.conn, "k"))

    def test_roundtrip_carries_the_sweep_date(self):
        # Without the date a pinned universe cannot be aged or audited.
        pit_cache.put_universe(self.conn, "k", "2026-08-27", ["NCT1", "NCT2"])
        got = pit_cache.get_universe(self.conn, "k")
        self.assertEqual(got, ("2026-08-27", ["NCT1", "NCT2"]))

    def test_an_empty_sweep_is_stored_and_read_back_as_empty(self):
        pit_cache.put_universe(self.conn, "k", "2026-08-27", [])
        self.assertEqual(pit_cache.get_universe(self.conn, "k"), ("2026-08-27", []))

    def test_different_keys_do_not_collide(self):
        pit_cache.put_universe(self.conn, "a", "2026-08-27", ["NCT1"])
        pit_cache.put_universe(self.conn, "b", "2026-08-27", ["NCT2"])
        self.assertEqual(pit_cache.get_universe(self.conn, "a")[1], ["NCT1"])
        self.assertEqual(pit_cache.get_universe(self.conn, "b")[1], ["NCT2"])

    def test_a_refresh_overwrites_in_place(self):
        pit_cache.put_universe(self.conn, "k", "2026-08-27", ["NCT1"])
        pit_cache.put_universe(self.conn, "k", "2026-09-01", ["NCT1", "NCT2"])
        self.assertEqual(pit_cache.get_universe(self.conn, "k"),
                         ("2026-09-01", ["NCT1", "NCT2"]))


class TestFreshness(CacheCase):
    """One rule, derived from each source's own semantics, not a guessed TTL.

    An entry is valid for a query whose ``as_of`` is at or before the entry's
    fetch time. That is exactly right for an APPEND-ONLY source: a list
    fetched on the 27th can answer "what was true on the 25th" forever, and
    can never answer "what is true on the 28th".

    Nothing recorded a fetch time before this, so nothing COULD detect
    staleness — the cache was not yet wrong (it was two days old) but was
    structurally unable to notice when it became wrong.
    """

    def test_versions_serve_a_past_as_of(self):
        pit_cache.put_versions(self.conn, "NCT1", [{"version": 0}],
                               fetched_at="2026-08-27")
        self.assertIsNotNone(
            pit_cache.get_versions(self.conn, "NCT1", as_of="2026-08-20"))

    def test_versions_serve_an_as_of_equal_to_the_fetch_date(self):
        pit_cache.put_versions(self.conn, "NCT1", [{"version": 0}],
                               fetched_at="2026-08-27")
        self.assertIsNotNone(
            pit_cache.get_versions(self.conn, "NCT1", as_of="2026-08-27"))

    def test_versions_REFUSE_a_future_as_of(self):
        # The list GROWS. An entry fetched on the 27th cannot know what was
        # amended on the 28th, so this must read as a MISS and be refetched.
        pit_cache.put_versions(self.conn, "NCT1", [{"version": 0}],
                               fetched_at="2026-08-27")
        self.assertIsNone(
            pit_cache.get_versions(self.conn, "NCT1", as_of="2026-08-28"))

    def test_no_as_of_means_no_freshness_constraint(self):
        pit_cache.put_versions(self.conn, "NCT1", [{"version": 0}],
                               fetched_at="2026-08-27")
        self.assertIsNotNone(pit_cache.get_versions(self.conn, "NCT1"))

    def test_a_legacy_row_without_a_fetch_date_uses_the_conservative_floor(self):
        # Rows written before this column existed. Their true fetch time is
        # unknown, so they are treated as fetched on the day the cache file
        # was created — the earliest they COULD have been written.
        self.conn.execute(
            "INSERT INTO pit_versions (nct_id, payload) VALUES ('OLD','[]')")
        self.conn.commit()
        self.assertIsNotNone(pit_cache.get_versions(
            self.conn, "OLD", as_of=pit_cache.LEGACY_FETCHED_AT))
        self.assertIsNone(pit_cache.get_versions(
            self.conn, "OLD", as_of="2099-01-01"))

    def test_facts_follow_the_same_rule(self):
        pit_cache.put_facts(self.conn, 1, {"facts": {}}, fetched_at="2026-08-27")
        self.assertIsNotNone(pit_cache.get_facts(self.conn, 1, as_of="2026-08-01"))
        self.assertIsNone(pit_cache.get_facts(self.conn, 1, as_of="2026-08-28"))

    def test_an_immutable_study_version_has_no_freshness_question(self):
        # (nct_id, version) is a frozen historical record. It never expires,
        # and takes no as_of at all.
        pit_cache.put_study(self.conn, "NCT1", 3, {"study": {}})
        self.assertIsNotNone(pit_cache.get_study(self.conn, "NCT1", 3))


class TestPriceCache(CacheCase):
    """Prices are append-and-revise: new bars arrive, recent ones are revised.

    A CLOSED window (end before the fetch date) is safe indefinitely — a later
    split rescales both endpoints of a return equally, so the ratio is
    unchanged. An OPEN window (end at or after the fetch date) is only good
    for the day it was taken.
    """

    SERIES = {"2026-01-02": 10.0, "2026-01-03": 11.0}

    def test_a_closed_window_is_served_regardless_of_age(self):
        pit_cache.put_prices(self.conn, "ABC", "2025-01-01", "2025-12-31",
                             self.SERIES, fetched_at="2026-01-05")
        self.assertEqual(
            pit_cache.get_prices(self.conn, "ABC", "2025-01-01", "2025-12-31",
                                 today="2099-01-01"), self.SERIES)

    def test_an_open_window_is_served_only_on_the_day_it_was_taken(self):
        pit_cache.put_prices(self.conn, "ABC", "2025-01-01", "2026-08-27",
                             self.SERIES, fetched_at="2026-08-27")
        self.assertEqual(
            pit_cache.get_prices(self.conn, "ABC", "2025-01-01", "2026-08-27",
                                 today="2026-08-27"), self.SERIES)
        self.assertIsNone(
            pit_cache.get_prices(self.conn, "ABC", "2025-01-01", "2026-08-27",
                                 today="2026-08-28"))

    def test_a_different_window_is_a_different_entry(self):
        pit_cache.put_prices(self.conn, "ABC", "2025-01-01", "2025-12-31",
                             self.SERIES, fetched_at="2026-01-05")
        self.assertIsNone(
            pit_cache.get_prices(self.conn, "ABC", "2024-01-01", "2025-12-31",
                                 today="2026-01-05"))

    def test_a_miss_is_none(self):
        # None means "never looked". Callers must never store an empty series
        # fetched through a swallowing fetcher — see `_prices`.
        self.assertIsNone(pit_cache.get_prices(self.conn, "ZZZ", "a", "b",
                                               today="2026-08-27"))


if __name__ == "__main__":
    unittest.main()
