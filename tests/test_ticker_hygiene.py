"""Tests for src/ticker_hygiene.py — dead-ticker detection + watchlist cleanup (offline)."""
import json
import os
import tempfile
import unittest

from src import ticker_hygiene as th


# fake liveness: SQ / SPACEX / DEADCO are dead, everything else live
def _fake_live(t):
    return t.upper() not in {"SQ", "SPACEX", "DEADCO"}


class AuditTest(unittest.TestCase):
    def test_partitions_live_dead(self):
        a = th.audit_tickers(["SPY", "SQ", "aapl", "SPACEX"], live_fn=_fake_live)
        self.assertEqual(a["live"], ["AAPL", "SPY"])
        self.assertEqual(a["dead"], ["SPACEX", "SQ"])

    def test_dedupes_and_uppercases(self):
        a = th.audit_tickers(["spy", "SPY", " spy "], live_fn=_fake_live)
        self.assertEqual(a["live"], ["SPY"])

    def test_no_dead(self):
        a = th.audit_tickers(["SPY", "XYZ"], live_fn=_fake_live)
        self.assertEqual(a["dead"], [])


class WatchlistTest(unittest.TestCase):
    def _cfg(self, d):
        path = os.path.join(d, "config.json")
        json.dump({"watchlists": {
            "a": ["SPY", "SQ", "AAPL"],
            "b": ["SPACEX", "MSFT"],
            "c": ["NVDA"],
        }}, open(path, "w"))
        return path

    def test_watchlist_tickers_union(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = json.load(open(self._cfg(d)))
        self.assertEqual(th.watchlist_tickers(cfg),
                         ["AAPL", "MSFT", "NVDA", "SPACEX", "SPY", "SQ"])

    def test_dry_run_reports_but_does_not_write(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._cfg(d)
            res = th.clean_watchlists(path, dry_run=True, live_fn=_fake_live)
            self.assertEqual(set(res["dead"]), {"SQ", "SPACEX"})
            self.assertEqual(res["removed_by_list"], {})
            # file unchanged
            self.assertIn("SQ", json.load(open(path))["watchlists"]["a"])

    def test_clean_removes_dead_from_each_list(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._cfg(d)
            res = th.clean_watchlists(path, dry_run=False, live_fn=_fake_live)
            cfg = json.load(open(path))
        self.assertEqual(cfg["watchlists"]["a"], ["SPY", "AAPL"])     # SQ removed
        self.assertEqual(cfg["watchlists"]["b"], ["MSFT"])           # SPACEX removed
        self.assertEqual(cfg["watchlists"]["c"], ["NVDA"])           # untouched
        self.assertEqual(res["removed_by_list"]["a"], ["SQ"])

    def test_clean_no_write_when_all_live(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            json.dump({"watchlists": {"a": ["SPY", "AAPL"]}}, open(path, "w"))
            res = th.clean_watchlists(path, dry_run=False, live_fn=_fake_live)
        self.assertEqual(res["dead"], [])
        self.assertEqual(res["removed_by_list"], {})


if __name__ == "__main__":
    unittest.main()
