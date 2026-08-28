"""The ticker->CIK map is re-read from disk on every lookup.

`cik_for` calls `_ticker_map()`, which opens and json.loads a ~10k-entry file
each time. Profiled 2026-08-28 on the catalyst backtest: 2,234 lookups cost
3.5s — 24% of a 14.7s run — and every one of them parsed the same file.

The disk cache already has a 30-day TTL. This adds an in-process memo on top,
keyed by the file's mtime so a refreshed cache is picked up rather than
shadowed: the memo is a speed layer, never a second, longer expiry.

No test here touches the network.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.insider import edgar


class TickerMapCase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = os.path.join(self._dir.name, "tickers.json")
        with open(self.path, "w") as fh:
            json.dump({"AAPL": 320193, "PFE": 78003}, fh)
        patcher = mock.patch.object(edgar, "TICKER_CACHE", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)
        edgar.reset_ticker_map()
        self.addCleanup(edgar.reset_ticker_map)


class TestTheMapIsReadOnce(TickerMapCase):
    def test_repeated_lookups_parse_the_file_once(self):
        real_load = json.load
        with mock.patch("json.load", side_effect=real_load) as load:
            for _ in range(50):
                edgar.cik_for("AAPL")
        self.assertEqual(load.call_count, 1)

    def test_the_lookup_still_returns_the_right_cik(self):
        self.assertEqual(edgar.cik_for("AAPL"), 320193)
        self.assertEqual(edgar.cik_for("pfe"), 78003)
        self.assertIsNone(edgar.cik_for("NOPE"))
        self.assertIsNone(edgar.cik_for(""))


class TestTheMemoDoesNotOutliveTheFile(TickerMapCase):
    def test_a_rewritten_cache_is_picked_up(self):
        # The memo must never become a second, longer TTL sitting on top of
        # the disk cache's own 30-day expiry.
        self.assertEqual(edgar.cik_for("AAPL"), 320193)
        with open(self.path, "w") as fh:
            json.dump({"AAPL": 999999}, fh)
        # A DIFFERENT but still-fresh mtime. Setting it to epoch 0 would trip
        # the disk cache's own 30-day TTL and test the wrong thing.
        import time
        past = time.time() - 3600
        os.utime(self.path, (past, past))
        self.assertEqual(edgar.cik_for("AAPL"), 999999)

    def test_an_empty_map_is_not_memoized(self):
        # `_ticker_map` returns {} when it cannot read OR fetch. Memoizing
        # that would make one transient failure poison every later lookup in
        # the process — the defect this codebase keeps paying for.
        os.remove(self.path)
        with mock.patch.object(edgar, "_get", side_effect=RuntimeError("down")):
            self.assertEqual(edgar._ticker_map(), {})
            self.assertEqual(edgar._ticker_map(), {})
        with open(self.path, "w") as fh:
            json.dump({"AAPL": 320193}, fh)
        self.assertEqual(edgar.cik_for("AAPL"), 320193)


if __name__ == "__main__":
    unittest.main()
