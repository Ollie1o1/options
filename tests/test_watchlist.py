"""Watchlist persistence — path anchoring and round-trip behaviour.

`src/watchlist.py` had no tests and a CWD-relative path. Both are fixed here.

The anchoring assertions test the module constant *directly* rather than a
path injected by the test. A test that always supplies its own path never
executes the default, which is the branch that ships — the same blind spot that
let two NameErrors reach production in this repo.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import watchlist as W


class TestWatchlistPathIsAnchored(unittest.TestCase):
    """The default path must name one file regardless of where you start.

    It was the bare string "watchlist.json". The launcher starts from the repo
    root so it worked in practice, but a scan started from anywhere else read
    an empty list — and because `save_watchlist` writes the whole list, the
    next add would have overwritten the real file with a single ticker.
    """

    def test_default_path_is_absolute(self):
        self.assertTrue(
            os.path.isabs(W._WATCHLIST_PATH),
            f"_WATCHLIST_PATH is relative ({W._WATCHLIST_PATH!r}) and so resolves "
            "against whatever directory the process happens to be in")

    def test_default_path_is_the_repo_root_file(self):
        expected = Path(__file__).resolve().parent.parent / "watchlist.json"
        self.assertEqual(Path(W._WATCHLIST_PATH), expected)

    def test_path_does_not_move_with_the_working_directory(self):
        before = W._WATCHLIST_PATH
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                # Re-import under the new CWD: the constant is computed at
                # import time, so this is the case a stale value would hide.
                import importlib
                reloaded = importlib.reload(W)
                self.assertEqual(reloaded._WATCHLIST_PATH, before)
            finally:
                os.chdir(cwd)
                importlib.reload(W)


class TestWatchlistRoundTrip(unittest.TestCase):
    """Load/save/add/remove against a temp file — never the real watchlist."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "watchlist.json")
        self._patch = patch.object(W, "_WATCHLIST_PATH", self.path)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_missing_file_is_an_empty_list_not_an_error(self):
        self.assertEqual(W.load_watchlist(), [])

    def test_round_trip(self):
        W.save_watchlist(["SPY", "AAPL"])
        self.assertEqual(W.load_watchlist(), ["SPY", "AAPL"])

    def test_tickers_come_back_uppercased(self):
        W.save_watchlist(["spy", "aapl"])
        self.assertEqual(W.load_watchlist(), ["SPY", "AAPL"])

    def test_non_string_entries_are_dropped(self):
        with open(self.path, "w") as fh:
            json.dump(["SPY", 42, None, {"x": 1}, "AAPL"], fh)
        self.assertEqual(W.load_watchlist(), ["SPY", "AAPL"])

    def test_corrupt_file_is_an_empty_list_not_a_crash(self):
        with open(self.path, "w") as fh:
            fh.write("{not json at all")
        self.assertEqual(W.load_watchlist(), [])

    def test_add_deduplicates_case_insensitively(self):
        W.add_to_watchlist("SPY")
        W.add_to_watchlist("spy")
        self.assertEqual(W.load_watchlist(), ["SPY"])

    def test_add_preserves_what_is_already_there(self):
        """The overwrite hazard the anchoring fix was about: save writes the
        whole list, so an add that started from an empty read would erase it."""
        W.save_watchlist(["SPY", "AAPL"])
        W.add_to_watchlist("MSFT")
        self.assertEqual(W.load_watchlist(), ["SPY", "AAPL", "MSFT"])

    def test_remove(self):
        W.save_watchlist(["SPY", "AAPL"])
        W.remove_from_watchlist("aapl")
        self.assertEqual(W.load_watchlist(), ["SPY"])

    def test_remove_of_an_absent_ticker_changes_nothing(self):
        W.save_watchlist(["SPY"])
        W.remove_from_watchlist("TSLA")
        self.assertEqual(W.load_watchlist(), ["SPY"])


if __name__ == "__main__":
    unittest.main()
