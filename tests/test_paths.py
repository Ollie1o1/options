"""Repo-root path resolution, and the one string that is not a path.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_paths -v
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.paths import PROJECT_ROOT, repo_path


class RepoPathTest(unittest.TestCase):
    def test_a_relative_name_resolves_against_the_repo_root(self):
        self.assertEqual(repo_path("config.json"),
                         str(PROJECT_ROOT / "config.json"))

    def test_an_absolute_path_is_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "config.json")
            self.assertEqual(repo_path(p), p)

    def test_a_pathlib_path_is_accepted(self):
        self.assertEqual(repo_path(Path("config.json")),
                         str(PROJECT_ROOT / "config.json"))


class SqliteMagicStringsTest(unittest.TestCase):
    """`:memory:` is a SQLite keyword, not a filename.

    Anchoring it produced a REAL FILE at the repo root literally named
    `:memory:` — a 36KB SQLite database at schema v21 that got committed to
    git in 54ec402 and churned on every test run. Seven test modules pass the
    literal expecting a private in-memory database and were instead sharing one
    file on disk, which is the isolation failure
    `feedback_tests_must_not_name_the_real_ledger` exists to prevent.

    The resolver is the right place to fix it: everything downstream hands the
    result to `sqlite3.connect`, which understands these strings, and a path
    helper must not mangle something that was never a path.

    KNOW THIS BEFORE USING `:memory:` IN A TEST. `PaperManager._get_connection`
    opens and CLOSES a connection per operation, so a real `:memory:` database
    is discarded between calls and every operation starts empty. The seven
    modules passing the literal today never relied on persistence and pass
    either way, but a test that writes and then reads will silently see
    nothing. When persistence is needed, use a file in a `TemporaryDirectory`
    — the pattern `feedback_tests_must_not_name_the_real_ledger` already asks
    for.
    """

    def test_memory_is_passed_through_unchanged(self):
        self.assertEqual(repo_path(":memory:"), ":memory:")

    def test_memory_does_not_become_a_file_under_the_repo_root(self):
        self.assertNotIn(str(PROJECT_ROOT), repo_path(":memory:"))

    def test_a_sqlite_uri_is_passed_through_unchanged(self):
        # Same class of mistake: a URI is not a relative path either, and
        # `_ro_uri`-style read-only handles are used across this codebase.
        uri = "file:/tmp/x.db?mode=ro"
        self.assertEqual(repo_path(uri), uri)

    def test_a_plain_name_beginning_with_file_is_still_a_path(self):
        # Guards over-matching: "filed_trades.db" must not be read as a URI.
        self.assertEqual(repo_path("filed_trades.db"),
                         str(PROJECT_ROOT / "filed_trades.db"))


class NoLeakedMemoryFileTest(unittest.TestCase):
    def test_the_repo_root_holds_no_file_named_memory(self):
        """The artifact itself, so a regression is caught as a failing test."""
        self.assertFalse((PROJECT_ROOT / ":memory:").exists(),
                         "a file named ':memory:' is in the repo root — "
                         "something resolved the SQLite keyword as a path")


if __name__ == "__main__":
    unittest.main()
