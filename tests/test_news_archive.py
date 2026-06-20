"""Tests for src.news_archive — point-in-time persistence of fetched news."""
import os
import tempfile
import unittest
from datetime import datetime, timezone

from src import news_archive as na


class _Item:
    """Minimal stand-in for news_fetcher.NewsItem (duck-typed)."""
    def __init__(self, headline, source="src", published=None,
                 sentiment=0.0, relevance=1.0, url="http://x"):
        self.headline = headline
        self.source = source
        self.published = published or datetime(2026, 6, 19, tzinfo=timezone.utc)
        self.sentiment = sentiment
        self.relevance = relevance
        self.url = url


class NewsArchiveTest(unittest.TestCase):
    def setUp(self):
        fd, self.db = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(self.db)  # let the module create it

    def tearDown(self):
        if os.path.exists(self.db):
            os.unlink(self.db)

    def test_archive_inserts_rows_and_returns_count(self):
        items = [_Item("Apple beats earnings", sentiment=0.3),
                 _Item("Apple sued in China", sentiment=-0.2)]
        n = na.archive_items(items, "AAPL", db_path=self.db)
        self.assertEqual(n, 2)
        stats = na.archive_stats(self.db)
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["symbols"], 1)

    def test_dedupe_same_item_not_double_counted(self):
        item = _Item("Apple beats earnings")
        na.archive_items([item], "AAPL", db_path=self.db)
        inserted = na.archive_items([item], "AAPL", db_path=self.db)
        self.assertEqual(inserted, 0)
        self.assertEqual(na.archive_stats(self.db)["total"], 1)

    def test_same_headline_different_symbol_kept_separately(self):
        item_a = _Item("Memory boom helps Micron")
        item_b = _Item("Memory boom helps Micron")
        na.archive_items([item_a], "AAPL", db_path=self.db)
        na.archive_items([item_b], "MU", db_path=self.db)
        self.assertEqual(na.archive_stats(self.db)["total"], 2)

    def test_stats_reports_latest_and_days(self):
        na.archive_items(
            [_Item("h1", published=datetime(2026, 6, 18, tzinfo=timezone.utc))],
            "AAPL", db_path=self.db,
            now=datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc))
        na.archive_items(
            [_Item("h2", published=datetime(2026, 6, 19, tzinfo=timezone.utc))],
            "AAPL", db_path=self.db,
            now=datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc))
        stats = na.archive_stats(self.db)
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["archive_days"], 2)
        self.assertEqual(stats["latest"][:10], "2026-06-19")

    def test_empty_items_noop(self):
        self.assertEqual(na.archive_items([], "AAPL", db_path=self.db), 0)

    def test_stats_on_missing_db_is_zeroed(self):
        stats = na.archive_stats(self.db + ".nope")
        self.assertEqual(stats["total"], 0)

    def test_format_stats_line_is_factual(self):
        na.archive_items([_Item("h1"), _Item("h2")], "AAPL", db_path=self.db)
        line = na.format_stats_line(na.archive_stats(self.db))
        self.assertIn("2", line)
        self.assertIn("news", line.lower())


if __name__ == "__main__":
    unittest.main()
