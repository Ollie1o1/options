"""Time-to-max and post-max give-back on the forward outcome window."""
import sqlite3
import tempfile
import unittest

from src.squeeze.backtest import panel


def _price_db(path, closes, symbol="TEST"):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE px (date TEXT, symbol TEXT, close REAL, volume REAL,"
                 " PRIMARY KEY (symbol,date))")
    rows = [(f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}", symbol, float(c), 1_000_000.0)
            for i, c in enumerate(closes)]
    conn.executemany("INSERT INTO px VALUES (?,?,?,?)", rows)
    conn.commit()
    conn.close()


class ForwardPathTest(unittest.TestCase):
    def _book(self, closes):
        fd = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        fd.close()
        _price_db(fd.name, closes)
        return panel.PriceBook(fd.name, ["TEST"])

    def test_max_on_first_bar_has_tmax_one(self):
        # entry at index 0 (price 100); path = [120, 110, 105]
        book = self._book([100.0, 120.0, 110.0, 105.0])
        got = book.forward("TEST", 0, 3)
        self.assertEqual(got.t_max, 1)
        self.assertAlmostEqual(got.max_up, 0.20, places=6)

    def test_give_back_measured_from_the_max(self):
        # max 150 at bar 2, then down to 90 -> give-back = 90/150 - 1 = -0.40
        book = self._book([100.0, 120.0, 150.0, 90.0, 95.0])
        got = book.forward("TEST", 0, 4)
        self.assertEqual(got.t_max, 2)
        self.assertAlmostEqual(got.dd_after_max, -0.40, places=6)

    def test_max_on_final_bar_gives_back_nothing(self):
        book = self._book([100.0, 110.0, 120.0, 130.0])
        got = book.forward("TEST", 0, 3)
        self.assertEqual(got.t_max, 3)
        self.assertAlmostEqual(got.dd_after_max, 0.0, places=6)

    def test_legacy_fields_unchanged(self):
        book = self._book([100.0, 120.0, 150.0, 90.0, 95.0])
        got = book.forward("TEST", 0, 4)
        self.assertAlmostEqual(got.max_up, 0.50, places=6)
        self.assertAlmostEqual(got.end, -0.05, places=6)
        self.assertAlmostEqual(got.min_dn, -0.10, places=6)
