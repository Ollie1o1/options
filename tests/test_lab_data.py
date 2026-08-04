"""Tests for src/lab/data.py — loading the price substrate."""
import os
import sqlite3
import tempfile
import unittest

from src.lab import data


class LoadUniverseTest(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "px.db")
        c = sqlite3.connect(self.db)
        c.execute("CREATE TABLE px (date TEXT, symbol TEXT, close REAL, volume REAL)")
        rows = [(f"2020-01-{i+1:02d}", "AAA", 100.0 + i, 1e6) for i in range(80)]
        rows += [(f"2020-01-{i+1:02d}", "BBB", 50.0, 1e6) for i in range(3)]
        c.executemany("INSERT INTO px VALUES (?,?,?,?)", rows)
        c.commit(); c.close()

    def test_loads_bars_in_date_order(self):
        u = data.load_universe(["AAA"], db_path=self.db, min_bars=10)
        self.assertEqual(len(u["AAA"]), 80)
        self.assertEqual(u["AAA"][0][0], "2020-01-01")
        self.assertLess(u["AAA"][0][1], u["AAA"][-1][1])

    def test_a_symbol_with_too_little_history_is_dropped(self):
        u = data.load_universe(["AAA", "BBB"], db_path=self.db, min_bars=10)
        self.assertIn("AAA", u)
        self.assertNotIn("BBB", u)

    def test_a_missing_symbol_is_simply_absent(self):
        u = data.load_universe(["ZZZ"], db_path=self.db, min_bars=10)
        self.assertEqual(u, {})

    def test_a_date_window_is_respected(self):
        u = data.load_universe(["AAA"], db_path=self.db, min_bars=5,
                               start="2020-01-10", end="2020-01-20")
        self.assertEqual(len(u["AAA"]), 11)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class SplitAdjustmentTest(unittest.TestCase):
    """`data/squeeze_prices.db` stores RAW closes. NVDA falls 89.9% on
    2024-06-10, AMZN 94.9% on 2022-06-06, GOOG 95.1% on 2022-07-18 — splits,
    not crashes. Unadjusted they destroy long calls and manufacture long puts,
    which is exactly the shape of a spurious backtest result."""

    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "px.db")
        c = sqlite3.connect(self.db)
        c.execute("CREATE TABLE px (date TEXT, symbol TEXT, close REAL, volume REAL)")
        # 40 bars at ~1000, then a 10-for-1 split to ~100, then 40 more.
        rows = [(f"2024-01-{i+1:02d}", "SPLT", 1000.0 + i, 1e6) for i in range(40)]
        rows += [(f"2024-02-{i+1:02d}", "SPLT", 104.0 + i * 0.1, 1e6) for i in range(40)]
        # A genuinely volatile name that never splits: -35% in one day.
        rows += [(f"2024-01-{i+1:02d}", "VOL", 100.0, 1e6) for i in range(40)]
        rows += [(f"2024-02-{i+1:02d}", "VOL", 65.0, 1e6) for i in range(40)]
        c.executemany("INSERT INTO px VALUES (?,?,?,?)", rows)
        c.commit(); c.close()

    def _max_drop(self, bars):
        return min(bars[i][1] / bars[i - 1][1] - 1 for i in range(1, len(bars)))

    def test_a_split_no_longer_looks_like_a_ninety_percent_crash(self):
        u = data.load_universe(["SPLT"], db_path=self.db, min_bars=10)
        self.assertGreater(self._max_drop(u["SPLT"]), -0.20)

    def test_the_adjusted_series_is_continuous_across_the_split(self):
        u = data.load_universe(["SPLT"], db_path=self.db, min_bars=10)
        closes = [c for _, c in u["SPLT"]]
        self.assertAlmostEqual(closes[40] / closes[39], 1.0, delta=0.05)

    def test_the_total_return_across_a_split_is_preserved(self):
        """A 10-for-1 split is not a loss. Pre-split 1000 -> post-split 104
        is really 1000 -> 1040, a +4% move."""
        u = data.load_universe(["SPLT"], db_path=self.db, min_bars=10)
        closes = [c for _, c in u["SPLT"]]
        self.assertAlmostEqual(closes[-1] / closes[0], 107.9 * 10 / 1000.0, delta=0.05)

    def test_a_real_thirty_five_percent_drop_is_left_alone(self):
        """Adjusting genuine crashes away would be worse than the bug."""
        u = data.load_universe(["VOL"], db_path=self.db, min_bars=10)
        self.assertAlmostEqual(self._max_drop(u["VOL"]), -0.35, places=3)

    def test_adjustment_can_be_switched_off_for_inspection(self):
        u = data.load_universe(["SPLT"], db_path=self.db, min_bars=10,
                               adjust_splits=False)
        self.assertLess(self._max_drop(u["SPLT"]), -0.85)
