"""Joining the panel, the price book, and shares outstanding into dhist rows."""
import math
import sqlite3
import tempfile
import unittest

import numpy as np

from src.squeeze.backtest import panel
from src.squeeze.sleeve import panel_rows


class _Shares:
    """Minimal SharesLookup stand-in: get(symbol, asof) -> shares or None."""

    def __init__(self, mapping):
        self._m = mapping

    def get(self, symbol, asof):
        return self._m.get(symbol)


def _book(closes, symbol="T0"):
    fd = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    fd.close()
    conn = sqlite3.connect(fd.name)
    conn.execute("CREATE TABLE px (date TEXT, symbol TEXT, close REAL,"
                 " volume REAL, PRIMARY KEY (symbol,date))")
    conn.executemany("INSERT INTO px VALUES (?,?,?,?)",
                     [(f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}", symbol,
                       float(c), 1_000_000.0) for i, c in enumerate(closes)])
    conn.commit()
    conn.close()
    return panel.PriceBook(fd.name, [symbol])


def _rec(symbol, entry_index, si_ratio, ret_5d=0.20, spot=100.0):
    return {"date": "2020-03-02", "symbol": symbol, "entry_index": entry_index,
            "si_ratio": si_ratio, "ret_5d": ret_5d, "spot": spot,
            "sigma_d": 0.05}


class PanelRowsTest(unittest.TestCase):
    def setUp(self):
        # 200 bars, rising; entry at index 100
        self.closes = [100.0 + i for i in range(200)]
        self.book = _book(self.closes)
        self.shares = _Shares({"T0": 50_000_000.0})

    def test_spot0_comes_from_the_panel_not_the_first_path_bar(self):
        rec = _rec("T0", entry_index=100, si_ratio=0.9, spot=self.closes[100])
        rows, _ = panel_rows.build([rec], self.book, self.shares, horizon=42)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["spot0"], self.closes[100])
        # the path starts one bar LATER, so the two must differ here
        self.assertNotAlmostEqual(float(rows[0]["path"][0]), rows[0]["spot0"])
        self.assertAlmostEqual(float(rows[0]["path"][0]), self.closes[101])

    def test_the_path_is_exactly_the_horizon_long(self):
        rec = _rec("T0", entry_index=100, si_ratio=0.9)
        rows, _ = panel_rows.build([rec], self.book, self.shares, horizon=42)
        self.assertEqual(len(rows[0]["path"]), 42)

    def test_the_path_is_a_view_and_not_a_copy(self):
        rec = _rec("T0", entry_index=100, si_ratio=0.9)
        rows, _ = panel_rows.build([rec], self.book, self.shares, horizon=42)
        base = self.book._close["T0"]
        self.assertTrue(np.shares_memory(base, rows[0]["path"]))

    def test_a_series_ending_inside_the_window_is_dropped_and_counted(self):
        rec = _rec("T0", entry_index=190, si_ratio=0.9)
        rows, stats = panel_rows.build([rec], self.book, self.shares, horizon=42)
        self.assertEqual(rows, [])
        self.assertEqual(stats["short_path"], 1)

    def test_short_path_drops_are_counted_per_arm(self):
        # The censoring is not arm-neutral — a treated name that just ran +10%
        # is likelier to delist mid-window than a low-SI control — so the
        # drops must be visible per arm, with the total kept for compatibility.
        recs = [_rec("A", entry_index=190, si_ratio=0.1),
                _rec("B", entry_index=190, si_ratio=0.9)]
        book = _book(self.closes, symbol="A")
        book._close["B"] = book._close["A"]
        book._dates["B"] = book._dates["A"]
        shares = _Shares({"A": 50_000_000.0, "B": 50_000_000.0})
        rows, stats = panel_rows.build(recs, book, shares, horizon=42)
        self.assertEqual(rows, [])
        self.assertEqual(stats["short_path"], 2)
        self.assertEqual(stats["short_path_treated"], 1)
        self.assertEqual(stats["short_path_control"], 1)

    def test_covariates_are_derived_as_specified(self):
        rec = _rec("T0", entry_index=100, si_ratio=0.9, spot=200.0)
        rows, _ = panel_rows.build([rec], self.book, self.shares, horizon=42)
        r = rows[0]
        self.assertAlmostEqual(r["rv"], 0.05 * math.sqrt(252.0))
        self.assertAlmostEqual(r["log_mcap"], math.log(50_000_000.0 * 200.0))
        self.assertAlmostEqual(r["log_price"], math.log(200.0))

    def test_rows_without_short_interest_are_counted_and_excluded(self):
        rec = _rec("T0", entry_index=100, si_ratio=None)
        rows, stats = panel_rows.build([rec], self.book, self.shares, horizon=42)
        self.assertEqual(rows, [])
        self.assertEqual(stats["ungradeable"], 1)

    def test_arms_are_assigned_and_counted(self):
        recs = [_rec(f"T{i}", 100, si_ratio=0.01 * (i + 1)) for i in range(100)]
        book = _book(self.closes, symbol="T0")
        for i in range(100):
            book._close[f"T{i}"] = book._close["T0"]
            book._dates[f"T{i}"] = book._dates["T0"]
        shares = _Shares({f"T{i}": 50_000_000.0 for i in range(100)})
        rows, stats = panel_rows.build(recs, book, shares, horizon=42)
        self.assertGreater(stats["treated"], 0)
        self.assertGreater(stats["control"], 0)
        self.assertEqual(stats["treated"] + stats["control"], len(rows))

    def test_a_gradeable_row_always_yields_a_market_cap(self):
        # si_ratio exists only because shares_out did, so mcap must be derivable
        rec = _rec("T0", entry_index=100, si_ratio=0.9)
        rows, _ = panel_rows.build([rec], self.book, self.shares, horizon=42)
        self.assertTrue(math.isfinite(rows[0]["log_mcap"]))
