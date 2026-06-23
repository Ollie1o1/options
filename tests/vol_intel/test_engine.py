"""Tests for the vol-intel engine — offline, synthetic chain + ohlcv DBs."""
from __future__ import annotations
import os, sqlite3, tempfile, unittest
import numpy as np
from src.breakout.data import upsert_ohlcv
from src.vol_intel.engine import build_rows


class EngineTests(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp()
        self.chain = os.path.join(d, "ca.db")
        self.ohlcv = os.path.join(d, "ohlcv.db")
        con = sqlite3.connect(self.chain)
        con.execute("""CREATE TABLE chain_snapshots (symbol TEXT, snap_date TEXT,
            type TEXT, strike REAL, expiration TEXT, iv REAL, spot REAL)""")
        rows = []
        for snap, iv in (("2026-06-15", 0.30), ("2026-06-22", 0.40)):
            for typ in ("call", "put"):
                rows.append(("AAA", snap, typ, 100, "2026-07-22", iv, 100))
        con.executemany("INSERT INTO chain_snapshots VALUES (?,?,?,?,?,?,?)", rows)
        con.commit(); con.close()
        rng = np.random.default_rng(0)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120)))
        upsert_ohlcv(self.ohlcv, "AAA", [
            {"date": f"d{i:05d}", "close": float(c), "high": float(c) + 1,
             "low": float(c) - 1, "volume": 1000.0} for i, c in enumerate(close)])

    def test_build_rows_produces_mover_and_vrp(self):
        movers, vrp_rows = build_rows(self.chain, self.ohlcv, "2026-06-22")
        self.assertEqual(len(movers), 1)
        self.assertAlmostEqual(movers[0]["d_iv"], 0.10, places=6)
        self.assertIn("rv_pctile", movers[0])
        self.assertEqual(len(vrp_rows), 1)
        self.assertEqual(vrp_rows[0]["symbol"], "AAA")
        self.assertIn("label", vrp_rows[0])

    def test_missing_ohlcv_symbol_skipped_not_crashed(self):
        # symbol present in chain but not ohlcv -> no vrp row, no exception
        empty = os.path.join(tempfile.mkdtemp(), "empty.db")
        upsert_ohlcv(empty, "ZZZ", [{"date": "d00000", "close": 1.0, "high": 1.0,
                                     "low": 1.0, "volume": 1.0}])
        movers, vrp_rows = build_rows(self.chain, empty, "2026-06-22")
        self.assertEqual(len(movers), 1)      # mover still computed
        self.assertEqual(len(vrp_rows), 0)    # no realized vol -> skipped


if __name__ == "__main__":
    unittest.main()
