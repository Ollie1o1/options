"""Tests for ATM-IV extraction from the chain archive — pure + temp-db, offline."""
from __future__ import annotations
import os, sqlite3, tempfile, unittest
from src.vol_intel.atm_iv import atm_iv, iv_move, TARGET_DTE


def _row(typ, strike, exp, iv, snap="2026-06-22"):
    return {"type": typ, "strike": strike, "expiration": exp, "iv": iv, "snap_date": snap}


class AtmIvPureTests(unittest.TestCase):
    def test_picks_nearest_strike_and_averages_call_put(self):
        rows = [_row("call", 100, "2026-07-22", 0.40), _row("put", 100, "2026-07-22", 0.50),
                _row("call", 130, "2026-07-22", 0.90)]
        self.assertAlmostEqual(atm_iv(rows, spot=101, target_dte=30), 0.45, places=6)

    def test_picks_expiry_nearest_target_dte(self):
        rows = [_row("call", 100, "2026-06-25", 0.20),   # ~3 DTE
                _row("call", 100, "2026-07-22", 0.40)]   # ~30 DTE
        self.assertAlmostEqual(atm_iv(rows, spot=100, target_dte=30), 0.40, places=6)

    def test_none_on_no_iv(self):
        self.assertIsNone(atm_iv([_row("call", 100, "2026-07-22", None)], spot=100))

    def test_none_on_empty(self):
        self.assertIsNone(atm_iv([], spot=100))

    def test_constant(self):
        self.assertEqual(TARGET_DTE, 30)


class IvMoveTests(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "ca.db")
        con = sqlite3.connect(self.db)
        con.execute("""CREATE TABLE chain_snapshots (symbol TEXT, snap_date TEXT,
            type TEXT, strike REAL, expiration TEXT, iv REAL, spot REAL)""")
        rows = [
            ("AAA", "2026-06-15", "call", 100, "2026-07-15", 0.30, 100),
            ("AAA", "2026-06-15", "put", 100, "2026-07-15", 0.30, 100),
            ("AAA", "2026-06-22", "call", 100, "2026-07-22", 0.40, 100),
            ("AAA", "2026-06-22", "put", 100, "2026-07-22", 0.40, 100),
        ]
        con.executemany("INSERT INTO chain_snapshots VALUES (?,?,?,?,?,?,?)", rows)
        con.commit(); con.close()

    def test_iv_move_computes_delta_vs_prior_snap(self):
        out = {r["symbol"]: r for r in iv_move(self.db, "2026-06-22")}
        self.assertIn("AAA", out)
        self.assertAlmostEqual(out["AAA"]["iv"], 0.40, places=6)
        self.assertAlmostEqual(out["AAA"]["prev_iv"], 0.30, places=6)
        self.assertAlmostEqual(out["AAA"]["d_iv"], 0.10, places=6)

    def test_d_iv_none_when_no_prior_snapshot(self):
        import sqlite3
        con = sqlite3.connect(self.db)
        con.execute("INSERT INTO chain_snapshots VALUES (?,?,?,?,?,?,?)",
                    ("BBB", "2026-06-22", "call", 50, "2026-07-22", 0.6, 50))
        con.execute("INSERT INTO chain_snapshots VALUES (?,?,?,?,?,?,?)",
                    ("BBB", "2026-06-22", "put", 50, "2026-07-22", 0.6, 50))
        con.commit(); con.close()
        out = {r["symbol"]: r for r in iv_move(self.db, "2026-06-22")}
        self.assertIsNone(out["BBB"]["prev_iv"])
        self.assertIsNone(out["BBB"]["d_iv"])


if __name__ == "__main__":
    unittest.main()
