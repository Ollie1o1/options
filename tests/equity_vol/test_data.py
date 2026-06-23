"""Tests for the dolt straddle/close loaders — offline, synthetic sqlite."""
from __future__ import annotations
import os, sqlite3, tempfile, unittest
from src.equity_vol.data import (Entry, closes, pick_entry, straddle_entries,
                                 days_between, TARGET_DTE, FREQ_DAYS)


def _row(typ, strike, exp, bid, ask, iv):
    return {"type": typ, "strike": strike, "expiration": exp, "bid": bid, "ask": ask, "iv": iv}


class PureTests(unittest.TestCase):
    def test_days_between(self):
        self.assertEqual(days_between("2022-01-01", "2022-01-31"), 30)

    def test_constants(self):
        self.assertEqual((TARGET_DTE, FREQ_DAYS), (30, 28))

    def test_pick_entry_nearest_dte_and_strike_pairs_call_put(self):
        rows = [_row("call", 100, "2022-02-01", 5.0, 5.4, 0.30),   # ~30 DTE
                _row("put", 100, "2022-02-01", 4.0, 4.4, 0.30),
                _row("call", 130, "2022-02-01", 0.5, 0.7, 0.50),
                _row("call", 100, "2022-01-05", 1.0, 1.2, 0.30)]   # ~3 DTE
        e = pick_entry(rows, spot=101, date="2022-01-02", target_dte=30)
        self.assertEqual(e.strike, 100)
        self.assertEqual(e.expiration, "2022-02-01")
        self.assertAlmostEqual(e.straddle_bid, 9.0, places=6)   # 5.0 + 4.0
        self.assertAlmostEqual(e.straddle_ask, 9.8, places=6)   # 5.4 + 4.4
        self.assertAlmostEqual(e.iv, 0.30, places=6)

    def test_pick_entry_none_without_matching_put(self):
        rows = [_row("call", 100, "2022-02-01", 5.0, 5.4, 0.30)]  # no put at 100
        self.assertIsNone(pick_entry(rows, spot=100, date="2022-01-02"))

    def test_pick_entry_rejects_split_mismatch_strike(self):
        # pre-split strikes (~2800) vs a post-split spot (~140) -> skip, not a fake ATM
        rows = [_row("call", 2800, "2022-02-01", 5.0, 5.4, 0.30),
                _row("put", 2800, "2022-02-01", 4.0, 4.4, 0.30)]
        self.assertIsNone(pick_entry(rows, spot=140, date="2022-01-02"))


class DbTests(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "dolt.db")
        con = sqlite3.connect(self.db)
        con.execute("CREATE TABLE dolt_chain (symbol TEXT, date TEXT, expiration TEXT, "
                    "strike REAL, type TEXT, bid REAL, ask REAL, mid REAL, iv REAL)")
        con.execute("CREATE TABLE stocks_close (symbol TEXT, date TEXT, close REAL)")
        chain = []
        for d in ("2022-01-03", "2022-02-03", "2022-03-03"):  # ~monthly
            exp = {"2022-01-03": "2022-02-02", "2022-02-03": "2022-03-05",
                   "2022-03-03": "2022-04-02"}[d]
            for typ, bid, ask in (("call", 5.0, 5.4), ("put", 4.0, 4.4)):
                chain.append((  "AAA", d, exp, 100, typ, bid, ask, (bid + ask) / 2, 0.30))
        con.executemany("INSERT INTO dolt_chain VALUES (?,?,?,?,?,?,?,?,?)", chain)
        con.executemany("INSERT INTO stocks_close VALUES (?,?,?)",
                        [("AAA", d, 100.0) for d in ("2022-01-03", "2022-02-03", "2022-03-03")])
        con.commit(); con.close()

    def test_closes(self):
        self.assertEqual(closes(self.db, "AAA")["2022-01-03"], 100.0)

    def test_closes_skips_null_close(self):
        con = sqlite3.connect(self.db)
        con.execute("INSERT INTO stocks_close VALUES ('AAA','2022-04-04',NULL)")
        con.commit(); con.close()
        px = closes(self.db, "AAA")     # must not raise on the NULL row
        self.assertNotIn("2022-04-04", px)
        self.assertEqual(px["2022-01-03"], 100.0)

    def test_straddle_entries_rejects_too_close(self):
        # with a 90-day freq, the ~monthly snapshots collapse to a single entry
        entries = straddle_entries(self.db, "AAA", target_dte=30, freq_days=90)
        self.assertEqual(len(entries), 1)

    def test_straddle_entries_spaced_and_real_marks(self):
        entries = straddle_entries(self.db, "AAA", target_dte=30, freq_days=28)
        self.assertGreaterEqual(len(entries), 2)
        self.assertTrue(all(isinstance(e, Entry) for e in entries))
        self.assertAlmostEqual(entries[0].straddle_bid, 9.0, places=6)


if __name__ == "__main__":
    unittest.main()
