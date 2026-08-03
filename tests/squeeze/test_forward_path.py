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


class EntryIndexTest(unittest.TestCase):
    """panel.build's record dropped feats["_i"], which the adapter needs to
    slice a path without recomputing it. Verified on a real built panel, not by
    inspecting source text — the point is that the value is correct, not that
    the key appears somewhere in the file."""

    def _dbs(self):
        import os
        import tempfile
        d = tempfile.mkdtemp()
        si_db = os.path.join(d, "si.db")
        px_db = os.path.join(d, "px.db")
        sh_db = os.path.join(d, "sh.db")

        dates = [f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(200)]

        conn = sqlite3.connect(px_db)
        conn.execute("CREATE TABLE px (date TEXT, symbol TEXT, close REAL,"
                     " volume REAL, PRIMARY KEY (symbol,date))")
        conn.executemany("INSERT INTO px VALUES (?,?,?,?)",
                         [(dates[i], "TEST", 100.0 + i, 1_000_000.0)
                          for i in range(200)])
        conn.commit(); conn.close()

        conn = sqlite3.connect(si_db)
        conn.execute("CREATE TABLE si (settlement_date TEXT NOT NULL,"
                     " symbol TEXT NOT NULL, shares_short REAL,"
                     " shares_prior REAL, adv REAL, dtc REAL,"
                     " market_class TEXT, PRIMARY KEY (settlement_date, symbol))")
        conn.executemany(
            "INSERT INTO si VALUES (?,?,?,?,?,?,?)",
            [(dates[70 + 11 * k], "TEST", 5_000_000.0, 4_800_000.0,
              1_000_000.0, 5.0, "NMS") for k in range(3)])
        conn.commit(); conn.close()

        conn = sqlite3.connect(sh_db)
        conn.execute("CREATE TABLE shares_out (symbol TEXT NOT NULL,"
                     " filed TEXT NOT NULL, end_date TEXT, shares REAL,"
                     " PRIMARY KEY (symbol, filed))")
        conn.execute("INSERT INTO shares_out VALUES ('TEST','2019-01-01',"
                     " '2019-01-01', 50000000.0)")
        conn.commit(); conn.close()
        return si_db, px_db, sh_db, dates

    def test_the_built_record_carries_a_usable_entry_index(self):
        si_db, px_db, sh_db, dates = self._dbs()
        recs = panel.build(db_path=si_db, prices_db=px_db, shares_db=sh_db,
                           horizons=(21,), verbose=False)
        self.assertTrue(recs)
        book = panel.PriceBook(px_db, ["TEST"])
        for rec in recs:
            i = rec["entry_index"]
            self.assertIsInstance(i, int)
            # the index must point at the bar whose close IS the record's spot
            self.assertAlmostEqual(float(book._close["TEST"][i]), rec["spot"])
