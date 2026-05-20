import os, sqlite3, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.backfill_crypto_pnl import compute_backfill

class TestBackfill(unittest.TestCase):
    def _db(self):
        d = tempfile.mkdtemp(); p = os.path.join(d, "c.db")
        c = sqlite3.connect(p)
        c.execute("""CREATE TABLE trades (entry_id INTEGER PRIMARY KEY,
            strategy_name TEXT, entry_price REAL, pnl_pct REAL, pnl_usd REAL,
            status TEXT, spread_width REAL, net_credit REAL,
            quantity REAL DEFAULT 1.0)""")
        c.execute("INSERT INTO trades VALUES (1,'Long Put',2561.03,0.2163,55402.0,'CLOSED',NULL,NULL,1.0)")
        c.execute("INSERT INTO trades VALUES (2,'Long Call',5000.0,3.1174,15586.8,'CLOSED',NULL,NULL,1.0)")
        c.commit(); c.close()
        return p

    def test_dryrun_diff_corrects_inflation_and_applies_cap(self):
        import math
        p = self._db()
        rows = compute_backfill(p)
        r1 = next(r for r in rows if r["entry_id"] == 1)
        qty = math.floor(999.0/2561.03*1e4)/1e4
        self.assertAlmostEqual(r1["new_quantity"], qty, places=6)
        self.assertAlmostEqual(r1["new_pnl_usd"], 0.2163*2561.03*qty, places=2)
        self.assertLess(r1["new_pnl_usd"], r1["old_pnl_usd"])

    def test_compute_does_not_write(self):
        p = self._db()
        compute_backfill(p)
        v = sqlite3.connect(p).execute("SELECT pnl_usd FROM trades WHERE entry_id=1").fetchone()[0]
        self.assertEqual(v, 55402.0)

    def test_credit_spread_row_uses_max_loss(self):
        import math, os, sqlite3, tempfile
        # Build a fresh DB with one credit-spread row (spread_width=1000, net_credit=250 → max_loss 750)
        d = tempfile.mkdtemp(); p = os.path.join(d, "c.db")
        c = sqlite3.connect(p)
        c.execute("""CREATE TABLE trades (entry_id INTEGER PRIMARY KEY,
            strategy_name TEXT, entry_price REAL, pnl_pct REAL, pnl_usd REAL,
            status TEXT, spread_width REAL, net_credit REAL,
            quantity REAL DEFAULT 1.0)""")
        # Bear Call: entry_price stores net_credit (250); spread_width=1000; pnl_pct=0.4 means kept 40% of credit
        c.execute("INSERT INTO trades VALUES (10,'Bear Call',250.0,0.4,5000.0,'CLOSED',1000.0,250.0,1.0)")
        # Also a NULL-data credit row that should be SKIPPED
        c.execute("INSERT INTO trades VALUES (11,'Iron Condor',100.0,0.3,3000.0,'CLOSED',NULL,NULL,1.0)")
        c.commit(); c.close()

        rows = compute_backfill(p)
        # Only the Bear Call row should appear; the NULL-data Iron Condor is skipped
        self.assertEqual([r["entry_id"] for r in rows], [10])
        expected_qty = math.floor(999.0 / 750.0 * 1e4) / 1e4   # max_loss = 1000-250 = 750
        self.assertAlmostEqual(rows[0]["new_quantity"], expected_qty, places=6)
        self.assertAlmostEqual(rows[0]["new_pnl_usd"], 0.4 * 250.0 * expected_qty, places=2)

if __name__ == "__main__":
    unittest.main()
