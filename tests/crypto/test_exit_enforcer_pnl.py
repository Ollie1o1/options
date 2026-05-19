import os, sqlite3, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.crypto.exit_enforcer import _close_row

class TestExitEnforcerPnl(unittest.TestCase):
    def _db(self):
        d = tempfile.mkdtemp(); p = os.path.join(d, "c.db")
        c = sqlite3.connect(p)
        c.execute("""CREATE TABLE trades (entry_id INTEGER PRIMARY KEY,
            strategy_name TEXT, entry_price REAL, exit_price REAL,
            pnl_pct REAL, pnl_usd REAL, exit_reason TEXT, status TEXT,
            exit_date TEXT, quantity REAL DEFAULT 1.0)""")
        c.execute("INSERT INTO trades (entry_id, status, quantity) VALUES (1,'OPEN',0.39)")
        c.commit()
        return p, c

    def test_close_row_scales_pnl_by_quantity(self):
        p, c = self._db()
        with sqlite3.connect(p) as conn:
            _close_row(conn, entry_id=1, exit_price=3115.05, pnl_pct=0.2163,
                       pnl_usd=554.02, reason="Time Exit",
                       strategy_name="Long Put", entry_price=2561.03)
        row = c.execute("SELECT pnl_usd, status FROM trades WHERE entry_id=1").fetchone()
        self.assertAlmostEqual(row[0], 554.02 * 0.39, places=2)
        self.assertEqual(row[1], "CLOSED")

if __name__ == "__main__":
    unittest.main()
