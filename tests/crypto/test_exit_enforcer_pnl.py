import os, sqlite3, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.crypto.exit_enforcer import _close_row
from src.paper_manager import _sanitize_close_values

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
        # After the sanitization fix, the stored value is sanitized_usd * qty.
        # sanitized_usd = round(2561.03 * 0.2163 * 1.0, 2) = 553.96
        _, _, expected_sanitized = _sanitize_close_values(
            "Long Put", 2561.03, 3115.05, 0.2163, multiplier=1.0)
        self.assertAlmostEqual(row[0], expected_sanitized * 0.39, places=4)
        self.assertEqual(row[1], "CLOSED")

    def test_close_row_applies_sanitization_then_quantity(self):
        """Sanitization (clamp/floor) must still run; scaling is on top."""
        p, c = self._db()
        # "Bull Put" is a credit spread: pnl_pct is clamped to [floor, 1.0].
        # Using pnl_pct=99.0 forces clamping to 1.0 — raw * qty != sanitized * qty.
        raw_pnl_pct = 99.0
        raw_pnl_usd = 1_000_000.0  # ignored by _sanitize_close_values
        entry_price = 2561.03
        with sqlite3.connect(p) as conn:
            _close_row(conn, entry_id=1, exit_price=3000.0, pnl_pct=raw_pnl_pct,
                       pnl_usd=raw_pnl_usd, reason="X",
                       strategy_name="Bull Put", entry_price=entry_price)
        stored = c.execute("SELECT pnl_usd FROM trades WHERE entry_id=1").fetchone()[0]
        # Compute expected the same way _close_row does after the fix
        _, expected_pct, expected_sanitized = _sanitize_close_values(
            "Bull Put", entry_price, 3000.0, raw_pnl_pct, multiplier=1.0)
        self.assertAlmostEqual(stored, expected_sanitized * 0.39, places=4)
        # Confirm clamping happened (pnl_pct should be 1.0, not 99.0)
        self.assertAlmostEqual(expected_pct, 1.0, places=4)
        # And confirm raw*qty was NOT stored (sanity)
        self.assertNotAlmostEqual(stored, raw_pnl_usd * 0.39, places=2)

if __name__ == "__main__":
    unittest.main()
