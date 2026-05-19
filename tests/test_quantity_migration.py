import os, sqlite3, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paper_manager import PaperManager

class TestQuantityMigration(unittest.TestCase):
    def test_quantity_column_exists_and_defaults_one(self):
        d = tempfile.mkdtemp()
        db = os.path.join(d, "pm.db")
        pm = PaperManager(db_path=db)            # runs migrations on init
        conn = sqlite3.connect(db)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(trades)")]
        self.assertIn("quantity", cols)
        conn.execute("INSERT INTO trades (ticker, status) VALUES ('BTC','OPEN')")
        conn.commit()
        q = conn.execute("SELECT quantity FROM trades").fetchone()[0]
        self.assertEqual(q, 1.0)
        conn.close()

    def test_idempotent_on_rerun(self):
        import os, sqlite3, tempfile
        from src.paper_manager import PaperManager
        d = tempfile.mkdtemp()
        db = os.path.join(d, "pm.db")
        PaperManager(db_path=db)            # first run: migrates 0 -> 11
        PaperManager(db_path=db)            # second run: must be a no-op (no raise)
        conn = sqlite3.connect(db)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(trades)")]
        self.assertEqual(cols.count("quantity"), 1)  # column not duplicated
        v = conn.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(v, 11)
        conn.close()

if __name__ == "__main__":
    unittest.main()
