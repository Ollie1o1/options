"""Tests for the long-term fills store (data/longterm.db)."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.longterm import fills as F


class TestFills(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "longterm.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_and_read_back(self):
        fid = F.record_fill("mu", 750, 2.5, 748.20, fill_date="2026-07-21",
                            note="tranche 1", db_path=self.db)
        rows = F.fills_for("MU", db_path=self.db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].id, fid)
        self.assertEqual(rows[0].ticker, "MU")           # uppercased
        self.assertEqual(rows[0].fill_date, "2026-07-21")

    def test_filled_levels(self):
        F.record_fill("MU", 750, 1, 749, db_path=self.db)
        F.record_fill("MU", 650, 1, 640, db_path=self.db)
        F.record_fill("ASML", 900, 1, 899, db_path=self.db)
        self.assertEqual(F.filled_levels("MU", db_path=self.db), {750.0, 650.0})

    def test_book_cost_basis(self):
        F.record_fill("MU", 750, 2, 750.0, db_path=self.db)
        F.record_fill("MU", 650, 2, 650.0, db_path=self.db)
        bk = F.book(db_path=self.db)
        self.assertAlmostEqual(bk["MU"]["shares"], 4.0)
        self.assertAlmostEqual(bk["MU"]["cost"], 2800.0)
        self.assertAlmostEqual(bk["MU"]["avg_price"], 700.0)

    def test_deployed_usd(self):
        F.record_fill("MU", 750, 2, 750.0, db_path=self.db)
        F.record_fill("KO", 60, 10, 59.0, db_path=self.db)
        self.assertAlmostEqual(F.deployed_usd(db_path=self.db), 2090.0)

    def test_rejects_nonpositive(self):
        with self.assertRaises(ValueError):
            F.record_fill("MU", 750, 0, 749, db_path=self.db)
        with self.assertRaises(ValueError):
            F.record_fill("MU", 750, 1, -1, db_path=self.db)

    def test_empty_db(self):
        self.assertEqual(F.fills_for(db_path=self.db), [])
        self.assertEqual(F.book(db_path=self.db), {})
        self.assertEqual(F.deployed_usd(db_path=self.db), 0.0)


if __name__ == "__main__":
    unittest.main()
