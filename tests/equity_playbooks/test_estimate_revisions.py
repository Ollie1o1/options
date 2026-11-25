"""Offline unit tests for scripts/estimate_revisions_test.py."""
from __future__ import annotations

import datetime as dt
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import estimate_revisions_test as mod


def _make_archive_db(path: Path) -> None:
    con = sqlite3.connect(str(path))
    con.execute(
        """
        CREATE TABLE estimate_snapshots (
            snap_date TEXT,
            symbol TEXT,
            dataset TEXT,
            period TEXT,
            field TEXT,
            value REAL
        )
        """
    )
    rows = [
        # Valid 2026-10-14: AAPL has 4 up, 1 down => (4-1)/5 = +0.6
        ("2026-10-14", "AAPL", "eps_revisions", "0q", "upLast30days", 2.0),
        ("2026-10-14", "AAPL", "eps_revisions", "+1y", "upLast30days", 2.0),
        ("2026-10-14", "AAPL", "eps_revisions", "0q", "downLast30days", 1.0),
        # Total < 2 should be ignored
        ("2026-10-14", "MSFT", "eps_revisions", "0q", "upLast30days", 1.0),
        # Before cutoff date should be ignored
        ("2026-10-01", "AAPL", "eps_revisions", "0q", "upLast30days", 5.0),
        # Second non-overlapping snapshot (>=21d later)
        ("2026-11-10", "AAPL", "eps_revisions", "0q", "downLast30days", 3.0),
        ("2026-11-10", "AAPL", "eps_revisions", "+1y", "upLast30days", 1.0),
    ]
    con.executemany("INSERT INTO estimate_snapshots VALUES (?, ?, ?, ?, ?, ?)", rows)
    con.commit()
    con.close()


class TestEstimateRevisions(unittest.TestCase):
    def test_compute_rev_breadth_30d(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "est.db"
            _make_archive_db(db)
            res = mod.compute_rev_breadth_30d(str(db))
            self.assertEqual(sorted(res.keys()), ["2026-10-14", "2026-11-10"])
            self.assertAlmostEqual(res["2026-10-14"]["AAPL"], 0.6, places=6)
            self.assertNotIn("MSFT", res["2026-10-14"])
            self.assertAlmostEqual(res["2026-11-10"]["AAPL"], -0.5, places=6)

    def test_select_non_overlapping_snapshots(self) -> None:
        dates = ["2026-10-14", "2026-10-20", "2026-11-04", "2026-11-09", "2026-11-30"]
        kept = mod.select_non_overlapping_snapshots(dates, min_cal_gap_days=21)
        self.assertEqual(kept, ["2026-10-14", "2026-11-04", "2026-11-30"])

    def test_date_gate_refuses_before_unlock_and_allows_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "est.db"
            _make_archive_db(db)
            rc_cov = mod.main(["--coverage-only", "--db", str(db)])
            self.assertEqual(rc_cov, 0)

            class _FakeDate(dt.date):
                @classmethod
                def today(cls) -> dt.date:
                    return cls(2026, 11, 24)

            with mock.patch.object(mod.dt, "date", _FakeDate):
                rc_refused = mod.main(["--db", str(db)])
                self.assertEqual(rc_refused, 2)


if __name__ == "__main__":
    unittest.main()
