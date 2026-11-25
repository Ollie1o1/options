"""Offline unit tests for src.own_picks v2 stock/ETF playbook journal & scorecard."""
from __future__ import annotations

import datetime as dt
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.own_picks import (
    build_equity_scorecard,
    close_equity_pick,
    list_equity_picks,
    log_equity_pick,
)


def _make_ohlcv_db(path: Path) -> None:
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE ohlcv (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)"
    )
    d0 = dt.date(2026, 10, 1)
    dates = []
    cur = d0
    while len(dates) < 35:
        if cur.weekday() < 5:
            dates.append(cur.isoformat())
        cur += dt.timedelta(days=1)
    rows = []
    for i, d in enumerate(dates):
        rows.append(("SPY", d, 500 + i * 0.5, 500 + i * 0.5, 500 + i * 0.5, 500 + i * 0.5, 1e7))
        rows.append(("AAPL", d, 200 + i * 1.5, 200 + i * 1.5, 200 + i * 1.5, 200 + i * 1.5, 5e6))
    con.executemany("INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    con.commit()
    con.close()


class TestOwnPicksEquityV2(unittest.TestCase):
    def test_log_close_and_scorecard_by_playbook(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "own_picks.db")
            eq_db = Path(td) / "eq.db"
            _make_ohlcv_db(eq_db)

            pid = log_equity_pick(
                "AAPL",
                200.0,
                shares=10,
                stop_price=188.0,
                target_price=215.0,
                playbook_id="equity_pullback",
                entry_date="2026-10-01",
                notes="RSI2 pullback",
                db_path=db,
            )
            self.assertEqual(pid, 1)
            self.assertEqual(len(list_equity_picks(status="OPEN", db_path=db)), 1)

            sc = build_equity_scorecard(db_path=db, ohlcv_db=str(eq_db))
            self.assertEqual(sc["total_picks"], 1)
            pb = sc["by_playbook"]["equity_pullback"]
            self.assertEqual(pb["n_picks"], 1)
            self.assertEqual(pb["horizons"][5]["verdict"], "too few to judge")
            self.assertGreater(pb["horizons"][5]["mean"], 0.0)

            ok = close_equity_pick(pid, 212.0, exit_date="2026-10-15", db_path=db)
            self.assertTrue(ok)
            sc2 = build_equity_scorecard(db_path=db, ohlcv_db=str(eq_db))
            self.assertAlmostEqual(sc2["by_playbook"]["equity_pullback"]["closed_mean_ret"], 0.06, places=4)


if __name__ == "__main__":
    unittest.main()
