"""Discover lens forward record: idempotent recording, the preregistered lock, the read."""
from __future__ import annotations

import datetime as dt
import math
import os
import sqlite3
import tempfile
import unittest
from typing import Dict, List, Tuple

import numpy as np

from src.topdown import forward_log as F
from src.topdown.lenses import LENS_BY_ID, LensResult, LensRow
from src.topdown.synthetic import business_days


def _res(lens_id: str, rows: List[Tuple[str, str]]) -> LensResult:
    r = LensResult(LENS_BY_ID[lens_id])
    r.rows = [LensRow(ticker=tk, sector_etf="XLK", why="t", as_of=d) for tk, d in rows]
    return r


def _market(n: int = 260) -> Tuple[List[str], Dict[str, Tuple[List[str], np.ndarray, np.ndarray]]]:
    days = business_days(n)
    rng = np.random.default_rng(1)
    spy = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    t = np.arange(n)
    aaa = spy * np.exp(0.002 * t)             # +0.2%/session over SPY, every window
    zzz = spy * np.exp(-0.002 * t)            # -0.2%/session
    xlk = spy.copy()
    vol = np.full(n, 1e6)
    return days, {"SPY": (days, spy, vol), "AAA": (days, aaa, vol), "ZZZ": (days, zzz, vol), "XLK": (days, xlk, vol)}


class RecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self._tmp.name, "lf.db")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_recording_is_idempotent_and_keys_on_the_rows_own_date(self) -> None:
        res = [_res("leaders", [("AAA", "2026-11-20"), ("ZZZ", "2026-11-20")]), _res("insiders", [])]
        a = F.record_results(res, db_path=self.db, record_date="2026-11-20")
        b = F.record_results(res, db_path=self.db, record_date="2026-11-23")  # stale cache re-recorded
        self.assertEqual((a["rows_new"], b["rows_new"]), (2, 0))
        con = sqlite3.connect(self.db)
        try:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM lens_rows").fetchone()[0], 2)
            runs = con.execute("SELECT record_date, lens_id, n_rows FROM lens_runs ORDER BY 1, 2").fetchall()
        finally:
            con.close()
        self.assertIn(("2026-11-20", "insiders", 0), runs)  # empty runs are data too
        self.assertEqual(len(runs), 4)

    def test_due_after_close_on_trading_days_once(self) -> None:
        self.assertFalse(F.due({}, "2026-11-25", 1500))
        self.assertFalse(F.due({}, "2026-11-25", 1759))
        self.assertTrue(F.due({}, "2026-11-25", 1800))
        self.assertFalse(F.due({"last_lens_snapshot": "2026-11-25"}, "2026-11-25", 1800))
        self.assertFalse(F.due({}, "2026-11-28", 1800))  # Saturday

    def test_scheduled_snapshot_rejects_stale_prices_without_writing(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        hub = SimpleNamespace(board=SimpleNamespace(as_of="2026-11-24"))
        with patch("src.topdown.hub.build_hub", return_value=hub):
            outcome = F.run_snapshot(db_path=self.db, expected_session="2026-11-25")
        self.assertFalse(outcome["complete"])
        self.assertFalse(os.path.exists(self.db))


class ReadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self._tmp.name, "lf.db")
        self.days, self.ohlcv = _market()
        fridays = [d for d in self.days[:-30] if dt.date.fromisoformat(d).weekday() == 4]
        self.fridays = fridays
        for d in fridays:
            F.record_results([_res("leaders", [("AAA", d)]), _res("landmines", [("ZZZ", d)])],
                             db_path=self.db, record_date=d)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_locked_before_the_preregistered_date_even_with_data(self) -> None:
        rep = {r.lens_id: r for r in F.forward_report(db_path=self.db, ohlcv=self.ohlcv, today="2027-05-24")}
        self.assertTrue(rep["leaders"].locked)
        self.assertIsNone(rep["leaders"].mean_excess)
        self.assertIn("locked until", rep["leaders"].lock_reason)
        self.assertGreaterEqual(rep["leaders"].weeks_recorded, F.MIN_WEEKS)

    def test_read_after_unlock_measures_excess_over_spy(self) -> None:
        rep = {r.lens_id: r for r in F.forward_report(db_path=self.db, ohlcv=self.ohlcv, today="2027-06-01")}
        lead, mine = rep["leaders"], rep["landmines"]
        self.assertFalse(lead.locked)
        expected = 0.002 * F.HORIZONS["leaders"] - F.COST
        self.assertAlmostEqual(lead.mean_excess, expected, places=9)
        assert lead.ci_bonf is not None and mine.ci_bonf is not None
        self.assertGreater(lead.ci_bonf[0], 0)
        self.assertEqual(lead.proposal, "propose PROMISING")      # CONTEXT -> PROMISING
        self.assertEqual(mine.proposal, "confirmed AVOID")
        self.assertLess(mine.ci_bonf[1], 0)
        self.assertTrue(rep["insiders"].locked)                    # nothing recorded

    def test_entry_is_the_session_after_the_data_date(self) -> None:
        # Put a +10% jump in AAA on a recorded data date: it must not be in the outcome.
        d0 = self.fridays[3]
        i = self.days.index(d0)
        dates, px, vol = self.ohlcv["AAA"]
        jumped = px.copy()
        jumped[i:] *= 1.10
        ohlcv = dict(self.ohlcv)
        ohlcv["AAA"] = (dates, jumped, vol)
        rep = {r.lens_id: r for r in F.forward_report(db_path=self.db, ohlcv=ohlcv, today="2027-06-01")}
        week = dict(rep["leaders"].weekly)[F._iso_week(d0)]
        self.assertAlmostEqual(week, 0.002 * F.HORIZONS["leaders"] - F.COST, places=9)

    def test_too_few_weeks_stays_locked_after_the_date(self) -> None:
        db2 = os.path.join(self._tmp.name, "few.db")
        for d in self.fridays[:5]:
            F.record_results([_res("leaders", [("AAA", d)])], db_path=db2, record_date=d)
        rep = {r.lens_id: r for r in F.forward_report(db_path=db2, ohlcv=self.ohlcv, today="2027-06-01")}
        self.assertTrue(rep["leaders"].locked)
        self.assertIn(f"5/{F.MIN_WEEKS}", rep["leaders"].lock_reason)
        self.assertIn("read unlocks", F.status_line(list(rep.values())))

    def test_late_capture_cannot_use_an_earlier_entry_date(self) -> None:
        db = os.path.join(self._tmp.name, "late.db")
        F.record_results([_res("leaders", [("AAA", "2026-11-20")])],
                         db_path=db, record_date="2026-11-23")
        week = F.lens_weeks(db)["leaders"][F._iso_week("2026-11-20")]
        self.assertEqual(week["AAA"], ("2026-11-23", "XLK"))

    def test_bonferroni_level(self) -> None:
        self.assertTrue(math.isclose(F.BONFERRONI_LEVEL, 1 - 0.05 / 6))


if __name__ == "__main__":
    unittest.main()
