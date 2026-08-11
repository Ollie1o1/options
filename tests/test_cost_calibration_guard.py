"""The calibration guard where it actually bites: the grade and the ledger.

Two live paths applied friction thresholds at any tenor:

  * `worth.assess` capped anything above 10% friction at THIN, and past 250 DTE
    the median candidate is above that — so STRONG and CLEAR were unreachable
    by cost alone, at any quality.
  * `PaperManager.log_trade` refused on the 25% gate with a message reading
    "the spread eats the trade", which is a verdict about the candidate rather
    than an admission that the threshold was never measured there.

Neither threshold changes. See `src/cost_calibration.py`.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_cost_calibration_guard -v
"""
from __future__ import annotations

import datetime as dt
import unittest

from src import worth
from src.cost_calibration import CALIBRATED_DTE
from src.paper_manager import PaperManager


def _exp(days: int) -> str:
    return (dt.date(2024, 1, 1) + dt.timedelta(days=days)).isoformat()


class WorthOutsideCalibrationTest(unittest.TestCase):
    """A grade that cannot be earned should say so, not say THIN."""

    def _row(self, days: int):
        return {"date": "2024-01-01", "expiration": _exp(days),
                "ev_per_contract": 50.0, "ev_noise": 10.0}

    def test_a_long_dated_candidate_is_ungraded_not_thin(self):
        w = worth.assess(self._row(600))
        self.assertEqual(w.grade, "UNGRADED")

    def test_it_says_why(self):
        w = worth.assess(self._row(600))
        self.assertIn("calibrat", (w.limiting or "").lower())

    def test_the_boundary_is_still_graded(self):
        w = worth.assess(self._row(CALIBRATED_DTE[1]))
        self.assertNotIn("calibrat", (w.limiting or "").lower())

    def test_an_ordinary_tenor_is_untouched(self):
        # 45 DTE is inside the live filter's own range; the guard must be
        # invisible to everything the book actually trades.
        w = worth.assess(self._row(45))
        self.assertNotIn("calibrat", (w.limiting or "").lower())

    def test_a_row_with_no_tenor_is_still_graded(self):
        w = worth.assess({"ev_per_contract": 50.0, "ev_noise": 10.0})
        self.assertNotIn("calibrat", (w.limiting or "").lower())


class LedgerRefusesOutsideCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.mgr = PaperManager(db_path=":memory:")

    def _trade(self, days: int):
        return {"strategy_name": "Bull Put", "ticker": "AAA",
                "date": "2024-01-01", "expiration": _exp(days),
                "strike": 100.0, "type": "put", "entry_price": 1.0,
                "contracts": 1, "capital_at_risk": 400.0, "status": "OPEN",
                "quality_score": 50.0,
                # Not what these tests are about; the affordability gate is
                # covered elsewhere and would otherwise refuse every fixture.
                "allow_unaffordable": True}

    def _logged(self) -> int:
        with self.mgr._get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

    def test_a_long_dated_trade_is_refused(self):
        self.assertFalse(self.mgr.log_trade_if_new(self._trade(600)))
        self.assertEqual(self._logged(), 0)

    def test_an_ordinary_tenor_is_still_logged(self):
        # The guard must refuse only what it is aimed at. All 972 real trades
        # are DTE 8-59.
        self.mgr.log_trade_if_new(self._trade(45))
        self.assertEqual(self._logged(), 1)

    def test_the_boundary_is_still_logged(self):
        self.mgr.log_trade_if_new(self._trade(CALIBRATED_DTE[1]))
        self.assertEqual(self._logged(), 1)

    def test_a_deliberate_manual_entry_can_override(self):
        # Same escape hatch the friction gate already honours: this refuses
        # auto-logging, it does not forbid the trade.
        t = self._trade(600)
        t["allow_untradeable"] = True
        self.mgr.log_trade_if_new(t)
        self.assertEqual(self._logged(), 1)


if __name__ == "__main__":
    unittest.main()
