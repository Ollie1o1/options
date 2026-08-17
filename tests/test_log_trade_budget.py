"""A trade may carry the budget that was in force when it was chosen.

`budget_at_entry` rides on the trade dict, exactly as `allow_unaffordable`
already does, because the budget is a property of the DECISION rather than of
the manager — and eight log sites funnel through `log_trade`.

KEY PRESENCE IS THE SIGNAL:
    key present, value None  -> operator chose NO LIMIT
    key present, value float -> that ceiling
    key ABSENT               -> fall back to config (the scheduler's path)

The last case must stay byte-identical to current behaviour: a non-interactive
run that never saw a budget prompt must never be treated as having chosen
"no limit".
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from src.paper_manager import PaperManager


def _trade(**over):
    t = {
        # expiration is 37 DTE from date — inside the cost model's calibrated
        # band (<= 67 DTE), so the untradeable-DTE gate never fires here and
        # these cases exercise the budget gate only.
        "date": "2026-08-14", "ticker": "TSTX", "expiration": "2026-09-20",
        "strike": 100.0, "type": "put", "entry_price": 2.00,
        "quality_score": 0.5, "strategy_name": "Long Put",
    }
    t.update(over)
    return t


class BudgetCase(unittest.TestCase):
    """Every case builds its own ledger in a temp dir. NEVER the real book."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self._tmp.name, "ledger.db")

    def tearDown(self):
        self._tmp.cleanup()

    def _mgr(self, cap):
        m = PaperManager(db_path=self.db)
        m._max_capital_at_risk = cap      # stand in for config
        return m

    def _stored(self):
        with sqlite3.connect(self.db) as c:
            return list(c.execute(
                "SELECT ticker, budget_at_entry FROM trades ORDER BY entry_id"))


class TestKeyAbsentFallsBackToConfig(BudgetCase):

    def test_absent_key_is_refused_by_the_config_cap(self):
        m = self._mgr(100.0)          # $100 cap, position risks $200
        self.assertFalse(m.log_trade(_trade()))

    def test_absent_key_is_allowed_under_a_generous_config_cap(self):
        m = self._mgr(100000.0)
        self.assertTrue(m.log_trade(_trade()))

    def test_absent_key_stores_the_config_cap(self):
        m = self._mgr(100000.0)
        m.log_trade(_trade())
        self.assertEqual(self._stored()[0][1], 100000.0)


class TestExplicitBudgetOverridesConfig(BudgetCase):

    def test_none_means_no_limit_even_when_config_would_refuse(self):
        m = self._mgr(100.0)          # config would refuse a $200 position
        self.assertTrue(m.log_trade(_trade(budget_at_entry=None)))

    def test_none_is_stored_as_null(self):
        m = self._mgr(100.0)
        m.log_trade(_trade(budget_at_entry=None))
        self.assertIsNone(self._stored()[0][1])

    def test_a_number_binds_even_when_config_is_generous(self):
        m = self._mgr(100000.0)
        self.assertFalse(m.log_trade(_trade(budget_at_entry=100.0)))

    def test_a_generous_number_admits_and_is_stored(self):
        m = self._mgr(100.0)
        self.assertTrue(m.log_trade(_trade(budget_at_entry=50000.0)))
        self.assertEqual(self._stored()[0][1], 50000.0)


class TestTheExistingEscapeHatchStillWorks(BudgetCase):

    def test_allow_unaffordable_bypasses_an_explicit_budget(self):
        m = self._mgr(100000.0)
        self.assertTrue(m.log_trade(
            _trade(budget_at_entry=1.0, allow_unaffordable=True)))


if __name__ == "__main__":
    unittest.main()
