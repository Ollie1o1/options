"""Schema v25 records who opened a trade: an automated feeder or the operator.

Until now nothing in the ledger separated the two, so the operator's own picks
could not be measured apart from the auto-logger's. `NULL` means the row
predates the column and the answer was never recorded — not "manual".
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta

from src import paper_manager as pm

# Inside the cost model's calibrated tenor range; a far-dated contract is
# refused before the insert and would test nothing.
_EXPIRY = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")


def _trade(**overrides):
    trade = {"ticker": "AAPL", "expiration": _EXPIRY, "strike": 150.0,
             "type": "call", "entry_price": 3.50, "quality_score": 0.75,
             "strategy_name": "Long Call"}
    trade.update(overrides)
    return trade


def _spread(**overrides):
    spread = {"ticker": "MSFT", "expiration": _EXPIRY, "type": "Bull Put",
              "short_strike": 400.0, "long_strike": 395.0, "net_credit": 1.20,
              "max_profit": 120.0, "max_loss": 380.0}
    spread.update(overrides)
    return spread


class TestTradeOrigin(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = os.path.join(self._tmp.name, "ledger.db")
        self.cfg = os.path.join(self._tmp.name, "config.json")
        with open(self.cfg, "w") as f:
            json.dump({"exit_rules": {"take_profit": 0.5, "stop_loss": -0.25,
                                      "time_exit_dte": 21},
                       "paper_trading": {"commission_per_contract": 0.65,
                                         "slippage_per_share": 0.0}}, f)
        self.mgr = pm.PaperManager(db_path=self.db, config_path=self.cfg)

    def _origins(self):
        with sqlite3.connect(self.db) as conn:
            return [r[0] for r in conn.execute(
                "SELECT origin FROM trades ORDER BY entry_id")]

    def test_migration_25_is_registered_and_current(self):
        self.assertIn(25, pm._MIGRATIONS)
        self.assertGreaterEqual(pm._SCHEMA_VERSION, 25)

    def test_a_fresh_ledger_has_the_column(self):
        # The version stamp is not evidence the column exists (2026-09-08);
        # look at the table.
        with sqlite3.connect(self.db) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(trades)")]
        self.assertIn("origin", cols)

    def test_a_direct_log_is_manual(self):
        self.assertTrue(self.mgr.log_trade(_trade()))
        self.assertEqual(self._origins(), ["manual"])

    def test_an_auto_logged_trade_is_auto(self):
        self.assertTrue(self.mgr.log_trade_if_new(_trade(), auto_log=True))
        self.assertEqual(self._origins(), ["auto"])

    def test_the_interactive_dedup_helper_stays_manual(self):
        # log_trade_if_new is also the interactive log-trades menu's entry
        # point; without the flag it is still the operator's trade.
        self.assertTrue(self.mgr.log_trade_if_new(_trade()))
        self.assertEqual(self._origins(), ["manual"])

    def test_spreads_carry_the_origin_through(self):
        self.assertTrue(self.mgr.log_spread(_spread()))
        self.assertTrue(self.mgr.log_spread_if_new(_spread(ticker="NVDA"), auto_log=True))
        self.assertEqual(self._origins(), ["manual", "auto"])

    def test_rows_that_predate_the_column_stay_null(self):
        tmp = os.path.join(self._tmp.name, "old.db")
        conn = sqlite3.connect(tmp)
        conn.execute("CREATE TABLE trades (date TEXT, ticker TEXT)")
        conn.execute("INSERT INTO trades VALUES ('2026-09-01', 'OLD')")
        for sql in pm._MIGRATIONS[25]:
            conn.execute(sql)
        conn.commit()
        self.assertEqual(conn.execute("SELECT origin FROM trades").fetchall(), [(None,)])
        conn.close()


if __name__ == "__main__":
    unittest.main()
