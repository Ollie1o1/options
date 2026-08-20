"""Sizing at the chokepoint: what `log_trade` does with a SizingDecision.

`log_trade` is the one place a capital decision cannot be routed around, so the
assertions here are about the ledger, not the arithmetic (tests/test_book_sizing.py
covers that): does a refused trade leave the table untouched, does the stored
`capital_at_risk` describe the position actually taken, and does `quantity`
survive the INSERT at all.

Every test builds its own temp database and temp config. Nothing here names the
real ledger — `PaperManager` migrates on construction, so a stray relative path
would migrate the real book.

Reference: docs/BOOK_SIZING_SPEC.md §6, tests 9-12.
"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paper_manager import PaperManager

# Inside the cost-calibration window (10-67 DTE) so the tradeability gate that
# runs BEFORE sizing cannot be what refuses these trades.
_EXPIRY = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
_TODAY = datetime.now().strftime("%Y-%m-%d")
_YESTERDAY = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


def _write_config(path, **sizing):
    block = {
        "enabled": True,
        "opening_balance": 50_000.0,
        "equity_basis_date": "2026-08-05",
        # The sized era starts before "now" so trades logged by these tests are
        # inside it; grandfathering is asserted separately.
        "sizing_start_date": "2000-01-01",
        "max_risk_pct": 0.02,
        "max_open_risk_pct": 0.10,
    }
    block.update(sizing)
    cfg = {
        "exit_rules": {"take_profit": 0.50, "stop_loss": -0.25},
        "paper_trading": {"commission_per_contract": 0.0,
                          "slippage_per_share": 0.0},
        # No budget cap and no friction cap: those gates run first and this file
        # is about the one that runs last.
        "auto_log": {"max_capital_at_risk": None, "max_friction_to_credit": None},
        "position_sizing": block,
    }
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


def _bull_put(**over):
    """$5 wide, $1.08 credit -> $392 of risk per contract."""
    trade = {
        "ticker": "WMT", "expiration": _EXPIRY, "strike": 100.0, "type": "put",
        "entry_price": 1.08, "quality_score": 0.75,
        "strategy_name": "Bull Put Spread",
        "long_strike": 95.0, "spread_width": 5.0, "net_credit": 1.08,
        "max_loss_usd": 392.0, "max_profit_usd": 108.0,
    }
    trade.update(over)
    return trade


class SizedInsert(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.dir.name, "book.db")
        self.cfg = _write_config(os.path.join(self.dir.name, "config.json"))

    def tearDown(self):
        self.dir.cleanup()

    def _pm(self, **sizing):
        if sizing:
            _write_config(self.cfg, **sizing)
        return PaperManager(db_path=self.db, config_path=self.cfg)

    def _rows(self):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM trades")]
        finally:
            conn.close()

    def test_quantity_survives_the_insert(self):
        # $50,000 x 2% = $1,000 budget; $392 per contract -> 2.
        self.assertTrue(self._pm().log_trade(_bull_put()))
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["quantity"], 2.0)

    def test_capital_at_risk_is_stored_at_the_sized_quantity(self):
        # Spec test 10. The column has to describe the position actually taken:
        # two contracts of a $392 spread tie up $784, not $392. A number that
        # describes something other than its label is the exact defect class
        # this whole change exists to remove.
        self._pm().log_trade(_bull_put())
        self.assertAlmostEqual(self._rows()[0]["capital_at_risk"], 784.0)

    def test_refusal_inserts_no_row(self):
        # Spec test 9. $1,468 of risk against a $1,000 budget sizes below one
        # contract, and sizing is a gate: the trade does not happen.
        pm = self._pm()
        self.assertFalse(pm.log_trade(_bull_put(
            ticker="CRM", spread_width=20.0, net_credit=5.32,
            max_loss_usd=1_468.0, entry_price=5.32)))
        self.assertEqual(self._rows(), [])
        self.assertEqual(pm.unsized_rejected, 1)

    def test_allow_unsized_logs_one_contract_despite_the_refusal(self):
        # Spec test 11. Same shape as allow_unaffordable / allow_untradeable:
        # a key in the trade dict, for a deliberate manual entry.
        pm = self._pm()
        self.assertTrue(pm.log_trade(_bull_put(
            ticker="CRM", spread_width=20.0, net_credit=5.32,
            max_loss_usd=1_468.0, entry_price=5.32, allow_unsized=True)))
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["quantity"], 1.0)
        self.assertAlmostEqual(rows[0]["capital_at_risk"], 1_468.0)
        self.assertEqual(pm.unsized_rejected, 0)

    def test_disabled_logs_exactly_one_contract(self):
        # Spec test 5 at the ledger: today's behaviour, unchanged and explicit.
        self.assertTrue(self._pm(enabled=False).log_trade(_bull_put()))
        rows = self._rows()
        self.assertEqual(rows[0]["quantity"], 1.0)
        self.assertAlmostEqual(rows[0]["capital_at_risk"], 392.0)

    def test_open_risk_from_earlier_trades_reduces_later_ones(self):
        # The concurrent cap reads the ledger, so it has to see the rows this
        # same manager just wrote. $5,000 ceiling, $784 per 2-lot entry: the
        # sixth fills to $4,704 and the seventh has no headroom left. Without
        # this cap a per-trade rule permits unbounded correlated exposure.
        pm = self._pm()
        for i, ticker in enumerate(["A", "B", "C", "D", "E", "F", "G", "H"]):
            pm.log_trade(_bull_put(ticker=ticker, strike=100.0 + i))
        rows = self._rows()
        self.assertEqual(len(rows), 6)
        self.assertEqual(pm.unsized_rejected, 2)
        self.assertLessEqual(sum(r["capital_at_risk"] for r in rows), 5_000.0)

    def test_grandfathered_positions_do_not_block_new_trades(self):
        # Spec test 8 at the ledger: a legacy position far over the ceiling was
        # opened before the sized era and must not refuse everything after it.
        pm = self._pm()
        conn = sqlite3.connect(self.db)
        conn.execute(
            "INSERT INTO trades (date, ticker, expiration, strike, type, "
            "entry_price, quality_score, strategy_name, status, capital_at_risk)"
            " VALUES ('1999-01-01', 'OLD', '1999-02-01', 1.0, 'put', 1.0, 0.5,"
            " 'Bull Put Spread', 'OPEN', 176323.0)")
        conn.commit()
        conn.close()
        self.assertTrue(pm.log_trade(_bull_put()))

    def test_realised_losses_shrink_the_next_position(self):
        # The account basis is the book's own equity, compounding: after a
        # $20,000 realised loss the 2% budget is $600 and a $392 spread sizes
        # to 1 contract instead of 2.
        pm = self._pm(equity_basis_date="2000-01-01")
        conn = sqlite3.connect(self.db)
        conn.execute(
            "INSERT INTO trades (date, ticker, expiration, strike, type, "
            "entry_price, quality_score, strategy_name, status, pnl_usd, "
            "exit_date) VALUES (?, 'OLD', '2026-02-01', 1.0, 'put', 1.0, 0.5,"
            " 'Bull Put Spread', 'CLOSED', -20000.0, ?)", (_YESTERDAY, _TODAY))
        conn.commit()
        conn.close()
        self.assertTrue(pm.log_trade(_bull_put()))
        self.assertEqual(self._rows()[-1]["quantity"], 1.0)

    def test_a_caller_supplied_quantity_is_honoured(self):
        # The crypto ledger sizes its own rows by unit risk (core.sizing.
        # capped_quantity) before calling, in fractional coins. Book sizing is
        # calibrated on the equity book and must not overwrite that.
        pm = self._pm()
        self.assertTrue(pm.log_trade(_bull_put(quantity=0.5)))
        self.assertEqual(self._rows()[0]["quantity"], 0.5)
        self.assertAlmostEqual(self._rows()[0]["capital_at_risk"], 196.0)

    def test_sizing_never_pushes_a_position_over_the_budget_gate(self):
        # The budget gate runs BEFORE sizing and sees one contract. Multiplying
        # up afterwards must not carry the position past the cap it just
        # cleared, or `capital_at_risk` would exceed `budget_at_entry` on a row
        # the budget gate had approved.
        _write_config(self.cfg)
        with open(self.cfg) as f:
            cfg = json.load(f)
        cfg["auto_log"]["max_capital_at_risk"] = 600.0
        with open(self.cfg, "w") as f:
            json.dump(cfg, f)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        self.assertTrue(pm.log_trade(_bull_put()))
        row = self._rows()[0]
        self.assertEqual(row["quantity"], 1.0)
        self.assertLessEqual(row["capital_at_risk"], row["budget_at_entry"])


class RefusalsAreCountable(unittest.TestCase):
    """What the auto-log summary line arithmetic depends on.

    The feeder counts every `False` as `_skipped` and then subtracts the
    manager's refusal counters to work out how many really were duplicates:
    `_dupes = _skipped - _refused - _near_dupes - _unsized`. If a sizing
    refusal did not increment its own counter it would be reported as a
    duplicate — the same defect that once printed "skipped 5 duplicates" for a
    window that logged nothing because everything was over budget.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.dir.name, "book.db")
        self.cfg = _write_config(os.path.join(self.dir.name, "config.json"))
        self.pm = PaperManager(db_path=self.db, config_path=self.cfg)

    def tearDown(self):
        self.dir.cleanup()

    def _too_big(self, **over):
        return _bull_put(ticker="CRM", spread_width=20.0, net_credit=5.32,
                         max_loss_usd=1_468.0, entry_price=5.32, **over)

    def test_the_if_new_wrapper_reports_a_sizing_refusal_as_a_refusal(self):
        self.assertFalse(self.pm.log_trade_if_new(self._too_big(), auto_log=True))
        self.assertEqual(self.pm.unsized_rejected, 1)
        self.assertEqual(self.pm.duplicate_rejected, 0)
        self.assertEqual(self.pm.unaffordable_rejected, 0)

    def test_a_spread_refusal_counts_once_through_log_spread(self):
        spread = {"ticker": "CRM", "expiration": _EXPIRY,
                  "short_strike": 100.0, "long_strike": 80.0, "type": "Bull Put",
                  "net_credit": 5.32, "max_profit": 532.0, "max_loss": 1_468.0}
        self.assertFalse(self.pm.log_spread(spread))
        self.assertEqual(self.pm.unsized_rejected, 1)

    def test_the_counter_is_per_manager_not_global(self):
        other = PaperManager(db_path=self.db, config_path=self.cfg)
        self.pm.log_trade(self._too_big())
        self.assertEqual(other.unsized_rejected, 0)


class RealLedgerUntouched(unittest.TestCase):
    """Spec test 12 — these tests write no book but their own.

    Asserted on the mechanism rather than on the source text: `PaperManager`
    resolves a RELATIVE db_path against the repo root and migrates it on
    construction, so the thing that keeps the real book safe is that every
    manager here is handed an absolute temp path that passes straight through.
    """

    def test_every_manager_here_is_anchored_on_a_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "book.db")
            pm = PaperManager(db_path=db,
                              config_path=_write_config(
                                  os.path.join(tmp, "config.json")))
            self.assertEqual(pm.db_path, db)
            self.assertTrue(os.path.exists(db))
            self.assertTrue(os.path.isabs(pm.db_path))


if __name__ == "__main__":
    unittest.main()
