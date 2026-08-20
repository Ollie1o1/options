"""Realised dollars must count the contracts actually held.

`_sanitize_close_values` derived `pnl_usd = entry_price x pnl_pct x multiplier`
and no exit path scaled it, so every realised dollar figure in this system
described ONE contract regardless of the position's size. That was invisible
while every row carried the migration default `quantity = 1.0`; position sizing
(src/book_sizing.py) writes 2 and 3, and from that moment a two-lot winner would
have been booked at half its value — including into `book_equity`, which
compounds the next position's size off exactly this column.

The crypto ledger already knew: `crypto/exit_enforcer.py` reads the row's
quantity and scales the value this function returns. The equity book did not.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_realised_pnl_quantity -v
"""
from __future__ import annotations

import sqlite3
import unittest
from datetime import date, timedelta

from src import check_pnl as cp
from src.paper_manager import _sanitize_close_values
from tests.test_mark_trustworthiness import _MarkTestCase


class SanitiseScalesByQuantity(unittest.TestCase):

    def test_two_contracts_realise_twice_the_dollars(self):
        one = _sanitize_close_values("Long Call", 3.50, 7.00, 1.0)[2]
        two = _sanitize_close_values("Long Call", 3.50, 7.00, 1.0, quantity=2.0)[2]
        self.assertAlmostEqual(one, 350.0)
        self.assertAlmostEqual(two, 700.0)

    def test_percentages_are_untouched_by_size(self):
        # Only the dollar figure scales: a return is a return at any size, and
        # pnl_pct feeds the IC sample, where scaling it would be a silent bug.
        _, pct_one, _ = _sanitize_close_values("Long Call", 3.50, 7.00, 1.0)
        exit_two, pct_two, _ = _sanitize_close_values(
            "Long Call", 3.50, 7.00, 1.0, quantity=2.0)
        self.assertEqual(pct_one, pct_two)
        self.assertEqual(exit_two, 7.00)

    def test_default_is_one_contract(self):
        # The crypto exit enforcer passes no quantity and scales the result
        # itself; changing the default would double-count its rows.
        self.assertAlmostEqual(
            _sanitize_close_values("Long Call", 3.50, 1.75, -0.5)[2], -175.0)

    def test_a_missing_or_absurd_quantity_falls_back_to_one(self):
        for bad in (None, 0.0, -3.0, float("nan"), "two"):
            self.assertAlmostEqual(
                _sanitize_close_values("Long Call", 3.50, 7.00, 1.0,
                                       quantity=bad)[2],
                350.0, msg=f"quantity={bad!r}")

    def test_a_credit_spread_scales_too(self):
        # $1.00 credit, closed at max loss on a $5 spread: -$400 a lot. The
        # name is the ledger's exact one — `_CREDIT_STRUCTURES` matches whole
        # strings, so "Bull Put Spread" would take the long-premium branch and
        # clamp at -100%. No row in either book is named that way.
        one = _sanitize_close_values("Bull Put", 1.00, 5.00, -4.0,
                                     max_loss_floor=-4.0)[2]
        three = _sanitize_close_values("Bull Put", 1.00, 5.00, -4.0,
                                       max_loss_floor=-4.0, quantity=3.0)[2]
        self.assertAlmostEqual(one, -400.0)
        self.assertAlmostEqual(three, -1_200.0)


class ExitPathsWriteTheWholePosition(_MarkTestCase):
    """End-to-end through `update_positions`, with the network stubbed out."""

    @staticmethod
    def _past_expiry():
        return str(date.today() - timedelta(days=1))

    def _closed(self, entry_id):
        with sqlite3.connect(self.db) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute("SELECT * FROM trades WHERE entry_id=?",
                                (entry_id,)).fetchone()

    def test_a_stopped_out_two_lot_books_twice_the_loss(self):
        # Same row twice, sized 1 and 2, closed in the same run at the same
        # mark: the only difference between them is the contract count.
        one = self._insert_open_row(quantity=1.0)
        two = self._insert_open_row(strike=101.0, quantity=2.0)
        self._manager(traded_mark=(1.00, "last")).update_positions()
        r1, r2 = self._closed(one), self._closed(two)
        self.assertEqual(r1["status"], "CLOSED")
        self.assertEqual(r2["status"], "CLOSED")
        self.assertAlmostEqual(r2["pnl_usd"], 2 * r1["pnl_usd"])
        self.assertAlmostEqual(r2["pnl_pct"], r1["pnl_pct"])

    def test_expiry_settlement_books_the_whole_position(self):
        # The deterministic settlement path prices off spot, not off a mark,
        # and is the backstop that closes anything the ladder cannot price.
        one = self._insert_open_row(expiration=self._past_expiry(), quantity=1.0)
        two = self._insert_open_row(expiration=self._past_expiry(), strike=101.0,
                                    quantity=2.0)
        self._manager().update_positions()
        r1, r2 = self._closed(one), self._closed(two)
        self.assertIn("Expired", r1["exit_reason"])
        self.assertAlmostEqual(r2["pnl_usd"], 2 * r1["pnl_usd"])

    def test_a_two_lot_spread_books_twice_the_credit(self):
        one = self._insert_open_row(
            strike=100.0, type="put", strategy_name="Bull Put Spread",
            long_strike=95.0, net_credit=1.00, entry_price=1.00, quantity=1.0)
        two = self._insert_open_row(
            strike=100.0, type="put", strategy_name="Bull Put Spread",
            long_strike=95.0, net_credit=1.00, entry_price=1.00, quantity=2.0)
        # Both legs quoted, so the structure marks and the take-profit fires.
        self._manager(chain_quotes={(100.0, "put"): (0.05, 0.07),
                                    (95.0, "put"): (0.01, 0.03)},
                      traded_mark=(None, None)).update_positions()
        r1, r2 = self._closed(one), self._closed(two)
        self.assertEqual(r1["status"], "CLOSED")
        self.assertAlmostEqual(r2["pnl_usd"], 2 * r1["pnl_usd"])


class PortfolioViewCountsContracts(unittest.TestCase):
    """The display side reads the same column through one helper.

    Every dollar figure in `check_pnl` is `per-contract x multiplier`, and each
    of those sites now folds `_row_lots` into the multiplier: open P&L, cost
    basis, concentration, portfolio max loss, the equity curve, the drawdown,
    the closed-trade rows and the portfolio Greeks. The helper is what they all
    share, so it is what is pinned here; the rendered screen needs live spot
    prices and is verified by running the app.
    """

    def test_a_missing_column_is_one_contract(self):
        self.assertEqual(cp._row_lots({}), 1.0)
        self.assertEqual(cp._row_lots({"quantity": None}), 1.0)

    def test_a_sized_row_reports_its_size(self):
        self.assertEqual(cp._row_lots({"quantity": 3.0}), 3.0)
        self.assertEqual(cp._row_lots({"quantity": 0.42}), 0.42)

    def test_nonsense_never_zeroes_a_position(self):
        # Zeroing would delete a real position from the portfolio totals.
        for bad in (0, -1, float("nan"), float("inf"), "two", object()):
            self.assertEqual(cp._row_lots({"quantity": bad}), 1.0,
                             msg=f"quantity={bad!r}")

    def test_the_two_ledgers_agree_on_what_a_lot_is(self):
        # check_pnl reconstructs dollar P&L from pnl_pct and must land on the
        # same number paper_manager stored, or the screen and the ledger would
        # disagree about the same trade.
        from src.paper_manager import _lots as pm_lots
        for value in (None, 0, -1, 1.0, 2.0, 0.42, "x"):
            self.assertEqual(cp._lots(value), pm_lots(value),
                             msg=f"quantity={value!r}")


if __name__ == "__main__":
    unittest.main()
