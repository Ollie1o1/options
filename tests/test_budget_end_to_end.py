"""The session budget reaches both the board and the ledger.

The whole point of using ONE quantity (capital at risk) on both sides is that
the board can never show a candidate the ledger would then refuse. This pins
that property, and pins the two things most likely to be broken by a later
edit: the refusal block must stay visible, and the board must not be re-sorted.
"""
from __future__ import annotations

import json
import unittest

import pandas as pd

from src import budget_view as bv
from src.paths import repo_path


class TestBoardAndLedgerAgree(unittest.TestCase):

    def test_everything_the_board_keeps_would_be_accepted_by_the_ledger(self):
        from src.capital_risk import within_budget
        df = pd.DataFrame([
            {"symbol": "A", "max_profit": 73.0, "max_loss": 127.0,
             "ev_per_contract": 5.0},
            {"symbol": "B", "max_profit": 500.0, "max_loss": 9500.0,
             "ev_per_contract": 40.0},
        ])
        budget = 1000.0
        kept = bv.annotate(bv.affordable(df, budget, "Bull Put"), "Bull Put")
        for _, row in kept.iterrows():
            self.assertTrue(within_budget(row["capital_at_risk"], budget),
                            f"{row['symbol']} shown but would be refused")


class TestTheBoardIsNarrowedNeverReordered(unittest.TestCase):
    """`_budget_board` is the only new thing that touches a live board."""

    def _board(self):
        # Deliberately NOT in quality order: the board arrives already ordered
        # by whatever the gate left, and the budget must not re-sort it.
        return pd.DataFrame([
            {"symbol": "AAA", "type": "call", "ask": 1.00,
             "max_profit": 300.0, "ev_per_contract": 5.0},
            {"symbol": "BBB", "type": "put", "ask": 40.00,
             "max_profit": 900.0, "ev_per_contract": 9.0},
            {"symbol": "CCC", "type": "call", "ask": 2.00,
             "max_profit": 100.0, "ev_per_contract": 1.0},
        ])

    def test_order_is_preserved_and_only_affordable_rows_survive(self):
        from src import options_screener as osx
        board = self._board()
        out = osx._budget_board(
            board, lambda r: osx._strategy_label_for_mode("Discovery scan",
                                                          r.get("type")),
            500.0, verbose=False)
        # BBB ties up $4,000 as long premium and is dropped; the other two keep
        # the order they arrived in.
        self.assertEqual(list(out["symbol"]), ["AAA", "CCC"])
        self.assertEqual(list(out["capital_at_risk"]), [100.0, 200.0])

    def test_no_budget_annotates_without_dropping_or_reordering(self):
        from src import options_screener as osx
        board = self._board()
        out = osx._budget_board(
            board, lambda r: osx._strategy_label_for_mode("Discovery scan",
                                                          r.get("type")),
            None, verbose=False)
        self.assertEqual(list(out["symbol"]), ["AAA", "BBB", "CCC"])
        self.assertIn("reward_per_risk", out.columns)
        self.assertIn("net_ev_per_risk", out.columns)

    def test_rows_are_sized_by_their_own_strategy_not_the_frames(self):
        """A Premium Selling board mixes two different risk definitions.

        A short put's risk is its collateral; a short call's cannot be bounded
        at all. Sizing the whole frame under one label would price one of them
        as the other — which is how a $31,850 cash-secured put once sized as a
        $50 debit.
        """
        from src import options_screener as osx
        board = pd.DataFrame([
            {"symbol": "PUT", "type": "put", "strike": 50.0, "ask": 1.00},
            {"symbol": "CALL", "type": "call", "strike": 50.0, "ask": 1.00},
        ])
        out = osx._budget_board(
            board, lambda r: osx._strategy_label_for_mode("Premium Selling",
                                                          r.get("type")),
            None, verbose=False)
        risks = dict(zip(out["symbol"], out["capital_at_risk"]))
        self.assertEqual(risks["PUT"], 4900.0)   # (50 - 1) x 100 collateral
        self.assertIsNone(risks["CALL"])         # unbounded, not zero

    def test_an_unsizable_row_never_passes_a_set_budget(self):
        from src import options_screener as osx
        board = pd.DataFrame([
            {"symbol": "CALL", "type": "call", "strike": 50.0, "ask": 1.00},
        ])
        out = osx._budget_board(
            board, lambda r: osx._strategy_label_for_mode("Premium Selling",
                                                          r.get("type")),
            1_000_000.0, verbose=False)
        self.assertEqual(len(out), 0)


class TestConfigNoteIsNarrowed(unittest.TestCase):
    """The value does not change; only what it governs."""

    def _auto_log(self):
        with open(repo_path("config.json")) as fh:
            return json.load(fh)["auto_log"]

    def test_the_value_is_unchanged(self):
        self.assertEqual(self._auto_log()["max_capital_at_risk"], 4000)

    def test_the_note_says_it_is_the_scheduler_budget(self):
        note = self._auto_log()["_max_capital_at_risk_note"].lower()
        self.assertIn("scheduler", note)

    def test_the_note_says_interactive_scans_choose_their_own(self):
        note = self._auto_log()["_max_capital_at_risk_note"].lower()
        self.assertIn("interactive", note)


class TestTheAutoLogPathsStayOnConfig(unittest.TestCase):
    """The scheduler must never inherit an operator's "no limit".

    `log_trade` reads `budget_at_entry` by KEY PRESENCE: absent means fall back
    to config. The two `--auto-log` trade dicts in `options_screener` therefore
    must not carry the key at all. This is a source-level assertion because the
    property being protected is the ABSENCE of a line, which no runtime test of
    the happy path can observe.
    """

    def test_no_auto_log_trade_dict_sets_budget_at_entry(self):
        with open(repo_path("src/options_screener.py")) as fh:
            src = fh.read()
        auto_log_start = src.index("Auto-log mode: bypass interactive save menu")
        auto_log_end = src.index("Collapsed post-scan prompt", auto_log_start)
        auto_log_block = src[auto_log_start:auto_log_end]
        self.assertNotIn(
            "budget_at_entry", auto_log_block,
            "an --auto-log trade dict sets budget_at_entry: an unattended "
            "scheduler run would inherit an operator's chosen budget instead "
            "of falling back to config")


if __name__ == "__main__":
    unittest.main()
