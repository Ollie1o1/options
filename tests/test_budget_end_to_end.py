"""The session budget reaches both the board and the ledger.

The whole point of using ONE quantity (capital at risk) on both sides is that
the board can never show a candidate the ledger would then refuse. This pins
that property, and pins the two things most likely to be broken by a later
edit: the refusal block must stay visible, and the board must not be re-sorted.
"""
from __future__ import annotations

import io
import json
import os
import contextlib
import tempfile
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


class TestTheSpreadLedgerIsBudgetGatedToo(unittest.TestCase):
    """A multi-leg entry must obey the session budget, not just a single leg.

    `within_budget` appears exactly once in `paper_manager`, inside
    `log_trade` — which reads as though spreads and condors were never gated.
    They are: `log_spread` and `log_iron_condor` both end in
    `return self.log_trade(trade_dict)`, and both set `max_loss_usd`, so the
    gate sees a real worst case. These tests pin that routing, because a
    future refactor that stopped funnelling through `log_trade` would remove
    the gate silently and the board would be the only thing still filtering.

    Never touches the real book — every ledger here is a tempfile.
    """

    def _pm(self, tmpdir):
        from src.paper_manager import PaperManager
        return PaperManager(os.path.join(tmpdir, "ledger.db"))

    def _spread(self, **kw):
        d = {"date": "2026-08-16", "ticker": "SPY", "expiration": "2026-09-18",
             "short_strike": 600.0, "long_strike": 500.0, "type": "Bull Put",
             "net_credit": 5.0, "max_profit": 500.0, "max_loss": 9500.0}
        d.update(kw)
        return d

    def _condor(self, **kw):
        d = {"date": "2026-08-16", "ticker": "SPY", "expiration": "2026-09-18",
             "short_put_strike": 600.0, "long_put_strike": 500.0,
             "short_call_strike": 700.0, "long_call_strike": 800.0,
             "total_credit": 5.0, "max_profit": 500.0, "max_risk": 9500.0}
        d.update(kw)
        return d

    def _log(self, method_name, payload):
        """Returns (inserted, refusals, printed output)."""
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = self._pm(tmpdir)
            with contextlib.redirect_stdout(buf):
                inserted = getattr(pm, method_name)(payload)
            return inserted, pm.unaffordable_rejected, buf.getvalue()

    def test_a_spread_over_the_session_budget_is_refused_out_loud(self):
        inserted, refused, out = self._log(
            "log_spread", self._spread(budget_at_entry=400.0))
        self.assertFalse(inserted)
        self.assertEqual(refused, 1)
        self.assertIn("$400", out)
        self.assertIn("capital at risk", out)

    def test_a_spread_under_the_session_budget_is_accepted(self):
        inserted, refused, _ = self._log(
            "log_spread", self._spread(budget_at_entry=10_000.0))
        self.assertTrue(inserted)
        self.assertEqual(refused, 0)

    def test_a_condor_over_the_session_budget_is_refused_out_loud(self):
        inserted, refused, out = self._log(
            "log_iron_condor", self._condor(budget_at_entry=400.0))
        self.assertFalse(inserted)
        self.assertEqual(refused, 1)
        self.assertIn("$400", out)

    def test_a_condor_under_the_session_budget_is_accepted(self):
        inserted, refused, _ = self._log(
            "log_iron_condor", self._condor(budget_at_entry=10_000.0))
        self.assertTrue(inserted)
        self.assertEqual(refused, 0)

    def test_an_absent_key_falls_back_to_the_config_cap(self):
        """No key = no prompt was ever answered = the scheduler's number."""
        from src.paper_manager import PaperManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cap = PaperManager(os.path.join(tmpdir, "probe.db"))._max_capital_at_risk
        self.assertEqual(cap, 4000.0, "config cap moved; this test's premise did")
        # $9,500 of risk is over the $4,000 config cap, and no key is present.
        inserted, refused, out = self._log("log_spread", self._spread())
        self.assertFalse(inserted)
        self.assertEqual(refused, 1)
        self.assertIn("$4,000", out)
        # ...and the same payload under an explicit "no limit" goes in.
        inserted, refused, _ = self._log(
            "log_spread", self._spread(budget_at_entry=None))
        self.assertTrue(inserted)
        self.assertEqual(refused, 0)

    def test_allow_unaffordable_is_still_an_escape_hatch(self):
        inserted, refused, _ = self._log(
            "log_spread",
            self._spread(budget_at_entry=400.0, allow_unaffordable=True))
        self.assertTrue(inserted)
        self.assertEqual(refused, 0)

    def test_a_spread_with_no_bounded_loss_never_passes_a_budget(self):
        payload = self._spread(budget_at_entry=400.0)
        payload.pop("max_loss")
        inserted, refused, out = self._log("log_spread", payload)
        self.assertFalse(inserted)
        self.assertEqual(refused, 1)
        self.assertIn("unbounded", out)


class TestEveryModeThatShouldPromptDoes(unittest.TestCase):
    """DISCOVER, MY LIST, TICKER, SELL, SPREADS and IRON all ask.

    MY LIST and TICKER were missed the first time: `elif is_my_list_mode`
    catches MY LIST before the branch that asked, and `elif is_ticker_mode`
    sits after it. The prompt is now a single call keyed off the mode flags,
    so this pins both the coverage and the fact that there is only ONE call
    site — two would let a mode be asked twice in one scan.
    """

    def _source(self):
        with open(repo_path("src/options_screener.py")) as fh:
            return fh.read()

    def test_the_prompt_has_exactly_one_call_site(self):
        src = self._source()
        self.assertEqual(
            src.count("session_budget = prompt_for_budget()"), 1,
            "more than one prompt call site — a mode could be asked twice")

    def test_the_gate_names_every_mode_the_spec_requires(self):
        src = self._source()
        start = src.index("session_budget = prompt_for_budget()")
        # The `if` guarding the single call site, immediately above it.
        guard = src[src.rindex("if (", 0, start):start]
        for flag in ("is_discovery_mode", "is_my_list_mode", "is_ticker_mode",
                     "is_premium_selling_mode", "is_credit_spread_mode",
                     "is_iron_condor_mode"):
            self.assertIn(flag, guard, f"{flag} cannot reach the budget prompt")

    def test_the_budget_scan_mode_is_not_swept_in(self):
        """`ALL` already asks a per-CONTRACT budget — a different quantity."""
        src = self._source()
        start = src.index("session_budget = prompt_for_budget()")
        guard = src[src.rindex("if (", 0, start):start]
        self.assertNotIn("is_budget_mode", guard)


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
