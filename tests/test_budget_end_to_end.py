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

    # `allow_unsized` for the same reason the fixtures elsewhere carry
    # `allow_unaffordable`: position sizing runs AFTER the budget gate and would
    # refuse a $9,500 spread on its own (2% of a $50,000 book is $1,000), which
    # is not what these tests are about. See src/book_sizing.py.
    def _spread(self, **kw):
        d = {"date": "2026-08-16", "ticker": "SPY", "expiration": "2026-09-18",
             "short_strike": 600.0, "long_strike": 500.0, "type": "Bull Put",
             "net_credit": 5.0, "max_profit": 500.0, "max_loss": 9500.0,
             "allow_unsized": True}
        d.update(kw)
        return d

    def _condor(self, **kw):
        d = {"date": "2026-08-16", "ticker": "SPY", "expiration": "2026-09-18",
             "short_put_strike": 600.0, "long_put_strike": 500.0,
             "short_call_strike": 700.0, "long_call_strike": 800.0,
             "total_credit": 5.0, "max_profit": 500.0, "max_risk": 9500.0,
             "allow_unsized": True}
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


class TestOnlyAnAnsweredPromptSetsTheKey(unittest.TestCase):
    """"Never asked" and "asked, chose no limit" must stay different claims.

    `log_trade` reads key PRESENCE. Setting the key unconditionally made every
    non-prompting mode — ALL, LOTTERY, SQUEEZE, a bare ticker at the menu —
    send `budget_at_entry=None`, i.e. an explicit "no limit", which silently
    uncapped modes that config had been holding at $4,000. The flag exists
    because `None` is a legitimate chosen value and cannot double as a
    sentinel for "never asked".
    """

    def test_a_mode_that_never_prompted_sends_no_key_at_all(self):
        from src.options_screener import _with_session_budget
        out = _with_session_budget({"ticker": "SPY"}, False, None)
        self.assertNotIn("budget_at_entry", out,
                         "a mode that never saw the prompt would be treated "
                         "as having chosen 'no limit'")

    def test_a_prompt_answered_with_no_limit_still_sets_the_key(self):
        from src.options_screener import _with_session_budget
        out = _with_session_budget({"ticker": "SPY"}, True, None)
        self.assertIn("budget_at_entry", out)
        self.assertIsNone(out["budget_at_entry"])

    def test_a_chosen_number_rides_through(self):
        from src.options_screener import _with_session_budget
        out = _with_session_budget({"ticker": "SPY"}, True, 400.0)
        self.assertEqual(out["budget_at_entry"], 400.0)

    def test_the_distinction_actually_changes_what_the_ledger_does(self):
        """End-to-end on a tempfile ledger, not just dict shape.

        Same $9,500 spread three ways: never asked (config's $4,000 binds and
        refuses), asked and answered "no limit" (logs), asked and answered
        $400 (refuses).
        """
        gate = TestTheSpreadLedgerIsBudgetGatedToo()
        from src.options_screener import _with_session_budget

        never_asked = _with_session_budget(gate._spread(), False, None)
        inserted, refused, out = gate._log("log_spread", never_asked)
        self.assertFalse(inserted, "config's cap must still bind")
        self.assertEqual(refused, 1)
        self.assertIn("$4,000", out)

        chose_no_limit = _with_session_budget(gate._spread(), True, None)
        inserted, refused, _ = gate._log("log_spread", chose_no_limit)
        self.assertTrue(inserted, "an explicit no-limit must log")

        chose_400 = _with_session_budget(gate._spread(), True, 400.0)
        inserted, refused, out = gate._log("log_spread", chose_400)
        self.assertFalse(inserted)
        self.assertIn("$400", out)


class TestEveryInteractiveLogSiteCarriesTheBudget(unittest.TestCase):
    """The PRESENCE of the key on the interactive paths, not just its absence
    on the auto-log ones.

    Delete the wrapper from the `[L]` spread call and nothing else in the
    suite notices, while the $4,000-vs-session divergence quietly returns.
    Structural rather than runtime because these four call sites live inside
    `main()`'s post-scan menu, and reaching them for real means driving
    `main()` — which runs `update_positions()` against the live book.
    """

    def _interactive_block(self):
        with open(repo_path("src/options_screener.py")) as fh:
            src = fh.read()
        start = src.index("Collapsed post-scan prompt")
        return src[start:]

    def test_all_four_interactive_log_sites_go_through_the_helper(self):
        block = self._interactive_block()
        self.assertEqual(
            block.count("_with_session_budget"), 4,
            "expected exactly 4 interactive log sites to attach the budget "
            "([P] single leg, [L] spread, [L] condor, [L] single leg)")

    def test_the_spread_and_condor_calls_are_among_them(self):
        block = self._interactive_block()
        self.assertIn("pm.log_spread(_with_session_budget(", block)
        self.assertIn("pm.log_iron_condor_if_new(_with_session_budget(", block)

    def test_nothing_sets_the_key_by_hand_any_more(self):
        """One chokepoint: the literal key must appear only in the helper."""
        with open(repo_path("src/options_screener.py")) as fh:
            src = fh.read()
        self.assertEqual(
            src.count('"budget_at_entry"'), 1,
            "budget_at_entry is set outside _with_session_budget, which can "
            "bypass the was-it-actually-asked check")


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
    to config. The three `--auto-log` payloads in `options_screener` therefore
    must not carry the key at all. This is a source-level assertion because the
    property being protected is the ABSENCE of a line, which no runtime test of
    the happy path can observe.

    This is not merely defensive. `prompt_for_budget()` is gated on the MODE,
    not on `_interactive`, and it never raises: an unattended `--auto-log`
    DISCOVER run reaches the prompt, reads nothing, and returns None with
    `budget_was_chosen = True` — a full "the operator chose NO LIMIT". Key
    absence in this block is the only thing standing between that and the
    unbounded feeder that logged $27k and $83k positions.
    """

    def _auto_log_block(self):
        with open(repo_path("src/options_screener.py")) as fh:
            src = fh.read()
        start = src.index("Auto-log mode: bypass interactive save menu")
        return src[start:src.index("Collapsed post-scan prompt", start)]

    def test_no_auto_log_trade_dict_sets_budget_at_entry(self):
        self.assertNotIn(
            "budget_at_entry", self._auto_log_block(),
            "an --auto-log trade dict sets budget_at_entry: an unattended "
            "scheduler run would inherit an operator's chosen budget instead "
            "of falling back to config")

    def test_no_auto_log_payload_is_routed_through_the_helper(self):
        """The key can also arrive without its own name being written.

        `_with_session_budget(_trade, budget_was_chosen, session_budget)` sets
        `budget_at_entry` without the literal ever appearing at the call site,
        so wrapping an auto-log payload passes the assertion above while
        handing cron the session value. Both doors have to be shut: the helper
        is the only writer of the key (pinned by
        `test_nothing_sets_the_key_by_hand_any_more`), so barring it from this
        block makes the key unreachable here by any route.
        """
        self.assertNotIn(
            "_with_session_budget", self._auto_log_block(),
            "an --auto-log payload is wrapped in _with_session_budget: the "
            "key never appears by name, but the scheduler still inherits the "
            "session budget — and in a cron run that value is an explicit "
            "'no limit'")


if __name__ == "__main__":
    unittest.main()


class TestTheSaveMenuReportsWhatTheLedgerDid(unittest.TestCase):
    """"Logged" has to mean a row was written.

    `log_trade` returns False and prints its own refusal when a candidate is
    over budget, a duplicate, or untradeable. Both save-menu paths threw that
    return away and announced success regardless, so a $500 session budget
    produced this on screen, in this order:

        Skipped Long Call on NVDA: capital at risk $2,400 exceeds the $500 budget
        ✓ Paper trade logged: NVDA CALL $180

    with nothing in the ledger. The budget work is what makes it routine —
    refusal used to need a $4,000 position — so the contradiction is fixed
    here rather than filed.

    Structural, because reaching these lines means driving `main()`, which
    runs `update_positions()` against the live book.
    """

    def _interactive_block(self):
        with open(repo_path("src/options_screener.py")) as fh:
            src = fh.read()
        return src[src.index("Collapsed post-scan prompt"):]

    def test_the_paper_trade_success_line_is_conditional_on_the_write(self):
        block = self._interactive_block()
        self.assertIn("if pm.log_trade(trade_dict):", block,
                      "[P] announces success without checking log_trade")

    def test_the_bulk_count_is_insertions_not_selections(self):
        """`len(picks_to_log)` counts what the operator chose, not what landed."""
        block = self._interactive_block()
        self.assertNotIn("Logged {len(picks_to_log)} trades", block)

    def test_every_bulk_log_call_feeds_the_counter(self):
        block = self._interactive_block()
        for call in ("pm.log_spread(", "pm.log_iron_condor_if_new(",
                     "pm.log_trade("):
            idx = block.index(call)
            window = block[max(0, idx - 60):idx]
            self.assertIn("if ", window,
                          f"{call} in the bulk path does not feed the count")


class TestTheMenuOffersOnlyWhatTheBoardShowed(unittest.TestCase):
    """`[P]`/`[L]` must not offer a candidate the budget hid.

    The board is narrowed by `_budget_board`, but `ScanResult.picks` is the
    RAW scan, and the save menu read that — so a $2,000 budget could show
    three affordable puts and then hand `[P]` the $34,680 cash-secured put it
    had just removed. The ledger refuses it, so nothing bad is written; the
    operator is simply offered a trade they were never shown, which is the
    board-vs-ledger divergence the budget exists to close.

    Same lesson the tearsheet already learned: `--tearsheet N` must mean the N
    the reader just read.
    """

    def test_scan_result_carries_the_board_frames(self):
        from src.schemas import ScanResult
        r = ScanResult()
        for f in ("board_picks", "board_credit_spreads", "board_iron_condors"):
            self.assertIsNone(getattr(r, f),
                              f"{f} must default to None — 'no board built'")

    def test_the_menu_reads_the_board_not_the_raw_scan(self):
        with open(repo_path("src/options_screener.py")) as fh:
            src = fh.read()
        block = src[src.index("Collapsed post-scan prompt"):]
        self.assertIn("elif not _menu_picks.empty:", block)
        self.assertIn("rank_single_legs_by_verdict(_menu_picks, mode)", block)
        self.assertIn("log_src = _menu_picks if not _menu_picks.empty", block)

    def test_an_empty_board_is_not_the_same_as_no_board(self):
        """`is None` chooses the fallback; `.empty` would defeat the filter.

        If the budget removes every candidate, `board_picks` is an EMPTY
        frame. Falling back to the raw scan on emptiness would offer the whole
        unfiltered board precisely when the budget had rejected all of it.
        """
        with open(repo_path("src/options_screener.py")) as fh:
            src = fh.read()
        self.assertIn(
            "_menu_picks = (picks if scan_results.board_picks is None", src,
            "the menu falls back on emptiness rather than on absence")

    def test_the_auto_log_path_still_reads_the_raw_scan(self):
        """The invariant this must not break.

        --auto-log has to keep applying CONFIG's cap. If the menu's rebinding
        had been done on `picks` itself, the unattended scheduler would have
        inherited the operator's session budget — the unbounded feeder again,
        through a different door.
        """
        with open(repo_path("src/options_screener.py")) as fh:
            src = fh.read()
        auto = src[src.index("Auto-log mode: bypass interactive save menu"):
                   src.index("Collapsed post-scan prompt")]
        self.assertNotIn("_menu_picks", auto)
        self.assertNotIn("_menu_spreads", auto)
        self.assertNotIn("_menu_condors", auto)
        self.assertIn("_log_src = picks if not picks.empty", auto)


class TestTheConfigNoteNamesTheModesItActuallyCovers(unittest.TestCase):
    """The note claimed more than the prompt delivers.

    "interactive scans ask for their own budget" reads as ALL of them. Six
    modes ask; ALL/Budget, LOTTERY, SQUEEZE and a bare ticker typed at the
    menu are interactive too and remain bound by this value through key
    absence. An operator reading the old note would believe a bare-ticker scan
    was uncapped when it was not.
    """

    def _note(self):
        with open(repo_path("config.json")) as fh:
            return json.load(fh)["auto_log"]["_max_capital_at_risk_note"]

    def test_it_names_the_modes_that_ask(self):
        note = self._note()
        for mode in ("DISCOVER", "MY LIST", "TICKER", "SELL", "SPREADS", "IRON"):
            self.assertIn(mode, note)

    def test_it_says_the_other_interactive_paths_are_still_capped(self):
        note = self._note().lower()
        self.assertIn("lottery", note)
        self.assertIn("squeeze", note)
        self.assertIn("bare ticker", note)

    def test_the_value_is_still_unchanged(self):
        with open(repo_path("config.json")) as fh:
            self.assertEqual(json.load(fh)["auto_log"]["max_capital_at_risk"], 4000)


class TestABudgetThatRemovesEverythingSaysSo(unittest.TestCase):
    """Silence reads as a broken app.

    Found by running every mode after the 2026-08-17 merges: a $2,000 budget
    against short puts that need $20k-$76k of collateral filtered the whole
    board, and the operator saw one line — "0 of 61 surviving candidates fit"
    — followed by nothing at all. No board, no summary, no indication of what
    budget would have worked.

    The refusal gates already handle this case properly ("Nothing here cleared
    the gates. That is the answer, not an error."), so this matches them, and
    names the cheapest candidate because that is the number the operator needs
    to act on.
    """

    def _board(self, budget):
        from src import options_screener as osx
        df = pd.DataFrame([
            {"symbol": "AVGO", "type": "put", "strike": 385.0, "ask": 2.00},
            {"symbol": "NVDA", "type": "put", "strike": 220.0, "ask": 1.50},
        ])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = osx._budget_board(
                df, lambda r: osx._strategy_label_for_mode("Premium Selling",
                                                           r.get("type")),
                budget, verbose=True)
        return out, buf.getvalue()

    def test_it_says_nothing_fits_rather_than_going_silent(self):
        out, text = self._board(2_000.0)
        self.assertEqual(len(out), 0)
        self.assertIn("0 of 2", text)
        self.assertIn("budget", text.lower())
        self.assertNotEqual(text.strip().count("\n"), 0,
                            "only one line printed — the operator sees silence")

    def test_it_names_the_cheapest_so_the_operator_can_act(self):
        _out, text = self._board(2_000.0)
        # NVDA is the cheaper collateral: (220 - 1.50) x 100 = $21,850
        self.assertIn("21,850", text)

    def test_it_does_not_fire_when_something_survives(self):
        out, text = self._board(1_000_000.0)
        self.assertEqual(len(out), 2)
        self.assertNotIn("Nothing fits", text)

    def test_it_does_not_fire_without_a_budget(self):
        out, text = self._board(None)
        self.assertEqual(len(out), 2)
        self.assertNotIn("Nothing fits", text)
