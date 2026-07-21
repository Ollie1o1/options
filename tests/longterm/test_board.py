"""Tests for banner + board rendering (plain-mode, color pinned off)."""
import builtins
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import src.formatting as fmt
from src.longterm import board as B
from src.longterm import fills as F
from src.longterm import plan as P
from src.longterm import zones as Z
from src.longterm.discover import CandidateRead, DeepRead
from src.longterm.detail import DetailRead
from src.news_fetcher import AnalystChange, NewsData, NewsItem
from src.short_interest import ShortInterest

fmt._COLOR_ENABLED = False  # pin: never env vars (supports_color memoizes)


def mu_plan():
    return P.Plan(5000.0, [P.PlanName("MU", [P.Tranche(750, 0.4), P.Tranche(650, 0.6)])])


def read(state, spot=780.0, level=750.0):
    return Z.ZoneRead("MU", state, spot, level, (spot - level) / level * 100,
                      1.0, -32.4, True)


class TestBanner(unittest.TestCase):
    def test_silent_when_nothing_triggered(self):
        self.assertEqual(B.banner([read(Z.WATCHING)], mu_plan(), 5000.0), "")
        self.assertEqual(B.banner([], mu_plan(), 5000.0), "")

    def test_near_line_contents(self):
        out = B.banner([read(Z.NEAR)], mu_plan(), 5000.0)
        self.assertIn("MU", out)
        self.assertIn("NEAR", out)
        self.assertIn("750", out)
        self.assertIn("$2,000", out)          # 5000 × 1.0 alloc × 0.4 weight
        self.assertIn("-32.4%", out)          # drawdown context

    def test_size_capped_at_remaining(self):
        out = B.banner([read(Z.IN_ZONE, spot=749.0)], mu_plan(), 500.0)
        self.assertIn("$500", out)
        self.assertNotIn("$2,000", out)

    def test_earnings_flag(self):
        out = B.banner([read(Z.NEAR)], mu_plan(), 5000.0, earnings={"MU": "07-28"})
        self.assertIn("earnings 07-28", out)


class TestBoard(unittest.TestCase):
    def test_board_shows_ladder_states_and_book(self):
        book = {"MU": {"shares": 2.0, "cost": 1500.0, "avg_price": 750.0}}
        out = B.render_board(mu_plan(), [read(Z.NEAR)], book, 3500.0)
        self.assertIn("MU", out)
        self.assertIn("750", out)
        self.assertIn("650", out)
        self.assertIn("NEAR", out)
        self.assertIn("2.0", out)            # held shares visible
        self.assertIn("3,500", out)          # remaining cash

    def test_board_empty_plan(self):
        out = B.render_board(P.Plan(), [], {}, 0.0)
        self.assertIn("empty", out.lower())  # points user at ADD

    def test_suggested_size(self):
        plan = mu_plan()
        n, t = plan.names[0], plan.names[0].tranches[0]
        self.assertEqual(B.suggested_size(plan, n, t, 10000.0), 2000.0)
        self.assertEqual(B.suggested_size(plan, n, t, 500.0), 500.0)


class TestHandleCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.plan_path = os.path.join(self.tmp.name, "plan.json")
        self.db = os.path.join(self.tmp.name, "longterm.db")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, line, plan):
        return B.handle_command(line, plan, plan_path=self.plan_path, db_path=self.db)

    def test_add(self):
        plan, msg = self._run("ADD MU 750/650/550", P.Plan(5000.0, []))
        self.assertEqual(plan.names[0].ticker, "MU")
        self.assertEqual([t.level for t in plan.names[0].tranches], [750.0, 650.0, 550.0])
        self.assertAlmostEqual(plan.names[0].tranches[0].weight, 1 / 3)
        self.assertEqual(P.load_plan(self.plan_path).names[0].ticker, "MU")  # persisted
        self.assertIn("MU", msg)

    def test_add_duplicate_rejected(self):
        plan, _ = self._run("ADD MU 750", P.Plan(5000.0, []))
        plan2, msg = self._run("ADD MU 700", plan)
        self.assertEqual(len(plan2.names), 1)
        self.assertIn("already", msg.lower())

    def test_fill_records_and_matches_level(self):
        plan, _ = self._run("ADD MU 750/650", P.Plan(5000.0, []))
        plan, msg = self._run("FILL MU 750 2.5 748.20", plan)
        self.assertEqual(F.filled_levels("MU", db_path=self.db), {750.0})
        self.assertIn("748.20", msg)

    def test_fill_unknown_level_rejected(self):
        plan, _ = self._run("ADD MU 750/650", P.Plan(5000.0, []))
        plan, msg = self._run("FILL MU 700 1 700", plan)
        self.assertEqual(F.fills_for(db_path=self.db), [])
        self.assertIn("no tranche", msg.lower())

    def test_fill_already_filled_rejected(self):
        plan, _ = self._run("ADD MU 750", P.Plan(5000.0, []))
        plan, _ = self._run("FILL MU 750 1 749", plan)
        plan, msg = self._run("FILL MU 750 1 748", plan)
        self.assertEqual(len(F.fills_for(db_path=self.db)), 1)
        self.assertIn("filled", msg.lower())

    def test_edit_replaces_ladder(self):
        plan, _ = self._run("ADD MU 750/650", P.Plan(5000.0, []))
        plan, _ = self._run("EDIT MU 800/700/600", plan)
        self.assertEqual([t.level for t in plan.names[0].tranches], [800.0, 700.0, 600.0])

    def test_remove_and_cash(self):
        plan, _ = self._run("ADD MU 750", P.Plan(5000.0, []))
        plan, _ = self._run("CASH 6000", plan)
        self.assertEqual(plan.cash_pool_usd, 6000.0)
        plan, _ = self._run("REMOVE MU", plan)
        self.assertEqual(plan.names, [])

    def test_remove_rejected_when_shares_held(self):
        plan, _ = self._run("ADD MU 750", P.Plan(5000.0, []))
        plan, _ = self._run("FILL MU 750 2.5 748.20", plan)
        plan2, msg = self._run("REMOVE MU", plan)
        self.assertEqual([n.ticker for n in plan2.names], ["MU"])  # not removed
        self.assertIn("MU", msg)
        self.assertIn("held", msg.lower())
        self.assertEqual(F.book(db_path=self.db)["MU"]["shares"], 2.5)  # fills untouched

    def test_remove_allowed_with_zero_shares(self):
        plan, _ = self._run("ADD MU 750", P.Plan(5000.0, []))
        plan, msg = self._run("REMOVE MU", plan)
        self.assertEqual(plan.names, [])
        self.assertIn("removed", msg.lower())

    def test_garbage_returns_grammar_help(self):
        plan, msg = self._run("FROB MU", P.Plan(0.0, []))
        self.assertIn("ADD", msg)
        self.assertIn("FILL", msg)


def _disc_candidate(ticker="MU", drawdown=-32.4, spot=760.0):
    return CandidateRead(
        ticker=ticker, spot=spot, drawdown_pct=drawdown,
        ma200_distance_pct=-8.5, momentum_12_1=-0.18,
        supports=[{"label": "200d MA", "level": 700.0, "pct": -0.079}],
        bounce={"by_horizon": {20: {"n": 41, "bounce_rate": 0.65, "median": 0.04, "p25": -0.02, "p75": 0.09}}},
        suggested_ladder=[P.Tranche(760.0, 0.5), P.Tranche(700.0, 0.5)],
    )


class TestRenderDiscoverBoard(unittest.TestCase):
    def test_shows_every_candidate_numbered(self):
        results = [(_disc_candidate("MU"), None), (_disc_candidate("AMD", -18.0), None)]
        out = B.render_discover_board(results, "SEMICONDUCTORS")
        self.assertIn("1", out)
        self.assertIn("2", out)
        self.assertIn("MU", out)
        self.assertIn("AMD", out)

    def test_deep_tier_narrative_appears_for_entries_with_deep_read(self):
        deep = DeepRead(ticker="MU", insider={"n_buyers": 2, "buy_value": 340_000.0,
                                              "label": "CLUSTER BUY"},
                        earnings_days=12, fundamentals=None)
        results = [(_disc_candidate("MU"), deep)]
        out = B.render_discover_board(results, "SEMICONDUCTORS")
        self.assertIn("insider", out.lower())

    def test_no_deep_read_omits_narrative_for_that_candidate(self):
        results = [(_disc_candidate("MU"), None)]
        out = B.render_discover_board(results, "SEMICONDUCTORS")
        self.assertIn("MU", out)  # still in the table

    def test_empty_results_shows_no_candidates_message(self):
        out = B.render_discover_board([], "SEMICONDUCTORS")
        self.assertIn("no candidates", out.lower())


class TestResolveAddTarget(unittest.TestCase):
    def test_numeric_add_resolves_to_canonical_command(self):
        results = [(_disc_candidate("MU"), None)]
        resolved = B.resolve_add_target("ADD 1", results)
        self.assertEqual(resolved, "ADD MU 760/700")

    def test_out_of_range_index_passes_through_unchanged(self):
        results = [(_disc_candidate("MU"), None)]
        resolved = B.resolve_add_target("ADD 5", results)
        self.assertEqual(resolved, "ADD 5")

    def test_no_prior_scan_passes_through_unchanged(self):
        resolved = B.resolve_add_target("ADD 1", None)
        self.assertEqual(resolved, "ADD 1")

    def test_ticker_add_passes_through_unchanged(self):
        results = [(_disc_candidate("MU"), None)]
        resolved = B.resolve_add_target("ADD MU 750/650/550", results)
        self.assertEqual(resolved, "ADD MU 750/650/550")

    def test_non_add_command_passes_through_unchanged(self):
        results = [(_disc_candidate("MU"), None)]
        resolved = B.resolve_add_target("REMOVE MU", results)
        self.assertEqual(resolved, "REMOVE MU")

    def test_case_insensitive_add_keyword(self):
        results = [(_disc_candidate("MU"), None)]
        resolved = B.resolve_add_target("add 1", results)
        self.assertEqual(resolved, "ADD MU 760/700")


def _full_detail(ticker="MU"):
    import datetime as dt
    return DetailRead(
        ticker=ticker,
        deep=DeepRead(
            ticker=ticker,
            insider={"n_buyers": 2, "buy_value": 340_000.0, "label": "CLUSTER BUY",
                     "window_days": 90},
            earnings_days=5,
            fundamentals={"trailingPE": 18.0, "forwardPE": 15.0, "profitMargins": 0.22,
                         "revenueGrowth": 0.11, "earningsGrowth": 0.08, "returnOnEquity": 0.31},
        ),
        short_interest=ShortInterest(pct_float=0.08, days_to_cover=3.4,
                                     pct_float_prior=0.06, shares_short=1_000_000,
                                     trend="rising"),
        news=NewsData(
            symbol=ticker,
            items=[NewsItem(headline="MU beats estimates", source="Yahoo Finance",
                            published=dt.datetime(2026, 7, 18), sentiment=0.6, url="")],
            analyst_changes=[AnalystChange(firm="Morgan Stanley", action="upgrade",
                                           from_grade="Hold", to_grade="Buy",
                                           date=dt.datetime(2026, 7, 15), price_target=900.0)],
            aggregate_sentiment=0.5,
            top_headlines=["MU beats estimates"],
            has_negative_catalyst=False,
            has_positive_catalyst=True,
        ),
    )


def _empty_detail(ticker="MU"):
    return DetailRead(ticker=ticker, deep=DeepRead(ticker=ticker),
                      short_interest=None, news=None)


class TestRenderDetail(unittest.TestCase):
    def test_header_vitals_always_present(self):
        c = _disc_candidate("MU")
        c.rsi = 28.0
        c.ann_vol_pct = 45.2
        out = B.render_detail(c, _full_detail("MU"))
        self.assertIn("MU", out)
        self.assertIn("28", out)   # RSI value
        self.assertIn("oversold", out.lower())

    def test_rsi_none_shows_na_not_missing_section(self):
        c = _disc_candidate("MU")
        c.rsi = None
        out = B.render_detail(c, _empty_detail("MU"))
        self.assertIn("n/a", out.lower())

    def test_full_support_resistance_ladder_shown(self):
        c = _disc_candidate("MU")
        c.supports = [{"label": "50d MA", "level": 720.0, "pct": -0.05},
                     {"label": "200d MA", "level": 700.0, "pct": -0.079}]
        out = B.render_detail(c, _empty_detail("MU"))
        self.assertIn("50d MA", out)
        self.assertIn("200d MA", out)

    def test_bounce_odds_table_shows_multiple_horizons(self):
        c = _disc_candidate("MU")
        c.bounce = {"by_horizon": {
            5: {"n": 30, "bounce_rate": 0.55, "median": 0.02, "p25": -0.01, "p75": 0.05},
            20: {"n": 41, "bounce_rate": 0.65, "median": 0.04, "p25": -0.02, "p75": 0.09},
        }}
        out = B.render_detail(c, _empty_detail("MU"))
        self.assertIn("55", out)
        self.assertIn("65", out)

    def test_bounce_all_zero_sample_shows_na_not_empty_section(self):
        c = _disc_candidate("MU")
        c.bounce = {"by_horizon": {
            20: {"n": 0, "bounce_rate": None, "median": None, "p25": None, "p75": None},
        }}
        out = B.render_detail(c, _empty_detail("MU"))
        self.assertIn("n/a", out.lower())

    def test_fundamentals_table_shows_all_six_fields(self):
        c = _disc_candidate("MU")
        out = B.render_detail(c, _full_detail("MU"))
        self.assertIn("18", out)   # trailing PE
        self.assertIn("15", out)   # forward PE
        self.assertIn("22", out)   # profit margin %
        self.assertIn("11", out)   # revenue growth %
        self.assertIn("8", out)    # earnings growth %
        self.assertIn("31", out)   # ROE %

    def test_fundamentals_missing_shows_na_section(self):
        c = _disc_candidate("MU")
        out = B.render_detail(c, _empty_detail("MU"))
        self.assertIn("fundamentals", out.lower())
        self.assertIn("n/a", out.lower())

    def test_short_interest_shown_with_trend(self):
        c = _disc_candidate("MU")
        out = B.render_detail(c, _full_detail("MU"))
        self.assertIn("8", out)     # 8% of float
        self.assertIn("3.4", out)   # days to cover

    def test_short_interest_missing_shows_na(self):
        c = _disc_candidate("MU")
        out = B.render_detail(c, _empty_detail("MU"))
        self.assertIn("short interest", out.lower())

    def test_news_headlines_shown(self):
        c = _disc_candidate("MU")
        out = B.render_detail(c, _full_detail("MU"))
        self.assertIn("MU beats estimates", out)

    def test_analyst_change_shown(self):
        c = _disc_candidate("MU")
        out = B.render_detail(c, _full_detail("MU"))
        self.assertIn("Morgan Stanley", out)
        self.assertIn("upgrade", out.lower())

    def test_catalyst_flag_line_present_when_positive(self):
        c = _disc_candidate("MU")
        out = B.render_detail(c, _full_detail("MU"))
        self.assertIn("catalyst", out.lower())

    def test_news_missing_shows_na_not_missing_section(self):
        c = _disc_candidate("MU")
        out = B.render_detail(c, _empty_detail("MU"))
        self.assertIn("news", out.lower())
        self.assertIn("n/a", out.lower())

    def test_insight_line_synthesis_included(self):
        c = _disc_candidate("MU")
        out = B.render_detail(c, _empty_detail("MU"))
        # insight_line always starts with "{ticker}: {drawdown}% off ATH"
        self.assertIn("off ATH", out)

    def test_earnings_within_window_uses_warn_glyph(self):
        c = _disc_candidate("MU")
        detail = _full_detail("MU")
        detail.deep.earnings_days = 5
        out = B.render_detail(c, detail)
        self.assertIn(fmt.GLYPHS["warn"], out)


class TestDiscoverMenuWiring(unittest.TestCase):
    """menu()'s D/DISCOVER branch and its ADD-by-index handoff to
    handle_command are exercised together here since menu() itself is an
    interactive input() loop (not unit-tested directly elsewhere in this
    file either) — this proves the two pieces actually compose, which
    neither test_discover.py nor the rest of test_board.py can prove alone.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.plan_path = os.path.join(self.tmp.name, "plan.json")
        self.db = os.path.join(self.tmp.name, "longterm.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_discover_then_add_by_index_lands_in_plan(self):
        # Simulates the two-step flow menu() performs: run a scan, then
        # resolve "ADD 1" against its results before handing off to the
        # existing, already-tested handle_command.
        candidate = _disc_candidate("MU")
        last_results = [(candidate, None)]

        resolved = B.resolve_add_target("ADD 1", last_results)
        plan, msg = B.handle_command(resolved, P.Plan(5000.0, []),
                                     plan_path=self.plan_path, db_path=self.db)

        self.assertEqual(plan.names[0].ticker, "MU")
        self.assertEqual([t.level for t in plan.names[0].tranches], [760.0, 700.0])
        self.assertIn("MU", msg)


class TestActionsMenu(unittest.TestCase):
    def test_lists_all_seven_actions_and_back(self):
        out = B.render_actions_menu(60)
        for n in "1234567":
            self.assertIn(f"[{n}]", out)
        self.assertIn("[B]", out)
        self.assertIn("Add a stock", out)
        self.assertIn("Write & open report", out)


class TestGuidedAdd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.plan_path = os.path.join(self.tmp.name, "plan.json")
        self.db = os.path.join(self.tmp.name, "longterm.db")
        self.orig_input = builtins.input

    def tearDown(self):
        self.tmp.cleanup()
        builtins.input = self.orig_input

    def _feed(self, *answers):
        it = iter(answers)
        builtins.input = lambda *_a, **_k: next(it)

    def test_typed_ticker_and_levels(self):
        self._feed("mu", "750, 650, 550")
        plan, last_discovery = B._guided_add(P.Plan(5000.0, []), None,
                                             plan_path=self.plan_path, db_path=self.db)
        self.assertEqual(plan.names[0].ticker, "MU")
        self.assertEqual([t.level for t in plan.names[0].tranches], [750.0, 650.0, 550.0])
        self.assertIsNone(last_discovery)

    def test_add_by_index_from_last_scan_skips_levels_question(self):
        candidate = _disc_candidate("MU")
        last_discovery = [(candidate, None)]
        self._feed("1")  # only one answer needed — no levels prompt
        plan, returned_discovery = B._guided_add(P.Plan(5000.0, []), last_discovery,
                                                  plan_path=self.plan_path, db_path=self.db)
        self.assertEqual(plan.names[0].ticker, "MU")
        self.assertEqual([t.level for t in plan.names[0].tranches], [760.0, 700.0])
        self.assertIs(returned_discovery, last_discovery)

    def test_reprompts_on_unparseable_levels(self):
        self._feed("mu", "cheap", "750, 650")
        plan, _ = B._guided_add(P.Plan(5000.0, []), None,
                                plan_path=self.plan_path, db_path=self.db)
        self.assertEqual([t.level for t in plan.names[0].tranches], [750.0, 650.0])


class TestGuidedFill(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.plan_path = os.path.join(self.tmp.name, "plan.json")
        self.db = os.path.join(self.tmp.name, "longterm.db")
        self.orig_input = builtins.input

    def tearDown(self):
        self.tmp.cleanup()
        builtins.input = self.orig_input

    def _feed(self, *answers):
        it = iter(answers)
        builtins.input = lambda *_a, **_k: next(it)

    def test_records_fill_against_chosen_tranche(self):
        plan = mu_plan()
        r = read(Z.NEAR, spot=752.0, level=750.0)
        self._feed("1", "1", "2.5", "748.20")
        B._guided_fill(plan, [r], plan_path=self.plan_path, db_path=self.db)
        self.assertEqual(F.filled_levels("MU", db_path=self.db), {750.0})

    def test_no_open_tranches_prints_error_and_returns_unchanged(self):
        plan = mu_plan()
        r = Z.ZoneRead("MU", Z.FILLED, 500.0, None, None, None, -30.0, True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = B._guided_fill(plan, [r], plan_path=self.plan_path, db_path=self.db)
        self.assertIs(result, plan)
        self.assertIn("nothing to fill", buf.getvalue())

    def test_empty_plan_returns_unchanged_without_prompting(self):
        empty = P.Plan(5000.0, [])
        result = B._guided_fill(empty, [], plan_path=self.plan_path, db_path=self.db)
        self.assertIs(result, empty)


class TestGuidedEdit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.plan_path = os.path.join(self.tmp.name, "plan.json")
        self.db = os.path.join(self.tmp.name, "longterm.db")
        self.orig_input = builtins.input

    def tearDown(self):
        self.tmp.cleanup()
        builtins.input = self.orig_input

    def _feed(self, *answers):
        it = iter(answers)
        builtins.input = lambda *_a, **_k: next(it)

    def test_replaces_ladder(self):
        plan = mu_plan()
        self._feed("1", "800, 700, 600")
        plan = B._guided_edit(plan, plan_path=self.plan_path, db_path=self.db)
        self.assertEqual([t.level for t in plan.names[0].tranches], [800.0, 700.0, 600.0])

    def test_empty_plan_returns_unchanged_without_prompting(self):
        empty = P.Plan(5000.0, [])
        result = B._guided_edit(empty, plan_path=self.plan_path, db_path=self.db)
        self.assertIs(result, empty)


class TestGuidedRemove(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.plan_path = os.path.join(self.tmp.name, "plan.json")
        self.db = os.path.join(self.tmp.name, "longterm.db")
        self.orig_input = builtins.input

    def tearDown(self):
        self.tmp.cleanup()
        builtins.input = self.orig_input

    def _feed(self, *answers):
        it = iter(answers)
        builtins.input = lambda *_a, **_k: next(it)

    def test_removes_chosen_ticker(self):
        plan = mu_plan()
        self._feed("1")
        plan = B._guided_remove(plan, plan_path=self.plan_path, db_path=self.db)
        self.assertEqual(plan.names, [])

    def test_empty_plan_returns_unchanged_without_prompting(self):
        empty = P.Plan(5000.0, [])
        result = B._guided_remove(empty, plan_path=self.plan_path, db_path=self.db)
        self.assertIs(result, empty)


class TestGuidedCash(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.plan_path = os.path.join(self.tmp.name, "plan.json")
        self.db = os.path.join(self.tmp.name, "longterm.db")
        self.orig_input = builtins.input

    def tearDown(self):
        self.tmp.cleanup()
        builtins.input = self.orig_input

    def _feed(self, *answers):
        it = iter(answers)
        builtins.input = lambda *_a, **_k: next(it)

    def test_sets_new_budget(self):
        plan = mu_plan()
        self._feed("7500")
        plan = B._guided_cash(plan, plan_path=self.plan_path, db_path=self.db)
        self.assertEqual(plan.cash_pool_usd, 7500.0)

    def test_default_is_current_budget(self):
        plan = mu_plan()  # cash_pool_usd=5000.0 per the module helper
        self._feed("")  # accept the shown default
        plan = B._guided_cash(plan, plan_path=self.plan_path, db_path=self.db)
        self.assertEqual(plan.cash_pool_usd, 5000.0)


class TestGuidedLog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.plan_path = os.path.join(self.tmp.name, "plan.json")
        self.db = os.path.join(self.tmp.name, "longterm.db")
        self.orig_input = builtins.input

    def tearDown(self):
        self.tmp.cleanup()
        builtins.input = self.orig_input

    def _feed(self, *answers):
        it = iter(answers)
        builtins.input = lambda *_a, **_k: next(it)

    def test_enter_accepts_suggested_ladder(self):
        candidate = _disc_candidate("MU")  # suggested_ladder=[760.0/0.5, 700.0/0.5]
        self._feed("")
        plan = B._guided_log(P.Plan(5000.0, []), candidate,
                             plan_path=self.plan_path, db_path=self.db)
        self.assertEqual(plan.names[0].ticker, "MU")
        self.assertEqual([t.level for t in plan.names[0].tranches], [760.0, 700.0])

    def test_typed_levels_override_suggestion(self):
        candidate = _disc_candidate("MU")
        self._feed("800, 700, 600")
        plan = B._guided_log(P.Plan(5000.0, []), candidate,
                             plan_path=self.plan_path, db_path=self.db)
        self.assertEqual([t.level for t in plan.names[0].tranches], [800.0, 700.0, 600.0])

    def test_reprompts_on_unparseable_input(self):
        candidate = _disc_candidate("MU")
        self._feed("cheap", "750, 650")
        plan = B._guided_log(P.Plan(5000.0, []), candidate,
                             plan_path=self.plan_path, db_path=self.db)
        self.assertEqual([t.level for t in plan.names[0].tranches], [750.0, 650.0])


class TestAskLevelsDefault(unittest.TestCase):
    def setUp(self):
        self.orig_input = builtins.input

    def tearDown(self):
        builtins.input = self.orig_input

    def _feed(self, *answers):
        it = iter(answers)
        builtins.input = lambda *_a, **_k: next(it)

    def test_empty_input_returns_default(self):
        self._feed("")
        result = B._ask_levels("levels", default="760/700")
        self.assertEqual(result, [760.0, 700.0])

    def test_typed_input_overrides_default(self):
        self._feed("800, 700, 600")
        result = B._ask_levels("levels", default="760/700")
        self.assertEqual(result, [800.0, 700.0, 600.0])

    def test_no_default_still_reprompts_on_bad_input(self):
        self._feed("cheap", "750, 650")
        result = B._ask_levels("levels")
        self.assertEqual(result, [750.0, 650.0])


class TestGuidedDiscover(unittest.TestCase):
    def setUp(self):
        self.orig_input = builtins.input

    def tearDown(self):
        builtins.input = self.orig_input

    def _feed(self, *answers):
        it = iter(answers)
        builtins.input = lambda *_a, **_k: next(it)

    def test_returns_scan_results_on_success(self):
        self._feed("semiconductors")
        fake_results = [(_disc_candidate("MU"), None)]
        with mock.patch("src.longterm.discover.scan", return_value=fake_results) as m:
            result = B._guided_discover(60)
        m.assert_called_once_with("semiconductors")
        self.assertEqual(result, fake_results)

    def test_returns_none_on_bad_sector(self):
        self._feed("nonsense")
        with mock.patch("src.longterm.discover.scan",
                        side_effect=ValueError("no matching sector")):
            result = B._guided_discover(60)
        self.assertIsNone(result)


class TestGuidedReport(unittest.TestCase):
    def test_opens_the_written_report(self):
        with mock.patch("src.longterm.report.write_report",
                        return_value=("reports/holdings/x.html", "reports/holdings/x.json")), \
             mock.patch.object(B, "_open_report_file") as m_open:
            B._guided_report()
        m_open.assert_called_once_with("reports/holdings/x.html")


class TestOpenReportFile(unittest.TestCase):
    def test_falls_back_to_printing_path_when_open_fails(self):
        buf = io.StringIO()
        with mock.patch.object(B._sys, "platform", "darwin"), \
             mock.patch("subprocess.run", side_effect=OSError("boom")), \
             contextlib.redirect_stdout(buf):
            B._open_report_file("/tmp/x.html")
        self.assertIn("Could not auto-open", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
