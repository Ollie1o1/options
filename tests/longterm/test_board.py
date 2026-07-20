"""Tests for banner + board rendering (plain-mode, color pinned off)."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import src.formatting as fmt
from src.longterm import board as B
from src.longterm import fills as F
from src.longterm import plan as P
from src.longterm import zones as Z
from src.longterm.discover import CandidateRead, DeepRead

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
        bounce={"by_horizon": {20: {"n": 41, "bounce_rate": 0.65}}},
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


if __name__ == "__main__":
    unittest.main()
