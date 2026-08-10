"""Coverage guards: the discredited metric must not creep back into ordering.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_ranking_coverage -v

`quality_score` measures -0.131 against return on capital on long calls and its
top quintile lost $10,173 across the book. Fifteen call sites once ordered
boards by it, or by keys measured worse (`return_on_risk` -0.216, `ev_score`
-0.325 on condors). These tests parse the source so a future edit cannot quietly
restore an ordering nobody re-measured.

Source-parsing rather than behavioural on purpose: the defect is "some module
sorts a display frame by X", which is a statement about the code, not about one
frame's output.
"""
import ast
import pathlib
import unittest


def _src(path: str) -> str:
    return pathlib.Path(path).read_text()


def _score_orderings(path: str):
    """Every `sort_values`/`nlargest` on a discredited key, in real code.

    Parsed rather than grepped: the first version of this guard matched its own
    explanatory comments and the docstrings describing the defect, so a module
    that had been fixed still failed.
    """
    bad = {"quality_score", "return_on_risk", "ev_score", "overall_score"}
    out = []
    for node in ast.walk(ast.parse(_src(path))):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "attr", None)
        if name not in ("sort_values", "nlargest"):
            continue
        consts = [a.value for a in node.args if isinstance(a, ast.Constant)]
        for a in node.args:
            if isinstance(a, ast.List):
                consts += [e.value for e in a.elts if isinstance(e, ast.Constant)]
        for kw in node.keywords:
            if isinstance(kw.value, ast.Constant):
                consts.append(kw.value.value)
        if bad & {c for c in consts if isinstance(c, str)}:
            out.append((name, node.lineno))
    return out


# Two places may still consult the composite, both documented inline:
#   1. `filter_and_score`'s funnel sort — SELECTION, not display, and the
#      replacement key tested on 2026-08-09 was a coin flip, so swapping it
#      would trade a bad key for an unmeasured one.
#   2. `rank_by_verdict`'s except branch — the failure-safe. A board rendered
#      in a discredited order beats a board that does not render.
_ALLOWED_SELECTION_SORTS = 2


class NoDiscreditedOrderingTest(unittest.TestCase):

    def test_no_display_module_orders_a_frame_by_a_discredited_key(self):
        for path in ("src/cli_display.py", "src/squeeze/board.py"):
            self.assertEqual(_score_orderings(path), [],
                             f"{path} orders a frame by a key measured negative")

    def test_the_screener_keeps_only_the_documented_selection_sorts(self):
        """Two remain, both deliberate and both documented inline: the
        `filter_and_score` funnel sort, and `rank_by_verdict`'s failure-safe
        fallback — a board that renders in a discredited order beats a board
        that does not render."""
        hits = _score_orderings("src/options_screener.py")
        self.assertLessEqual(
            len(hits), _ALLOWED_SELECTION_SORTS, f"undocumented orderings: {hits}")

    def test_the_condor_board_no_longer_sorts_by_return_on_risk(self):
        """-0.216 against return on capital, n=139: its top pick was its worst."""
        src = _src("src/options_screener.py")
        self.assertNotIn('final_condors = iron_condors_df.sort_values("return_on_risk"', src)

    def test_the_squeeze_stash_is_not_ordered_by_the_composite(self):
        src = _src("src/options_screener.py")
        self.assertNotIn('_sq_calls.sort_values("quality_score"', src)


class WorthCoverageTest(unittest.TestCase):
    """Every card that can carry a WORTH line does."""

    def test_one_shared_implementation(self):
        from src import cli_display
        self.assertTrue(callable(cli_display.worth_text))

    def test_single_leg_spread_and_condor_cards_all_use_it(self):
        src = _src("src/cli_display.py")
        # decision zone (single leg), spread detail card, condor detail card
        self.assertGreaterEqual(src.count("worth_text("), 4)

    def test_no_card_renders_stars_for_the_composite(self):
        """Stars read as a verdict. This metric has not earned one."""
        src = _src("src/cli_display.py")
        live = [ln for ln in src.splitlines()
                if "format_quality_score" in ln and not ln.strip().startswith("#")]
        self.assertEqual(
            live, [],
            f"a card still renders quality_score as stars: {live}")


class GateCoverageTest(unittest.TestCase):
    """Every board that produces tradeable candidates is gated."""

    def test_each_board_branch_routes_through_the_gate(self):
        src = _src("src/options_screener.py")
        for board in ("BUDGET", "CREDIT SPREADS", "IRON CONDOR",
                      "PREMIUM SELLING", "TICKER"):
            self.assertIn(f'"{board}"', src, f"{board} board is not gated")

    def test_the_top_n_path_is_gated(self):
        self.assertIn('gate_and_report(combined', _src("src/options_screener.py"))

    def test_the_gate_module_is_the_only_place_gates_are_defined(self):
        from src import pick_ranking
        self.assertEqual(len(pick_ranking.GATES), 6)


if __name__ == "__main__":
    unittest.main()
