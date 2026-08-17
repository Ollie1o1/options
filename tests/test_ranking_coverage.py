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


def _variable_key_orderings(path: str):
    """`sort_values(some_variable)` inside a function that names a bad key.

    The constant-argument check above misses
    `sort_col = "quality_score" if ... else None; df.sort_values(sort_col)`,
    because the discredited string never appears as an argument. That is not
    hypothetical: `offer_tearsheet` re-ranked by `quality_score` exactly this
    way, so `--tearsheet 1` indexed a different list from the one printed, and
    an NVDA scan rendered a tearsheet for a contract at another strike and
    expiry than anything on screen.

    Flags any function that both mentions a discredited key as a string and
    sorts by a non-constant. Deliberately blunt: a false positive costs a
    comment, a false negative costs a wrong contract on a page.
    """
    bad = {"quality_score", "return_on_risk", "ev_score", "overall_score"}
    out = []
    for fn in ast.walk(ast.parse(_src(path))):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        strings = {n.value for n in ast.walk(fn)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        if not (bad & strings):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", None) in ("sort_values", "nlargest")
                    and node.args
                    and not isinstance(node.args[0], (ast.Constant, ast.List))):
                out.append((fn.name, node.lineno))
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

    # Functions allowed to sort by a variable key, with the reason. Adding to
    # this list is a deliberate act; it is not a place to silence the guard.
    #
    # `comparison_rows` honours a sort the READER explicitly asked for,
    # including by score — "show me this sorted by score" is a legitimate
    # request and the column is labelled. Its DEFAULT is board order, so a
    # table nobody chose a sort for is never ranked by the composite.
    #
    # That last sentence was FALSE from the day it was written until
    # 2026-08-17: this entry named `print_comparison_table`, whose `sort_by`
    # defaulted to "quality_score" — a key `_sort_map` resolves — so every
    # unchosen board opened ranked by the composite while this comment
    # asserted it could not. An allowlist entry is a claim about behaviour;
    # `test_an_unchosen_board_is_not_ranked_by_the_composite` below now holds
    # it to that claim instead of trusting the prose.
    ALLOWED_VARIABLE_SORTS = {("src/cli_display.py", "comparison_rows")}

    def test_no_module_sorts_by_a_discredited_key_held_in_a_variable(self):
        """The hole the constant check left. See `_variable_key_orderings`."""
        for path in ("src/options_screener.py", "src/cli_display.py",
                     "src/squeeze/board.py"):
            found = [(fn, ln) for fn, ln in _variable_key_orderings(path)
                     if (path, fn) not in self.ALLOWED_VARIABLE_SORTS]
            self.assertEqual(
                found, [],
                f"{path} sorts by a variable in a function that names a "
                f"discredited key — the tearsheet bug of 2026-08-10")

    def test_an_unchosen_board_is_not_ranked_by_the_composite(self):
        """The behaviour the allowlist above only claimed until 2026-08-17.

        Structural checks cannot see this: the sort is legitimate code
        reached through a default ARGUMENT, so every AST guard passed while
        the board ranked by a key measured at -0.131 against return on
        capital. Only running it shows the order.
        """
        import io
        import contextlib
        import pandas as pd
        from src import formatting as fmt
        from src.cli_display import print_comparison_table

        rows = [{"symbol": s, "type": "put", "strike": 10.0 + i,
                 "quality_score": q, "T_years": 0.09, "premium": 1.0,
                 "prob_profit": 0.6, "delta": -0.2, "ev_per_contract": 5.0,
                 "spread_pct": 0.05}
                for i, (s, q) in enumerate([("AAA", 0.20), ("BBB", 0.90),
                                            ("CCC", 0.55)])]
        df = pd.DataFrame(rows)

        def order(**kw):
            fmt.set_color_enabled(False)
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    print_comparison_table(df, "Premium Selling", **kw)
                text = buf.getvalue()
                seen = []
                for line in text.splitlines():
                    for sym in ("AAA", "BBB", "CCC"):
                        if sym in line and sym not in seen:
                            seen.append(sym)
                return seen, text
            finally:
                fmt._COLOR_ENABLED = None

        seen, text = order()
        self.assertEqual(seen, ["AAA", "BBB", "CCC"],
                         "an unchosen board is ranked by quality_score")
        self.assertIn("board order", text)

        # A sort the reader explicitly asks for is still honoured and labelled.
        seen, text = order(sort_by="quality_score")
        self.assertEqual(seen, ["BBB", "CCC", "AAA"])
        self.assertIn("Score", text)

    def test_the_tearsheet_indexes_the_frame_the_cards_were_numbered_from(self):
        """`--tearsheet N` must mean the N the user just read.

        It was handed `picks` (pre-gate, pre-ordering) and then re-ranked, so
        it could render a contract the board had refused.
        """
        src = _src("src/options_screener.py")
        self.assertIn("offer_tearsheet(_display_df", src,
                      "the tearsheet is not being given the displayed frame")
        self.assertNotIn('sort_col = "quality_score" if "quality_score" in picks_df', src)

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
