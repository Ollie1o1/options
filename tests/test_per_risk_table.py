"""The per-dollar-of-risk comparison table (design spec section 4).

The board can already REFUSE on a budget. What it could not do until now is
let a $127 spread and a $34,680 cash-secured put be compared at all: raw
premium and raw `ev_per_contract` are per contract, so the large position wins
by construction. Dividing every reward by that candidate's own capital at risk
is the whole point of the table.

Three rules the spec puts on it, each pinned below:

* it is DISPLAY-ONLY — it may not re-sort the board, because ranking was
  disproven out of sample (Wilcoxon p=0.89) and a table sorted by
  `Net EV/$risk` would be a ranking claim the evidence does not support;
* an unanswerable cell prints "n/a", never 0 and never "nan" — `budget_view`
  returns None to mean "not answerable", and 0 would be a different claim;
* it never raises, because a display defect must not kill a scan that has
  already done all its network work.
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import pandas as pd

from src import formatting as fmt


def _legs():
    """Two single legs with the SAME return on risk at very different sizes."""
    return pd.DataFrame([
        {"symbol": "CHEAP", "type": "put", "strike": 5.0,
         "max_profit": 73.0, "max_loss": 127.0, "ev_per_contract": 5.0,
         "capital_at_risk": 127.0,
         "reward_per_risk": 73.0 / 127.0, "net_ev_per_risk": 5.0 / 127.0},
        {"symbol": "RICH", "type": "put", "strike": 346.0,
         "max_profit": 19900.0, "max_loss": 34600.0,
         "ev_per_contract": 1362.2, "capital_at_risk": 34600.0,
         "reward_per_risk": 19900.0 / 34600.0,
         "net_ev_per_risk": 1362.2 / 34600.0},
    ])


def _label(row):
    return "Short Put"


def _render(df, budget=None, label_fn=_label):
    from src.cli_display import print_per_risk_table
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_per_risk_table(df, label_fn, budget)
    return buf.getvalue()


class TestTheAxisIsCommon(unittest.TestCase):
    """The reason the table exists at all."""

    def setUp(self):
        fmt.set_color_enabled(False)

    def tearDown(self):
        fmt._COLOR_ENABLED = None

    def test_both_sizes_appear(self):
        out = _render(_legs())
        self.assertIn("CHEAP", out)
        self.assertIn("RICH", out)

    def test_equal_return_on_risk_prints_the_same_number(self):
        """The comparison raw premium cannot make.

        The two rows differ by 272x in premium and by 272x in EV per contract.
        On the common axis they are the same candidate, and the table has to
        say so — same digits on both lines.
        """
        out = _render(_legs())
        cheap = [l for l in out.splitlines() if "CHEAP" in l][0]
        rich = [l for l in out.splitlines() if "RICH" in l][0]
        self.assertIn("0.575", cheap)
        self.assertIn("0.575", rich)

    def test_three_decimals_so_short_premium_does_not_collapse(self):
        """A cash-secured put returns ~0.006 on collateral, a credit spread
        ~0.60. At two decimals every short put on the board reads 0.01 and the
        axis stops discriminating exactly where the budget bites hardest."""
        df = pd.DataFrame([
            {"symbol": "AVGO", "capital_at_risk": 34800.0,
             "reward_per_risk": 200.0 / 34800.0, "net_ev_per_risk": 0.002},
            {"symbol": "F", "capital_at_risk": 1082.0,
             "reward_per_risk": 18.0 / 1082.0, "net_ev_per_risk": 0.004},
        ])
        out = _render(df)
        self.assertIn("0.006", out)
        self.assertIn("0.017", out)

    def test_the_risk_column_is_capital_not_premium(self):
        """For a cash-secured put the two differ ~170x — see prompt_for_budget."""
        out = _render(_legs())
        self.assertIn("$34,600", out)


class TestItIsDisplayOnly(unittest.TestCase):
    """Never a ranking, never a mutation."""

    def setUp(self):
        fmt.set_color_enabled(False)

    def tearDown(self):
        fmt._COLOR_ENABLED = None

    def test_row_order_is_the_boards_order_not_ev_descending(self):
        """CHEAP arrives first and has the LOWER net_ev_per_risk.

        Sorting by the new column would swap them, and that is precisely the
        ranking claim Wilcoxon p=0.89 refused.
        """
        out = _render(_legs())
        self.assertLess(out.index("CHEAP"), out.index("RICH"),
                        "the table re-sorted the board")

    def test_the_frame_is_not_mutated(self):
        df = _legs()
        before = df.copy(deep=True)
        _render(df)
        pd.testing.assert_frame_equal(df, before)

    def test_it_says_it_is_not_a_ranking(self):
        self.assertIn("not a ranking", _render(_legs()).lower())


class TestUnanswerableCells(unittest.TestCase):
    """None means "not answerable" and must not read as zero."""

    def setUp(self):
        fmt.set_color_enabled(False)

    def tearDown(self):
        fmt._COLOR_ENABLED = None

    def _unsizable(self):
        """A naked short call: risk is unbounded, so nothing per-risk exists."""
        return pd.DataFrame([
            {"symbol": "NAKED", "type": "call", "strike": 100.0,
             "max_profit": 250.0, "ev_per_contract": 40.0,
             "capital_at_risk": None,
             "reward_per_risk": None, "net_ev_per_risk": None},
        ])

    def test_unbounded_risk_prints_na(self):
        out = _render(self._unsizable())
        self.assertIn("NAKED", out)
        line = [l for l in out.splitlines() if "NAKED" in l][0]
        self.assertIn("n/a", line)

    def test_it_never_prints_zero_for_an_unknown(self):
        line = [l for l in _render(self._unsizable()).splitlines()
                if "NAKED" in l][0]
        self.assertNotIn("0.00", line)
        self.assertNotIn("$0", line)

    def test_it_never_prints_nan(self):
        """pandas turns a mixed None/float column into NaN if anyone assigns a
        plain list; `budget_view.annotate` guards that with object dtype, and
        the renderer must not undo it by formatting NaN as a number."""
        out = _render(self._unsizable())
        self.assertNotIn("nan", out.lower())


class TestItNeverBreaksAScan(unittest.TestCase):

    def setUp(self):
        fmt.set_color_enabled(False)

    def tearDown(self):
        fmt._COLOR_ENABLED = None

    def test_a_frame_that_was_never_annotated_prints_nothing(self):
        """Lottery and any future board that skips `_budget_board`.

        Without the columns there is nothing to say, and a header over an
        empty table would imply the axis was computed when it was not.
        """
        df = pd.DataFrame([{"symbol": "SPY", "type": "call", "strike": 5.0}])
        self.assertEqual(_render(df), "")

    def test_an_empty_frame_prints_nothing(self):
        self.assertEqual(_render(pd.DataFrame()), "")

    def test_none_prints_nothing(self):
        self.assertEqual(_render(None), "")

    def test_a_row_of_garbage_does_not_raise(self):
        df = pd.DataFrame([{"symbol": None, "capital_at_risk": "not a number",
                            "reward_per_risk": object(),
                            "net_ev_per_risk": float("nan")}])
        _render(df)  # must not raise

    def test_a_label_function_that_throws_does_not_raise(self):
        def boom(row):
            raise RuntimeError("no label")
        _render(_legs(), label_fn=boom)  # must not raise


class TestTheStructureColumn(unittest.TestCase):
    """Names the strategy whose risk definition produced the Risk cell."""

    def setUp(self):
        fmt.set_color_enabled(False)

    def tearDown(self):
        fmt._COLOR_ENABLED = None

    def test_a_single_leg_shows_its_strike(self):
        self.assertIn("Short Put 5", _render(_legs()))

    def test_a_spread_shows_both_strikes(self):
        df = pd.DataFrame([{"symbol": "GLD", "short_strike": 395.0,
                            "long_strike": 390.0, "max_profit": 60.0,
                            "ev_per_contract": 12.0, "capital_at_risk": 313.0,
                            "reward_per_risk": 0.60,
                            "net_ev_per_risk": 0.041}])
        self.assertIn("395/390", _render(df, label_fn=lambda r: "Bull Put"))

    def test_a_condor_shows_its_short_strikes(self):
        df = pd.DataFrame([{"symbol": "SPY", "short_put_strike": 400.0,
                            "long_put_strike": 395.0,
                            "short_call_strike": 440.0,
                            "long_call_strike": 445.0, "max_profit": 120.0,
                            "ev_per_contract": 8.0, "capital_at_risk": 380.0,
                            "reward_per_risk": 0.32,
                            "net_ev_per_risk": 0.021}])
        out = _render(df, label_fn=lambda r: "Iron Condor")
        self.assertIn("400/440", out)


class TestTheBudgetHeader(unittest.TestCase):

    def setUp(self):
        fmt.set_color_enabled(False)

    def tearDown(self):
        fmt._COLOR_ENABLED = None

    def test_a_budget_is_named_in_the_header(self):
        self.assertIn("$2,000", _render(_legs(), budget=2000.0))

    def test_no_budget_still_prints_the_table(self):
        """The axis is useful with no budget at all — it is what makes two
        differently sized candidates comparable, budget or not."""
        out = _render(_legs(), budget=None)
        self.assertIn("CHEAP", out)
        self.assertIn("RICH", out)


class TestTheWorthColumnsAreReallyPopulated(unittest.TestCase):
    """Cost% and WORTH must come out of the grader, not out of "n/a".

    Every other test here feeds hand-built rows that carry no quotes, and
    `worth.assess` correctly grades those UNGRADED — so all of them would
    still pass if `_per_risk_worth` were broken and returned nothing for
    everything. This one uses the shared preview fixture, which carries real
    quote fields, and pins that a gradeable row actually grades.
    """

    def setUp(self):
        fmt.set_color_enabled(False)
        import os
        import sys
        from src.paths import repo_path
        scripts = repo_path("scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)

    def tearDown(self):
        fmt._COLOR_ENABLED = None

    def test_a_row_with_quotes_gets_a_grade_and_a_cost(self):
        from ui_preview import df as preview_df
        from src.cli_display import _per_risk_worth
        row = preview_df().iloc[0]
        friction, _breakeven, grade = _per_risk_worth(row)
        self.assertIn(grade, ("STRONG", "CLEAR", "THIN", "UNGRADED"))
        self.assertNotEqual(grade, "", "the grader raised and was swallowed")
        self.assertIsNotNone(friction, "no round-trip cost came back")

    def test_the_grade_matches_the_card_above_it(self):
        """One grader, so the table cannot contradict the detail card."""
        from ui_preview import df as preview_df
        from src.cli_display import _per_risk_worth, worth_text
        row = preview_df().iloc[0]
        _f, _b, grade = _per_risk_worth(row)
        self.assertIn(grade, worth_text(row))


class TestItIsWiredToEveryAnnotatedBoard(unittest.TestCase):
    """Structural: the six `_budget_board` sites each print the table.

    Runtime coverage would mean driving `main()`, which runs
    `update_positions()` against the live book — see the SDD ledger for the
    two real positions that closed during exactly that probe.
    """

    def _source(self):
        from src.paths import repo_path
        with open(repo_path("src/options_screener.py")) as fh:
            return fh.read()

    def test_every_budget_use_line_is_accompanied_by_the_table(self):
        src = self._source()
        self.assertEqual(src.count("_print_budget_use("),
                         src.count("_print_per_risk_table("),
                         "a board prints the budget-use line but not the "
                         "per-risk table, or the reverse")

    def test_the_table_reaches_all_six_annotated_boards(self):
        src = self._source()
        # 6 call sites + 1 definition
        self.assertEqual(src.count("_print_per_risk_table("), 7)


if __name__ == "__main__":
    unittest.main()
