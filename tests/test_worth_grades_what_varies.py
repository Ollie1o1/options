"""WORTH grades the checks that vary; the one that never does is reported, not graded.

WORTH took the WEAKEST of three margins: edge vs its own error bar, trading
cost, and win rate vs what the strategy needs. Taking the weakest means one
degenerate check pins the badge.

Once the error bar was measured (2026-08-17) the first check became degenerate
by nature: no single option trade's edge clears the uncertainty in the vol
forecast behind it. That is a fact about the asset class, not about any
contract, and it dragged every row on every single-leg board to THIN. The two
columns an operator reads first — Score and WORTH — then said nothing, which
is a design defect rather than honesty:

    "if the worth doesn't tell me its worth and the score doesn't tell me its
     score then like what?"

So sigma comes out of the grade and stays visible as the `Edge/err` NUMBER,
which does vary (-0.71 to +0.95 across 65 live rows) and orders candidates
sensibly. The universal truth it carries is stated once at board level instead
of being repeated as a constant on every row.

`Score` is dropped from the boards entirely: `quality_score` measures OOS IC
-0.12, "not distinguishable from zero", and a column called Score invites
precisely the misreading above.
"""
from __future__ import annotations

import unittest

from src.worth import Worth, assess


def _row(**kw):
    r = {"ev_per_contract": 40.0, "vega_dollar": 20.0,
         "hv_252d": 0.25, "hv_30d": 0.25,
         "expiration": "2026-09-18", "date": "2026-08-17"}
    r.update(kw)
    return r


class TestSigmaNoLongerCapsTheGrade(unittest.TestCase):

    def test_a_cheap_contract_clearing_its_bar_is_not_pinned_to_thin(self):
        """Friction 3% and a +16pp family margin should not read THIN just
        because no option trade ever clears its vol uncertainty."""
        w = assess(_row(strategy_name="Bull Put"),
                   historical_win_rate=0.664, required_win_rate=0.509)
        self.assertNotEqual(w.grade, "THIN")

    def test_the_sigma_is_still_reported(self):
        """Dropped from the grade, kept as a number — it is what `Edge/err`
        renders and it is the column that actually discriminates."""
        w = assess(_row(strategy_name="Bull Put"),
                   historical_win_rate=0.664, required_win_rate=0.509)
        self.assertIsNotNone(w.sigma)

    def test_a_tiny_edge_no_longer_drags_a_good_candidate_down(self):
        big_edge = assess(_row(strategy_name="Bull Put", ev_per_contract=400.0),
                          historical_win_rate=0.664, required_win_rate=0.509)
        small_edge = assess(_row(strategy_name="Bull Put", ev_per_contract=4.0),
                            historical_win_rate=0.664, required_win_rate=0.509)
        self.assertEqual(big_edge.grade, small_edge.grade)


class TestTheGradeStillDiscriminates(unittest.TestCase):
    """The two checks that vary per candidate must still decide it."""

    def test_a_family_below_its_bar_grades_worse(self):
        good = assess(_row(strategy_name="Bull Put"),
                      historical_win_rate=0.664, required_win_rate=0.509)
        bad = assess(_row(strategy_name="Bear Call"),
                     historical_win_rate=0.593, required_win_rate=0.667)
        self.assertNotEqual(good.grade, bad.grade)

    def test_the_limiting_margin_is_never_the_error_bar(self):
        w = assess(_row(strategy_name="Bear Call"),
                   historical_win_rate=0.593, required_win_rate=0.667)
        self.assertNotIn("error bar", w.limiting)

    def test_an_ungradeable_row_still_says_ungraded(self):
        w = assess({"ev_per_contract": 1.0}, historical_win_rate=None)
        self.assertEqual(w.grade, "UNGRADED")


class TestScoreIsOffTheBoards(unittest.TestCase):

    def _src(self):
        from src.paths import repo_path
        with open(repo_path("src/cli_display.py")) as fh:
            return fh.read()

    def test_the_comparison_table_has_no_score_column(self):
        src = self._src()
        i = src.index("def print_comparison_table(")
        body = src[i:src.index("\ndef ", i + 10)]
        self.assertNotIn("'Score'", body,
                         "quality_score is back on the board under a name that "
                         "reads as a verdict; its OOS IC is -0.12")

    def test_the_top_n_table_has_no_score_column(self):
        src = self._src()
        i = src.index("def print_top_n_table(")
        body = src[i:src.index("\ndef ", i + 10)]
        self.assertNotIn("'Score'", body)


class TestTheUncertaintyIsStatedOnce(unittest.TestCase):
    """Removed from every row, so it must appear at board level instead."""

    def test_the_per_risk_table_says_it(self):
        import io
        from contextlib import redirect_stdout
        import pandas as pd
        from src import formatting as fmt
        from src.cli_display import print_per_risk_table
        fmt.set_color_enabled(False)
        try:
            df = pd.DataFrame([{"symbol": "SPY", "type": "put", "strike": 700.0,
                                "capital_at_risk": 70000.0,
                                "reward_per_risk": 0.01,
                                "net_ev_per_risk": 0.001}])
            buf = io.StringIO()
            with redirect_stdout(buf):
                print_per_risk_table(df, lambda r: "Short Put", None)
            out = buf.getvalue().lower()
            self.assertIn("uncertainty", out)
        finally:
            fmt._COLOR_ENABLED = None


if __name__ == "__main__":
    unittest.main()
