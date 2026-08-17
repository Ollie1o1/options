"""Tests for src/worth.py — the graded "is this worth it" read.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_worth -v

The property that matters most is that the grade is the WEAKEST margin and
never a blend: a blend of plausible components is what produced a composite
whose top quintile lost $10,173.
"""
import unittest

from src import worth


def _row(**over):
    """A long call with a tight market: friction ~2% round trip."""
    row = {"strategy_name": "Long Call", "symbol": "AAPL",
           "bid": 9.90, "ask": 10.10, "premium": 10.0,
           "ev_per_contract": 100.0, "ev_noise": 25.0}
    row.update(over)
    return row


class SigmaGradeTest(unittest.TestCase):
    """Edge measured in units of its own error bar — REPORTED, not graded.

    These four cases pinned sigma as a grade input until 2026-08-17. It was
    removed because the measured error bar made the check degenerate: no single
    option trade's edge clears its vol-forecast uncertainty, so taking the
    weakest of three margins pinned every row on every single-leg board to
    THIN, and the two columns an operator reads first said nothing at all.

    The NUMBER still varies and is what `Edge/err` renders. See
    tests/test_worth_grades_what_varies.py.
    """

    def test_sigma_is_still_computed(self):
        w = worth.assess(_row(ev_per_contract=100.0, ev_noise=25.0))
        self.assertAlmostEqual(w.sigma, 4.0, places=6)

    def test_sigma_no_longer_decides_the_grade(self):
        big = worth.assess(_row(ev_per_contract=100.0, ev_noise=25.0))
        tiny = worth.assess(_row(ev_per_contract=9.0, ev_noise=40.0))
        self.assertEqual(big.grade, tiny.grade)

    def test_a_missing_ev_cannot_be_graded_on_sigma(self):
        w = worth.assess(_row(ev_per_contract=None, ev_noise=25.0))
        self.assertIsNone(w.sigma)


class WeakestLinkTest(unittest.TestCase):
    """The grade is the minimum of the margins. This is the whole design."""

    def test_a_huge_edge_with_ruinous_friction_is_not_strong(self):
        # bid/ask 8.00/12.00 -> 20% one crossing, 40% round trip
        w = worth.assess(_row(ev_per_contract=1000.0, ev_noise=10.0,
                              bid=8.00, ask=12.00))
        self.assertEqual(w.grade, "THIN")
        self.assertEqual(w.limiting, "trading cost")

    def test_a_noisy_edge_no_longer_drags_cheap_trading_down(self):
        """Was: "cheap trading does not rescue a noisy edge" — sigma capped
        the grade. It no longer does; the noise is reported as `Edge/err`
        instead, because every option trade's edge is noisy by that measure
        and grading on it made the badge a constant."""
        w = worth.assess(_row(ev_per_contract=5.0, ev_noise=50.0))
        self.assertEqual(w.grade, worth._grade_friction(w.friction))
        self.assertIsNotNone(w.sigma)

    def test_the_grade_is_never_better_than_any_single_margin(self):
        """Still the weakest link — over the margins that GRADE. Sigma is
        excluded here because it is reported rather than graded."""
        for ev, noise, bid, ask in [(100.0, 25.0, 9.90, 10.10),
                                    (45.0, 30.0, 9.90, 10.10),
                                    (9.0, 40.0, 9.90, 10.10),
                                    (1000.0, 10.0, 8.00, 12.00)]:
            w = worth.assess(_row(ev_per_contract=ev, ev_noise=noise,
                                  bid=bid, ask=ask))
            margins = [g for g in (worth._grade_friction(w.friction),
                                   worth._grade_breakeven(w.breakeven_margin))
                       if g is not None]
            self.assertEqual(worth.GRADES.index(w.grade),
                             min(worth.GRADES.index(m) for m in margins))

    def test_the_limiting_margin_is_named(self):
        w = worth.assess(_row(ev_per_contract=1000.0, ev_noise=10.0,
                              bid=8.00, ask=12.00))
        self.assertEqual(w.limiting, "trading cost")

    def test_nothing_is_limiting_when_every_margin_agrees(self):
        """A STRONG contract read 'STRONG, limited by edge vs its error bar'
        on a live scan — naming a constraint that was not constraining."""
        w = worth.assess(_row(ev_per_contract=500.0, ev_noise=25.0,
                              bid=9.90, ask=10.10))
        self.assertEqual(w.grade, "STRONG")
        self.assertEqual(w.limiting, "")
        self.assertNotIn("limited by", w.line())

    def test_friction_alone_can_no_longer_be_out_ranked_by_sigma(self):
        """With sigma out of the grade, a lone live margin IS the grade —
        nothing else is left to be weaker than it."""
        w = worth.assess(_row(ev_per_contract=500.0, ev_noise=25.0,
                              bid=9.60, ask=10.40))     # 8% round trip
        self.assertEqual(w.grade, worth._grade_friction(w.friction))


class BreakevenMarginTest(unittest.TestCase):
    """Arithmetic against your own history, not a prediction."""

    def _spread(self, **over):
        row = {"strategy_name": "Bull Put", "symbol": "SPY",
               "short_bid": 2.00, "short_ask": 2.05,
               "long_bid": 1.00, "long_ask": 1.05,
               "spread_width": 5.0, "premium": 1.0,
               "ev_per_contract": 100.0, "ev_noise": 25.0}
        row.update(over)
        return row

    def test_a_structure_needing_more_than_you_run_is_demoted(self):
        w = worth.assess(self._spread(), historical_win_rate=0.30)
        self.assertEqual(w.grade, "THIN")
        self.assertEqual(w.limiting, "breakeven margin")

    def test_a_comfortable_margin_does_not_demote(self):
        w = worth.assess(self._spread(), historical_win_rate=0.95)
        self.assertNotEqual(w.limiting, "breakeven margin")

    def test_without_a_win_rate_the_margin_is_absent_not_zero(self):
        w = worth.assess(self._spread())
        self.assertIsNone(w.breakeven_margin)


class UngradeableTest(unittest.TestCase):
    """An absent basis says so rather than defaulting to a middle grade."""

    def test_a_row_with_no_ev_and_no_quotes_is_ungraded(self):
        w = worth.assess({"strategy_name": "Long Call"})
        self.assertEqual(w.grade, "UNGRADED")

    def test_an_ungraded_row_says_why(self):
        self.assertIn("no trustworthy vol basis",
                      worth.assess({"strategy_name": "Long Call"}).line())

    def test_a_junk_row_never_raises(self):
        for bad in ({}, {"ev_per_contract": "abc"}, {"ev_noise": float("nan")}):
            self.assertIn(worth.assess(bad).grade,
                          ("UNGRADED", "THIN", "CLEAR", "STRONG"))


class LineTest(unittest.TestCase):
    """What lands on the card."""

    def test_the_line_names_the_grade_and_the_margins(self):
        line = worth.assess(_row()).line()
        self.assertIn("STRONG", line)
        self.assertIn("error bar", line)
        self.assertIn("of reward", line)

    def test_every_grade_has_a_distinct_pip_string(self):
        pips = {worth._PIPS[g] for g in worth.GRADES}
        self.assertEqual(len(pips), len(worth.GRADES))


class NoiseConsolidationTest(unittest.TestCase):
    """cli_display must not carry its own copy of the noise constants."""

    def test_cli_display_delegates_to_the_one_implementation(self):
        from src import cli_display
        from src.tearsheet.collect import ev_noise
        row = {"iv_confidence": "low", "vega_dollar": 30.0}
        self.assertEqual(cli_display._ev_noise_for_row(row), ev_noise(row))

    def test_the_sigma_table_exists_in_exactly_one_module(self):
        import pathlib
        src = pathlib.Path("src/cli_display.py").read_text()
        self.assertNotIn('"medium": 1.5', src)


if __name__ == "__main__":
    unittest.main()
