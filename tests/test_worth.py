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
    """Edge measured in units of its own error bar."""

    def test_an_edge_clearing_two_error_bars_is_strong(self):
        w = worth.assess(_row(ev_per_contract=100.0, ev_noise=25.0))
        self.assertEqual(w.grade, "STRONG")
        self.assertAlmostEqual(w.sigma, 4.0, places=6)

    def test_an_edge_inside_one_error_bar_is_thin(self):
        w = worth.assess(_row(ev_per_contract=9.0, ev_noise=40.0))
        self.assertEqual(w.grade, "THIN")

    def test_an_edge_between_one_and_two_bars_is_clear(self):
        w = worth.assess(_row(ev_per_contract=45.0, ev_noise=30.0))
        self.assertEqual(w.grade, "CLEAR")

    def test_the_boundary_is_inclusive_at_two_bars(self):
        self.assertEqual(worth.assess(_row(ev_per_contract=50.0,
                                           ev_noise=25.0)).grade, "STRONG")

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

    def test_cheap_trading_does_not_rescue_a_noisy_edge(self):
        w = worth.assess(_row(ev_per_contract=5.0, ev_noise=50.0))
        self.assertEqual(w.grade, "THIN")
        self.assertEqual(w.limiting, "edge vs its error bar")

    def test_the_grade_is_never_better_than_any_single_margin(self):
        for ev, noise, bid, ask in [(100.0, 25.0, 9.90, 10.10),
                                    (45.0, 30.0, 9.90, 10.10),
                                    (9.0, 40.0, 9.90, 10.10),
                                    (1000.0, 10.0, 8.00, 12.00)]:
            w = worth.assess(_row(ev_per_contract=ev, ev_noise=noise,
                                  bid=bid, ask=ask))
            margins = [g for g in (worth._grade_sigma(w.sigma),
                                   worth._grade_friction(w.friction))
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

    def test_a_genuine_constraint_is_still_named(self):
        w = worth.assess(_row(ev_per_contract=500.0, ev_noise=25.0,
                              bid=9.60, ask=10.40))     # 8% round trip
        self.assertEqual(w.grade, "CLEAR")
        self.assertIn("limited by trading cost", w.line())


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
