"""Rendering tests. Assert on the OUTPUT STRING, never on source content."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import src.formatting as fmt
from src.catalyst import board
from src.catalyst.design import Amendments
from src.catalyst.implied import ImpliedMove
from src.catalyst.models import CatalystEvent, Coverage, Trial
from src.catalyst.runway import Runway

fmt._COLOR_ENABLED = False  # pin: never env vars (supports_color memoizes)


def a_trial(nct="NCT06510816", date="2026-10-31", precision="day",
            dtype="ESTIMATED", phase="PHASE3"):
    return Trial(nct_id=nct, sponsor_name="Annexon, Inc.",
                 brief_title="Vonaprument in Dry AMD With Geographic Atrophy",
                 phase=phase, event_date=date, date_precision=precision,
                 date_type=dtype, status="ACTIVE_NOT_RECRUITING",
                 enrollment=400, allocation="RANDOMIZED", masking="QUADRUPLE",
                 primary_outcome="Change From Baseline in GA Lesion Area",
                 conditions=("Geographic Atrophy",))


def a_row(**kw):
    base = dict(
        event=CatalystEvent(trial=a_trial(), ticker="ANNX", mcap=976_332_558.0),
        other_events=2,
        amendments=Amendments(versions=11, outcomes_updated=3,
                              status_now="ACTIVE_NOT_RECRUITING",
                              flags=("outcome measures edited 3x",),
                              available=True),
        runway=Runway(cash=412_000_000.0, burn_per_quarter=58_000_000.0,
                      quarters=7.1, runway_end="2028-05-01",
                      funded_through=True),
        implied=ImpliedMove(expiry="2026-11-20", spot=22.10, straddle=8.40,
                            move_pct=0.38),
    )
    base.update(kw)
    return board.BoardRow(**base)


def coverage():
    return Coverage(swept=599, resolved=162, dropped_unresolved=437,
                    dropped_out_of_band=40, deep_failures=3)


class TestFormatEventDate(unittest.TestCase):
    def test_day_precision_estimate(self):
        self.assertEqual(board.format_event_date("2026-10-31", "day", "ESTIMATED"),
                         "2026-10-31 (est)")

    def test_actual_date_carries_no_qualifier(self):
        self.assertEqual(board.format_event_date("2026-10-31", "day", "ACTUAL"),
                         "2026-10-31")

    def test_month_precision_is_visibly_different(self):
        out = board.format_event_date("2027-03", "month", "ESTIMATED")
        self.assertIn("~", out)
        self.assertIn("2027-03", out)
        self.assertNotEqual(out, "2027-03 (est)")


class TestCollapse(unittest.TestCase):
    def test_one_row_per_ticker(self):
        events = [
            CatalystEvent(trial=a_trial("NCT1", "2026-12-01"), ticker="ANNX"),
            CatalystEvent(trial=a_trial("NCT2", "2026-10-31"), ticker="ANNX"),
            CatalystEvent(trial=a_trial("NCT3", "2026-11-15"), ticker="SRPT"),
        ]
        out = board.collapse(events)
        self.assertEqual(len(out), 2)

    def test_keeps_the_soonest_event_and_counts_the_rest(self):
        events = [
            CatalystEvent(trial=a_trial("NCT1", "2026-12-01"), ticker="ANNX"),
            CatalystEvent(trial=a_trial("NCT2", "2026-10-31"), ticker="ANNX"),
        ]
        (event, others), = board.collapse(events)
        self.assertEqual(event.trial.nct_id, "NCT2")
        self.assertEqual(others, 1)

    def test_tie_breaks_to_the_later_phase(self):
        events = [
            CatalystEvent(trial=a_trial("NCT1", "2026-10-31", phase="PHASE2"),
                          ticker="ANNX"),
            CatalystEvent(trial=a_trial("NCT2", "2026-10-31", phase="PHASE3"),
                          ticker="ANNX"),
        ]
        (event, _), = board.collapse(events)
        self.assertEqual(event.trial.nct_id, "NCT2")


class TestRender(unittest.TestCase):
    def setUp(self):
        self.out = board.render([a_row()], coverage())

    def test_shows_ticker_and_date(self):
        self.assertIn("ANNX", self.out)
        self.assertIn("2026-10-31 (est)", self.out)

    def test_labels_the_date_as_primary_completion_never_readout(self):
        self.assertIn("PRIMARY COMPLETION", self.out.upper())
        self.assertNotIn("READOUT", self.out.upper())

    def test_shows_the_funded_through_verdict(self):
        self.assertIn("FUNDED THROUGH", self.out.upper())

    def test_shows_a_raise_first_warning_when_underfunded(self):
        row = a_row(runway=Runway(cash=40_000_000.0, burn_per_quarter=30_000_000.0,
                                  quarters=1.3, runway_end="2026-12-01",
                                  funded_through=False))
        self.assertIn("RAISE BEFORE", board.render([row], coverage()).upper())

    def test_unknown_runway_says_unknown_not_zero(self):
        out = board.render([a_row(runway=Runway())], coverage())
        self.assertNotIn("0.0 q", out)
        self.assertIn("unknown", out.lower())

    def test_shows_amendment_flag(self):
        self.assertIn("outcome measures edited 3x", self.out)

    def test_unavailable_amendments_say_so(self):
        out = board.render([a_row(amendments=Amendments())], coverage())
        self.assertIn("amendment history unavailable", out.lower())

    def test_shows_implied_move(self):
        self.assertIn("38", self.out)

    def test_missing_implied_move_omits_the_line_rather_than_showing_zero(self):
        # Asserting "0%" is absent would match "27.0%" in the coverage footer;
        # the real claim is that the implied row is not rendered at all.
        out = board.render([a_row(implied=ImpliedMove())], coverage())
        self.assertNotIn("implied", out)
        self.assertIn("implied", self.out)  # present when the move IS known

    def test_shows_other_event_count(self):
        self.assertIn("+2 more", self.out)

    def test_singular_when_one_other_event(self):
        self.assertIn("+1 more event", board.render([a_row(other_events=1)],
                                                    coverage()))

    def test_omits_the_counter_when_there_are_no_others(self):
        self.assertNotIn("more event", board.render([a_row(other_events=0)],
                                                    coverage()))

    def test_prints_coverage_footer(self):
        self.assertIn("599", self.out)
        self.assertIn("27.0%", self.out)

    def test_prints_the_base_rate_with_its_caveat(self):
        self.assertIn("other drugs", self.out.lower())

    def test_empty_board_says_so_rather_than_printing_nothing(self):
        out = board.render([], coverage())
        self.assertIn("no catalysts", out.lower())

    def test_no_ansi_when_color_disabled(self):
        self.assertNotIn("\033[", self.out)

    def test_truncation_is_stated_on_the_board(self):
        c = coverage()
        c.shown, c.truncated = 40, 97
        out = board.render([a_row()], c)
        self.assertIn("40", out)
        self.assertIn("137", out)
        self.assertIn("--limit", out)

    def test_no_truncation_notice_when_nothing_was_withheld(self):
        self.assertNotIn("--limit", self.out)

    def test_truncation_is_stated_even_on_an_empty_board(self):
        # An empty board with withheld names is the worst case to hide.
        c = coverage()
        c.shown, c.truncated = 0, 12
        self.assertIn("12", board.render([], c))


if __name__ == "__main__":
    unittest.main()
