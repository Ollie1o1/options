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
        #
        # Rendered without the legend, because the legend always DEFINES
        # "implied" — the claim under test is that the ROW is absent, not
        # that the word never appears on screen.
        out = board.render([a_row(implied=ImpliedMove())], coverage(),
                           legend=False)
        self.assertNotIn("implied", out)
        # present when the move IS known
        self.assertIn("implied", board.render([a_row()], coverage(),
                                              legend=False))

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

    def test_multi_phase_trial_shows_both_phases_not_just_the_first(self):
        # Live 2026-08-25: 40 of 130 trials in a PHASE2 sweep were registered
        # across phases. Rendering the first alone printed a bare "PH1", which
        # reads as a Phase 1 trial leaking past a Ph2/Ph3 filter.
        ev = CatalystEvent(trial=a_trial(), ticker="VSTM", mcap=641_000_000.0)
        object.__setattr__(ev.trial, "phases", ("PHASE1", "PHASE2"))
        out = board.render([board.BoardRow(event=ev)], coverage())
        self.assertIn("PH1/2", out)

    def test_phase_label_for_a_single_phase_is_unchanged(self):
        self.assertIn("PH3 PRIMARY COMPLETION", self.out)

    def test_truncation_is_stated_even_on_an_empty_board(self):
        # An empty board with withheld names is the worst case to hide.
        c = coverage()
        c.shown, c.truncated = 0, 12
        self.assertIn("12", board.render([], c))


class TestPdufaSection(unittest.TestCase):
    """PDUFA rows render SEPARATELY from trial rows.

    They come from an 8-K, not CT.gov, so they have no phase, enrollment or
    masking. Forcing them through the trial row would mean inventing those
    fields — the same defect as every other number in this repo that described
    something other than its label.
    """

    def _row(self, date="2026-11-14", funded=True):
        from src.catalyst.pdufa import PdufaEvent
        return board.PdufaRow(
            event=PdufaEvent(ticker="CYTK", cik=1, event_date=date,
                             filed="2026-05-05", doc_url="https://x/y.htm"),
            runway=Runway(cash=4.2e8, burn_per_quarter=6e7, quarters=7.0,
                          runway_end="2028-01-01", funded_through=funded),
            implied=ImpliedMove(expiry="2026-11-20", spot=50.0, straddle=9.0,
                                move_pct=0.18))

    def test_section_appears_with_its_own_heading(self):
        out = board.render([], coverage(), pdufa=[self._row()])
        self.assertIn("PDUFA", out.upper())
        self.assertIn("CYTK", out)

    def test_the_date_is_shown_without_an_estimated_qualifier(self):
        # A PDUFA date is firm. Rendering it "(est)" would understate it.
        out = board.render([], coverage(), pdufa=[self._row()])
        self.assertIn("2026-11-14", out)
        self.assertNotIn("2026-11-14 (est)", out)

    def test_shows_the_funded_verdict(self):
        out = board.render([], coverage(), pdufa=[self._row()])
        self.assertIn("FUNDED THROUGH", out.upper())

    def test_shows_a_raise_before_warning(self):
        out = board.render([], coverage(), pdufa=[self._row(funded=False)])
        self.assertIn("RAISE BEFORE", out.upper())

    def test_shows_the_implied_move(self):
        self.assertIn("18", board.render([], coverage(), pdufa=[self._row()]))

    def test_states_when_it_was_announced(self):
        self.assertIn("2026-05-05",
                      board.render([], coverage(), pdufa=[self._row()]))

    def test_no_section_when_there_are_no_pdufa_rows(self):
        self.assertNotIn("PDUFA", board.render([a_row()], coverage()).upper())

    def test_pdufa_rows_alone_do_not_trigger_the_empty_board_message(self):
        out = board.render([], coverage(), pdufa=[self._row()])
        self.assertNotIn("no catalysts", out.lower())

    def test_rows_are_sorted_by_date(self):
        out = board.render([], coverage(),
                           pdufa=[self._row("2027-01-05"), self._row("2026-09-26")])
        self.assertLess(out.index("2026-09-26"), out.index("2027-01-05"))


if __name__ == "__main__":
    unittest.main()


class TestBandedRender(unittest.TestCase):
    TODAY = "2026-08-26"

    def rows(self, n_near=2, n_mid=3, n_far=2):
        out = []
        for i in range(n_near):
            out.append(a_row(event=CatalystEvent(
                trial=a_trial(f"NCT-N{i}", f"2026-09-{5 + i:02d}"),
                ticker=f"NEAR{i}", mcap=5e8)))
        for i in range(n_mid):
            out.append(a_row(event=CatalystEvent(
                trial=a_trial(f"NCT-M{i}", f"2026-10-{5 + i:02d}"),
                ticker=f"MID{i}", mcap=5e8)))
        for i in range(n_far):
            out.append(a_row(event=CatalystEvent(
                trial=a_trial(f"NCT-F{i}", f"2027-01-{5 + i:02d}"),
                ticker=f"FAR{i}", mcap=5e8)))
        return out

    def test_prints_a_header_for_each_populated_band(self):
        out = board.render(self.rows(), coverage(), today=self.TODAY)
        self.assertIn("NEXT 30 DAYS", out)
        self.assertIn("31–90 DAYS", out)
        self.assertIn("BEYOND 90 DAYS", out)

    def test_an_empty_band_prints_no_header(self):
        # Dead chrome is worse than no chrome: a permanently empty section
        # trains a reader to skip the region it lives in.
        out = board.render(self.rows(n_far=0), coverage(), today=self.TODAY)
        self.assertNotIn("BEYOND 90 DAYS", out)

    def test_bands_appear_soonest_first(self):
        out = board.render(self.rows(), coverage(), today=self.TODAY)
        self.assertLess(out.index("NEXT 30 DAYS"), out.index("31–90 DAYS"))
        self.assertLess(out.index("31–90 DAYS"), out.index("BEYOND 90 DAYS"))

    def test_detail_top_bounds_the_number_of_full_blocks(self):
        out = board.render(self.rows(n_near=6, n_mid=6, n_far=6), coverage(),
                           today=self.TODAY, detail_top=3)
        self.assertEqual(out.count("design "), 3)

    def test_rows_past_detail_top_still_appear_compactly(self):
        out = board.render(self.rows(n_near=6, n_mid=6, n_far=6), coverage(),
                           today=self.TODAY, detail_top=3)
        self.assertIn("FAR5", out)

    def test_detail_goes_to_the_soonest_rows(self):
        out = board.render(self.rows(n_near=2, n_mid=6, n_far=6), coverage(),
                           today=self.TODAY, detail_top=2)
        near_block = out.index("NEAR0")
        self.assertIn("design", out[near_block:near_block + 400])

    def test_month_precision_survives_into_a_compact_row(self):
        # "~2027-03" and "2027-03-15" are different objects. Compacting a row
        # must not quietly promote an estimated month to a date.
        rows = [a_row(event=CatalystEvent(
            trial=a_trial("NCT-X", "2027-03", precision="month"),
            ticker="MONTHY", mcap=5e8))]
        out = board.render(rows, coverage(), today=self.TODAY, detail_top=0)
        self.assertIn("~2027-03", out)

    def test_compact_rows_carry_the_asset_so_the_drug_is_identifiable(self):
        out = board.render(self.rows(), coverage(), today=self.TODAY,
                           detail_top=0)
        self.assertIn("Vonaprument", out)

    def test_states_per_band_coverage_at_the_top(self):
        from src.catalyst.models import BandCoverage
        c = coverage()
        c.bands = [BandCoverage(band="NEXT_30", found=6, shown=6),
                   BandCoverage(band="D31_90", found=21, shown=14),
                   BandCoverage(band="BEYOND_90", found=70, shown=20)]
        out = board.render(self.rows(), c, today=self.TODAY)
        head = out[:out.index("NEXT 30 DAYS", out.index("NEXT 30 DAYS") + 1)]
        self.assertIn("14", head)
        self.assertIn("21", head)

    def test_no_ansi_when_color_disabled(self):
        out = board.render(self.rows(), coverage(), today=self.TODAY)
        self.assertNotIn("\033[", out)


class TestSuperlativeAnnotations(unittest.TestCase):
    TODAY = "2026-08-26"

    def test_names_the_shortest_runway_on_the_board(self):
        rows = []
        for i, q in enumerate((2.0, 9.0, 15.0)):
            rows.append(a_row(
                event=CatalystEvent(trial=a_trial(f"NCT{i}", f"2026-09-{5+i:02d}"),
                                    ticker=f"TK{i}", mcap=5e8),
                runway=Runway(cash=1e8, burn_per_quarter=2e7, quarters=q,
                              runway_end="2027-01-01", funded_through=True)))
        out = board.render(rows, coverage(), today=self.TODAY, detail_top=3)
        self.assertIn("shortest runway shown", out)

    def test_a_single_row_board_claims_no_superlative(self):
        out = board.render([a_row()], coverage(), today=self.TODAY)
        self.assertNotIn("shortest runway", out)
        self.assertNotIn("most-amended", out)


class TestLegendAndFooter(unittest.TestCase):
    def test_defines_every_field_it_prints(self):
        out = board.render([a_row()], coverage())
        for label in ("runway", "implied", "amend", "prior"):
            self.assertIn(label, out)

    def test_states_the_month_precision_convention(self):
        # The mid-month rule decides which band a row lands in, so a reader
        # must be able to see the rule being applied.
        out = board.render([a_row()], coverage())
        self.assertIn("mid-month", out.lower())

    def test_the_base_rate_caveat_appears_exactly_once(self):
        # It used to ride on every row — ~30 repetitions of the same 9 words.
        rows = [a_row(event=CatalystEvent(
            trial=a_trial(f"NCT{i}", f"2026-09-{5 + i:02d}"),
            ticker=f"TK{i}", mcap=5e8)) for i in range(5)]
        out = board.render(rows, coverage(), detail_top=5)
        self.assertEqual(out.lower().count("not a forecast"), 1)

    def test_says_it_is_not_ranked(self):
        self.assertIn("NOT ranked", board.render([a_row()], coverage()))

    def test_states_the_no_evidence_finding(self):
        out = board.render([a_row()], coverage())
        self.assertIn("NO EVIDENCE", out.upper())

    def test_legend_can_be_suppressed(self):
        out = board.render([a_row()], coverage(), legend=False)
        self.assertNotIn("mid-month", out.lower())

    def test_coverage_still_prints_when_the_legend_is_off(self):
        # Coverage is not decoration. A board that hides how little it saw
        # is the failure this package was built to avoid.
        out = board.render([a_row()], coverage(), legend=False)
        self.assertIn("599", out)

    def test_truncation_hint_appears_when_names_were_withheld(self):
        c = coverage()
        c.shown, c.truncated = 40, 57
        out = board.render([a_row()], c)
        self.assertIn("--limit", out)
        self.assertIn("57", out)

    def test_window_label_is_shown_when_given(self):
        out = board.render([a_row()], coverage(),
                           window_label="6m window · 2026-08-26 → 2027-02-22")
        self.assertIn("2027-02-22", out)


class TestLiveRunRegressions(unittest.TestCase):
    """Two defects a green suite missed and one live run caught, 2026-08-26."""

    TODAY = "2026-08-26"

    def test_the_legend_box_borders_line_up(self):
        # ui.card pads but never truncates, so a description longer than the
        # box pushes the right border out. The `prior` row overflowed by 40
        # columns and printed a ragged box.
        out = board.render([a_row()], coverage(), width=100)
        box = [ln for ln in out.splitlines() if ln.startswith("│")]
        self.assertTrue(box, "legend box did not render")
        self.assertEqual(len({len(ln) for ln in box}), 1,
                         f"ragged legend box: widths {sorted({len(ln) for ln in box})}")

    def test_no_line_exceeds_the_requested_width(self):
        out = board.render([a_row()], coverage(), width=100)
        too_long = [ln for ln in out.splitlines() if len(ln) > 100]
        self.assertEqual(too_long, [], f"lines over width: {too_long}")

    def test_a_superlative_on_a_compact_row_is_still_shown(self):
        # Superlatives are computed across EVERY row but were rendered only
        # in detail blocks, so on a live 40-name board zero of four appeared.
        rows = []
        for i, q in enumerate((2.0, 9.0, 15.0, 20.0)):
            rows.append(a_row(
                event=CatalystEvent(
                    trial=a_trial(f"NCT{i}", f"2026-09-{5 + i:02d}"),
                    ticker=f"TK{i}", mcap=5e8),
                runway=Runway(cash=1e8, burn_per_quarter=2e7, quarters=q,
                              runway_end="2027-01-01", funded_through=True)))
        # detail_top=1 pushes TK3 (the longest runway) onto a compact row.
        out = board.render(rows, coverage(), today=self.TODAY, detail_top=1)
        self.assertIn("longest runway shown", out)

    def test_the_compact_asset_still_renders_alongside_a_note(self):
        rows = []
        for i, q in enumerate((2.0, 9.0, 15.0)):
            rows.append(a_row(
                event=CatalystEvent(
                    trial=a_trial(f"NCT{i}", f"2026-09-{5 + i:02d}"),
                    ticker=f"TK{i}", mcap=5e8),
                runway=Runway(cash=1e8, burn_per_quarter=2e7, quarters=q,
                              runway_end="2027-01-01", funded_through=True)))
        out = board.render(rows, coverage(), today=self.TODAY, detail_top=0)
        self.assertIn("Vonaprument", out)
        self.assertIn("shortest runway shown", out)

    def test_compact_columns_align_across_date_precisions(self):
        # "~2027-03 (est, month)" is 21 chars and "2026-09-05 (est)" is 16.
        # Sizing the date column to the short form let every month-precision
        # row shove phase, runway and implied rightwards.
        rows = [
            a_row(event=CatalystEvent(
                trial=a_trial("NCT-D", "2026-09-05"), ticker="DAYY", mcap=5e8)),
            a_row(event=CatalystEvent(
                trial=a_trial("NCT-M", "2026-09", precision="month"),
                ticker="MONY", mcap=5e8)),
        ]
        out = board.render(rows, coverage(), today=self.TODAY, detail_top=0)
        lines = out.splitlines()
        day = next(ln for ln in lines if ln.startswith("  DAYY"))
        month = next(ln for ln in lines if ln.startswith("  MONY"))
        self.assertEqual(day.index("FUND"), month.index("FUND"),
                         f"runway column misaligned:\n{day!r}\n{month!r}")
