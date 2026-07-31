"""Stop-overshoot measurement: how far stopped trades ran past their rule.

The point of these tests is that the measurement cannot flatter itself. A stop
with no numeric level must be excluded rather than scored as a perfect exit,
and a trade that never overshot must not be able to cancel out one that did.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.overshoot_report import (  # noqa: E402
    format_summary,
    is_stop_exit,
    overshoot_for,
    parse_stop_level,
    summarize,
)


def _row(**kw):
    base = {
        "date": "2026-07-01", "exit_date": "2026-07-10", "ticker": "AAPL",
        "strategy_name": "Long Call", "pnl_pct": -0.75, "pnl_usd": -300.0,
        "exit_reason": "Stop Loss (-50%)", "capital_at_risk": 400.0,
    }
    base.update(kw)
    return base


class TestStopLevelParsing(unittest.TestCase):
    def test_premium_stop_level(self):
        self.assertAlmostEqual(parse_stop_level("Stop Loss (-50%)"), 0.50)

    def test_credit_stop_level(self):
        self.assertAlmostEqual(
            parse_stop_level("Stop Loss (100% of credit)"), 1.00)

    def test_strike_breach_has_no_numeric_level(self):
        self.assertIsNone(parse_stop_level("Stop Loss (strike breached)"))
        # ...but it is still a stop, so it is counted, just not distributed
        self.assertTrue(is_stop_exit("Stop Loss (strike breached)"))

    def test_non_stop_reasons_are_not_stops(self):
        for reason in ("Take Profit (50% of credit)",
                       "Time Exit (21d to expiry)",
                       "Expired (settled at intrinsic)", None, ""):
            self.assertIsNone(parse_stop_level(reason))
            self.assertFalse(is_stop_exit(reason))


class TestOvershoot(unittest.TestCase):
    def test_overshoot_is_the_excess_beyond_the_stated_stop(self):
        self.assertAlmostEqual(overshoot_for(-0.75, "Stop Loss (-50%)"), 0.25)
        self.assertAlmostEqual(
            overshoot_for(-1.575, "Stop Loss (100% of credit)"), 0.575)

    def test_a_stop_honoured_exactly_scores_zero(self):
        self.assertAlmostEqual(overshoot_for(-0.50, "Stop Loss (-50%)"), 0.0)

    def test_an_early_stop_scores_negative_not_none(self):
        # Exited before the rule demanded it: real information, kept as a
        # negative overshoot so the median reflects it.
        self.assertAlmostEqual(overshoot_for(-0.40, "Stop Loss (-50%)"), -0.10)

    def test_a_missing_level_is_excluded_not_treated_as_zero(self):
        # The failure this guards: scoring an unlevelled stop as 0.0 overshoot
        # would drag the median toward "rules were followed".
        self.assertIsNone(overshoot_for(-2.0, "Stop Loss (strike breached)"))

    def test_a_profitable_stop_row_is_excluded(self):
        self.assertIsNone(overshoot_for(0.10, "Stop Loss (-50%)"))


class TestSummary(unittest.TestCase):
    def test_split_on_the_scheduler_death_date(self):
        rows = [
            _row(exit_date="2026-05-01", pnl_pct=-0.55),   # before, +5%
            _row(exit_date="2026-07-01", pnl_pct=-0.90),   # after,  +40%
            _row(exit_date="2026-07-02", pnl_pct=-1.00),   # after,  +50%
        ]
        s = summarize(rows, cutoff="2026-06-15")
        self.assertEqual(s["before"]["n"], 1)
        self.assertEqual(s["after"]["n"], 2)
        self.assertAlmostEqual(s["before"]["median"], 0.05)
        self.assertAlmostEqual(s["after"]["median"], 0.45)
        self.assertEqual(s["all"]["n"], 3)

    def test_unlevelled_stops_are_counted_separately(self):
        rows = [_row(), _row(exit_reason="Stop Loss (strike breached)")]
        s = summarize(rows)
        self.assertEqual(s["n_stop_exits"], 2)
        self.assertEqual(s["n_levelled"], 1)
        self.assertEqual(s["n_unlevelled"], 1)

    def test_share_overshot_counts_only_genuine_excess(self):
        rows = [
            _row(pnl_pct=-0.50),   # exactly on the rule
            _row(pnl_pct=-0.40),   # early
            _row(pnl_pct=-0.90),   # overshot
        ]
        s = summarize(rows)
        self.assertEqual(s["all"]["n"], 3)
        self.assertAlmostEqual(s["all"]["share_overshot"], 1 / 3)

    def test_rows_without_an_exit_date_fall_back_to_the_entry_date(self):
        rows = [_row(exit_date=None, date="2026-05-01", pnl_pct=-0.90)]
        s = summarize(rows, cutoff="2026-06-15")
        self.assertEqual(s["before"]["n"], 1)

    def test_worst_list_is_ordered_by_overshoot(self):
        rows = [_row(pnl_pct=-0.60), _row(pnl_pct=-1.20), _row(pnl_pct=-0.80)]
        s = summarize(rows)
        overshoots = [w["overshoot"] for w in s["worst"]]
        self.assertEqual(overshoots, sorted(overshoots, reverse=True))


class TestNoteRendering(unittest.TestCase):
    def test_the_note_states_the_artifact_and_refuses_to_retro_edit(self):
        s = summarize([_row(pnl_pct=-1.00, exit_date="2026-07-01")],
                      cutoff="2026-06-15")
        text = format_summary(s)
        self.assertIn("2026-06-15", text)
        self.assertIn("not corrected", text)
        self.assertIn("as-traded", text)
        # the removal condition is stated, not left implicit
        self.assertIn("verifiably alive", text)

    def test_the_note_is_attached_to_the_published_record(self):
        from scripts.publish_track_record import methodology_notes

        rows = [_row(pnl_pct=-1.00, exit_date="2026-07-01",
                     strategy_name="Bear Call",
                     exit_reason="Stop Loss (100% of credit)")]
        headings = [h for h, _ in methodology_notes(rows, {}, [], [])]
        self.assertIn("Stops overshot because exits were checked by hand",
                      headings)

    def test_no_stop_exits_means_no_note(self):
        from scripts.publish_track_record import methodology_notes

        rows = [_row(exit_reason="Take Profit (50% of credit)")]
        headings = [h for h, _ in methodology_notes(rows, {}, [], [])]
        self.assertNotIn("Stops overshot because exits were checked by hand",
                         headings)


if __name__ == "__main__":
    unittest.main()
