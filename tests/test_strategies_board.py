"""The board must show signal, account, friction and status at a glance."""
from __future__ import annotations

import unittest

from src import formatting as fmt
from src.strategies import friction as fr
from src.strategies.board import format_board, format_detail
from src.strategies.seed import LIBRARY


class BoardTest(unittest.TestCase):
    def setUp(self):
        self._prev = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = False

    def tearDown(self):
        fmt._COLOR_ENABLED = self._prev

    def test_lists_every_setup(self):
        out = format_board(LIBRARY)
        for r in LIBRARY:
            self.assertIn(r.spec.id, out)

    def test_marks_controls_distinctly(self):
        self.assertIn("CONTROL", format_board(LIBRARY).upper())

    def test_shows_account_eligibility(self):
        out = format_board(LIBRARY).lower()
        self.assertIn("tfsa", out)
        self.assertIn("taxable", out)

    def test_account_filter_hides_ineligible_setups(self):
        out = format_board(LIBRARY, account="tfsa")
        self.assertNotIn("naked_call_extended", out)

    def test_respects_width(self):
        for line in format_board(LIBRARY, width=100).splitlines():
            self.assertLessEqual(len(line), 100, f"too wide: {line!r}")

    def test_default_arguments_branch(self):
        self.assertTrue(format_board(LIBRARY))


class FrictionColumnTest(unittest.TestCase):
    """The cost wall is the binding constraint, so it is a column, not a footnote."""

    def setUp(self):
        self._prev = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = False

    def tearDown(self):
        fmt._COLOR_ENABLED = self._prev

    def test_the_board_has_a_friction_column(self):
        self.assertIn("FRICTION", format_board(LIBRARY, table=fr.RECORDED).upper())

    def test_a_credit_spread_shows_its_measured_toll(self):
        out = format_board(LIBRARY, table=fr.RECORDED)
        row = [ln for ln in out.splitlines() if "call_spread_extended" in ln][0]
        self.assertIn("23%", row)

    def test_holding_to_expiry_shows_the_cheaper_figure(self):
        out = format_board(LIBRARY, table=fr.RECORDED)
        managed = [ln for ln in out.splitlines() if "put_spread_ivr50 " in ln][0]
        held = [ln for ln in out.splitlines() if "put_spread_ivr50_hold" in ln][0]
        self.assertIn("68%", managed)
        self.assertIn("34%", held)

    def test_an_unmeasured_structure_shows_a_dash_not_a_zero(self):
        out = format_board(LIBRARY, table=fr.RECORDED)
        row = [ln for ln in out.splitlines() if "covered_call_holdings" in ln][0]
        self.assertIn("—", row)
        self.assertNotIn("0%", row)

    def test_the_ceiling_is_stated_somewhere_on_the_board(self):
        self.assertIn("ceiling", format_board(LIBRARY, table=fr.RECORDED).lower())

    def test_the_detail_view_explains_where_the_number_came_from(self):
        r = [x for x in LIBRARY if x.spec.id == "put_spread_ivr50"][0]
        out = format_detail(r, table=fr.RECORDED)
        self.assertIn("credit", out.lower())
        self.assertIn("2026-08-06", out)


class DetailTest(unittest.TestCase):
    def setUp(self):
        self._prev = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = False

    def tearDown(self):
        fmt._COLOR_ENABLED = self._prev

    def test_shows_hypothesis_and_signal(self):
        r = [x for x in LIBRARY if x.signal.get("iv_rank_min")][0]
        out = format_detail(r)
        self.assertIn(r.hypothesis[:30], out)
        self.assertIn("iv_rank_min", out)

    def test_shows_the_capital_requirement(self):
        out = format_detail(LIBRARY[0]).lower()
        self.assertIn("cash-secured", out)

    def test_states_when_evidence_is_absent(self):
        out = format_detail(LIBRARY[0]).lower()
        self.assertTrue("not yet" in out or "no evidence" in out,
                        "an unevaluated setup must SAY so")

    def test_shows_amendments(self):
        r = LIBRARY[0].amend("status", "dead", reason="cost wall ate it",
                             date="2026-09-01")
        self.assertIn("cost wall ate it", format_detail(r))

    def test_respects_width(self):
        for line in format_detail(LIBRARY[0], width=100).splitlines():
            self.assertLessEqual(len(line), 100, f"too wide: {line!r}")


if __name__ == "__main__":
    unittest.main()
