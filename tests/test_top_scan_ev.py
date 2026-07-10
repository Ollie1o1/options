"""The cross-ticker top-N table must show what the tearsheet decides on.

`run_top_scan` ranks by `quality_score`, whose out-of-sample IC is +0.04 at
p=0.35 — no demonstrated edge. The tearsheet's verdict, by contrast, comes from
round-trip net EV and nothing else. On 2026-07-10 the two disagreed on five of
the eight top picks, and the table showed no EV column with which to notice.

These tests pin the table to the SAME verdict rule the tearsheet uses, so the
disagreement is visible on the screen where the picks are chosen.
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from src import cli_display  # noqa: E402


def _pick(**over):
    row = {
        "symbol": "XOM", "type": "call", "strike": 140.0, "expiration": "2026-07-31",
        "T_years": 20 / 365.0, "delta": 0.42, "iv_percentile_30": 0.73,
        "prob_profit": 0.295, "premium": 2.72, "quality_score": 0.633,
        "score_drivers": "+Theta(0.27) +IV vel(0.07)",
        "ev_per_contract": -27.69, "ev_gross_per_contract": 12.0,
        "ev_cost_per_contract": 39.69, "vega_dollar": 12.0, "iv_confidence": "High",
        "spread_pct": 0.084,
    }
    row.update(over)
    return row


def _render(rows):
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli_display.print_top_n_table(pd.DataFrame(rows), len(rows))
    return buf.getvalue()


class TestTopTableShowsNetEv(unittest.TestCase):
    def test_table_has_a_net_ev_column(self):
        out = _render([_pick()])
        self.assertIn("Net EV", out)
        self.assertIn("-28", out)

    def test_the_top_ranked_pick_can_be_marked_skip(self):
        # The headline finding: rank 1 by score, negative EV after cost.
        out = _render([_pick()])
        self.assertIn("SKIP", out)

    def test_a_positive_ev_pick_clear_of_its_band_is_marked_take(self):
        out = _render([_pick(symbol="WMT", ev_per_contract=31.23,
                             ev_gross_per_contract=70.0, vega_dollar=8.0)])
        self.assertIn("TAKE", out)

    def test_a_positive_ev_pick_inside_its_band_is_marked_marginal(self):
        # +$8 of edge on a contract whose vega is worth $10 per IV point.
        out = _render([_pick(symbol="WMT", ev_per_contract=7.89,
                             ev_gross_per_contract=50.0, vega_dollar=10.0,
                             iv_confidence="Low")])
        self.assertIn("MARG", out)
        self.assertNotIn("TAKE", out)

    def test_a_contract_with_no_gross_edge_is_skip_not_take(self):
        out = _render([_pick(ev_per_contract=-778.74, ev_gross_per_contract=-700.0)])
        self.assertIn("SKIP", out)

    def test_missing_ev_is_shown_as_indeterminate_not_as_zero(self):
        out = _render([_pick(ev_per_contract=float("nan"))])
        self.assertIn("INDET", out)
        self.assertNotIn("$0", out)

    def test_iv_column_is_not_labelled_as_implied_vol(self):
        # The column holds iv_percentile_30, not impliedVolatility. Calling it
        # "IV%" invites reading 73% as a 73-vol contract; XOM's IV was 28.7%.
        out = _render([_pick()])
        self.assertIn("IV%ile", out)

    def test_footer_counts_the_picks_the_verdict_rule_rejects(self):
        out = _render([_pick(), _pick(symbol="WMT", ev_per_contract=31.23,
                                      ev_gross_per_contract=70.0, vega_dollar=8.0)])
        self.assertIn("1 of 2", out)
        self.assertIn("net EV", out)

    def test_footer_is_silent_when_every_pick_clears(self):
        out = _render([_pick(ev_per_contract=200.0, ev_gross_per_contract=240.0)])
        self.assertNotIn("of 1 ranked pick", out)

    def test_verdict_rule_is_the_tearsheets_rule_not_a_copy(self):
        # One rule, one place. A second implementation is a second thing to drift.
        from src.tearsheet.render import decide_verdict
        self.assertIs(cli_display._verdict_for_row.__globals__["decide_verdict"],
                      decide_verdict)


class TestEmptyAndDegraded(unittest.TestCase):
    def test_empty_frame_still_says_so(self):
        self.assertIn("No contracts", _render([]) or _render([]))

    def test_a_row_with_no_ev_columns_at_all_does_not_crash(self):
        row = _pick()
        for k in ("ev_per_contract", "ev_gross_per_contract", "ev_cost_per_contract"):
            row.pop(k)
        out = _render([row])
        self.assertIn("XOM", out)
        self.assertIn("INDET", out)


if __name__ == "__main__":
    unittest.main()
