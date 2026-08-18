"""A heading that says CLEARED THE GATES must only show rows that did.

Found by running a real 111-ticker discovery scan on 2026-08-18. The scan's
own board reported `219 refused — negative expected value after cost`, and the
executive summary directly above it printed:

    1.  XOM $170.0 CALL @ $3.73  26.7% PoP  0.3x RR  -$78.21 (-21.0%) EV  ●●●○
    2.  XOM $167.5 CALL @ $3.20  30.6% PoP  0.5x RR  -$82.76 (-25.9%) EV  ●●○○
    3.  XOM $167.5 CALL @ $2.23  28.4% PoP  0.4x RR  -$66.69 (-29.8%) EV  ●○○○

Three negative-EV contracts under a heading asserting they cleared the gates,
on a scan that had just refused 219 rows for exactly that reason — and the
first one carrying the strongest WORTH pips on the block.

The cause was a frame mismatch. Every `gate_and_report` call in `run_scan`
assigns to a NEW name (`final_df`, `final_spreads`, `final_condors`,
`_gated`), so `picks` is never itself replaced by a gated frame, and
`print_executive_summary(picks, ...)` received the raw one. `cli_display`
carried a comment stating "Rows arrive already gated and ordered by carry
cost" — a claim about a caller, and it was false.

The gate now runs inside the renderer. The function that makes the claim is
the one that has to enforce it, for every caller, rather than trusting each
call site to have done it first.
"""
from __future__ import annotations

import contextlib
import io
import re
import unittest

import pandas as pd


def _row(symbol, ev, strike=100.0, premium=2.0, quality=0.50):
    return {
        "symbol": symbol, "strike": strike, "type": "call",
        "strategy_name": "Long Call", "premium": premium,
        "bid": premium - 0.02, "ask": premium + 0.02,
        "prob_profit": 0.55, "rr_ratio": 1.5,
        "ev_per_contract": ev,
        "ev_gross_per_contract": ev + 8.0,
        "ev_cost_per_contract": 8.0,
        "quality_score": quality, "max_loss": premium * 100,
        "expiration": "2026-09-18", "dte": 31,
        "spread_pct": 0.02, "volume": 500, "openInterest": 1000,
        "impliedVolatility": 0.30, "underlying_price": strike,
    }


def _render(df):
    """Capture what the executive summary actually prints."""
    from src.cli_display import print_executive_summary
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            print_executive_summary(
                df, {}, mode="Discovery scan", market_trend="Bull",
                volatility_regime="Low", macro_risk=False)
        except TypeError:
            print_executive_summary(df, {})
    return re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())


class TestTheSummaryOnlyShowsWhatCleared(unittest.TestCase):

    def test_a_negative_ev_row_is_not_displayed(self):
        """The live defect, reduced. XOM was refused `negative_ev` by
        `gate_board` and displayed anyway."""
        df = pd.DataFrame([_row("XOM", -78.21), _row("GOOD", +120.0)])
        out = _render(df)
        self.assertIn("GOOD", out, "the row that cleared should still show")
        self.assertNotIn("XOM", out,
                         "a row the gate refused is under CLEARED THE GATES")

    def test_the_first_row_is_not_a_refused_one(self):
        """Order matters as much as membership: the block shows `.head(3)`, so
        a refused row at position 0 becomes the headline pick."""
        df = pd.DataFrame([_row("BAD", -90.0), _row("OK", +50.0)])
        out = _render(df)
        self.assertNotIn("BAD", out)

    def test_when_everything_is_refused_it_says_so(self):
        """An empty block under a positive heading reads as 'nothing found'
        when the truth is 'everything was rejected'. Those are different
        answers and the operator acts differently on each."""
        df = pd.DataFrame([_row("BAD1", -90.0), _row("BAD2", -70.0)])
        out = _render(df)
        self.assertNotIn("BAD1", out)
        self.assertNotIn("BAD2", out)
        self.assertTrue(
            re.search(r"refus|none .*clear|nothing .*clear", out, re.I),
            f"silent empty block; said nothing about the refusals:\n{out}")

    def test_a_clean_board_is_unchanged(self):
        df = pd.DataFrame([_row("AAA", +100.0), _row("BBB", +90.0)])
        out = _render(df)
        for sym in ("AAA", "BBB"):
            self.assertIn(sym, out)

    def test_an_empty_frame_does_not_raise(self):
        self.assertIsInstance(_render(pd.DataFrame()), str)


class TestTheStaleClaimIsGone(unittest.TestCase):
    """The comment asserted the caller had already gated. It had not, and the
    comment is why nobody checked."""

    def test_no_comment_claims_rows_arrive_already_gated(self):
        import pathlib
        src = pathlib.Path("src/cli_display.py").read_text()
        self.assertNotIn("Rows arrive already gated", src)


if __name__ == "__main__":
    unittest.main()
