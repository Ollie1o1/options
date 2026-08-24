"""What reaches the board is chosen by EV per dollar at risk, not quality_score.

`quality_score` was removed as a board SORT — its top quintile is the worst
cell in the ledger (31.6% win rate, -19.9% return on capital, against +5.2%
for the [0.55, 0.65) bucket) and it lost at the #1 slot on a Wilcoxon test at
p=0.89. A pre-declared race on 2026-08-24 put its rank IC at -0.0194,
95% CI [-0.2587, +0.1454].

It kept choosing the board anyway. `pick_top_per_bucket` selects which
contracts are shown, and BOTH its branches sorted by `quality_score` first:

    _sort_cols = ["quality_score", "spread_pct", "volume", "openInterest", ...]

So fixing the display sort and `rank_by_verdict` changed the ORDER of a list
whose MEMBERSHIP was still being decided by the discredited metric. This is
the same defect the project already recorded once — "the board was ranked by
quality_score all along" — surviving in a second function nobody re-checked.
Proved live: after `rank_by_verdict` moved to EV/$risk, the concentration
warning changed (MU 12/15 -> WMT 5/15) while the displayed nine picks did not
move at all.

Selection is now EV per dollar AT RISK, with `quality_score` demoted to a
tie-break, matching `candidate_verdict.rank`.
"""
from __future__ import annotations

import unittest

import pandas as pd

from src.filters import pick_top_per_bucket


def _row(symbol, premium, ev, quality, bucket="LOW", strike=50.0):
    return {
        "symbol": symbol, "type": "call", "strike": strike,
        "expiration": "2026-12-18", "premium": premium,
        "ev_per_contract": ev, "quality_score": quality,
        "price_bucket": bucket, "spread_pct": 0.03,
        "volume": 1000, "openInterest": 5000, "T_years": 0.25,
        "strategy_name": "Long Call",
    }


class TestSelectionUsesEdgeNotScore(unittest.TestCase):

    def test_the_better_edge_per_dollar_is_shown_over_the_higher_score(self):
        """The whole point. `poor_edge` wins on quality_score and loses badly
        on edge per dollar; it must not be the contract shown."""
        df = pd.DataFrame([
            _row("SCORE", premium=1.00, ev=2.0, quality=0.99),   # +0.02/$
            _row("EDGE", premium=1.00, ev=40.0, quality=0.10),   # +0.40/$
        ])
        out = pick_top_per_bucket(df, per_bucket=1, diversify_tickers=True)
        self.assertEqual(list(out["symbol"]), ["EDGE"])

    def test_it_holds_on_the_single_stock_branch_too(self):
        """Both branches had the defect; fixing one would leave single-stock
        mode still choosing by score."""
        df = pd.DataFrame([
            _row("SCORE", premium=1.00, ev=2.0, quality=0.99),
            _row("EDGE", premium=1.00, ev=40.0, quality=0.10),
        ])
        out = pick_top_per_bucket(df, per_bucket=1, diversify_tickers=False)
        self.assertEqual(list(out["symbol"]), ["EDGE"])

    def test_a_big_ticket_does_not_win_on_size(self):
        """The live MU case: a large position with a weaker per-dollar edge
        must not displace a small one with a stronger edge."""
        df = pd.DataFrame([
            _row("BIG", premium=43.67, ev=1288.0, quality=0.52),  # +0.29/$
            _row("SMALL", premium=1.72, ev=74.0, quality=0.57),   # +0.43/$
        ])
        out = pick_top_per_bucket(df, per_bucket=1, diversify_tickers=True)
        self.assertEqual(list(out["symbol"]), ["SMALL"])

    def test_score_still_breaks_a_tie(self):
        """Demoted, not deleted — it remains a reasonable tie-break."""
        df = pd.DataFrame([
            _row("LOWSCORE", premium=1.00, ev=20.0, quality=0.10),
            _row("HISCORE", premium=1.00, ev=20.0, quality=0.90),
        ])
        out = pick_top_per_bucket(df, per_bucket=1, diversify_tickers=True)
        self.assertEqual(list(out["symbol"]), ["HISCORE"])


class TestItStillBehavesLikeItself(unittest.TestCase):

    def test_buckets_are_still_respected(self):
        df = pd.DataFrame([
            _row("A", 1.0, ev=5.0, quality=0.5, bucket="LOW"),
            _row("B", 5.0, ev=50.0, quality=0.5, bucket="MEDIUM", strike=60.0),
            _row("C", 20.0, ev=200.0, quality=0.5, bucket="HIGH", strike=70.0),
        ])
        out = pick_top_per_bucket(df, per_bucket=1, diversify_tickers=True)
        self.assertEqual(set(out["price_bucket"]), {"LOW", "MEDIUM", "HIGH"})

    def test_ticker_diversification_still_happens(self):
        df = pd.DataFrame([
            _row("DUP", 1.00, ev=90.0, quality=0.9, strike=50.0),
            _row("DUP", 1.00, ev=80.0, quality=0.8, strike=51.0),
            _row("OTHER", 1.00, ev=10.0, quality=0.1, strike=52.0),
        ])
        out = pick_top_per_bucket(df, per_bucket=2, diversify_tickers=True)
        self.assertEqual(set(out["symbol"]), {"DUP", "OTHER"},
                         "the second slot went to a repeat ticker")

    def test_an_empty_frame_is_handled(self):
        self.assertTrue(pick_top_per_bucket(pd.DataFrame(), per_bucket=3).empty)

    def test_a_frame_without_ev_does_not_crash(self):
        """Older frames and other modes may not carry ev_per_contract."""
        df = pd.DataFrame([{
            "symbol": "A", "type": "call", "strike": 50.0,
            "expiration": "2026-12-18", "quality_score": 0.5,
            "price_bucket": "LOW", "spread_pct": 0.03, "volume": 10,
            "openInterest": 10, "T_years": 0.25,
        }])
        out = pick_top_per_bucket(df, per_bucket=1, diversify_tickers=True)
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
