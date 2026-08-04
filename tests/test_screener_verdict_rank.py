"""The screener orders candidates by what survives their costs.

`quality_score` correlates -0.132 with return on the long-premium book and its
top bucket is the worst cell in the ledger, so sorting by it actively selects
the worst candidates. It is kept as a tie-breaker only.
"""
import unittest

import pandas as pd

from src.options_screener import rank_by_verdict


class RankByVerdictTest(unittest.TestCase):
    def _df(self):
        return pd.DataFrame([
            # high score, terrible market
            {"symbol": "AAA", "strategy_name": "Long Call", "bid": 5.0, "ask": 15.0,
             "quality_score": 0.95},
            # low score, tight market
            {"symbol": "BBB", "strategy_name": "Long Call", "bid": 9.9, "ask": 10.1,
             "quality_score": 0.55},
        ])

    def test_the_tighter_market_ranks_first_despite_the_lower_score(self):
        out = rank_by_verdict(self._df())
        self.assertEqual(out.iloc[0]["symbol"], "BBB")

    def test_the_verdict_columns_are_attached(self):
        out = rank_by_verdict(self._df())
        for col in ("verdict_passed", "verdict_reason", "friction_pct"):
            self.assertIn(col, out.columns)

    def test_the_unaffordable_market_is_marked_refused(self):
        out = rank_by_verdict(self._df())
        row = out[out["symbol"] == "AAA"].iloc[0]
        self.assertFalse(bool(row["verdict_passed"]))
        self.assertIn("friction", row["verdict_reason"])

    def test_an_empty_frame_is_returned_unchanged(self):
        self.assertTrue(rank_by_verdict(pd.DataFrame()).empty)

    def test_a_frame_with_no_quotes_falls_back_to_the_old_ordering(self):
        """Failure-safe: a scan whose quotes did not arrive must still produce
        a report rather than an exception."""
        df = pd.DataFrame([{"symbol": "AAA", "quality_score": 0.4},
                           {"symbol": "BBB", "quality_score": 0.9}])
        out = rank_by_verdict(df)
        self.assertEqual(list(out["symbol"]), ["BBB", "AAA"])

    def test_ties_are_broken_by_quality_score(self):
        df = pd.DataFrame([
            {"symbol": "LOW", "strategy_name": "Long Call", "bid": 9.9, "ask": 10.1,
             "quality_score": 0.30},
            {"symbol": "HIGH", "strategy_name": "Long Call", "bid": 9.9, "ask": 10.1,
             "quality_score": 0.80},
        ])
        self.assertEqual(rank_by_verdict(df).iloc[0]["symbol"], "HIGH")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
