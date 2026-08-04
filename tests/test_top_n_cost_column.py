"""The cross-ticker table shows what it now ranks on.

`rank_by_verdict` orders these rows by round-trip cost, but the table showed
only quality_score and told the reader that was the ranking. A table that
hides its own sort key is worse than one that sorts badly.
"""
import io
import unittest
from contextlib import redirect_stdout

import pandas as pd

from src.cli_display import print_top_n_table


def _df():
    return pd.DataFrame([
        {"symbol": "QQQ", "type": "call", "strike": 733.0, "expiration": "2026-09-18",
         "T_years": 45 / 365, "delta": 0.42, "premium": 20.0, "quality_score": 0.50,
         "friction_pct": 0.004, "verdict_passed": True, "prob_profit": 0.4},
        {"symbol": "COIN", "type": "call", "strike": 170.0, "expiration": "2026-09-18",
         "T_years": 45 / 365, "delta": 0.40, "premium": 8.0, "quality_score": 0.95,
         "friction_pct": 0.34, "verdict_passed": False, "prob_profit": 0.3},
    ])


def _render(df):
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_top_n_table(df, len(df))
    return buf.getvalue()


class CostColumnTest(unittest.TestCase):
    def test_the_table_has_a_cost_column(self):
        self.assertIn("Cost", _render(_df()))

    def test_a_cheap_candidate_shows_a_low_cost(self):
        self.assertIn("0%", _render(_df()))

    def test_an_expensive_candidate_shows_its_real_cost(self):
        self.assertIn("34%", _render(_df()))

    def test_the_footer_no_longer_claims_the_rank_is_quality_score(self):
        self.assertNotIn("Rank is quality_score", _render(_df()))

    def test_the_footer_names_the_actual_sort_key(self):
        out = _render(_df()).lower()
        self.assertTrue("cost" in out and "rank" in out)

    def test_a_frame_without_verdict_columns_still_renders(self):
        """Failure-safe: a scan whose quotes did not arrive must still print."""
        df = _df().drop(columns=["friction_pct", "verdict_passed"])
        self.assertIn("QQQ", _render(df))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
