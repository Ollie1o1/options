"""The failure-safe must not quietly restore the discredited ordering.

`rank_by_verdict` is failure-safe by design: if quotes are missing or anything
raises, a board still renders. It used to do that by falling back to
`df.sort_values("quality_score", ascending=False)`, and the ranking guard
allowlisted it with the reasoning "a board rendered in a discredited order
beats a board that does not render."

That reasoning has a hole. `quality_score`'s TOP quintile is the worst cell in
the ledger — 31.6% win rate and -19.9% return on capital against +5.2% for the
[0.55, 0.65) bucket — so the fallback did not render a merely *neutral* order.
It rendered an actively adverse one, and every caller then truncates with
`.head(N)`. A degraded board therefore showed the worst candidates first, with
nothing on screen saying the ordering had changed at all.

So the fallback now sorts by nothing. Scan order carries no claim, which is
the honest state when the key that would order the board cannot be computed,
and the degradation is announced rather than inferred.
"""
from __future__ import annotations

import unittest

import pandas as pd

from src import ranking_health as rh


def _rows():
    """Ascending quality_score, so a quality_score sort would REVERSE this."""
    return pd.DataFrame([
        {"symbol": "AAA", "quality_score": 0.10, "strategy_name": "Long Call"},
        {"symbol": "BBB", "quality_score": 0.50, "strategy_name": "Long Call"},
        {"symbol": "CCC", "quality_score": 0.90, "strategy_name": "Long Call"},
    ])


class TestTheFallbackDoesNotRank(unittest.TestCase):

    def setUp(self):
        rh.reset()

    def tearDown(self):
        rh.reset()

    def _force_failure(self, df):
        """Drive rank_by_verdict down its except branch."""
        from src import candidate_verdict as cv
        from src import options_screener as osc
        orig = cv.rank
        cv.rank = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no quotes"))
        try:
            return osc.rank_by_verdict(df)
        finally:
            cv.rank = orig

    def test_scan_order_is_preserved_not_reversed_by_score(self):
        out = self._force_failure(_rows())
        self.assertEqual(list(out["symbol"]), ["AAA", "BBB", "CCC"],
                         "the fallback re-ranked by the discredited key")

    def test_the_top_row_is_not_the_highest_score(self):
        """The specific harm: .head(N) after a quality_score sort surfaces the
        worst candidates, because its top quintile is the worst ledger cell."""
        out = self._force_failure(_rows())
        self.assertNotEqual(out.iloc[0]["symbol"], "CCC")

    def test_the_failure_is_recorded(self):
        self._force_failure(_rows())
        self.assertTrue(rh.is_degraded())
        self.assertIn("no quotes", (rh.reason() or ""))

    def test_a_successful_rank_leaves_health_clean(self):
        from src import options_screener as osc
        osc.rank_by_verdict(_rows())
        self.assertFalse(rh.is_degraded())

    def test_an_empty_frame_is_not_a_degradation(self):
        from src import options_screener as osc
        osc.rank_by_verdict(pd.DataFrame())
        self.assertFalse(rh.is_degraded())


class TestTheAnnouncement(unittest.TestCase):

    def setUp(self):
        rh.reset()

    def tearDown(self):
        rh.reset()

    def test_first_mark_announces_and_repeats_do_not(self):
        """Six display call sites can each fall back inside one scan; the
        operator should be told once, not six times."""
        self.assertTrue(rh.mark_degraded("boom"))
        self.assertFalse(rh.mark_degraded("boom"))
        self.assertFalse(rh.mark_degraded("different"))

    def test_reset_re_arms_the_announcement(self):
        rh.mark_degraded("boom")
        rh.reset()
        self.assertTrue(rh.mark_degraded("boom"))

    def test_render_says_the_rows_are_not_ranked(self):
        rh.mark_degraded("no quotes")
        text = " ".join(rh.render()).lower()
        self.assertIn("not ranked", text)

    def test_render_names_the_cause(self):
        rh.mark_degraded("no quotes")
        self.assertIn("no quotes", " ".join(rh.render()))

    def test_render_is_empty_when_healthy(self):
        self.assertEqual(rh.render(), [])


class TestTheGuardAllowlistShrank(unittest.TestCase):
    """The allowlist is a claim about behaviour. One of its two entries was
    this fallback; removing the sort must remove the entry, or the guard keeps
    permitting a sort that no longer exists and would re-permit a new one."""

    def test_the_screener_no_longer_needs_two_selection_sorts(self):
        import tests.test_ranking_coverage as guard
        self.assertEqual(guard._ALLOWED_SELECTION_SORTS, 1)

    def test_rank_by_verdict_contains_no_score_sort(self):
        import ast
        import inspect
        from src import options_screener as osc
        tree = ast.parse(inspect.getsource(osc))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "rank_by_verdict":
                for call in ast.walk(node):
                    if (isinstance(call, ast.Call)
                            and getattr(call.func, "attr", None) == "sort_values"):
                        self.fail("rank_by_verdict still sorts a frame")
                return
        self.fail("rank_by_verdict not found")


if __name__ == "__main__":
    unittest.main()
