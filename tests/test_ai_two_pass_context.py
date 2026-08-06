"""Tests that the AI pipeline actually receives per-ticker context.

`_run_ai_pipeline` used to rebuild `ticker_contexts` by looking symbols up in
`data_fetching._CHAIN_CACHE`. That cache is keyed by the tuple
`(symbol, min_dte, max_dte)`, so a bare-symbol lookup never hit, the dict was
always empty, and two-pass scoring silently degraded to single-pass — with the
"two-pass" label suppressed by the same emptiness, so nothing ever said so.

The scan already builds this mapping correctly (`ScanResults.ticker_contexts`),
which is what the dashboard reads. These tests pin that the CLI reads it too.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_ai_two_pass_context -v
"""
from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from src import options_screener as osc
# Imported here, not lazily inside a test: `mock.patch.dict("sys.modules", ...)`
# restores the mapping wholesale on exit, so any module first imported *inside*
# that patch gets evicted and re-imported fresh by the next test — handing it a
# different AI_CONFIG object than the one the test patched.
from src import config_ai  # noqa: F401
from src import ranking  # noqa: F401


def _picks() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "quality_score": [0.9, 0.4],
    })


class TwoPassContextTest(unittest.TestCase):
    """The contexts the scan computed must reach score_and_rank unchanged."""

    def setUp(self):
        self._captured = {}

        def _fake_score_and_rank(candidates, ticker_contexts, vix_regime, sector_ctx=None):
            self._captured["contexts"] = ticker_contexts
            return candidates

        self._fake_score_and_rank = _fake_score_and_rank

    def _run(self, ticker_contexts):
        fake_ai_rank = mock.MagicMock()
        fake_ai_rank.score_and_rank = self._fake_score_and_rank
        with mock.patch.dict("sys.modules", {"ai_rank": fake_ai_rank}), \
             mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), \
             mock.patch.object(osc, "print_ranked_table", create=True), \
             mock.patch("src.ranking.print_ranked_table"):
            return osc._run_ai_pipeline(
                _picks(), "Normal", verbose=False,
                ticker_contexts=ticker_contexts,
            )

    def test_supplied_contexts_reach_the_scorer(self):
        ctx = {"AAPL": {"iv_rank": 0.62}, "MSFT": {"iv_rank": 0.31}}
        self._run(ctx)
        self.assertEqual(self._captured["contexts"], ctx)

    def test_contexts_are_not_silently_empty(self):
        """The regression itself: a populated scan must not produce {}."""
        self._run({"AAPL": {"iv_rank": 0.62}, "MSFT": {"iv_rank": 0.31}})
        self.assertTrue(
            self._captured["contexts"],
            "ticker_contexts arrived empty — two-pass scoring is degraded",
        )

    def test_only_candidate_symbols_are_passed(self):
        """Contexts for tickers that produced no picks are not worth tokens."""
        ctx = {"AAPL": {"iv_rank": 0.62}, "MSFT": {"iv_rank": 0.31},
               "NVDA": {"iv_rank": 0.55}}
        self._run(ctx)
        self.assertEqual(set(self._captured["contexts"]), {"AAPL", "MSFT"})

    def test_missing_contexts_degrade_without_raising(self):
        self.assertIsNotNone(self._run({}))
        self.assertEqual(self._captured["contexts"], {})

    def test_default_arg_omitted_entirely(self):
        """The default branch: callers that pass nothing must still work.

        Covered explicitly because a default branch no test exercises is how
        two live NameErrors shipped here before.
        """
        fake_ai_rank = mock.MagicMock()
        fake_ai_rank.score_and_rank = self._fake_score_and_rank
        with mock.patch.dict("sys.modules", {"ai_rank": fake_ai_rank}), \
             mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), \
             mock.patch("src.ranking.print_ranked_table"):
            osc._run_ai_pipeline(_picks(), "Normal", verbose=False)
        self.assertEqual(self._captured["contexts"], {})


class TwoPassDisabledTest(unittest.TestCase):
    """With two-pass off, context must be withheld even when available."""

    def test_disabled_config_sends_no_context(self):
        captured = {}

        def _fake_score_and_rank(candidates, ticker_contexts, vix_regime, sector_ctx=None):
            captured["contexts"] = ticker_contexts
            return candidates

        fake_ai_rank = mock.MagicMock()
        fake_ai_rank.score_and_rank = _fake_score_and_rank

        with mock.patch.dict("sys.modules", {"ai_rank": fake_ai_rank}), \
             mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), \
             mock.patch.dict(config_ai.AI_CONFIG, {"two_pass_enabled": False}), \
             mock.patch("src.ranking.print_ranked_table"):
            osc._run_ai_pipeline(
                _picks(), "Normal", verbose=False,
                ticker_contexts={"AAPL": {"iv_rank": 0.62}},
            )
        self.assertEqual(captured["contexts"], {})


if __name__ == "__main__":
    unittest.main()
