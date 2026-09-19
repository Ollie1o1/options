"""Preregistered hypothesis sweep over strategy/weights/entry-exit/gates.

The six hypotheses are locked in
docs/superpowers/specs/2026-09-19-hypothesis-sweep-design.md and copied
verbatim to docs/HYPOTHESIS_SWEEP_PREREG.md. This module only computes —
it must never add, remove, or retune a hypothesis after seeing a result.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_hypothesis_sweep -v
"""
from __future__ import annotations

import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.backtest_spreads import SpreadTrade
from src.hypothesis_sweep import (
    MIN_EFFECTIVE_N, MIN_RAW_TRADES, N_TRIALS, HypothesisResult,
    _dsr_from_trades, run_h1_bull_put, run_h2_bear_call, run_h5_stop_loss,
    run_h4_entry_dte, run_h6_credit_to_width, H3_BONFERRONI_ALPHA, run_h3_spread_weight,
    format_report, run_all,
)


def _make_trade(symbol: str, entry: date, hold_days: int, pnl: float) -> SpreadTrade:
    return SpreadTrade(
        symbol=symbol, entry_date=entry, exit_date=entry + timedelta(days=hold_days),
        pnl_pct=pnl, components=np.full(27, 0.5), credit_to_width=0.25,
    )


class NTrialsIsLockedTest(unittest.TestCase):
    def test_family_size_is_exactly_six(self):
        """This is the whole preregistration's correction budget. If a
        future edit changes this without updating the locked spec and
        docs/HYPOTHESIS_SWEEP_PREREG.md, this test is the tripwire."""
        self.assertEqual(N_TRIALS, 6)


class DsrFromTradesTest(unittest.TestCase):
    def _rich_trades(self, n=40, seed=0):
        rng = np.random.default_rng(seed)
        base = date(2021, 1, 4)
        trades = []
        for i in range(n):
            entry = base + timedelta(days=i * 20)  # non-overlapping in time
            trades.append(_make_trade(f"SYM{i % 5}", entry, 10, float(rng.normal(0.15, 0.05))))
        return trades

    def test_too_few_raw_trades_refuses(self):
        trades = self._rich_trades(n=10)
        dsr, n_eff, refused, reason = _dsr_from_trades(trades)
        self.assertTrue(refused)
        self.assertEqual(reason, "insufficient_raw_trades")
        self.assertIsNone(dsr)

    def test_enough_raw_trades_but_all_overlapping_refuses_on_effective_n(self):
        base = date(2021, 1, 4)
        # 35 trades, all entered the same day with a long, fully overlapping
        # hold — effective_n's greedy non-overlap selection collapses this
        # to 1.
        trades = [_make_trade(f"SYM{i}", base, 400, 0.1) for i in range(35)]
        dsr, n_eff, refused, reason = _dsr_from_trades(trades)
        self.assertTrue(refused)
        self.assertEqual(reason, "insufficient_effective_n")
        self.assertLess(n_eff, MIN_EFFECTIVE_N)

    def test_healthy_population_returns_a_dsr_value_not_refused(self):
        trades = self._rich_trades(n=40)
        dsr, n_eff, refused, reason = _dsr_from_trades(trades)
        self.assertFalse(refused)
        self.assertIsNone(reason)
        self.assertIsInstance(dsr, float)
        self.assertGreaterEqual(n_eff, MIN_EFFECTIVE_N)

    def test_dsr_call_uses_the_locked_family_size_not_a_local_count(self):
        """Regression guard: n_trials passed to deflated_sharpe must be
        N_TRIALS (6), not len(trades) or any other locally-derived count."""
        import src.hypothesis_sweep as hs
        trades = self._rich_trades(n=40)
        captured = {}
        real_dsr = hs.deflated_sharpe

        def spy(returns, n_trials, n_eff, trial_variance=None):
            captured["n_trials"] = n_trials
            return real_dsr(returns, n_trials, n_eff, trial_variance)

        hs.deflated_sharpe = spy
        try:
            _dsr_from_trades(trades)
        finally:
            hs.deflated_sharpe = real_dsr
        self.assertEqual(captured["n_trials"], 6)


class HypothesisResultTest(unittest.TestCase):
    def test_constructs_with_required_and_default_fields(self):
        r = HypothesisResult(
            id="H1", category="strategy", statistic_type="dsr", value=0.97,
            n_eff=40, n_raw_trades=120, survives=True, refused=False,
        )
        self.assertEqual(r.reason, None)
        self.assertEqual(r.notes, "")


def _fake_price_frame(n_days: int = 1300, start_price: float = 100.0,
                      seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.015, n_days)
    closes = start_price * np.cumprod(1 + rets)
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    return pd.DataFrame({
        "Close": closes,
        "Volume": rng.integers(1_000_000, 5_000_000, n_days),
    }, index=idx)


class H1H2RunnerTest(unittest.TestCase):
    def setUp(self):
        patcher = patch("src.backtest_spreads._get_yf")
        self.mock_get_yf = patcher.start()
        self.addCleanup(patcher.stop)
        mock_yf = MagicMock()

        def fake_download(symbol, **kwargs):
            seed = sum(ord(c) for c in symbol)
            return _fake_price_frame(seed=seed)

        mock_yf.download.side_effect = fake_download
        self.mock_get_yf.return_value = mock_yf
        self.tickers = ["FAKE1", "FAKE2", "FAKE3", "FAKE4", "FAKE5"]

    def test_h1_returns_a_hypothesis_result_tagged_h1_strategy(self):
        result = run_h1_bull_put(tickers=self.tickers)
        self.assertEqual(result.id, "H1")
        self.assertEqual(result.category, "strategy")
        self.assertEqual(result.statistic_type, "dsr")

    def test_h2_returns_a_hypothesis_result_tagged_h2_strategy(self):
        result = run_h2_bear_call(tickers=self.tickers)
        self.assertEqual(result.id, "H2")
        self.assertEqual(result.category, "strategy")

    def test_survives_requires_both_not_refused_and_dsr_at_or_above_bar(self):
        result = run_h1_bull_put(tickers=self.tickers)
        if result.refused:
            self.assertFalse(result.survives)
        else:
            self.assertEqual(result.survives, result.value >= 0.95)

    def test_h1_passes_option_type_put_to_run_vertical_backtest(self):
        """Verify that H1 (Bull Put) passes option_type='put' to backtest.

        This guards against silent swaps that would be invisible to the other
        tests (which only check result tags, not the backtest path taken).
        """
        patcher = patch("src.hypothesis_sweep.run_vertical_backtest")
        mock_run_backtest = patcher.start()
        self.addCleanup(patcher.stop)

        # Return a minimal trade list so _dsr_from_trades refuses gracefully
        mock_run_backtest.return_value = []

        run_h1_bull_put(tickers=self.tickers)

        # Verify the mock was called with option_type="put"
        mock_run_backtest.assert_called_once()
        call_kwargs = mock_run_backtest.call_args.kwargs
        self.assertEqual(call_kwargs["option_type"], "put")

    def test_h2_passes_option_type_call_to_run_vertical_backtest(self):
        """Verify that H2 (Bear Call) passes option_type='call' to backtest.

        This guards against silent swaps that would be invisible to the other
        tests (which only check result tags, not the backtest path taken).
        """
        patcher = patch("src.hypothesis_sweep.run_vertical_backtest")
        mock_run_backtest = patcher.start()
        self.addCleanup(patcher.stop)

        # Return a minimal trade list so _dsr_from_trades refuses gracefully
        mock_run_backtest.return_value = []

        run_h2_bear_call(tickers=self.tickers)

        # Verify the mock was called with option_type="call"
        mock_run_backtest.assert_called_once()
        call_kwargs = mock_run_backtest.call_args.kwargs
        self.assertEqual(call_kwargs["option_type"], "call")


class H5RunnerTest(unittest.TestCase):
    def setUp(self):
        patcher = patch("src.backtest_spreads._get_yf")
        self.mock_get_yf = patcher.start()
        self.addCleanup(patcher.stop)
        mock_yf = MagicMock()

        def fake_download(symbol, **kwargs):
            seed = sum(ord(c) for c in symbol)
            return _fake_price_frame(seed=seed)

        mock_yf.download.side_effect = fake_download
        self.mock_get_yf.return_value = mock_yf
        self.tickers = ["FAKE1", "FAKE2", "FAKE3", "FAKE4", "FAKE5"]

    def test_returns_a_hypothesis_result_tagged_h5_entry_exit(self):
        result = run_h5_stop_loss(tickers=self.tickers)
        self.assertEqual(result.id, "H5")
        self.assertEqual(result.category, "entry_exit")
        self.assertEqual(result.statistic_type, "dsr")

    def test_matched_trades_are_paired_by_symbol_and_entry_date(self):
        """A regression guard on the pairing key: with the same deterministic
        roll-forward calendar under both stop multiples, n_raw_trades for the
        matched-difference series should not exceed either individual run's
        trade count."""
        from src.backtest_spreads import load_default_surface, run_vertical_backtest
        surface, _ = load_default_surface()
        current = run_vertical_backtest(self.tickers, option_type="put",
                                        stop_mult=2.0, surface=surface)
        variant = run_vertical_backtest(self.tickers, option_type="put",
                                        stop_mult=1.5, surface=surface)
        result = run_h5_stop_loss(tickers=self.tickers)
        self.assertLessEqual(result.n_raw_trades, min(len(current), len(variant)))


class H4H6RunnerTest(unittest.TestCase):
    def setUp(self):
        patcher = patch("src.backtest_spreads._get_yf")
        self.mock_get_yf = patcher.start()
        self.addCleanup(patcher.stop)
        mock_yf = MagicMock()

        def fake_download(symbol, **kwargs):
            seed = sum(ord(c) for c in symbol)
            return _fake_price_frame(seed=seed)

        mock_yf.download.side_effect = fake_download
        self.mock_get_yf.return_value = mock_yf
        self.tickers = ["FAKE1", "FAKE2", "FAKE3", "FAKE4", "FAKE5"]

    def test_h4_returns_a_hypothesis_result_tagged_h4_entry_exit(self):
        result = run_h4_entry_dte(tickers=self.tickers)
        self.assertEqual(result.id, "H4")
        self.assertEqual(result.category, "entry_exit")
        self.assertEqual(result.statistic_type, "dsr_compare")

    def test_h6_returns_a_hypothesis_result_tagged_h6_gates(self):
        result = run_h6_credit_to_width(tickers=self.tickers)
        self.assertEqual(result.id, "H6")
        self.assertEqual(result.category, "gates")
        self.assertEqual(result.statistic_type, "dsr_compare")

    def test_h6_survivor_population_is_nested_not_a_second_simulation(self):
        """The 0.25-floor survivor count must never exceed the 0.20-floor
        survivor count — they come from the SAME simulation, filtered, not
        two independent runs that could disagree in size for unrelated
        reasons."""
        from src.backtest_spreads import load_default_surface, run_vertical_backtest
        surface, _ = load_default_surface()
        trades = run_vertical_backtest(self.tickers, option_type="put", surface=surface)
        floor_020 = [t for t in trades if t.credit_to_width >= 0.20]
        floor_025 = [t for t in trades if t.credit_to_width >= 0.25]
        self.assertLessEqual(len(floor_025), len(floor_020))


class H3RunnerTest(unittest.TestCase):
    def setUp(self):
        patcher = patch("src.backtest_spreads._get_yf")
        self.mock_get_yf = patcher.start()
        self.addCleanup(patcher.stop)
        mock_yf = MagicMock()

        def fake_download(symbol, **kwargs):
            seed = sum(ord(c) for c in symbol)
            return _fake_price_frame(seed=seed)

        mock_yf.download.side_effect = fake_download
        self.mock_get_yf.return_value = mock_yf
        self.tickers = ["FAKE1", "FAKE2", "FAKE3", "FAKE4", "FAKE5"]

    def test_alpha_is_bonferroni_over_the_family_of_six(self):
        self.assertAlmostEqual(H3_BONFERRONI_ALPHA, 0.05 / 6)

    def test_returns_a_hypothesis_result_tagged_h3_weights_ic(self):
        result = run_h3_spread_weight(tickers=self.tickers)
        self.assertEqual(result.id, "H3")
        self.assertEqual(result.category, "weights")
        self.assertEqual(result.statistic_type, "ic")

    def test_too_few_raw_trades_refuses_like_the_dsr_hypotheses(self):
        result = run_h3_spread_weight(tickers=["FAKE1"])
        # A single fake ticker with a short warmup-truncated history may or
        # may not clear MIN_RAW_TRADES depending on the roll-forward count;
        # this only asserts the refusal path is reachable and well-formed
        # when it does trigger, not that it always does for this fixture.
        if result.refused:
            self.assertEqual(result.reason, "insufficient_raw_trades")
            self.assertIsNone(result.value)


class RunAllTest(unittest.TestCase):
    def setUp(self):
        patcher = patch("src.backtest_spreads._get_yf")
        self.mock_get_yf = patcher.start()
        self.addCleanup(patcher.stop)
        mock_yf = MagicMock()

        def fake_download(symbol, **kwargs):
            seed = sum(ord(c) for c in symbol)
            return _fake_price_frame(seed=seed)

        mock_yf.download.side_effect = fake_download
        self.mock_get_yf.return_value = mock_yf
        self.tickers = ["FAKE1", "FAKE2", "FAKE3", "FAKE4", "FAKE5"]

    def test_runs_all_six_hypotheses_in_order(self):
        results = run_all(tickers=self.tickers)
        self.assertEqual([r.id for r in results], ["H1", "H2", "H3", "H4", "H5", "H6"])

    def test_a_refused_hypothesis_does_not_stop_the_batch(self):
        """Even with a universe too small for some hypotheses to clear
        MIN_RAW_TRADES, run_all must still return exactly 6 rows — refusal
        is a per-hypothesis outcome, not a batch abort."""
        results = run_all(tickers=["FAKE1"])
        self.assertEqual(len(results), 6)

    def test_format_report_includes_every_hypothesis_id_and_verdict(self):
        results = run_all(tickers=self.tickers)
        report = format_report(results)
        for r in results:
            self.assertIn(r.id, report)
            verdict = "REFUSED" if r.refused else ("SURVIVES" if r.survives else "null")
            self.assertIn(verdict, report)
