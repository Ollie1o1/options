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
    _dsr_from_trades, run_h1_bull_put, run_h2_bear_call,
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
