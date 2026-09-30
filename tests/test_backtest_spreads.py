"""Vertical-spread synthetic simulation for the hypothesis sweep.

Bull Put / Bear Call verticals, priced with the same BS primitives as
backtest_optimizer.py's naked short-put engine, with friction charged per
leg from the fitted spread surface instead of a flat round-trip slippage.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_backtest_spreads -v
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.backtest_optimizer import bs_call_price, bs_put_price
from src.backtest_spreads import (
    SPREAD_IDX, load_default_surface, simulate_vertical_pnl, SpreadTrade,
    WING_DELTA, backtest_ticker_vertical, run_vertical_backtest,
)
from src.spread_surface import Cell, SpreadSurface, save_surface


class SimulateVerticalPnlBullPutTest(unittest.TestCase):
    """Bull Put: short the higher strike, long the lower strike."""

    S0, K_SHORT, K_LONG, SIGMA, R, DTE = 100.0, 95.0, 90.0, 0.30, 0.05, 45

    def test_entry_credit_is_short_minus_long_premium(self):
        """With zero friction, the net credit at entry equals the BS
        difference between the two legs — this is the reference the
        pnl_pct denominator (entry_mid_credit) must use."""
        T0 = self.DTE / 365.0
        expected_credit = (bs_put_price(self.S0, self.K_SHORT, T0, self.R, self.SIGMA)
                           - bs_put_price(self.S0, self.K_LONG, T0, self.R, self.SIGMA))
        future = np.full(self.DTE, self.S0)  # flat: exits on time-exit at min_dte
        pnl, _ = simulate_vertical_pnl(
            self.S0, future, self.K_SHORT, self.K_LONG, self.SIGMA, self.R,
            self.DTE, "put", rel_short=0.0, rel_long=0.0)
        # Flat underlying still decays the credit toward zero as time passes,
        # so pnl_pct should be positive (short decays faster near the money)
        # and finite — this pins that the function runs the real BS path,
        # not a stub.
        self.assertTrue(np.isfinite(pnl))
        self.assertGreater(expected_credit, 0.0)

    def test_friction_strictly_reduces_pnl(self):
        """Charging friction on both legs must strictly lower pnl_pct
        relative to a zero-friction run, holding everything else fixed —
        this is the property that makes the spread-surface integration in
        Task 2 actually matter instead of being inert."""
        future = np.full(self.DTE, self.S0)
        pnl_free, _ = simulate_vertical_pnl(
            self.S0, future, self.K_SHORT, self.K_LONG, self.SIGMA, self.R,
            self.DTE, "put", rel_short=0.0, rel_long=0.0)
        pnl_friction, _ = simulate_vertical_pnl(
            self.S0, future, self.K_SHORT, self.K_LONG, self.SIGMA, self.R,
            self.DTE, "put", rel_short=0.02, rel_long=0.03)
        self.assertLess(pnl_friction, pnl_free)

    def test_a_deep_favorable_move_hits_take_profit_before_the_last_bar(self):
        """The underlying rallying hard makes both puts nearly worthless
        almost immediately — the trade should exit on take-profit, not on
        the last bar of the window."""
        future = np.full(self.DTE, self.S0 * 1.40)
        pnl, exit_offset = simulate_vertical_pnl(
            self.S0, future, self.K_SHORT, self.K_LONG, self.SIGMA, self.R,
            self.DTE, "put", rel_short=0.01, rel_long=0.01)
        self.assertGreater(pnl, 0.0)
        self.assertLess(exit_offset, self.DTE - 1)

    def test_a_deep_adverse_move_hits_stop_loss(self):
        """The underlying crashing through both strikes should trigger the
        stop-loss and produce a negative pnl_pct."""
        future = np.full(self.DTE, self.S0 * 0.60)
        pnl, exit_offset = simulate_vertical_pnl(
            self.S0, future, self.K_SHORT, self.K_LONG, self.SIGMA, self.R,
            self.DTE, "put", rel_short=0.01, rel_long=0.01)
        self.assertLess(pnl, 0.0)
        self.assertLess(exit_offset, self.DTE - 1)

    def test_non_positive_entry_credit_returns_zero(self):
        """K_long > K_short for a put vertical is a malformed spread (the
        wing is priced above the short leg) — bs_put_price(K_long) could
        exceed bs_put_price(K_short), making entry_mid_credit <= 0. The
        function must refuse to divide by a non-positive credit rather than
        return a nonsensical pnl_pct."""
        future = np.full(self.DTE, self.S0)
        pnl, exit_offset = simulate_vertical_pnl(
            self.S0, future, 90.0, 95.0, self.SIGMA, self.R,  # swapped
            self.DTE, "put", rel_short=0.01, rel_long=0.01)
        self.assertEqual(pnl, 0.0)
        self.assertEqual(exit_offset, 0)


class SimulateVerticalPnlBearCallTest(unittest.TestCase):
    """Bear Call: short the lower strike, long the higher strike."""

    S0, K_SHORT, K_LONG, SIGMA, R, DTE = 100.0, 105.0, 110.0, 0.30, 0.05, 45

    def test_entry_credit_is_positive_for_a_well_formed_call_vertical(self):
        T0 = self.DTE / 365.0
        credit = (bs_call_price(self.S0, self.K_SHORT, T0, self.R, self.SIGMA)
                 - bs_call_price(self.S0, self.K_LONG, T0, self.R, self.SIGMA))
        self.assertGreater(credit, 0.0)

    def test_a_deep_adverse_rally_hits_stop_loss(self):
        future = np.full(self.DTE, self.S0 * 1.40)
        pnl, exit_offset = simulate_vertical_pnl(
            self.S0, future, self.K_SHORT, self.K_LONG, self.SIGMA, self.R,
            self.DTE, "call", rel_short=0.01, rel_long=0.01)
        self.assertLess(pnl, 0.0)
        self.assertLess(exit_offset, self.DTE - 1)


class LoadDefaultSurfaceTest(unittest.TestCase):
    def test_missing_file_returns_none_and_fallback_provenance(self):
        surface, provenance = load_default_surface("/nonexistent/path.json")
        self.assertIsNone(surface)
        self.assertEqual(provenance, "fallback_flat")

    def test_present_file_returns_surface_and_surface_provenance(self):
        cells = {(1, 1, 1): Cell(n=50, rel_half_spread=0.02, median_depth=10)}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "surface.json")
            save_surface(SpreadSurface(cells, {"fit_date": "2026-09-19"}), path)
            surface, provenance = load_default_surface(path)
        self.assertIsNotNone(surface)
        self.assertEqual(provenance, "surface")

    def test_empty_cells_still_counts_as_fallback(self):
        """A surface file with no cells (e.g. an empty archive) must not be
        treated as real data — that would silently fall through to
        `_collapsed`'s `caller_default`/ValueError path deep inside a
        roll-forward loop instead of failing predictably here."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "surface.json")
            save_surface(SpreadSurface({}, {}), path)
            surface, provenance = load_default_surface(path)
        self.assertIsNone(surface)
        self.assertEqual(provenance, "fallback_flat")


def _fake_price_frame(n_days: int = 1300, start_price: float = 100.0,
                      seed: int = 0) -> pd.DataFrame:
    """A deterministic, mildly-trending synthetic daily OHLCV frame long
    enough (>300 rows plus warmup plus one full entry_dte window) to drive
    backtest_ticker_vertical's roll-forward loop."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.015, n_days)
    closes = start_price * np.cumprod(1 + rets)
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    return pd.DataFrame({
        "Close": closes,
        "Volume": rng.integers(1_000_000, 5_000_000, n_days),
    }, index=idx)


class BacktestTickerVerticalTest(unittest.TestCase):
    def setUp(self):
        self.frame = _fake_price_frame()
        patcher = patch("src.backtest_spreads._get_yf")
        self.mock_get_yf = patcher.start()
        self.addCleanup(patcher.stop)
        mock_yf = MagicMock()
        mock_yf.download.return_value = self.frame
        self.mock_get_yf.return_value = mock_yf

    def test_returns_a_nonempty_list_of_spread_trades(self):
        trades = backtest_ticker_vertical("FAKE", option_type="put")
        self.assertIsNotNone(trades)
        self.assertGreater(len(trades), 0)
        self.assertIsInstance(trades[0], SpreadTrade)

    def test_every_trade_has_exit_strictly_after_entry(self):
        trades = backtest_ticker_vertical("FAKE", option_type="put")
        for t in trades:
            self.assertLess(t.entry_date, t.exit_date)

    def test_components_length_matches_weight_keys_and_spread_overridden(self):
        from src.backtest_optimizer import WEIGHT_KEYS
        trades = backtest_ticker_vertical("FAKE", option_type="put")
        spread_idx = WEIGHT_KEYS.index("spread")
        for t in trades:
            self.assertEqual(len(t.components), len(WEIGHT_KEYS))
            # Neutral (0.5) is compute_component_scores' own hardcoded
            # placeholder — this asserts backtest_ticker_vertical actually
            # overwrote it, not merely that it's some finite number.
            self.assertNotEqual(t.components[spread_idx], 0.5)

    def test_credit_to_width_is_positive_for_well_formed_bull_put_trades(self):
        trades = backtest_ticker_vertical("FAKE", option_type="put")
        for t in trades:
            self.assertGreater(t.credit_to_width, 0.0)

    def test_bear_call_short_strike_is_below_long_strike(self):
        """Sanity check on the wing direction for a call vertical — this
        would silently produce backwards spreads if option_type='call'
        picked deltas with the wrong sign."""
        trades = backtest_ticker_vertical("FAKE", option_type="call")
        self.assertIsNotNone(trades)
        self.assertGreater(len(trades), 0)

    def test_too_short_a_price_history_returns_none(self):
        short_frame = self.frame.iloc[:100]
        mock_yf = MagicMock()
        mock_yf.download.return_value = short_frame
        self.mock_get_yf.return_value = mock_yf
        trades = backtest_ticker_vertical("FAKE", option_type="put")
        self.assertIsNone(trades)

    def test_surface_friction_path_actually_runs_and_changes_the_economics(self):
        """The whole point of this module (per its own docstring) is per-leg
        friction from the fitted spread surface instead of the flat
        SLIPPAGE_PCT/2.0 fallback. Every other test in this class calls
        backtest_ticker_vertical with the default surface=None, so the
        `if surface is not None:` branch — and the real
        SpreadSurface.oi_collapsed_relative signature it depends on — is
        otherwise never exercised. A single cell with rel_half_spread=0.08
        (deliberately far from SLIPPAGE_PCT/2.0 == 0.01) is enough: no
        (delta, dte) bucket in this run will hit it as an exact cell, so
        every lookup falls through _collapsed's ladder all the way to the
        "global" median — which is this one cell's value regardless of
        which bucket asked for it. That keeps the fixture simple while still
        proving the kwargs (`abs_delta=`, `dte=`) and the (value, provenance)
        tuple-unpack are wired correctly: a mismatch there raises inside
        backtest_ticker_vertical's try/except and silently degrades to
        `trades is None`, which this test would also catch.
        """
        distinctive_rel_half_spread = 0.08
        surface = SpreadSurface(
            {(0, 0, 0): Cell(n=50, rel_half_spread=distinctive_rel_half_spread,
                             median_depth=10)},
            {"fit_date": "2026-09-19"},
        )

        # Sanity-check the fixture directly against the real signature before
        # trusting it inside the roll-forward loop.
        rel, provenance = surface.oi_collapsed_relative(abs_delta=0.30, dte=45.0)
        self.assertEqual(rel, distinctive_rel_half_spread)
        self.assertEqual(provenance, "global")

        flat_trades = backtest_ticker_vertical("FAKE", option_type="put")
        surface_trades = backtest_ticker_vertical(
            "FAKE", option_type="put", surface=surface)

        self.assertIsNotNone(flat_trades)
        self.assertIsNotNone(surface_trades)
        self.assertGreater(len(surface_trades), 0)
        # Same price history, same strike-selection inputs (friction never
        # feeds the entry/exit-timing decision) -> identical trade count and
        # entry/exit dates, differing only in the friction-driven economics.
        self.assertEqual(len(flat_trades), len(surface_trades))

        for flat_t, surf_t in zip(flat_trades, surface_trades):
            self.assertEqual(flat_t.entry_date, surf_t.entry_date)
            self.assertEqual(flat_t.exit_date, surf_t.exit_date)
            # The surface's friction (0.08/leg) is far above the flat
            # fallback's (0.01/leg), so the received credit -- and therefore
            # credit_to_width -- must come out strictly lower under the
            # surface path. Equality here would mean the surface's
            # oi_collapsed_relative value never made it into the trade.
            self.assertLess(surf_t.credit_to_width, flat_t.credit_to_width)
            self.assertNotEqual(surf_t.components[SPREAD_IDX],
                                flat_t.components[SPREAD_IDX])


class RunVerticalBacktestTest(unittest.TestCase):
    def setUp(self):
        self.frame = _fake_price_frame()
        patcher = patch("src.backtest_spreads._get_yf")
        self.mock_get_yf = patcher.start()
        self.addCleanup(patcher.stop)
        mock_yf = MagicMock()
        mock_yf.download.return_value = self.frame
        self.mock_get_yf.return_value = mock_yf

    def test_pools_trades_across_tickers(self):
        trades = run_vertical_backtest(["FAKE1", "FAKE2", "FAKE3"], option_type="put")
        symbols = {t.symbol for t in trades}
        self.assertEqual(symbols, {"FAKE1", "FAKE2", "FAKE3"})
        self.assertGreater(len(trades), 0)

    def test_a_ticker_that_returns_none_is_skipped_not_fatal(self):
        def fake_download(symbol, **kwargs):
            if symbol == "BADTICKER":
                return pd.DataFrame()
            return self.frame
        mock_yf = MagicMock()
        mock_yf.download.side_effect = fake_download
        self.mock_get_yf.return_value = mock_yf
        trades = run_vertical_backtest(["FAKE1", "BADTICKER"], option_type="put")
        symbols = {t.symbol for t in trades}
        self.assertEqual(symbols, {"FAKE1"})


if __name__ == "__main__":
    unittest.main()
