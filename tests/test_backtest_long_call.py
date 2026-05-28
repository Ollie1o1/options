"""Tests for the long-call extension to src/backtest_optimizer.

The existing optimizer simulates short puts only. The auto-log on the live
screener is dominated by long calls (the `auto_log_skip_long_puts` filter),
so the optimizer's outputs were never strategy-matched to the actual book.
These tests pin the behavior of the new long-call simulator and the
strategy-aware scoring switch.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_backtest_long_call -v
"""
from __future__ import annotations

import math
import unittest

import numpy as np


class LongCallSimulator(unittest.TestCase):
    """`simulate_long_call_pnl` is the buyer-side mirror of `simulate_pnl`.

    Buy the call at `entry_price`. Future closes determine the call value
    via BS pricing. Exit on:
      - take profit (call value >= entry_price * (1 + take_profit_mult))
      - stop loss  (call value <= entry_price * (1 - stop_loss_mult))
      - time exit  (dte <= min_dte)
      - last bar of the simulation window
    Returns pnl_pct as (exit_proceeds - entry_cost) / entry_cost.
    """

    def _bs_call_price(self, S, K, T, r, sigma):
        """Inline BS call for test math (avoids depending on internals)."""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return max(S - K, 0.0)
        from math import erfc, log, sqrt, exp
        def N(x): return 0.5 * erfc(-x / sqrt(2))
        sq = sigma * sqrt(T)
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / sq
        d2 = d1 - sq
        return S * N(d1) - K * exp(-r * T) * N(d2)

    def test_flat_future_returns_negative_pnl_from_theta_decay(self):
        """When the underlying doesn't move, a long call loses to theta."""
        from src.backtest_optimizer import simulate_long_call_pnl
        S0, K, sigma, r, dte = 100.0, 105.0, 0.30, 0.05, 45
        entry = self._bs_call_price(S0, K, dte / 365.0, r, sigma)
        future = np.full(dte, S0)  # underlying never moves
        pnl = simulate_long_call_pnl(entry, future, K, sigma, r, dte)
        self.assertLess(pnl, 0.0, f"expected negative P&L from theta, got {pnl:+.3f}")
        # Should not be a total loss when underlying held flat
        self.assertGreater(pnl, -1.0)

    def test_take_profit_fires_on_strong_rally(self):
        """If the call doubles in value, take-profit at +100% should fire."""
        from src.backtest_optimizer import simulate_long_call_pnl
        S0, K, sigma, r, dte = 100.0, 105.0, 0.30, 0.05, 45
        entry = self._bs_call_price(S0, K, dte / 365.0, r, sigma)
        # Underlying ramps up sharply over the first 5 bars then holds high
        future = np.concatenate([
            np.linspace(S0, S0 * 1.20, 5),  # +20% rally
            np.full(dte - 5, S0 * 1.20),
        ])
        pnl = simulate_long_call_pnl(entry, future, K, sigma, r, dte,
                                     take_profit_mult=1.0, stop_loss_mult=0.5)
        self.assertGreater(pnl, 0.5,
                          f"expected take-profit (>+50% net of slippage), got {pnl:+.3f}")

    def test_stop_loss_caps_downside(self):
        """If the underlying tanks, the loss should cap near the stop level."""
        from src.backtest_optimizer import simulate_long_call_pnl
        S0, K, sigma, r, dte = 100.0, 105.0, 0.30, 0.05, 45
        entry = self._bs_call_price(S0, K, dte / 365.0, r, sigma)
        # Underlying drops 15% over first 10 bars then stays down
        future = np.concatenate([
            np.linspace(S0, S0 * 0.85, 10),
            np.full(dte - 10, S0 * 0.85),
        ])
        pnl = simulate_long_call_pnl(entry, future, K, sigma, r, dte,
                                     take_profit_mult=1.0, stop_loss_mult=0.5)
        # Should be capped near -50% by stop_loss_mult=0.5 (not -100%)
        self.assertLess(pnl, 0.0)
        self.assertGreater(pnl, -0.7,
                          f"stop loss should cap loss around -50% + slippage, got {pnl:+.3f}")


class StrategyAwareScoring(unittest.TestCase):
    """`compute_component_scores(mode=...)` must flip signs on direction-aware
    factors so long calls aren't scored as if they were short puts."""

    def _baseline_inputs(self):
        return dict(
            S=100.0, K=95.0, T=45/365.0, r=0.05, sigma=0.30,
            hv_pct_rank=0.6, hv_ratio=1.3, rsi_score=0.7, vol_rank=0.5,
        )

    def test_vrp_sign_inverts_between_modes(self):
        """vrp_score: short put benefits from HV>6m-avg (premium-rich); long
        call benefits from HV<6m-avg (premium-cheap). Same input ⇒ opposite
        sides of 0.5."""
        from src.backtest_optimizer import compute_component_scores, WEIGHT_KEYS
        idx = WEIGHT_KEYS.index("vrp")
        kw_high = self._baseline_inputs(); kw_high["hv_ratio"] = 1.5  # elevated
        kw_low  = self._baseline_inputs(); kw_low["hv_ratio"]  = 0.6  # depressed
        sp_high = compute_component_scores(mode="short_put", **kw_high)[idx]
        lc_high = compute_component_scores(mode="long_call", **kw_high)[idx]
        sp_low  = compute_component_scores(mode="short_put", **kw_low)[idx]
        lc_low  = compute_component_scores(mode="long_call", **kw_low)[idx]
        # Sellers love elevated HV; buyers hate it.
        self.assertGreater(sp_high, sp_low, "short_put: high HV ratio should score higher")
        self.assertLess(lc_high, lc_low, "long_call: high HV ratio should score lower")

    def test_momentum_sign_inverts_between_modes(self):
        """High RSI (bullish momentum) should help long calls and hurt short
        puts (a put seller wants the underlying to *not* keep climbing past
        their strike — and historically post-rally puts have less premium)."""
        from src.backtest_optimizer import compute_component_scores, WEIGHT_KEYS
        idx = WEIGHT_KEYS.index("momentum")
        kw_bull = self._baseline_inputs(); kw_bull["rsi_score"] = 0.9
        kw_bear = self._baseline_inputs(); kw_bear["rsi_score"] = 0.2
        sp_bull = compute_component_scores(mode="short_put", **kw_bull)[idx]
        lc_bull = compute_component_scores(mode="long_call", **kw_bull)[idx]
        sp_bear = compute_component_scores(mode="short_put", **kw_bear)[idx]
        lc_bear = compute_component_scores(mode="long_call", **kw_bear)[idx]
        self.assertGreater(lc_bull, lc_bear, "long_call: bullish momentum should score higher")
        self.assertGreater(sp_bear, sp_bull, "short_put: bearish momentum should score higher")

    def test_default_mode_is_short_put_for_backwards_compat(self):
        """Existing callers don't pass `mode=`. They must keep getting the
        short-put scoring they had before."""
        from src.backtest_optimizer import compute_component_scores, WEIGHT_KEYS
        kw = self._baseline_inputs()
        no_mode = compute_component_scores(**kw)
        explicit = compute_component_scores(mode="short_put", **kw)
        np.testing.assert_array_equal(no_mode, explicit)


class ZeroVarianceMasking(unittest.TestCase):
    """The current optimizer's L2 regularizes toward UNIFORM, which forces
    weight onto factors whose score column has zero variance in the dataset
    (catalyst, gamma_theta, pcr, etc. in the synthetic backtest). That's
    nonsensical — a factor with no information cannot have a signed effect.

    The fix: detect zero-variance columns and freeze them at the current
    weight while letting the optimizer move only the live-signal factors.
    """

    def _make_bt(self, n_trades=100):
        from src.backtest_optimizer import BacktestResult, WEIGHT_KEYS
        rng = np.random.default_rng(seed=7)
        scores = rng.uniform(0, 1, size=(n_trades, len(WEIGHT_KEYS)))
        # Zero-variance: catalyst, pcr, gex set to a constant
        for col_name in ("catalyst", "pcr", "gex"):
            scores[:, WEIGHT_KEYS.index(col_name)] = 0.5
        pnl = scores[:, WEIGHT_KEYS.index("vrp")] * 0.3 + rng.normal(0, 0.05, n_trades)
        return BacktestResult(component_scores=scores, pnl_pct=pnl, symbols=["X"] * n_trades)

    def test_zero_variance_factors_keep_their_current_weight(self):
        """After optimization with mask_zero_variance=True, the three constant
        columns should have weights identical to their `current_weights`
        values (renormalized) — they're frozen, not reshuffled by L2."""
        from src.backtest_optimizer import optimize_weights, WEIGHT_KEYS
        bt = self._make_bt()
        current = {k: 1.0 / len(WEIGHT_KEYS) for k in WEIGHT_KEYS}
        current["catalyst"] = 0.01
        current["pcr"] = 0.02
        current["gex"] = 0.03
        # renormalize
        s = sum(current.values()); current = {k: v / s for k, v in current.items()}

        out = optimize_weights(
            bt, method="minimize", n_trials=50, l2_lambda=0.10, verbose=False,
            current_weights=current, mask_zero_variance=True,
        )

        # Each zero-variance factor's *relative* share among zero-variance
        # factors must equal the current relative share. (Absolute share may
        # rescale because the live-signal factors took budget.)
        zero_var_keys = ["catalyst", "pcr", "gex"]
        cur_zv_sum = sum(current[k] for k in zero_var_keys)
        out_zv_sum = sum(out[k] for k in zero_var_keys)
        for k in zero_var_keys:
            cur_share = current[k] / cur_zv_sum
            out_share = out[k] / out_zv_sum if out_zv_sum > 0 else 0
            self.assertAlmostEqual(
                cur_share, out_share, places=3,
                msg=f"{k}: zero-var factor's relative share changed "
                    f"({cur_share:.4f} → {out_share:.4f}) — masking failed",
            )


if __name__ == "__main__":
    unittest.main()
