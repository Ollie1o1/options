"""Portfolio risk math must respect direction, quantity, and correlation.

2026-07-13 audit findings pinned here:
  - Portfolio Greeks summed raw per-contract values: a short put showed up as
    delta-NEGATIVE (it is delta-positive for the book), and the GEX RISK-OFF
    gate keyed off a sign-scrambled number. Direction must flip the sign.
  - The `quantity` column was ignored everywhere in this module — latent while
    every open row is qty 1.0, wrong the moment a fractional/multi-lot row
    lands (core.sizing.capped_quantity produces fractional quantities).
  - MC VaR drew independent shocks per position, so a book of correlated long
    calls got full diversification credit and VaR was understated exactly when
    it matters. Positions must share a market factor (config var_correlation).

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_portfolio_risk_math -v
"""
from __future__ import annotations

import unittest

from src import portfolio_risk
from src.portfolio_risk import RiskAggregator


class _Stub(RiskAggregator):
    """RiskAggregator with the network stubbed out (same pattern as
    tests/test_portfolio_spot_cache.py)."""

    def __init__(self, trades, spot=100.0, config=None):
        super().__init__(db_path=":memory:", config=config)
        self._trades = trades
        self._spot = spot

    def _load_open_trades(self):
        return list(self._trades)

    def _fetch_spot_uncached(self, ticker):
        return self._spot

    def _get_current_iv(self, ticker, expiration, strike, opt_type):
        return 0.30, "market"


def _trade(ticker="NVDA", expiration="2030-01-18", strike=100.0,
           opt_type="call", strategy_name="Long Call", quantity=1.0,
           entry_price=5.0):
    return {"ticker": ticker, "expiration": expiration, "strike": strike,
            "type": opt_type, "strategy_name": strategy_name,
            "quantity": quantity, "entry_price": entry_price}


class TestGreeksDirection(unittest.TestCase):
    def setUp(self):
        portfolio_risk.reset_spot_cache()

    def test_short_put_book_is_delta_positive(self):
        # A cash-secured short put is a bullish position: the BOOK's delta is
        # positive even though the put contract's delta is negative.
        g = _Stub([_trade(opt_type="put",
                          strategy_name="Short Put (cash-secured)")]
                  ).get_portfolio_greeks(rfr=0.04)
        self.assertGreater(g["portfolio_delta"], 0.0)

    def test_long_call_positive_short_call_negative_delta(self):
        long_g = _Stub([_trade(strategy_name="Long Call")]
                       ).get_portfolio_greeks(rfr=0.04)
        portfolio_risk.reset_spot_cache()
        short_g = _Stub([_trade(strategy_name="Covered Call (sell)")]
                        ).get_portfolio_greeks(rfr=0.04)
        self.assertGreater(long_g["portfolio_delta"], 0.0)
        self.assertLess(short_g["portfolio_delta"], 0.0)
        self.assertAlmostEqual(long_g["portfolio_delta"],
                               -short_g["portfolio_delta"], places=10)

    def test_short_position_has_negative_gamma_and_vega(self):
        g = _Stub([_trade(opt_type="put", strategy_name="Short Put")]
                  ).get_portfolio_greeks(rfr=0.04)
        self.assertLess(g["portfolio_gamma"], 0.0)
        self.assertLess(g["portfolio_vega"], 0.0)

    def test_gex_follows_direction_not_option_type(self):
        # The book's own gamma dollars: long anything = +gamma exposure,
        # short anything = -gamma exposure. A long put is NOT negative GEX.
        long_put = _Stub([_trade(opt_type="put", strategy_name="Long Put")]
                         ).get_portfolio_greeks(rfr=0.04)
        portfolio_risk.reset_spot_cache()
        short_call = _Stub([_trade(strategy_name="Short Call (naked)")]
                           ).get_portfolio_greeks(rfr=0.04)
        self.assertGreater(long_put["portfolio_gex"], 0.0)
        self.assertLess(short_call["portfolio_gex"], 0.0)


class TestGreeksQuantity(unittest.TestCase):
    def setUp(self):
        portfolio_risk.reset_spot_cache()

    def test_quantity_scales_every_greek(self):
        g1 = _Stub([_trade(quantity=1.0)]).get_portfolio_greeks(rfr=0.04)
        portfolio_risk.reset_spot_cache()
        g3 = _Stub([_trade(quantity=3.0)]).get_portfolio_greeks(rfr=0.04)
        for key in ("portfolio_delta", "portfolio_gamma",
                    "portfolio_vega", "portfolio_gex"):
            self.assertAlmostEqual(g3[key], 3.0 * g1[key], places=8,
                                   msg=f"{key} must scale with quantity")

    def test_missing_or_zero_quantity_defaults_to_one(self):
        t = _trade()
        t["quantity"] = None
        g_none = _Stub([t]).get_portfolio_greeks(rfr=0.04)
        portfolio_risk.reset_spot_cache()
        g_one = _Stub([_trade(quantity=1.0)]).get_portfolio_greeks(rfr=0.04)
        self.assertAlmostEqual(g_none["portfolio_delta"],
                               g_one["portfolio_delta"], places=10)

    def test_fractional_quantity_supported(self):
        g1 = _Stub([_trade(quantity=1.0)]).get_portfolio_greeks(rfr=0.04)
        portfolio_risk.reset_spot_cache()
        gh = _Stub([_trade(quantity=0.5)]).get_portfolio_greeks(rfr=0.04)
        self.assertAlmostEqual(gh["portfolio_delta"],
                               0.5 * g1["portfolio_delta"], places=10)


class TestMemoSignature(unittest.TestCase):
    """The priced-positions memo is keyed on the open-book signature; quantity
    and direction now affect the priced values, so they must key the memo too
    (same contract, different qty/side must NOT serve each other's frame)."""

    def setUp(self):
        portfolio_risk.reset_spot_cache()

    def test_quantity_change_invalidates_memo(self):
        g1 = _Stub([_trade(quantity=1.0)]).get_portfolio_greeks(rfr=0.04)
        # No reset: a qty-2 book must not be served the qty-1 frame.
        g2 = _Stub([_trade(quantity=2.0)]).get_portfolio_greeks(rfr=0.04)
        self.assertAlmostEqual(g2["portfolio_delta"],
                               2.0 * g1["portfolio_delta"], places=8)

    def test_direction_change_invalidates_memo(self):
        g_long = _Stub([_trade(strategy_name="Long Call")]
                       ).get_portfolio_greeks(rfr=0.04)
        g_short = _Stub([_trade(strategy_name="Short Call (sell)")]
                        ).get_portfolio_greeks(rfr=0.04)
        self.assertLess(g_short["portfolio_delta"], 0.0)
        self.assertGreater(g_long["portfolio_delta"], 0.0)


class TestVarQuantityAndDirection(unittest.TestCase):
    def setUp(self):
        portfolio_risk.reset_spot_cache()

    def _var(self, trades, config=None):
        return _Stub(trades, config=config).calculate_portfolio_var(
            n_simulations=2000, rfr=0.04)

    def test_var_scales_with_quantity(self):
        v1 = self._var([_trade(quantity=1.0)])
        portfolio_risk.reset_spot_cache()
        v2 = self._var([_trade(quantity=2.0)])
        self.assertAlmostEqual(v2["var_95"], 2.0 * v1["var_95"], places=6)

    def test_var_short_position_loses_when_price_rises(self):
        # Deterministic seed: a long call and its short mirror have exactly
        # opposite P&L distributions, so mean P&L must be opposite.
        v_long = self._var([_trade(strategy_name="Long Call")])
        portfolio_risk.reset_spot_cache()
        v_short = self._var([_trade(strategy_name="Short Call (sell)")])
        self.assertAlmostEqual(v_long["mean_pnl"], -v_short["mean_pnl"],
                               places=6)


class TestVarCorrelation(unittest.TestCase):
    """Positions must share a market factor: with var_correlation=1 two
    identical positions are exactly one doubled position; with 0 they get
    diversification credit; higher correlation may never lower VaR."""

    def setUp(self):
        portfolio_risk.reset_spot_cache()

    def _var(self, trades, rho):
        agg = _Stub(trades, config={"var_correlation": rho})
        return agg.calculate_portfolio_var(n_simulations=2000, rfr=0.04)

    def test_full_correlation_doubles_single_position_var(self):
        one = self._var([_trade(ticker="NVDA")], rho=1.0)
        portfolio_risk.reset_spot_cache()
        two = self._var([_trade(ticker="NVDA"), _trade(ticker="AAPL")],
                        rho=1.0)
        self.assertAlmostEqual(two["var_95"], 2.0 * one["var_95"], places=4)

    def test_zero_correlation_gives_diversification_credit(self):
        indep = self._var([_trade(ticker="NVDA"), _trade(ticker="AAPL")],
                          rho=0.0)
        portfolio_risk.reset_spot_cache()
        full = self._var([_trade(ticker="NVDA"), _trade(ticker="AAPL")],
                         rho=1.0)
        self.assertLess(indep["var_95"], full["var_95"])

    def test_default_correlation_is_conservative_not_independent(self):
        # No config: the default must NOT be the old independent-draws model.
        agg = _Stub([_trade(ticker="NVDA"), _trade(ticker="AAPL")])
        default = agg.calculate_portfolio_var(n_simulations=2000, rfr=0.04)
        portfolio_risk.reset_spot_cache()
        indep = self._var([_trade(ticker="NVDA"), _trade(ticker="AAPL")],
                          rho=0.0)
        self.assertGreater(default["var_95"], indep["var_95"])


if __name__ == "__main__":
    unittest.main()
