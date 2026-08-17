"""EV must describe the POSITION, not the instrument.

`calculate_metrics` prices every contract as `fair_value - market_price`, which
is the edge to a BUYER. On a Premium Selling board you are the seller, so that
number has the wrong sign — and it is what the `negative_ev` gate, the verdict,
the WORTH grade, the displayed "Net EV" column and the persisted `entry_ev_net`
all read.

Measured live on 2026-08-17: every short put shown had IV BELOW realized vol —
i.e. the option was cheap, good to buy, bad to sell — and the board reported
positive EV for selling it, while refusing 56 of 94 candidates whose EV was
"negative" precisely because they were the RICH ones worth selling.

The codebase already knew. `calculate_scores` negates the value before ranking
it, with the comment "seller's edge = prem_vals - hv_payoff". That flip was
applied only to the RANK, never to the level, so one frame carried two numbers
describing opposite positions.

That flip is also not quite right on its own terms: negating the NET
(`-(gross - cost)`) negates the cost too, which turns a cost the seller pays
into a credit they earn. Costs never flip:

    buyer:  net =  gross - cost
    seller: net = -gross - cost
"""
from __future__ import annotations

import unittest

import pandas as pd

from src.trade_analysis import position_ev_per_contract


class TestTheArithmetic(unittest.TestCase):

    def test_a_buyer_keeps_the_instrument_edge(self):
        gross, net = position_ev_per_contract(50.0, 8.0, is_short=False)
        self.assertAlmostEqual(gross, 50.0)
        self.assertAlmostEqual(net, 42.0)

    def test_a_seller_earns_the_opposite_edge(self):
        gross, net = position_ev_per_contract(50.0, 8.0, is_short=True)
        self.assertAlmostEqual(gross, -50.0)

    def test_the_seller_still_pays_the_cost(self):
        """The whole point. `-(gross - cost)` would be -42, which quietly pays
        the seller the spread instead of charging it."""
        _gross, net = position_ev_per_contract(50.0, 8.0, is_short=True)
        self.assertAlmostEqual(net, -58.0)
        self.assertNotAlmostEqual(net, -42.0,
                                  msg="cost was flipped along with the edge")

    def test_selling_a_rich_option_is_positive(self):
        """Instrument overpriced (fair < market) => buyer's edge negative =>
        the seller's is positive, less the cost."""
        gross, net = position_ev_per_contract(-60.0, 10.0, is_short=True)
        self.assertAlmostEqual(gross, 60.0)
        self.assertAlmostEqual(net, 50.0)

    def test_selling_a_cheap_option_is_negative(self):
        """The live case: IV below realized vol on every row shown."""
        _gross, net = position_ev_per_contract(24.2, 4.0, is_short=True)
        self.assertLess(net, 0.0)

    def test_cost_is_never_negative_for_either_side(self):
        _g1, buyer = position_ev_per_contract(0.0, 12.0, is_short=False)
        _g2, seller = position_ev_per_contract(0.0, 12.0, is_short=True)
        self.assertAlmostEqual(buyer, -12.0)
        self.assertAlmostEqual(seller, -12.0)

    def test_absent_stays_absent(self):
        self.assertEqual(position_ev_per_contract(None, 5.0, is_short=True),
                         (None, None))
        self.assertEqual(position_ev_per_contract(50.0, None, is_short=True),
                         (None, None))

    def test_nan_is_absent_not_zero(self):
        self.assertEqual(
            position_ev_per_contract(float("nan"), 5.0, is_short=True),
            (None, None))


class TestTheFrameCarriesThePositionsEv(unittest.TestCase):
    """Through the real `enrich_and_score` pipeline, on the shared chain
    fixture, with realized vol dialled either side of implied."""

    def _chain(self, n, hv):
        """A minimal options chain. Self-contained on purpose: the equivalent
        fixture in `test_scoring` lives in a pytest-only module, which the
        canonical unittest runner skips."""
        from datetime import datetime, timedelta
        exp = (datetime.today() + timedelta(days=30)).strftime("%Y-%m-%d")
        return pd.DataFrame({
            "symbol": ["AAPL"] * n,
            # All puts, all OTM: `Premium Selling` keeps only puts, and an
            # ATM/ITM strike is filtered before scoring. This is the shape a
            # short-put board actually has.
            "type": ["put"] * n,
            "strike": [150.0 - i * 5 for i in range(n)],
            "expiration": [exp] * n,
            "impliedVolatility": [0.25 + i * 0.02 for i in range(n)],
            "volume": [100 + i * 50 for i in range(n)],
            "openInterest": [500 + i * 100 for i in range(n)],
            "bid": [2.0 + i * 0.5 for i in range(n)],
            "ask": [2.2 + i * 0.5 for i in range(n)],
            "lastPrice": [2.1 + i * 0.5 for i in range(n)],
            "underlying": [155.0] * n,
            "hv_30d": [hv] * n, "hv_252d": [hv] * n,
            "sentiment_score": [0.0] * n,
            "sma_20": [150.0] * n, "sma_50": [148.0] * n,
            "ret_5d": [0.01] * n, "rsi_14": [55.0] * n,
            "atr_trend": [1.5] * n, "high_20": [160.0] * n,
            "low_20": [145.0] * n, "rvol": [1.0] * n,
            "is_squeezing": [False] * n, "short_interest": [0.05] * n,
            "seasonal_win_rate": [0.5] * n, "vwap": [154.0] * n,
            "fib_50": [152.0] * n, "fib_618": [153.0] * n,
            "iv_rank_30": [0.5] * n, "iv_percentile_30": [0.5] * n,
            "iv_rank_90": [0.5] * n, "iv_percentile_90": [0.5] * n,
            "iv_confidence": ["Normal"] * n,
        })

    CONFIG = {
        "filters": {"min_volume": 10, "min_open_interest": 10,
                    "delta_min": 0.05, "delta_max": 0.95,
                    "max_bid_ask_spread_pct": 0.50, "min_iv_percentile": 0},
        "composite_weights": {"pop_weight": 0.30, "ev_weight": 0.20,
                              "iv_rank_weight": 0.15, "spread_weight": 0.10,
                              "trend_weight": 0.10, "hv_iv_weight": 0.15},
        "min_pop": 0.40, "max_delta": 0.50, "iv_outlier_threshold": 0.50,
        "iv_outlier_min_volume": 5, "moneyness_band": 0.30,
    }

    def _scored(self, mode, hv):
        from src.options_screener import enrich_and_score
        df = self._chain(4, hv)
        cfg = self.CONFIG
        return enrich_and_score(
            df=df, min_dte=1, max_dte=90, risk_free_rate=0.05, config=cfg,
            vix_regime_weights=cfg.get("composite_weights", {}),
            trader_profile="swing", mode=mode, iv_rank=0.5, iv_percentile=0.5,
            earnings_date=None, sentiment_score=0.0, seasonal_win_rate=None,
            term_structure_spread=None, macro_risk_active=False,
            sector_perf={}, tnx_change_pct=0.0, short_interest=None,
            next_ex_div=None, earnings_move_data=None, hv_ewma=None,
            news_data=None)

    # Realized vol is set well inside `implausible_vol_gap`'s [0.55, 1.80]
    # band. Note the pipeline RE-SOLVES implied vol from each mid price
    # (`cross_validate_iv`), so the fixture cannot dictate which contracts end
    # up rich or cheap. These tests therefore assert the INVARIANTS, which hold
    # whatever IV the pipeline settles on, rather than absolute signs that
    # would depend on fighting a filter.
    HV = 0.40

    def _rows(self, mode):
        out = self._scored(mode, self.HV)
        cols = ["ev_gross_instrument_per_contract", "ev_gross_per_contract",
                "ev_cost_per_contract", "ev_per_contract"]
        for c in cols:
            self.assertIn(c, out.columns)
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.dropna(subset=cols)
        # An empty frame makes every `.all()` below vacuously true — which is
        # exactly how the first draft of this file "passed".
        self.assertGreater(len(out), 0, f"{mode} scored nothing to assert on")
        return out

    def test_a_short_position_earns_the_opposite_of_the_instrument_edge(self):
        r = self._rows("Premium Selling")
        self.assertTrue(
            ((r["ev_gross_per_contract"] + r["ev_gross_instrument_per_contract"]).abs()
             < 1e-6).all(),
            "position gross is not the negated instrument edge")

    def test_a_short_position_still_pays_the_cost(self):
        """net == gross - cost, never gross + cost. This is what makes the
        seller's EV different from a simple negation of the buyer's."""
        r = self._rows("Premium Selling")
        self.assertTrue(
            ((r["ev_gross_per_contract"] - r["ev_cost_per_contract"]
              - r["ev_per_contract"]).abs() < 1e-6).all())

    def test_an_overpriced_option_is_worth_selling_before_costs(self):
        """The semantic, on whichever contracts the pipeline judged rich."""
        r = self._rows("Premium Selling")
        rich = r[r["ev_gross_instrument_per_contract"] < 0]
        if rich.empty:
            self.skipTest("no overpriced contract in this fixture run")
        self.assertTrue((rich["ev_gross_per_contract"] > 0).all(),
                        "selling an overpriced option reads as negative edge")

    def test_an_underpriced_option_is_not_worth_selling(self):
        r = self._rows("Premium Selling")
        cheap = r[r["ev_gross_instrument_per_contract"] > 0]
        if cheap.empty:
            self.skipTest("no underpriced contract in this fixture run")
        self.assertTrue((cheap["ev_per_contract"] < 0).all(),
                        "selling an underpriced option reads as positive EV")

    def test_a_long_position_keeps_the_instrument_edge(self):
        """The long side was always correct and must not move."""
        r = self._rows("Lottery Ticket")
        self.assertTrue(
            ((r["ev_gross_per_contract"] - r["ev_gross_instrument_per_contract"]).abs()
             < 1e-6).all())
        self.assertTrue(
            ((r["ev_gross_per_contract"] - r["ev_cost_per_contract"]
              - r["ev_per_contract"]).abs() < 1e-6).all())

    def test_the_two_sides_are_not_mirror_images(self):
        """Both sides are charged the cost, so the nets sum to -2x it rather
        than to zero. A naive `-(gross - cost)` would sum them to zero."""
        short = self._rows("Premium Selling").set_index("strike")
        long_ = self._rows("Lottery Ticket").set_index("strike")
        common = short.index.intersection(long_.index)
        self.assertGreater(len(common), 0)
        total = (short.loc[common, "ev_per_contract"]
                 + long_.loc[common, "ev_per_contract"])
        expected = -(short.loc[common, "ev_cost_per_contract"]
                     + long_.loc[common, "ev_cost_per_contract"])
        # Tolerance scaled to the magnitude: the instrument edge is recomputed
        # per mode and agrees only to floating-point noise on values in the
        # hundreds, so an absolute 1e-6 would fail on arithmetic that is right.
        self.assertTrue(
            ((total - expected).abs() <= 1e-6 * total.abs().clip(lower=1.0)).all(),
            f"nets sum to {list(total)} rather than {list(expected)}")


class TestTheRankNoLongerDoubleFlips(unittest.TestCase):
    """`calculate_scores` negated the level before ranking it. Now that the
    level is already the position's, negating again would restore the bug."""

    def test_no_ad_hoc_negation_survives(self):
        from src.paths import repo_path
        with open(repo_path("src/options_screener.py")) as fh:
            src = fh.read()
        self.assertNotIn("_ev_for_rank = -_ev_for_rank", src,
                         "the rank flips a level that is already the "
                         "position's — shorts are ranked backwards again")


if __name__ == "__main__":
    unittest.main()
