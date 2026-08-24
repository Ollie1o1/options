"""A seller's PoP must be the SELLER's, on every short board.

`calculate_metrics` builds the probability of profit twice and the two halves
disagree about who is holding the position:

    pop_sim       Monte Carlo, `is_short=mode in _short_modes` — where
                  `_short_modes` is {Premium Selling, Credit Spreads, Iron
                  Condor}. Correctly the SELLER's on all three.
    prob_profit   analytical, `calculate_probability_of_profit`, which is
                  documented as and always returns the BUYER's.

They were then averaged, 60/40, and the average of two opposite conventions is
not a probability of anything:

    blend = 0.6*s + 0.4*(1 - s) = 0.4 + 0.2*s

which maps every seller probability s in [0.5, 1.0] onto [0.50, 0.60]. That is
the entire observed range of the shipped number: measured over 909 closed
trades on 2026-08-23, Bull Put spanned 0.53-0.58, Bear Call 0.53-0.60, Iron
Condor 0.30-0.61, and NOTHING anywhere in the book exceeded 0.6139 against a
predicted ceiling of 0.60.

A post-blend flip then fired for Premium Selling alone, giving

    1 - (0.4 + 0.2*s) = 0.6 - 0.2*s

which DECREASES as the true probability rises. That inversion is visible in
the outcomes: Short Put's high-PoP half won 37.0% while its low-PoP half won
61.8%.

The tests below assert the properties, not the constants. A level test can be
satisfied by a new fudge factor; monotonicity cannot.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

AS_OF = datetime(2026, 8, 17, 16, 0, tzinfo=timezone.utc)
UNDERLYING = 150.0

CONFIG = {
    "filters": {"min_volume": 10, "min_open_interest": 10, "delta_min": 0.01,
                "delta_max": 0.99, "max_bid_ask_spread_pct": 0.60,
                "min_iv_percentile": 0},
    "composite_weights": {"pop_weight": 0.30, "ev_weight": 0.20,
                          "iv_rank_weight": 0.15, "spread_weight": 0.10,
                          "trend_weight": 0.10, "hv_iv_weight": 0.15},
    "min_pop": 0.0, "max_delta": 0.99, "iv_outlier_threshold": 0.60,
    "iv_outlier_min_volume": 5, "moneyness_band": 0.40,
}


def _chain(strikes, opt_type="put", dte=30, iv=0.30):
    base = AS_OF.replace(tzinfo=None)
    rows = []
    for k in strikes:
        # Rough intrinsic-plus-time premium; the exact level does not matter,
        # only that it is positive and ordered.
        intrinsic = max(0.0, (k - UNDERLYING) if opt_type == "put"
                        else (UNDERLYING - k))
        mid = max(0.35, intrinsic + 2.0 - abs(k - UNDERLYING) * 0.05)
        rows.append({
            "symbol": "AAPL", "type": opt_type, "strike": float(k),
            "expiration": (base + timedelta(days=dte)).strftime("%Y-%m-%d"),
            "impliedVolatility": iv, "volume": 800, "openInterest": 3000,
            "bid": round(mid - 0.05, 2), "ask": round(mid + 0.05, 2),
            "lastPrice": round(mid, 2), "underlying": UNDERLYING,
            "hv_30d": 0.25, "hv_252d": 0.25, "sentiment_score": 0.0,
            "sma_20": 150.0, "sma_50": 148.0, "ret_5d": 0.01, "rsi_14": 55.0,
            "atr_trend": 1.5, "high_20": 160.0, "low_20": 145.0, "rvol": 1.0,
            "is_squeezing": False, "short_interest": 0.05,
            "seasonal_win_rate": 0.5, "vwap": 149.0, "fib_50": 152.0,
            "fib_618": 153.0, "iv_rank_30": 0.5, "iv_percentile_30": 0.5,
            "iv_rank_90": 0.5, "iv_percentile_90": 0.5,
            "iv_confidence": "Normal",
        })
    return pd.DataFrame(rows)


def _pops(strikes, mode, opt_type="put"):
    """{strike: prob_profit} for one board, as the screener actually builds it."""
    from src.options_screener import enrich_and_score
    out = enrich_and_score(
        as_of=AS_OF, df=_chain(strikes, opt_type=opt_type), min_dte=1,
        max_dte=200, risk_free_rate=0.045, config=CONFIG,
        vix_regime_weights=CONFIG["composite_weights"], trader_profile="swing",
        mode=mode, iv_rank=0.5, iv_percentile=0.5, earnings_date=None,
        sentiment_score=0.0, seasonal_win_rate=None,
        term_structure_spread=None, macro_risk_active=False, sector_perf={},
        tnx_change_pct=0.0, short_interest=None, next_ex_div=None,
        earnings_move_data=None, hv_ewma=None, news_data=None)
    if out is None or out.empty or "prob_profit" not in out.columns:
        return {}
    return {float(r["strike"]): float(r["prob_profit"])
            for _, r in out.iterrows() if pd.notna(r.get("prob_profit"))}


# Strikes on a 150 underlying. A put seller's probability of profit RISES as
# the strike falls. These three are chosen because Premium Selling applies its
# own delta band and drops anything further out — a fixture whose strikes the
# board discards tests nothing.
FAR, MID, NEAR = 140.0, 145.0, 148.0
#: Credit Spreads and Iron Condor keep the far wing, so the ceiling that the
#: old blend could not exceed is only reachable on those boards.
DEEP = 120.0
SELLER_MODES = ("Premium Selling", "Credit Spreads", "Iron Condor")


class TestSellerPopRisesAsTheStrikeGetsSafer(unittest.TestCase):
    """The property. A short put further from the money is more likely to
    expire worthless, so its seller's PoP must be higher — on every board that
    the Monte Carlo already treats as short."""

    def test_every_seller_mode_is_monotone_in_safety(self):
        for mode in SELLER_MODES:
            with self.subTest(mode=mode):
                pops = _pops([FAR, MID, NEAR], mode)
                self.assertEqual(len(pops), 3, f"{mode}: board did not build")
                self.assertGreater(
                    pops[FAR], pops[NEAR],
                    f"{mode}: the SAFER short put reports the LOWER "
                    f"probability of profit — the number is inverted")
                self.assertGreater(pops[FAR], pops[MID])
                self.assertGreater(pops[MID], pops[NEAR])

    def test_a_far_otm_short_put_is_actually_likely_to_win(self):
        """Level check. A 20%-OTM put at 30 DTE and 30 vol expires worthless
        the large majority of the time; anything near 0.5 is the two
        conventions cancelling out."""
        for mode in SELLER_MODES:
            with self.subTest(mode=mode):
                self.assertGreater(_pops([FAR, NEAR], mode)[FAR], 0.70)

    def test_the_shipped_ceiling_is_gone(self):
        """Nothing in 909 closed trades exceeded 0.6139, because the blend
        could not produce more than 0.60."""
        for mode in ("Credit Spreads", "Iron Condor"):
            with self.subTest(mode=mode):
                self.assertGreater(_pops([DEEP, NEAR], mode)[DEEP], 0.62)


class TestBuyerBoardsAreUnchanged(unittest.TestCase):
    """The fix must not flip the boards that were already right. Long Call
    read 0.25-0.32 and Long Put 0.26-0.37 — plausible buyer probabilities."""

    def test_a_call_buyer_is_not_handed_a_sellers_probability(self):
        pops = _pops([140.0, 145.0, 148.0], "Discovery scan", opt_type="call")
        self.assertEqual(len(pops), 3, "buyer board did not build")
        self.assertLess(pops[148.0], 0.5,
                        "a long call reading above 0.5 has been flipped")

    def test_a_buyer_pop_falls_as_the_strike_gets_further_away(self):
        pops = _pops([140.0, 145.0, 148.0], "Discovery scan", opt_type="call")
        self.assertLess(pops[148.0], pops[140.0])


class TestTheTwoHalvesAgree(unittest.TestCase):
    """`pop_sim` and `prob_profit` must describe the same position. When they
    do not, their average is the compressed band this file exists to remove."""

    def test_the_simulated_and_analytical_halves_do_not_oppose_each_other(self):
        from src.options_screener import enrich_and_score
        out = enrich_and_score(
            as_of=AS_OF, df=_chain([FAR, NEAR]), min_dte=1, max_dte=200,
            risk_free_rate=0.045, config=CONFIG,
            vix_regime_weights=CONFIG["composite_weights"],
            trader_profile="swing", mode="Credit Spreads", iv_rank=0.5,
            iv_percentile=0.5, earnings_date=None, sentiment_score=0.0,
            seasonal_win_rate=None, term_structure_spread=None,
            macro_risk_active=False, sector_perf={}, tnx_change_pct=0.0,
            short_interest=None, next_ex_div=None, earnings_move_data=None,
            hv_ewma=None, news_data=None)
        self.assertIsNotNone(out)
        assert out is not None
        if "pop_sim" not in out.columns or out["pop_sim"].isna().all():
            self.skipTest("simulation unavailable in this environment")
        row = out[out["strike"] == FAR].iloc[0]
        sim = float(row["pop_sim"])
        blended = float(row["prob_profit"])
        self.assertLess(
            abs(sim - blended), 0.25,
            f"pop_sim says {sim:.2f} and prob_profit says {blended:.2f} — "
            f"they are describing opposite sides of the same trade")


if __name__ == "__main__":
    unittest.main()
