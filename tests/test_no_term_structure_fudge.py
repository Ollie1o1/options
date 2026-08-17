"""A contract's EV must not depend on OTHER expirations' implied vols.

`calculate_metrics` multiplied the EV's vol basis by 1.05 / 0.95 / 1.0
depending on whether an expiration's MEAN IV sat above or below the mean IV of
the whole scan frame. So the same contract, priced in the same market, got a
different fair value depending on which other expirations happened to be
fetched alongside it. That alone is a bad property; the measurement decided it.

`scripts/term_structure_study.py` — pre-registered, SPY optionsDX 2010-2023,
144 NON-OVERLAPPING 21-day windows, signal from the chain at t, forecast from
prices up to t, no lookahead:

  H1 THE SHIPPED RULE — REJECTED
    fires on 89.6% of windows, and fires "down" on 120 of 144 — nearly a
    constant. Accuracy 62.0% against a 52% break-even looks like skill, but a
    rule that ALWAYS said "down" scores 63.9% on this sample, so the rule is
    WORSE than the trivial constant and its apparent edge is a base-rate
    artifact. RMSE 0.0904 -> 0.0902: -0.02%, nothing.

  H2 A PROPER ATM SLOPE — real signal, OPPOSITE sign
    Spearman(far-minus-near ATM IV, forecast error) = -0.2460, and it survives
    a split it was not chosen on (early -0.3111, late -0.2404). Contango means
    realized vol comes in BELOW the forecast, so the estimate should be
    LOWERED. The shipped rule RAISED it. Nudging in the shipped direction
    costs 3.04% of RMSE.

The adjustment is therefore deleted rather than re-tuned. The slope finding is
recorded as a lead for a properly pre-registered follow-up, not smuggled in
here with another hand-set constant — this repo already paid for those
([[project_adjustment_stack_carries_the_negative]], IC -0.096).
"""
from __future__ import annotations

import unittest

import pandas as pd


def _chain(rows):
    """A minimal frame with two expirations at controllable IV levels."""
    from datetime import datetime, timedelta
    base = datetime.today()
    out = []
    for dte, iv, strike in rows:
        out.append({
            "symbol": "AAPL", "type": "call", "strike": strike,
            "expiration": (base + timedelta(days=dte)).strftime("%Y-%m-%d"),
            "impliedVolatility": iv, "volume": 500, "openInterest": 2000,
            "bid": 4.90, "ask": 5.10, "lastPrice": 5.00, "underlying": 150.0,
            "hv_30d": 0.25, "hv_252d": 0.25, "sentiment_score": 0.0,
            "sma_20": 150.0, "sma_50": 148.0, "ret_5d": 0.01, "rsi_14": 55.0,
            "atr_trend": 1.5, "high_20": 160.0, "low_20": 145.0, "rvol": 1.0,
            "is_squeezing": False, "short_interest": 0.05,
            "seasonal_win_rate": 0.5, "vwap": 149.0, "fib_50": 152.0,
            "fib_618": 153.0, "iv_rank_30": 0.5, "iv_percentile_30": 0.5,
            "iv_rank_90": 0.5, "iv_percentile_90": 0.5, "iv_confidence": "Normal",
        })
    return pd.DataFrame(out)


CONFIG = {
    "filters": {"min_volume": 10, "min_open_interest": 10, "delta_min": 0.05,
                "delta_max": 0.95, "max_bid_ask_spread_pct": 0.50,
                "min_iv_percentile": 0},
    "composite_weights": {"pop_weight": 0.30, "ev_weight": 0.20,
                          "iv_rank_weight": 0.15, "spread_weight": 0.10,
                          "trend_weight": 0.10, "hv_iv_weight": 0.15},
    "min_pop": 0.40, "max_delta": 0.50, "iv_outlier_threshold": 0.50,
    "iv_outlier_min_volume": 5, "moneyness_band": 0.30,
}


def _target_expiration(dte=30):
    from datetime import datetime, timedelta
    return (datetime.today() + timedelta(days=dte)).strftime("%Y-%m-%d")


def _ev_of_target(rows, dte=30):
    """Net EV of the `dte`-day contract, whatever else is in the frame.

    Selected by EXPIRATION, not strike: the fixture deliberately puts the
    target and its neighbour at the SAME strike, so a strike-only lookup can
    silently read the neighbour's row instead and compare two different
    contracts.

    Driven through `enrich_and_score` rather than `calculate_metrics` directly:
    the quote columns (`premium`, `mid`, `spread_pct`) are assembled by the
    former and are a precondition of the latter.
    """
    from datetime import datetime, timezone
    from src.options_screener import enrich_and_score
    # Pin `as_of`: T_years is otherwise taken from wall-clock, so two calls
    # milliseconds apart price the same contract at fractionally different
    # tenors and the comparison below wobbles in the 7th significant figure.
    # See project_scorer_not_reproducible_20260810.
    out = enrich_and_score(
        as_of=datetime(2026, 8, 17, 16, 0, tzinfo=timezone.utc),
        df=_chain(rows), min_dte=1, max_dte=200, risk_free_rate=0.045,
        config=CONFIG, vix_regime_weights=CONFIG["composite_weights"],
        trader_profile="swing", mode="Discovery scan", iv_rank=0.5,
        iv_percentile=0.5, earnings_date=None, sentiment_score=0.0,
        seasonal_win_rate=None, term_structure_spread=None,
        macro_risk_active=False, sector_perf={}, tnx_change_pct=0.0,
        short_interest=None, next_ex_div=None, earnings_move_data=None,
        hv_ewma=None, news_data=None)
    if out is None or out.empty or "ev_per_contract" not in out.columns:
        return None
    hit = out[out["expiration"].astype(str) == _target_expiration(dte)]
    if hit.empty:
        return None
    v = pd.to_numeric(hit["ev_per_contract"], errors="coerce").dropna()
    return float(v.iloc[0]) if len(v) else None


class TestOneContractsEvIsItsOwn(unittest.TestCase):

    def test_a_far_expirations_iv_does_not_move_this_contracts_ev(self):
        """The property the fudge violated.

        Same 30-DTE contract both times. Only the OTHER expiration's IV
        changes — enough to flip the shipped rule between 1.05 and 0.95.
        """
        low_neighbour = _ev_of_target([(30, 0.30, 150.0), (60, 0.20, 150.0)])
        high_neighbour = _ev_of_target([(30, 0.30, 150.0), (60, 0.60, 150.0)])
        self.assertIsNotNone(low_neighbour)
        self.assertIsNotNone(high_neighbour)
        self.assertAlmostEqual(
            low_neighbour, high_neighbour, places=6,
            msg="this contract's EV changed because a DIFFERENT expiration's "
                "IV changed — the term-structure fudge is back")

    def test_it_holds_when_the_neighbour_is_removed_entirely(self):
        alone = _ev_of_target([(30, 0.30, 150.0)])
        paired = _ev_of_target([(30, 0.30, 150.0), (60, 0.60, 150.0)])
        self.assertIsNotNone(alone)
        self.assertAlmostEqual(alone, paired, places=6)


class TestTheFudgeIsGoneFromTheSource(unittest.TestCase):
    """Structural backstop: the property test above can only fire if the frame
    survives filtering, and a future refactor could change that."""

    def _src(self):
        from src.paths import repo_path
        with open(repo_path("src/options_screener.py")) as fh:
            return fh.read()

    def test_no_ts_signal_multiplier(self):
        self.assertNotIn("ts_signal", self._src())

    def test_hv_is_not_scaled_after_the_basis_is_chosen(self):
        self.assertNotIn("hv_arr = hv_arr *", self._src())


if __name__ == "__main__":
    unittest.main()
