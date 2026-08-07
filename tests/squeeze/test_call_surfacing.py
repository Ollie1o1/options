"""The squeeze long side must survive the generic delta filter.

The 2026-08-07 live run graded four SETUP names (ONDS 40.5% SI, SOUN 42.4%,
NTLA 41.3%, QUBT 32.4%) and then showed "Best call: —" for every one, because
`call_board` re-ranks only rows that reached `picks`. Squeeze Hunt inherits
Discovery's |delta| 0.15-0.35 band, and the natural squeeze expression is a
near-ATM call at |delta| ~0.5 — so the long side is filtered out by
construction while puts on a name that just ran +20% land inside the band.

Raw chains that day carried ONDS $9C at a 5.6% spread on 26,722 OI and SOUN
$8C at 2.9% on 53,925 OI: tighter than the puts the scan did surface. This is
a filter artifact, not a liquidity fact.
"""
import os
import sys
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd


def _chain(symbol="ONDS", spot=8.69):
    """Synthetic chain with an ATM call that the delta band would drop."""
    exp = (datetime.today() + timedelta(days=21)).strftime("%Y-%m-%d")
    strikes = [7.0, 8.0, 9.0, 10.0, 12.0]
    rows = []
    for kind in ("call", "put"):
        for k in strikes:
            rows.append({
                "symbol": symbol, "type": kind, "strike": k, "expiration": exp,
                "impliedVolatility": 1.14, "volume": 500.0, "openInterest": 20000.0,
                "bid": max(0.05, abs(spot - k) * 0.5 + 0.30),
                "ask": max(0.07, abs(spot - k) * 0.5 + 0.34),
                "lastPrice": max(0.06, abs(spot - k) * 0.5 + 0.32),
                "underlying": spot, "hv_30d": 0.85, "sentiment_score": 0.0,
                "sma_20": 7.9, "sma_50": 7.2, "ret_5d": 0.227, "rsi_14": 68.0,
                "atr_trend": 0.4, "high_20": 9.2, "low_20": 6.1, "rvol": 2.1,
                "is_squeezing": True, "short_interest": 0.405,
                "seasonal_win_rate": 0.5, "vwap": 8.5, "fib_50": 8.0,
                "fib_618": 8.3, "iv_rank_30": 0.6, "iv_percentile_30": 0.6,
                "iv_rank_90": 0.6, "iv_percentile_90": 0.6,
                "iv_confidence": "Normal",
            })
    return pd.DataFrame(rows)


def _config():
    return {
        "filters": {
            "min_volume": 10, "min_open_interest": 10,
            "delta_min": 0.15, "delta_max": 0.35,
            "max_bid_ask_spread_pct": 0.40, "min_iv_percentile": 0,
        },
        "composite_weights": {
            "pop_weight": 0.30, "ev_weight": 0.20, "iv_rank_weight": 0.15,
            "spread_weight": 0.10, "trend_weight": 0.10, "hv_iv_weight": 0.15,
        },
        "min_pop": 0.40, "max_delta": 0.50, "iv_outlier_threshold": 0.50,
        "iv_outlier_min_volume": 5, "moneyness_band": 0.30,
    }


def _score(df, config, mode, squeeze_out=None):
    from src.options_screener import enrich_and_score
    return enrich_and_score(
        squeeze_out=squeeze_out,
        df=df, min_dte=1, max_dte=90, risk_free_rate=0.05, config=config,
        vix_regime_weights=config["composite_weights"], trader_profile="swing",
        mode=mode, iv_rank=0.6, iv_percentile=0.6, earnings_date=None,
        sentiment_score=0.0, seasonal_win_rate=None, term_structure_spread=None,
        macro_risk_active=False, sector_perf={}, tnx_change_pct=0.0,
        short_interest=0.405, next_ex_div=None, earnings_move_data=None,
        hv_ewma=None, news_data=None,
    )


class TestStashTransportIsConcatSafe(unittest.TestCase):
    """A multi-ticker scan concatenates every ticker's scored frame.

    pandas `__finalize__` compares `obj.attrs == attrs` across the frames
    being concatenated, so a DataFrame parked in attrs raises
    "Can only compare identically-labeled DataFrame objects" — which is
    exactly how the first version of this fix crashed a live 15-ticker scan.
    attrs can carry scalars only.
    """

    def test_two_scored_squeeze_frames_concatenate(self):
        import pandas as _pd
        a = _score(_chain("ONDS", 8.69), _config(), "Squeeze Hunt")
        b = _score(_chain("SOUN", 7.52), _config(), "Squeeze Hunt")
        frames = [f for f in (a, b) if not f.empty]
        if len(frames) < 2:
            self.skipTest("fixture produced fewer than two non-empty frames")
        _pd.concat(frames, ignore_index=True)  # must not raise

    def test_attrs_carries_no_dataframe(self):
        scored = _score(_chain(), _config(), "Squeeze Hunt")
        import pandas as _pd
        for key, val in scored.attrs.items():
            self.assertNotIsInstance(
                val, _pd.DataFrame,
                f"attrs[{key!r}] holds a DataFrame — concat will raise")


class TestSqueezeCallSurfacing(unittest.TestCase):
    def test_squeeze_mode_stashes_calls_the_delta_band_drops(self):
        stash = {}
        scored = _score(_chain(), _config(), "Squeeze Hunt", squeeze_out=stash)
        stash = stash.get("calls")
        self.assertIsInstance(stash, pd.DataFrame)
        self.assertFalse(stash.empty, "squeeze long side was not captured")
        self.assertTrue((stash["type"] == "call").all())
        # The point of the stash: it holds calls that picks does not.
        surfaced = set(scored[scored["type"] == "call"]["strike"]) if not scored.empty else set()
        self.assertTrue(set(stash["strike"]) - surfaced,
                        "stash adds nothing over the filtered picks")

    def test_stashed_calls_carry_the_columns_the_board_renders(self):
        out = {}
        _score(_chain(), _config(), "Squeeze Hunt", squeeze_out=out)
        stash = out["calls"]
        for col in ("strike", "expiration", "dte", "delta", "premium",
                    "spread_pct", "quality_score"):
            self.assertIn(col, stash.columns)

    def test_stashed_calls_still_respect_the_liquidity_filters(self):
        # Relaxing delta must not become "show any call" — a 200%-spread
        # contract is not a tradeable long side.
        out = {}
        _score(_chain(), _config(), "Squeeze Hunt", squeeze_out=out)
        self.assertTrue((out["calls"]["spread_pct"] <= 0.40).all())

    def test_other_modes_do_not_pay_for_the_stash(self):
        out = {}
        _score(_chain(), _config(), "Discovery scan", squeeze_out=out)
        self.assertEqual(out, {})


class TestCallBoardConsumesTheStash(unittest.TestCase):
    def test_board_renders_a_call_that_the_ranked_picks_lack(self):
        from src.squeeze.board import call_board
        out = {}
        _score(_chain(), _config(), "Squeeze Hunt", squeeze_out=out)
        text = call_board(out["calls"], "ONDS")
        self.assertIsNotNone(text)
        self.assertIn("ONDS", text)


class TestStashReachesTheDisplay(unittest.TestCase):
    """The stash is useless unless it survives the scan's plumbing."""

    def test_scan_result_carries_a_per_symbol_call_map(self):
        from src.schemas import ScanResult
        r = ScanResult()
        self.assertEqual(r.squeeze_calls, {})

    def test_scoring_step_lifts_the_stash_out_of_attrs(self):
        # attrs survive enrich_and_score, but nothing downstream reads a
        # DataFrame's attrs — the result dict is the transport.
        self.assertTrue('result["squeeze_calls"]' in _read_screener(),
                        "_score_fetched_data does not export the stash")

    def test_run_scan_collects_the_stash_per_symbol(self):
        self.assertTrue("squeeze_calls_map" in _read_screener(),
                        "run_scan does not collect the stash per symbol")

    def test_call_board_is_fed_the_stash_not_the_filtered_picks(self):
        # The regression this whole file exists for: call_board(_sq_rows, ...)
        # re-ranks post-filter rows and finds no calls.
        self.assertFalse("_sq_call_board(_sq_rows" in _read_screener(),
                         "call board still reads the post-filter picks")


def _read_screener():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "src",
                        "options_screener.py")
    with open(path, encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    unittest.main()
