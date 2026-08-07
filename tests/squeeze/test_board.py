"""Tests for squeeze display boards (render smoke + content)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd

from src import formatting as fmt
from src.squeeze import board as B
from src.squeeze import detector as D
from src.squeeze import universe as U


def _setup_nbis():
    return D.assess_squeeze({
        "short_interest": 0.2797,
        "short_interest_dtc": 3.5,
        "short_interest_trend": "rising",
        "iv_skew": -0.089,
        "ret_5d": -18.2,
    })


class TestBanner(unittest.TestCase):
    def setUp(self):
        # fmt.supports_color memoizes — pin the flag, never env vars
        self._saved = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = False

    def tearDown(self):
        fmt._COLOR_ENABLED = self._saved

    def test_setup_banner_contains_evidence_and_caveat(self):
        text = B.banner(_setup_nbis(), "NBIS")
        self.assertIn("SHORT-SQUEEZE SETUP", text)
        self.assertIn("NBIS", text)
        self.assertIn("28.0%", text)
        self.assertIn("rising", text)
        self.assertIn("FINRA", text)  # staleness caveat present

    def test_none_grade_renders_nothing(self):
        self.assertIsNone(B.banner(D.assess_squeeze({}), "MU"))

    def test_watch_banner_says_watch(self):
        watch = D.assess_squeeze({"short_interest": 0.16,
                                  "short_interest_trend": "rising"})
        text = B.banner(watch, "XYZ")
        self.assertIn("SQUEEZE WATCH", text)


class TestCallBoard(unittest.TestCase):
    def setUp(self):
        self._saved = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = False

    def tearDown(self):
        fmt._COLOR_ENABLED = self._saved

    def _df(self):
        return pd.DataFrame([
            {"type": "put", "strike": 170.0, "expiration": "2026-08-14",
             "dte": 29, "delta": -0.38, "premium": 23.7, "spread_pct": 8.4,
             "ev_per_contract": -544.0, "quality_score": 0.40},
            {"type": "call", "strike": 185.0, "expiration": "2026-08-14",
             "dte": 29, "delta": 0.42, "premium": 14.2, "spread_pct": 6.0,
             "ev_per_contract": -120.0, "quality_score": 0.35},
            {"type": "call", "strike": 195.0, "expiration": "2026-08-14",
             "dte": 29, "delta": 0.30, "premium": 9.8, "spread_pct": 7.2,
             "ev_per_contract": -80.0, "quality_score": 0.31},
        ])

    def test_only_calls_shown_ranked(self):
        text = B.call_board(self._df(), "NBIS")
        self.assertIn("SQUEEZE CALLS", text)
        self.assertIn("185.0", text)
        self.assertIn("195.0", text)
        self.assertNotIn("170.0", text)  # the put stays out

    def test_no_calls_returns_none(self):
        df = self._df()
        self.assertIsNone(B.call_board(df[df["type"] == "put"], "NBIS"))
        self.assertIsNone(B.call_board(None, "NBIS"))


class TestScanBoard(unittest.TestCase):
    def setUp(self):
        self._saved = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = False

    def tearDown(self):
        fmt._COLOR_ENABLED = self._saved

    def test_setup_rows_sort_first(self):
        per = [
            {"ticker": "AAA", "setup": D.assess_squeeze({"short_interest": 0.16,
                                                         "short_interest_trend": "rising"}),
             "best_call": None},
            {"ticker": "NBIS", "setup": _setup_nbis(), "best_call": "$185C 08/14"},
        ]
        text = B.squeeze_scan_board(per)
        self.assertLess(text.index("NBIS"), text.index("AAA"))
        self.assertIn("SETUP", text)
        self.assertIn("WATCH", text)
        self.assertIn("$185C 08/14", text)


def _ladder(spot=8.69):
    """One expiry across strikes, priced so premium falls as strike rises."""
    rows = []
    for strike, prem, qs in [(6.0, 2.85, 0.70), (7.5, 1.54, 0.65),
                             (9.0, 0.71, 0.50), (10.0, 0.40, 0.45),
                             (13.0, 0.10, 0.40)]:
        rows.append({
            "type": "call", "strike": strike, "expiration": "2026-08-21",
            "dte": 13, "delta": 0.9 - (strike - 6.0) * 0.1, "premium": prem,
            "spread_pct": 0.08, "ev_per_contract": -12.0,
            "quality_score": qs, "underlying": spot,
        })
    return pd.DataFrame(rows)


class TestConvexityRanking(unittest.TestCase):
    """Rank the squeeze long side by payoff on the move, not by PoP.

    quality_score rewards probability of profit, which on a call ladder means
    deep ITM — the 2026-08-07 board surfaced delta +0.79 contracts, nearly
    stock replacement. A squeeze is a right-tail trade (the backtest's own
    framing: fatter right tail, *worse* median), so the long side should be
    ranked by what it pays if the measured move happens.
    """

    def setUp(self):
        self._saved = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = False

    def tearDown(self):
        fmt._COLOR_ENABLED = self._saved

    def test_multiple_is_intrinsic_at_target_over_premium(self):
        # spot 10 -> +20% = 12; a $9 call pays 3.00 on a 1.50 premium.
        got = B.convexity_multiple({"strike": 9.0, "premium": 1.50,
                                    "underlying": 10.0})
        self.assertAlmostEqual(got, 2.0, places=6)

    def test_a_strike_the_move_never_reaches_scores_zero(self):
        # spot 10 -> 12; a $13 call is still worthless.
        got = B.convexity_multiple({"strike": 13.0, "premium": 0.10,
                                    "underlying": 10.0})
        self.assertEqual(got, 0.0)

    def test_missing_inputs_score_none_rather_than_zero(self):
        # None means "cannot rank", which must not read as "pays nothing".
        self.assertIsNone(B.convexity_multiple({"strike": 9.0, "premium": 0.0,
                                                "underlying": 10.0}))
        self.assertIsNone(B.convexity_multiple({"strike": 9.0, "premium": 1.5,
                                                "underlying": None}))

    def test_board_leads_with_convexity_not_the_deepest_itm(self):
        # $6.0 has the best quality_score (0.70) and the deepest delta (+0.90);
        # $9.0 pays 2.0x on the move against its 1.6x. The deep ITM contract
        # may still appear, but it must not lead.
        text = B.call_board(_ladder(), "ONDS", top_n=3)
        self.assertLess(text.index("$9.0"), text.index("$6.0"))

    def test_board_ranks_the_ladder_by_payoff_on_the_move(self):
        text = B.call_board(_ladder(), "ONDS", top_n=5)
        order = [ln for ln in text.splitlines() if "$" in ln and "C " not in ln]
        pos = {k: text.index(f"${k}") for k in ("9.0", "10.0", "13.0")}
        self.assertLess(pos["9.0"], pos["10.0"])
        self.assertLess(pos["10.0"], pos["13.0"])
        self.assertTrue(order)

    def test_footnote_warns_that_short_dated_is_not_the_measured_trade(self):
        # Dividing by premium favours cheap, short-dated contracts, and the
        # cohort's hit rate is measured over 42 trading days. A 13-DTE call
        # priced at 3.3x is not the trade the 50.5% describes.
        text = B.call_board(_ladder(), "ONDS", top_n=1)
        self.assertIn("42 trading days", text)

    def test_board_shows_the_multiple_it_ranked_on(self):
        text = B.call_board(_ladder(), "ONDS", top_n=3)
        self.assertIn("+20%", text)
        # $9.0 strike, spot 8.69 -> target 10.43, intrinsic 1.43 on 0.71 = 2.0x
        self.assertIn("2.0x", text)

    def test_summary_best_call_agrees_with_the_detail_board(self):
        # Two surfaces, one answer: the board's top row and the scan board's
        # "Best call" column must not name different contracts.
        df = _ladder()
        label = B.best_call_label(df)
        text = B.call_board(df, "ONDS", top_n=1)
        self.assertEqual(label, "$9C 2026-08-21")
        self.assertIn("$9.0", text)

    def test_best_call_label_is_none_without_calls(self):
        self.assertIsNone(B.best_call_label(pd.DataFrame()))
        self.assertIsNone(B.best_call_label(
            pd.DataFrame([{"type": "put", "strike": 9.0, "premium": 1.0,
                           "underlying": 10.0}])))

    def test_falls_back_to_quality_score_when_spot_is_missing(self):
        df = _ladder().drop(columns=["underlying"])
        text = B.call_board(df, "ONDS", top_n=1)
        self.assertIsNotNone(text)
        self.assertIn("$6.0", text)  # best quality_score wins again


class TestCallBoardSpreadUnits(unittest.TestCase):
    """`spread_pct` is a FRACTION everywhere in the pipeline.

    enrich_and_score sets (ask-bid)/mid, the filter compares it to 0.40 for
    40%, and cli_display multiplies by 100 before printing. call_board printed
    it raw under a "Sprd%" header, so a 10% spread read as 0.1 — a 100×
    understatement that makes the widest contract look like the tightest. It
    went unnoticed because the board rendered no rows until the delta-band
    stash was wired up.
    """

    def setUp(self):
        self._saved = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = False

    def tearDown(self):
        fmt._COLOR_ENABLED = self._saved

    def test_a_ten_percent_spread_renders_as_ten(self):
        df = pd.DataFrame([{
            "type": "call", "strike": 9.0, "expiration": "2026-08-21", "dte": 13,
            "delta": 0.45, "premium": 0.71, "spread_pct": 0.10,
            "ev_per_contract": -14.0, "quality_score": 0.55,
        }])
        text = B.call_board(df, "ONDS")
        self.assertIn("10.0", text)
        self.assertNotIn(" 0.1 ", text)


class TestSourcingLines(unittest.TestCase):
    def setUp(self):
        self._saved = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = False

    def tearDown(self):
        fmt._COLOR_ENABLED = self._saved

    def test_reports_how_many_names_cleared_the_momentum_screen(self):
        uni = U.SqueezeUniverse(tickers=["AAA", "BBB", "CCC"], momentum=["AAA", "BBB"])
        text = "\n".join(B.sourcing_lines(uni))
        self.assertIn("2 of 3", text)
        self.assertIn("AAA", text)

    def test_quiet_week_says_the_momentum_cohort_is_empty(self):
        uni = U.SqueezeUniverse(tickers=["AAA", "BBB"], momentum=[])
        text = "\n".join(B.sourcing_lines(uni))
        self.assertIn("0 of 2", text)

    def test_fallback_universe_is_called_out_as_stale(self):
        # An 8-name hardcoded list must never read as a live Finviz screen.
        uni = U.SqueezeUniverse(tickers=U.FALLBACK_TICKERS, source="fallback")
        text = "\n".join(B.sourcing_lines(uni))
        self.assertIn("fallback", text.lower())


if __name__ == "__main__":
    unittest.main()
