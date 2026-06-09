"""Tests for src/intel/playbook.py.

The core guarantee: the right scenario line fires for the right situation. We
build an archetypal state for every scenario and assert its id is the primary
match, then check structural properties (priority order, catch-all, fillable).
"""
from __future__ import annotations

import unittest

from src.intel import playbook as P
from src.intel.playbook import PlaybookState


_FULL_FMT = {
    "ma50": "$205", "ma200": "$189", "nearest_support": "$189",
    "support_200d": "$189", "support_50d": "$205", "resist": "$235",
}


def st(**kw) -> PlaybookState:
    fmt = dict(_FULL_FMT)
    fmt.update(kw.pop("fmt", {}))
    return PlaybookState(fmt=fmt, **kw)


# (expected_id, state) — one archetype per scenario.
CASES = [
    ("earnings_imminent", st(days_to_earnings=2)),
    ("earnings_soon_iv_ramp", st(days_to_earnings=7)),
    ("earnings_just_passed", st(days_to_earnings=-1)),
    ("falling_knife", st(new_60d_low=True, momentum=-0.3)),
    ("primary_downtrend", st(below_200d=True, trend=-0.3)),
    ("counter_trend_bounce", st(below_200d=True, near_resistance=True, trend=-0.1)),
    ("fresh_death_cross", st(death_cross=True, trend=-0.1)),
    ("oversold_at_support_bounce",
     st(rsi=20, support_dist=0.01, bounce_rate=0.7, bounce_n=15, trend=0.0)),
    ("oversold_in_downtrend", st(rsi=25, trend=-0.3, support_dist=0.1)),
    ("big_drop_strong_bounce",
     st(price_down_5d=True, bounce_rate=0.7, bounce_n=15, rsi=50, support_dist=0.1)),
    ("big_drop_thin_sample",
     st(price_down_5d=True, bounce_rate=0.7, bounce_n=5, rsi=50)),
    ("pullback_to_200d_support",
     st(trend=0.5, momentum=-0.3, support_dist=0.01, near_200d=True)),
    ("pullback_to_support_falling",
     st(verdict="WAIT", trend=0.5, momentum=-0.3, support_dist=0.025)),
    ("shallow_dip_wait_for_support",
     st(trend=0.5, momentum=-0.3, support_dist=0.05)),
    ("trend_repairing_reclaimed_50d",
     st(trend=0.4, reclaimed_50d=True, momentum=0.0)),
    ("parabolic_let_cool", st(parabolic=True, trend=0.0)),
    ("overbought_extended", st(rsi=80, trend=0.5, momentum=0.0)),
    ("at_resistance_buy_breakout", st(near_resistance=True, trend=0.1)),
    ("iv_rich_prefer_spreads", st(iv_rank=0.8)),
    ("iv_cheap_long_premium", st(iv_rank=0.2)),
    ("skew_rich_puts", st(skew=0.2)),
    ("term_backwardated", st(term_backwardated=True)),
    ("analyst_cuts_headwind", st(analyst_cuts=2, analyst_raises=0)),
    ("analyst_raises_tailwind", st(analyst_raises=2, analyst_cuts=0)),
    ("bad_news_already_down",
     st(news_sentiment=-0.3, price_down_5d=True)),  # no bounce sample
    ("healthy_uptrend_entry", st(trend=0.5, momentum=0.3, rsi=50)),
    ("uptrend_defensive_regime", st(trend=0.2, momentum=0.0, regime="DEFENSIVE")),
    ("sideways_no_edge", st(trend=0.1, momentum=0.05, regime="")),
    ("signals_conflict",
     st(verdict="NEUTRAL", trend=0.3, momentum=-0.3, top_driver="trend up vs momentum down")),
    ("catch_all", st(verdict="WAIT", trend=0.25, momentum=0.0)),
]


class ScenarioTableTests(unittest.TestCase):
    def test_each_archetype_fires_expected(self):
        for expected_id, state in CASES:
            with self.subTest(scenario=expected_id):
                self.assertEqual(P.select_id(state), expected_id)

    def test_table_covers_every_scenario(self):
        covered = {cid for cid, _ in CASES}
        defined = {sc.id for sc in P.SCENARIOS}
        self.assertEqual(covered, defined,
                         f"untested scenarios: {defined - covered}")


class StructuralTests(unittest.TestCase):
    def test_priorities_unique(self):
        prios = [sc.priority for sc in P.SCENARIOS]
        self.assertEqual(len(prios), len(set(prios)), "duplicate priorities")

    def test_catch_all_always_matches(self):
        primary, _ = P.select(PlaybookState(fmt=_FULL_FMT))
        self.assertIsNotNone(primary)

    def test_select_returns_filled_text(self):
        primary, _ = P.select(st(below_200d=True, trend=-0.3))
        self.assertIn("$189", primary)        # {ma200} filled
        self.assertNotIn("{", primary)         # no leftover template fields

    def test_secondary_is_lower_priority(self):
        # A drop with a strong bounce also matches several lower-priority lines.
        primary, secondary = P.select(
            st(price_down_5d=True, bounce_rate=0.7, bounce_n=15, rsi=50, support_dist=0.1))
        self.assertIsNotNone(secondary)
        self.assertNotEqual(primary, secondary)

    def test_missing_fmt_field_skips_scenario_not_crash(self):
        # State that would hit pullback_to_200d (needs support_200d) but fmt lacks it.
        state = PlaybookState(trend=0.5, momentum=-0.3, support_dist=0.01,
                              fmt={"nearest_support": "$189", "ma50": "$205",
                                   "resist": "$235"})
        # Should not raise, and should fall through to a fillable line.
        primary, _ = P.select(state)
        self.assertIsNotNone(primary)
        self.assertNotIn("{", primary)


if __name__ == "__main__":
    unittest.main()
