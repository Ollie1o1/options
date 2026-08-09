"""Signals must be computed from the past only.

A signal that peeks even one day ahead manufactures an edge out of nothing, and
it is the easiest way to fool a backtest. These tests exist mostly to make that
impossible rather than to check arithmetic.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_alloc_signals -v
"""
from __future__ import annotations

import unittest

from src.alloc.signals import (Snapshot, SignalHistory, atm_iv, passes,
                               skew_25d, snapshot, term_slope)

EXP = "2024-03-15"


def _row(strike, typ, bid, ask, iv, expiration=EXP):
    return {"expiration": expiration, "strike": float(strike), "type": typ,
            "bid": bid, "ask": ask, "iv": iv}


class AtmIvTest(unittest.TestCase):
    def test_picks_the_strike_nearest_spot(self):
        chain = [_row(100, "call", 2.9, 3.1, 0.20),
                 _row(100, "put", 2.9, 3.1, 0.22),
                 _row(150, "call", 0.1, 0.2, 0.90)]
        self.assertAlmostEqual(atm_iv(chain, 100.0), 0.21, places=3)

    def test_ignores_missing_or_zero_iv(self):
        chain = [_row(100, "call", 2.9, 3.1, None),
                 _row(100, "put", 2.9, 3.1, 0.0),
                 _row(105, "call", 1.0, 1.2, 0.30)]
        self.assertAlmostEqual(atm_iv(chain, 100.0), 0.30, places=3)

    def test_no_usable_iv_returns_none(self):
        self.assertIsNone(atm_iv([_row(100, "call", 1, 2, None)], 100.0))

    def test_empty_chain_returns_none(self):
        self.assertIsNone(atm_iv([], 100.0))

    def test_no_spot_returns_none(self):
        self.assertIsNone(atm_iv([_row(100, "call", 1, 2, 0.2)], None))


class SnapshotTest(unittest.TestCase):
    def test_snapshot_recovers_spot_and_iv(self):
        chain = [_row(100, "call", 2.9, 3.1, 0.20),
                 _row(100, "put", 2.9, 3.1, 0.20)]
        s = snapshot(chain, "2024-01-05")
        self.assertAlmostEqual(s.spot, 100.0, places=2)
        self.assertAlmostEqual(s.atm_iv, 0.20, places=3)

    def test_empty_chain_snapshot_is_all_none(self):
        s = snapshot([], "2024-01-05")
        self.assertIsNone(s.spot)
        self.assertIsNone(s.atm_iv)


class CausalityTest(unittest.TestCase):
    """THE tests. Nothing may be known before it happens."""

    def _feed(self, ivs, start_day=1):
        h = SignalHistory(lookback=52)
        for i, iv in enumerate(ivs):
            h.update("AAA", Snapshot(f"2024-01-{start_day+i:02d}", 100.0, iv))
        return h

    def test_features_reflect_only_what_was_fed(self):
        h = self._feed([0.10] * 20)
        first = h.features("AAA")["iv_rank"]
        h.update("AAA", Snapshot("2024-02-01", 100.0, 0.99))   # a later spike
        self.assertNotEqual(first, h.features("AAA")["iv_rank"],
                            "adding a later day must change nothing retroactively")

    def test_an_out_of_order_update_is_ignored(self):
        h = self._feed([0.10] * 20)
        before = h.features("AAA")
        h.update("AAA", Snapshot("2023-01-01", 999.0, 999.0))  # older date
        self.assertEqual(h.features("AAA"), before)

    def test_a_repeated_date_is_ignored(self):
        h = self._feed([0.10] * 20)
        before = h.features("AAA")
        h.update("AAA", Snapshot("2024-01-20", 500.0, 500.0))  # same last date
        self.assertEqual(h.features("AAA"), before)

    def test_no_history_gives_no_features(self):
        self.assertEqual(SignalHistory().features("NOPE"), {})

    def test_symbols_do_not_share_history(self):
        h = SignalHistory()
        for i in range(15):
            h.update("AAA", Snapshot(f"2024-01-{i+1:02d}", 100.0, 0.10))
        self.assertEqual(h.features("BBB"), {})


class IvRankTest(unittest.TestCase):
    def test_highest_iv_in_the_window_ranks_at_the_top(self):
        h = SignalHistory()
        for i, iv in enumerate([0.10] * 19 + [0.50]):
            h.update("AAA", Snapshot(f"2024-01-{i+1:02d}", 100.0, iv))
        self.assertGreater(h.features("AAA")["iv_rank"], 95)

    def test_lowest_iv_in_the_window_ranks_at_the_bottom(self):
        h = SignalHistory()
        for i, iv in enumerate([0.50] * 19 + [0.10]):
            h.update("AAA", Snapshot(f"2024-01-{i+1:02d}", 100.0, iv))
        self.assertLess(h.features("AAA")["iv_rank"], 5)

    def test_too_little_history_gives_no_rank(self):
        h = SignalHistory()
        for i in range(5):
            h.update("AAA", Snapshot(f"2024-01-{i+1:02d}", 100.0, 0.2))
        self.assertIsNone(h.features("AAA")["iv_rank"])


class RealizedVolTest(unittest.TestCase):
    """Trailing realized vol, and the variance premium it implies.

    This is the feature that decides whether BUYING premium can work: you are
    paid for owning an option when implied underprices what the underlying
    actually does. It must be computed from the spot path already observed.
    """

    def _feed(self, h, spots, iv=0.20, step=1):
        import datetime as dt
        d = dt.date(2024, 1, 1)
        for s in spots:
            h.update("AAA", Snapshot(d.isoformat(), s, iv))
            d += dt.timedelta(days=step)

    def test_a_flat_price_path_has_almost_no_realized_vol(self):
        h = SignalHistory()
        self._feed(h, [100.0] * 20)
        self.assertLess(h.features("AAA")["rv"], 0.01)

    def test_a_choppy_path_realizes_more_than_a_calm_one(self):
        calm, wild = SignalHistory(), SignalHistory()
        self._feed(calm, [100.0 + (i % 2) * 0.5 for i in range(20)])
        self._feed(wild, [100.0 + (i % 2) * 8.0 for i in range(20)])
        self.assertGreater(wild.features("AAA")["rv"],
                           calm.features("AAA")["rv"])

    def test_too_little_history_gives_no_realized_vol(self):
        h = SignalHistory()
        self._feed(h, [100.0, 101.0, 102.0])
        self.assertIsNone(h.features("AAA")["rv"])

    def test_variance_premium_is_implied_minus_realized(self):
        h = SignalHistory()
        self._feed(h, [100.0] * 20, iv=0.30)
        f = h.features("AAA")
        self.assertAlmostEqual(f["iv_minus_rv"], 0.30 - f["rv"], places=6)

    def test_variance_premium_is_none_when_either_side_is_unknown(self):
        h = SignalHistory()
        self._feed(h, [100.0, 101.0])
        self.assertIsNone(h.features("AAA")["iv_minus_rv"])

    def test_sampling_gaps_do_not_inflate_realized_vol(self):
        # The cache is every-other-day in 2022-24 and daily in 2025. A return
        # measured over a 2-day gap is sqrt(2) larger than a 1-day return, so
        # scaling by elapsed time is what keeps the two eras comparable —
        # otherwise the backfill itself would look like a vol regime change.
        daily, every_other = SignalHistory(), SignalHistory()
        path = [100.0 * (1.01 ** i) for i in range(20)]
        self._feed(daily, path, step=1)
        self._feed(every_other, path, step=2)
        a = daily.features("AAA")["rv"]
        b = every_other.features("AAA")["rv"]
        self.assertLess(abs(a - b) / max(a, b, 1e-9), 0.35)

    def test_a_long_data_hole_is_skipped_not_annualised(self):
        # 2020-03-20 -> 2022-01-03 is a 21-month hole in this cache. Treating
        # it as one return would report a fabricated vol explosion.
        h = SignalHistory()
        self._feed(h, [100.0] * 20)
        h.update("AAA", Snapshot("2026-01-01", 250.0, 0.20))
        self.assertLess(h.features("AAA")["rv"], 0.05)


class TrendTest(unittest.TestCase):
    def test_price_above_its_average_is_a_positive_trend(self):
        h = SignalHistory()
        for i in range(15):
            h.update("AAA", Snapshot(f"2024-01-{i+1:02d}", 100.0 + i, 0.2))
        self.assertGreater(h.features("AAA")["trend"], 0)

    def test_price_below_its_average_is_a_negative_trend(self):
        h = SignalHistory()
        for i in range(15):
            h.update("AAA", Snapshot(f"2024-01-{i+1:02d}", 100.0 - i, 0.2))
        self.assertLess(h.features("AAA")["trend"], 0)

    def test_four_week_return_uses_the_fourth_prior_observation(self):
        h = SignalHistory()
        for i, spot in enumerate([100.0, 100.0, 100.0, 100.0, 110.0]):
            h.update("AAA", Snapshot(f"2024-01-{i+1:02d}", spot, 0.2))
        self.assertAlmostEqual(h.features("AAA")["ret_4w"], 10.0, places=6)


class PassesTest(unittest.TestCase):
    def test_no_conditions_always_passes(self):
        self.assertTrue(passes({"iv_rank": 50.0}, {}))

    def test_threshold_met_passes(self):
        self.assertTrue(passes({"iv_rank": 60.0}, {"iv_rank_min": 50}))

    def test_threshold_missed_fails(self):
        self.assertFalse(passes({"iv_rank": 40.0}, {"iv_rank_min": 50}))

    def test_an_uncomputable_feature_FAILS_rather_than_passing(self):
        """Treating unknown as a pass would silently turn a signalled strategy
        back into the unconditional one and make the comparison meaningless."""
        self.assertFalse(passes({"iv_rank": None}, {"iv_rank_min": 50}))
        self.assertFalse(passes({}, {"iv_rank_min": 50}))

    def test_every_condition_must_hold(self):
        f = {"iv_rank": 60.0, "trend": -5.0}
        self.assertFalse(passes(f, {"iv_rank_min": 50, "trend_min": 0}))
        self.assertTrue(passes(f, {"iv_rank_min": 50, "trend_max": 0}))

    def test_max_conditions_work(self):
        self.assertTrue(passes({"iv_rank": 20.0}, {"iv_rank_max": 30}))
        self.assertFalse(passes({"iv_rank": 40.0}, {"iv_rank_max": 30}))


if __name__ == "__main__":
    unittest.main()


class SplitForgetTest(unittest.TestCase):
    """A 20:1 split is not a 95% crash, and must not be read as one."""

    def test_forget_clears_the_series(self):
        h = SignalHistory()
        for i in range(15):
            h.update("AAA", Snapshot(f"2024-01-{i+1:02d}", 2000.0, 0.2))
        self.assertTrue(h.features("AAA"))
        h.forget("AAA")
        self.assertEqual(h.features("AAA"), {})

    def test_history_restarts_cleanly_after_a_split(self):
        h = SignalHistory()
        for i in range(15):
            h.update("AAA", Snapshot(f"2024-01-{i+1:02d}", 2000.0, 0.2))
        h.forget("AAA")
        for i in range(15):
            h.update("AAA", Snapshot(f"2024-02-{i+1:02d}", 100.0, 0.2))
        # trend is measured against the POST-split series only
        self.assertAlmostEqual(h.features("AAA")["trend"], 0.0, places=6)

    def test_without_forget_a_split_looks_like_a_crash(self):
        """Documents the bug this prevents."""
        h = SignalHistory()
        for i in range(15):
            h.update("AAA", Snapshot(f"2024-01-{i+1:02d}", 2000.0, 0.2))
        h.update("AAA", Snapshot("2024-02-01", 100.0, 0.2))
        self.assertLess(h.features("AAA")["trend"], -80)

    def test_forgetting_an_unknown_symbol_is_harmless(self):
        SignalHistory().forget("NOPE")


# --------------------------------------------------------------------------
# Shape and rate-of-change features (docs/LEADING_INDICATORS_20260809.md §3)
#
# Every feature measured before these was a LEVEL of one name at one moment,
# and every one of them was tested and killed. These are the shape of the
# surface across expirations and across strikes, and the rate at which the
# level is moving — which is what distinguishes "premium is rich and calming
# down" from "premium is rich because a crash is underway".
# --------------------------------------------------------------------------

def _drow(strike, typ, iv, delta, expiration=EXP):
    """A chain row carrying a delta, which `_row` deliberately does not."""
    r = _row(strike, typ, 1.0, 1.2, iv, expiration)
    r["delta"] = delta
    return r


class TermStructureTest(unittest.TestCase):
    """Near-dated ATM IV against far-dated.

    IV rank is a level against a name's own history and is COINCIDENT with
    stress. This is a shape, and it inverts as stress arrives — which is why
    it is worth testing after the levels came back empty.
    """

    def test_backwardation_is_positive(self):
        # near expiry richer than far: the market is pricing near-term stress
        chain = [_row(100, "call", 2.9, 3.1, 0.40, "2024-02-01"),
                 _row(100, "put", 2.9, 3.1, 0.40, "2024-02-01"),
                 _row(100, "call", 5.0, 5.2, 0.25, "2024-04-01"),
                 _row(100, "put", 5.0, 5.2, 0.25, "2024-04-01")]
        self.assertAlmostEqual(term_slope(chain, 100.0), 0.15, places=6)

    def test_contango_is_negative(self):
        chain = [_row(100, "call", 2.9, 3.1, 0.20, "2024-02-01"),
                 _row(100, "put", 2.9, 3.1, 0.20, "2024-02-01"),
                 _row(100, "call", 5.0, 5.2, 0.30, "2024-04-01"),
                 _row(100, "put", 5.0, 5.2, 0.30, "2024-04-01")]
        self.assertAlmostEqual(term_slope(chain, 100.0), -0.10, places=6)

    def test_one_expiration_has_no_slope(self):
        chain = [_row(100, "call", 2.9, 3.1, 0.20),
                 _row(100, "put", 2.9, 3.1, 0.20)]
        self.assertIsNone(term_slope(chain, 100.0))

    def test_expirations_too_close_together_are_one_point_not_a_slope(self):
        # A 2-day separation is noise in the surface, not term structure.
        chain = [_row(100, "call", 2.9, 3.1, 0.40, "2024-02-01"),
                 _row(100, "put", 2.9, 3.1, 0.40, "2024-02-01"),
                 _row(100, "call", 3.0, 3.2, 0.25, "2024-02-03"),
                 _row(100, "put", 3.0, 3.2, 0.25, "2024-02-03")]
        self.assertIsNone(term_slope(chain, 100.0))

    def test_an_unusable_iv_on_one_leg_gives_no_slope(self):
        chain = [_row(100, "call", 2.9, 3.1, None, "2024-02-01"),
                 _row(100, "put", 2.9, 3.1, None, "2024-02-01"),
                 _row(100, "call", 5.0, 5.2, 0.25, "2024-04-01"),
                 _row(100, "put", 5.0, 5.2, 0.25, "2024-04-01")]
        self.assertIsNone(term_slope(chain, 100.0))

    def test_no_spot_gives_no_slope(self):
        self.assertIsNone(term_slope([_row(100, "call", 1, 2, 0.2)], None))


class SkewTest(unittest.TestCase):
    """25-delta put IV against 25-delta call IV.

    Selling a bull put IS selling the put wing, and how rich that wing is
    relative to the call side is the price of the thing being sold. No level
    feature captures it: `atm_iv` is measured at the money and `iv_rank` is a
    time-series rank of that same at-the-money number.
    """

    def _chain(self, put_iv, call_iv, expiration=EXP):
        return [_drow(90, "put", put_iv, -0.25, expiration),
                _drow(100, "put", 0.20, -0.50, expiration),
                _drow(100, "call", 0.20, 0.50, expiration),
                _drow(110, "call", call_iv, 0.25, expiration)]

    def test_a_rich_put_wing_is_positive_skew(self):
        self.assertAlmostEqual(skew_25d(self._chain(0.35, 0.22)),
                               0.13, places=6)

    def test_a_symmetric_smile_has_no_skew(self):
        self.assertAlmostEqual(skew_25d(self._chain(0.25, 0.25)),
                               0.0, places=6)

    def test_a_chain_without_deltas_gives_no_skew(self):
        chain = [_row(90, "put", 1, 2, 0.35), _row(110, "call", 1, 2, 0.22)]
        self.assertIsNone(skew_25d(chain))

    def test_no_strike_near_25_delta_gives_no_skew(self):
        # A ladder that stops at 45-delta cannot price a 25-delta wing, and
        # substituting the nearest listed contract however far away it is
        # is the same defect `_nearest_delta` had before DELTA_TOLERANCE.
        chain = [_drow(95, "put", 0.35, -0.45), _drow(105, "call", 0.22, 0.45)]
        self.assertIsNone(skew_25d(chain))

    def test_one_wing_missing_gives_no_skew(self):
        chain = [_drow(90, "put", 0.35, -0.25)]
        self.assertIsNone(skew_25d(chain))

    def test_skew_is_read_off_the_nearest_expiration(self):
        # Two expirations with opposite skews: the near one must win, so the
        # feature has one consistent horizon rather than a blend of two.
        chain = (self._chain(0.35, 0.20, "2024-02-01")
                 + self._chain(0.20, 0.35, "2024-06-01"))
        self.assertAlmostEqual(skew_25d(chain), 0.15, places=6)


class IvVelocityTest(unittest.TestCase):
    """The rate the level is moving, which the level itself cannot say.

    §4f of ATTRIBUTION_20260808 found high IV rank selects INTO a crash. That
    is what a level does. "Rich and falling" is the textbook short-premium
    entry; "rich and rising" is a crash in progress; IV rank cannot tell them
    apart and this can.
    """

    def _feed(self, h, ivs, step=1, start=(2024, 1, 1)):
        import datetime as dt
        d = dt.date(*start)
        for iv in ivs:
            h.update("AAA", Snapshot(d.isoformat(), 100.0, iv))
            d += dt.timedelta(days=step)

    def test_rising_implied_vol_is_positive_velocity(self):
        h = SignalHistory()
        self._feed(h, [0.20 + 0.01 * i for i in range(20)])
        self.assertGreater(h.features("AAA")["iv_velocity"], 0)

    def test_falling_implied_vol_is_negative_velocity(self):
        h = SignalHistory()
        self._feed(h, [0.40 - 0.01 * i for i in range(20)])
        self.assertLess(h.features("AAA")["iv_velocity"], 0)

    def test_a_flat_level_has_no_velocity(self):
        h = SignalHistory()
        self._feed(h, [0.25] * 20)
        self.assertAlmostEqual(h.features("AAA")["iv_velocity"], 0.0, places=9)

    def test_sampling_density_does_not_change_the_velocity(self):
        # Same IV path in vol-points-per-calendar-day, sampled daily and every
        # other day. The cache changes cadence in 2025, and a velocity that
        # read the cadence would make the backfill itself look like a signal —
        # the same trap `_realized_vol` scales for.
        daily, sparse = SignalHistory(), SignalHistory()
        self._feed(daily, [0.20 + 0.005 * i for i in range(24)], step=1)
        self._feed(sparse, [0.20 + 0.010 * i for i in range(12)], step=2)
        a = daily.features("AAA")["iv_velocity"]
        b = sparse.features("AAA")["iv_velocity"]
        self.assertAlmostEqual(a, b, places=6)

    def test_too_little_history_gives_no_velocity(self):
        h = SignalHistory()
        self._feed(h, [0.20, 0.21])
        self.assertIsNone(h.features("AAA")["iv_velocity"])

    def test_a_long_data_hole_is_not_a_velocity(self):
        # Comparing across a 21-month hole fabricates an event, which is the
        # same failure that made split detection read gap-drift as a split.
        h = SignalHistory()
        h.update("AAA", Snapshot("2020-03-20", 100.0, 0.80))
        h.update("AAA", Snapshot("2022-01-03", 100.0, 0.20))
        self.assertIsNone(h.features("AAA")["iv_velocity"])


class VolOfVolTest(unittest.TestCase):
    def test_a_steady_level_has_almost_no_vol_of_vol(self):
        h = SignalHistory()
        for i in range(20):
            h.update("AAA", Snapshot(f"2024-01-{i+1:02d}", 100.0, 0.25))
        self.assertAlmostEqual(h.features("AAA")["vol_of_vol"], 0.0, places=9)

    def test_a_swinging_level_reads_higher_than_a_calm_one(self):
        calm, wild = SignalHistory(), SignalHistory()
        for i in range(20):
            calm.update("AAA", Snapshot(f"2024-01-{i+1:02d}", 100.0,
                                        0.25 + (i % 2) * 0.01))
            wild.update("AAA", Snapshot(f"2024-01-{i+1:02d}", 100.0,
                                        0.25 + (i % 2) * 0.20))
        self.assertGreater(wild.features("AAA")["vol_of_vol"],
                           calm.features("AAA")["vol_of_vol"])

    def test_too_little_history_gives_no_vol_of_vol(self):
        h = SignalHistory()
        h.update("AAA", Snapshot("2024-01-01", 100.0, 0.25))
        self.assertIsNone(h.features("AAA")["vol_of_vol"])


class NewFeaturesAreCausalTest(unittest.TestCase):
    """The whole module's contract: nothing may peek ahead."""

    def test_velocity_ignores_a_snapshot_fed_out_of_order(self):
        h = SignalHistory()
        for i in range(20):
            h.update("AAA", Snapshot(f"2024-02-{i+1:02d}", 100.0, 0.20))
        before = h.features("AAA")["iv_velocity"]
        h.update("AAA", Snapshot("2024-01-01", 100.0, 9.99))   # earlier date
        self.assertEqual(h.features("AAA")["iv_velocity"], before)

    def test_shape_features_survive_a_split_forget(self):
        h = SignalHistory()
        h.update("AAA", Snapshot("2024-01-01", 100.0, 0.2))
        h.forget("AAA")
        self.assertEqual(h.features("AAA"), {})


class NewConditionsTest(unittest.TestCase):
    """The new features must be usable as entry conditions, on the same terms.

    Chiefly the standing one: an uncomputable feature FAILS. Most symbol-days
    will have no `skew_25d` (85.4% of symbol-day-expiries carry both wings, so
    ~15% do not), and an unknown that passed would silently turn a conditioned
    arm back into the unconditional one.
    """

    def test_conditions_on_the_new_features_are_honoured(self):
        f = {"term_slope": 0.05, "skew_25d": 0.10, "iv_velocity": -0.02}
        self.assertTrue(passes(f, {"term_slope_min": 0.0}))
        self.assertFalse(passes(f, {"term_slope_max": 0.0}))
        self.assertTrue(passes(f, {"skew_25d_min": 0.05}))
        self.assertTrue(passes(f, {"iv_velocity_max": 0.0}))
        self.assertFalse(passes(f, {"iv_velocity_min": 0.0}))

    def test_an_uncomputable_new_feature_still_FAILS(self):
        f = {"term_slope": None, "skew_25d": None, "iv_velocity": None}
        self.assertFalse(passes(f, {"term_slope_min": 0.0}))
        self.assertFalse(passes(f, {"skew_25d_min": 0.0}))
        self.assertFalse(passes(f, {"iv_velocity_max": 0.0}))

    def test_the_rich_and_falling_arm_composes(self):
        # The interaction docs/LEADING_INDICATORS_20260809.md §2b argues for:
        # high IV rank AND falling is the textbook entry, high AND rising is a
        # crash in progress. IV rank alone cannot separate them.
        cond = {"iv_rank_min": 70.0, "iv_velocity_max": 0.0}
        calming = {"iv_rank": 85.0, "iv_velocity": -0.03}
        erupting = {"iv_rank": 85.0, "iv_velocity": +0.04}
        self.assertTrue(passes(calming, cond))
        self.assertFalse(passes(erupting, cond))
