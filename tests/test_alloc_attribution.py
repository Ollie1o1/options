"""Attribution: which entry-time features separated winners from losers.

The hazard this file guards is not arithmetic, it is self-deception. Ranking
features by correlation and reporting the best one is a search, and a search
finds something in noise every time. These tests pin the guards that make the
output honest: unknown features never silently pass, the split is by TIME and
never by shuffle, and the trial count is carried so deflation is possible.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_alloc_attribution -v
"""
from __future__ import annotations

import unittest

from src.alloc.attribution import (benjamini_hochberg, disaster_auc,
                                   feature_ic, bucket_table,
                                   format_ranking, monotonicity,
                                   rank_features, residual_ic, split_at_date,
                                   split_by_time)


class _T:
    """Minimal stand-in for engine.Trade."""

    def __init__(self, entry_date, roc, capital_at_risk=100.0, **feats):
        self.entry_date = entry_date
        self.capital_at_risk = capital_at_risk
        self.pnl = roc * capital_at_risk
        self.features = feats
        self.exit_date = "2024-12-31"


def _monotone(n=60, feature="iv_rank", noise=0.0):
    """Trades whose outcome rises with the feature."""
    out = []
    for i in range(n):
        out.append(_T(f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}",
                      roc=(i / n) - 0.5 + noise * ((-1) ** i),
                      **{feature: float(i)}))
    return out


class FeatureIcTest(unittest.TestCase):
    def test_a_monotone_feature_scores_a_high_ic(self):
        r = feature_ic(_monotone(), "iv_rank")
        self.assertGreater(r["ic"], 0.95)
        self.assertEqual(r["n"], 60)

    def test_an_inverted_feature_scores_a_negative_ic(self):
        trades = _monotone()
        for t in trades:
            t.features["iv_rank"] = -t.features["iv_rank"]
        self.assertLess(feature_ic(trades, "iv_rank")["ic"], -0.95)

    def test_a_feature_nobody_has_is_reported_as_unmeasurable(self):
        # Silently returning 0.0 would read as "measured, no effect" — a
        # different claim from "could not measure", and only one of them means
        # go and get more data.
        r = feature_ic(_monotone(), "does_not_exist")
        self.assertEqual(r["n"], 0)
        self.assertIsNone(r["ic"])

    def test_trades_missing_the_feature_are_dropped_not_zero_filled(self):
        trades = _monotone(n=40)
        for t in trades[:10]:
            t.features.pop("iv_rank")
        self.assertEqual(feature_ic(trades, "iv_rank")["n"], 30)

    def test_open_trades_are_excluded(self):
        trades = _monotone(n=40)
        for t in trades[:10]:
            t.exit_date = None
        self.assertEqual(feature_ic(trades, "iv_rank")["n"], 30)

    def test_a_constant_feature_has_no_ic(self):
        trades = [_T(f"2024-01-{i+1:02d}", roc=i / 20.0, iv_rank=50.0)
                  for i in range(20)]
        self.assertIsNone(feature_ic(trades, "iv_rank")["ic"])

    def test_clustered_t_is_reported_and_is_not_larger_than_naive(self):
        # Same-day trades share that day's move; clustering must not flatter.
        trades = [_T("2024-03-01", roc=(i % 5) / 10.0, iv_rank=float(i))
                  for i in range(40)]
        r = feature_ic(trades, "iv_rank")
        self.assertIn("t_clustered", r)
        self.assertLessEqual(abs(r["t_clustered"]), abs(r["t"]) + 1e-9)


class BucketTableTest(unittest.TestCase):
    def test_buckets_are_ordered_and_cover_every_trade(self):
        rows = bucket_table(_monotone(n=60), "iv_rank", n_buckets=4)
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(r["n"] for r in rows), 60)
        self.assertLess(rows[0]["mean_roc"], rows[-1]["mean_roc"])

    def test_too_few_trades_yields_no_buckets(self):
        self.assertEqual(bucket_table(_monotone(n=3), "iv_rank", 4), [])


class SplitByTimeTest(unittest.TestCase):
    def test_split_is_chronological_never_shuffled(self):
        trades = [_T(f"2024-01-{i+1:02d}", roc=0.1, iv_rank=float(i))
                  for i in range(10)]
        train, test = split_by_time(trades, frac=0.6)
        self.assertEqual(len(train), 6)
        self.assertEqual(len(test), 4)
        self.assertTrue(max(t.entry_date for t in train)
                        <= min(t.entry_date for t in test))

    def test_a_shuffled_input_still_splits_chronologically(self):
        import random
        trades = [_T(f"2024-01-{i+1:02d}", roc=0.1, iv_rank=float(i))
                  for i in range(10)]
        random.Random(0).shuffle(trades)
        train, test = split_by_time(trades, frac=0.5)
        self.assertTrue(max(t.entry_date for t in train)
                        <= min(t.entry_date for t in test))


class RankFeaturesTest(unittest.TestCase):
    def test_the_real_feature_outranks_the_noise_ones(self):
        trades = _monotone(n=80)
        for i, t in enumerate(trades):
            t.features["junk"] = float((i * 37) % 11)
        ranked = rank_features(trades, ["iv_rank", "junk"])
        self.assertEqual(ranked[0]["feature"], "iv_rank")

    def test_the_trial_count_is_carried_so_results_can_be_deflated(self):
        # Ranking k features IS a k-way search. A caller that cannot see k
        # cannot deflate, and an undeflated best-of-k is the exact mistake
        # ALLOCATION_BACKTEST_FINDINGS 4b records twice.
        ranked = rank_features(_monotone(n=40), ["iv_rank", "dte", "width"])
        self.assertTrue(all(r["n_trials"] == 3 for r in ranked))

    def test_unmeasurable_features_are_reported_not_dropped(self):
        ranked = rank_features(_monotone(n=40), ["iv_rank", "absent"])
        by = {r["feature"]: r for r in ranked}
        self.assertIsNone(by["absent"]["ic"])
        self.assertEqual(by["absent"]["n"], 0)


class ThresholdEffectTest(unittest.TestCase):
    """A feature whose value is TAIL AVOIDANCE must not read as "no signal".

    This is the blind spot that produced a wrong conclusion on 2026-08-08.
    `iv_rank` scored a Spearman IC of -0.029 and was written up as flat, while
    the same data as a conditional mean was monotone and strong (-2.85% /
    -1.15% / +2.51% / +3.86% across IVR<=30 / baseline / >=50 / >=70, DSR
    0.797). Both numbers were right. A rank correlation asks whether the
    feature orders EVERY trade; this is a threshold effect on a variable with
    skew -1.7 to -2.0, where the gain is fewer catastrophic trades. That moves
    the MEAN a long way and the median ordering barely at all.
    """

    def _threshold_trades(self, n=400):
        """Disasters concentrated at low feature values; everything else is
        noise unrelated to the feature. Rank IC stays modest because 85% of
        trades carry no information, while the mean gap is enormous because a
        -100% trade dwarfs a +5% one."""
        import random
        rng = random.Random(7)
        out = []
        for i in range(n):
            f = i / n
            rate = 0.30 if f < 0.4 else 0.06     # disaster probability
            roc = -1.0 if rng.random() < rate else rng.uniform(0.0, 0.10)
            out.append(_T(f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}",
                          roc=roc, feat=f))
        return out

    def test_the_rank_ic_understates_a_tail_avoidance_effect(self):
        # The whole reason the extra screen exists: an IC that reads as weak
        # sitting alongside a difference in outcome that is enormous. The two
        # are not comparable as numbers — an IC is a dimensionless correlation
        # and q_spread is in units of return on capital — so they are asserted
        # separately against what each one means.
        r = feature_ic(self._threshold_trades(), "feat")
        self.assertLess(abs(r["ic"]), 0.30)            # "weak" by IC convention
        self.assertGreater(r["q_spread"], 0.25)        # 25+ points of RoC

    def test_the_quantile_spread_catches_what_the_ic_misses(self):
        r = feature_ic(self._threshold_trades(), "feat")
        self.assertGreater(r["q_spread"], 0.15)
        self.assertGreater(r["q_top"], r["q_bot"])

    def test_quantile_spread_is_none_when_unmeasurable(self):
        self.assertIsNone(feature_ic(_monotone(n=4), "iv_rank")["q_spread"])

    def test_a_genuinely_flat_feature_shows_no_spread(self):
        import random
        rng = random.Random(3)
        trades = [_T(f"2024-01-{1 + i % 28:02d}", roc=rng.uniform(0, 0.1),
                     junk=rng.random()) for i in range(300)]
        self.assertLess(abs(feature_ic(trades, "junk")["q_spread"]), 0.05)

    def test_ranking_can_surface_a_threshold_feature_over_a_noisy_one(self):
        import random
        rng = random.Random(11)
        trades = self._threshold_trades()
        for t in trades:
            t.features["junk"] = rng.random()
        ranked = rank_features(trades, ["feat", "junk"], by="q_spread")
        self.assertEqual(ranked[0]["feature"], "feat")


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------
# The two screens the IC and the quantile spread still cannot supply.
# docs/LEADING_INDICATORS_20260809.md §2 lists four things a search should
# report; IC and q_spread were already here, these are the other two.
# --------------------------------------------------------------------------


class MonotonicityTest(unittest.TestCase):
    """The shape across buckets, as one number.

    `bucket_table` has always produced the shape and nothing summarised it, so
    judging it stayed a manual read of a table. It is the property that
    separates a graded effect from one good cell: ATTRIBUTION_20260808 §3
    threw out `credit_pct_width` on exactly this basis (Q1 -0.011, Q2 +0.019,
    Q3 +0.090, Q4 -0.068 — up then down) despite an IC of 0.56.
    """

    def test_a_rising_relationship_is_fully_monotone(self):
        self.assertGreater(monotonicity(_monotone(), "iv_rank"), 0.95)

    def test_a_falling_relationship_is_monotone_the_other_way(self):
        trades = _monotone()
        for t in trades:
            t.features["iv_rank"] = -t.features["iv_rank"]
        self.assertLess(monotonicity(trades, "iv_rank"), -0.95)

    def test_the_up_then_down_shape_that_fooled_credit_pct_width(self):
        # Deliberately built to score a strong IC while being non-monotone,
        # which is the artifact signature this screen exists to catch.
        trades = []
        for i in range(100):
            x = float(i)
            roc = (i / 100.0) if i < 75 else (0.75 - (i - 75) / 100.0)
            trades.append(_T(f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}",
                             roc=roc, credit_pct_width=x))
        self.assertLess(monotonicity(trades, "credit_pct_width"), 0.95)

    def test_too_few_trades_cannot_be_shaped(self):
        self.assertIsNone(monotonicity(_monotone(n=4), "iv_rank"))

    def test_a_feature_nobody_has_is_unmeasurable(self):
        self.assertIsNone(monotonicity(_monotone(), "does_not_exist"))


class DisasterAucTest(unittest.TestCase):
    """Does anything at entry flag the trades that lose most of the capital?

    For a short-premium book this is THE question and it is not the same one
    the IC answers: ATTRIBUTION_20260808 §5 found bull_put's 25 disasters were
    13.4% of trades and 1,577% of total absolute RoC. The entire P&L is the
    tail. §5 ran this by hand; it belongs in the harness.

    AUC is P(a disaster scored higher on this feature than a survivor), so
    0.5 is a coin flip and a value far from 0.5 in EITHER direction is a
    warning light.
    """

    def _mixed(self, flag_the_disasters: bool):
        trades = []
        for i in range(60):
            disaster = i % 5 == 0
            roc = -0.9 if disaster else 0.05
            x = (1.0 if (disaster and flag_the_disasters) else 0.0) + i * 1e-6
            trades.append(_T(f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}",
                             roc=roc, warn=x))
        return trades

    def test_a_feature_that_marks_every_disaster_scores_near_one(self):
        r = disaster_auc(self._mixed(True), "warn")
        self.assertGreater(r["auc"], 0.95)
        self.assertEqual(r["n_disasters"], 12)

    def test_a_feature_unrelated_to_the_disasters_scores_near_a_coin_flip(self):
        r = disaster_auc(self._mixed(False), "warn")
        self.assertLess(abs(r["auc"] - 0.5), 0.15)

    def test_no_disasters_means_nothing_to_separate(self):
        trades = [_T(f"2024-01-{1 + i % 28:02d}", roc=0.05, warn=float(i))
                  for i in range(40)]
        self.assertIsNone(disaster_auc(trades, "warn")["auc"])

    def test_all_disasters_means_nothing_to_separate(self):
        trades = [_T(f"2024-01-{1 + i % 28:02d}", roc=-0.9, warn=float(i))
                  for i in range(40)]
        self.assertIsNone(disaster_auc(trades, "warn")["auc"])

    def test_the_disaster_threshold_is_the_documented_one(self):
        # "more than half their capital", ATTRIBUTION_20260808 §5.
        trades = ([_T("2024-01-01", roc=-0.6, warn=1.0)] * 10
                  + [_T("2024-01-02", roc=-0.4, warn=0.0)] * 30)
        self.assertEqual(disaster_auc(trades, "warn")["n_disasters"], 10)


class RankingCarriesTheNewScreensTest(unittest.TestCase):
    def test_rank_features_reports_shape_and_tail_alongside_the_ic(self):
        rows = rank_features(_monotone(), ["iv_rank"])
        self.assertIn("mono", rows[0])
        self.assertIn("tail_auc", rows[0])

    def test_the_formatted_table_shows_them(self):
        text = format_ranking(rank_features(_monotone(), ["iv_rank"]))
        self.assertIn("mono", text)
        self.assertIn("tailAUC", text)


class ResidualIcTest(unittest.TestCase):
    """The IC that survives removing credit richness. The standing screen.

    On a held-to-expiry credit spread, return on capital is close to a function
    of the credit received and the credit is close to a function of implied
    vol. So ANY entry feature correlated with implied vol posts a large IC on
    this book and it means nothing. Measured 2026-08-11 on optionsDX SPY,
    n=1,598: `credit_pct_width` scores +0.6991 and `atm_iv` +0.5455, and both
    hypotheses tested that day collapsed under this control —

        term_slope_1m3m   raw +0.3654  ->  +0.0422  p=0.09
        entry_depth       raw -0.1907  ->  -0.0099  p=0.69

    ...having each passed the sign-holds and both-windows-significant tests
    first. This is the same class of screen as `mono` and `tail_auc`, and like
    both of those it is being added because something got through without it.
    """

    def _mixed(self, n=200, copy_strength=0.85, seed=7):
        """Outcome driven by TWO things: the control, and something else.

        `echo` is a noisy copy of the control carrying nothing of its own — it
        must lose its IC. `own` is independent of the control and genuinely
        drives the outcome — it must KEEP its IC. A screen that only satisfies
        the first is one that flattens everything.
        """
        import random
        rng = random.Random(seed)
        out = []
        for i in range(n):
            rich, own = rng.random(), rng.random()
            echo = copy_strength * rich + (1 - copy_strength) * rng.random()
            out.append(_T(f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}",
                          roc=(rich - 0.5) + 0.8 * (own - 0.5),
                          credit_pct_width=rich, atm_iv=rich,
                          echo=echo, own=own))
        return out

    def test_a_feature_that_is_the_control_keeps_no_ic(self):
        r = residual_ic(self._mixed(), "echo", controls=("credit_pct_width",))
        self.assertLess(abs(r["ic"]), 0.15)

    def test_the_same_feature_has_a_large_raw_ic(self):
        # Without this the test above would pass on a feature that never had
        # an IC to lose.
        self.assertGreater(feature_ic(self._mixed(), "echo")["ic"], 0.65)

    def test_a_feature_independent_of_the_controls_keeps_its_own_ic(self):
        # Guards over-correction, which is the way this screen would fail
        # silently: everything reads as noise and a real effect is discarded.
        r = residual_ic(self._mixed(), "own", controls=("credit_pct_width",))
        self.assertGreater(abs(r["ic"]), 0.45)

    def test_a_control_is_never_used_against_itself(self):
        # Residualising a feature on itself would report exactly 0 for every
        # control, which reads as "disproven" when it means "not asked".
        r = residual_ic(self._mixed(), "credit_pct_width",
                        controls=("credit_pct_width", "atm_iv"))
        self.assertNotIn("credit_pct_width", r["controls"])

    def test_a_renamed_control_is_not_measurable(self):
        # An EXACT duplicate under another name leaves a residual of pure
        # floating-point noise, and ranking that noise scored +0.97 before the
        # collinearity guard existed — the worst possible failure here, since
        # the screen would have endorsed the very thing it exists to catch.
        trades = self._mixed(copy_strength=1.0)
        r = residual_ic(trades, "echo", controls=("credit_pct_width",))
        self.assertIsNone(r["ic"])

    def test_a_feature_equal_to_every_control_cannot_be_measured(self):
        r = residual_ic(self._mixed(), "credit_pct_width",
                        controls=("credit_pct_width",))
        self.assertIsNone(r["ic"])
        self.assertEqual(r["controls"], [])

    def test_controls_absent_from_the_data_cannot_be_measured(self):
        # NOT the raw IC. Printing an uncontrolled number in a column labelled
        # "IC|ctl" is a false statement about what was measured, and it reads
        # as "the control made no difference" when it means "no control ran".
        trades = self._mixed()
        for t in trades:
            t.features.pop("credit_pct_width")
            t.features.pop("atm_iv")
        r = residual_ic(trades, "echo", controls=("credit_pct_width", "atm_iv"))
        self.assertIsNone(r["ic"])
        self.assertEqual(r["controls"], [])

    def test_one_trade_missing_a_control_does_not_discard_the_control(self):
        # Shipped broken and caught on real output: on the long_call tables a
        # single absent value dropped the control for the whole 3,000-trade
        # sample, and IC|ctl silently printed the raw IC instead. Dropping the
        # affected TRADES keeps the control and keeps the column honest.
        trades = self._mixed()
        trades[0].features["atm_iv"] = None
        r = residual_ic(trades, "echo", controls=("credit_pct_width", "atm_iv"))
        self.assertEqual(r["controls"], ["credit_pct_width", "atm_iv"])
        self.assertEqual(r["n"], len(trades) - 1)
        self.assertLess(abs(r["ic"]), 0.15)

    def test_a_control_missing_almost_everywhere_is_dropped_not_the_sample(self):
        # The other direction: if keeping a control would cost most of the
        # sample, the control goes rather than the evidence.
        trades = self._mixed()
        for t in trades[10:]:
            t.features["atm_iv"] = None
        r = residual_ic(trades, "echo", controls=("credit_pct_width", "atm_iv"))
        self.assertEqual(r["controls"], ["credit_pct_width"])
        self.assertEqual(r["n"], len(trades))

    def test_too_few_trades_is_none_not_zero(self):
        r = residual_ic(self._mixed(n=4), "echo")
        self.assertIsNone(r["ic"])

    def test_feature_ic_carries_the_residual(self):
        row = feature_ic(self._mixed(), "echo")
        self.assertIn("ic_resid", row)
        self.assertLess(abs(row["ic_resid"]), 0.15)

    def test_the_formatted_table_shows_it(self):
        text = format_ranking(rank_features(self._mixed(), ["echo", "own"]))
        self.assertIn("IC|ctl", text)


class SplitAtDateTest(unittest.TestCase):
    """A holdout boundary fixed as a DATE, not as a fraction of the trades.

    `split_by_time(trades, 0.7)` puts the cut wherever 70% of the trade COUNT
    falls, which moves when the trade count moves. The optionsDX
    pre-registration fixes the boundary at 2017-01-01 specifically so that
    Volmageddon, Q4 2018, COVID and the 2022 bear land in the holdout. A
    fractional cut cannot express that, and a boundary that drifts with the
    data is not a pre-registered boundary.
    """

    def test_the_boundary_date_belongs_to_the_holdout(self):
        trades = [_T("2016-12-31", roc=0.1), _T("2017-01-01", roc=0.1)]
        train, test = split_at_date(trades, "2017-01-01")
        self.assertEqual(len(train), 1)
        self.assertEqual(len(test), 1)
        self.assertEqual(train[0].entry_date, "2016-12-31")

    def test_it_splits_on_entry_date_not_on_position(self):
        # Fed out of order, the split must still be chronological.
        trades = [_T("2019-05-01", roc=0.1), _T("2011-05-01", roc=0.1),
                  _T("2018-05-01", roc=0.1)]
        train, test = split_at_date(trades, "2017-01-01")
        self.assertEqual([t.entry_date for t in train], ["2011-05-01"])
        self.assertEqual([t.entry_date for t in test],
                         ["2018-05-01", "2019-05-01"])

    def test_an_all_in_sample_window_yields_an_empty_holdout(self):
        train, test = split_at_date([_T("2011-01-01", roc=0.1)], "2017-01-01")
        self.assertEqual(len(train), 1)
        self.assertEqual(test, [])

    def test_only_closed_trades_are_split(self):
        # An open trade has no outcome to attribute, and counting it in the
        # window sizes would misreport how much evidence each half holds.
        opened = _T("2011-01-01", roc=0.1)
        opened.exit_date = None
        train, test = split_at_date([opened, _T("2011-02-01", roc=0.1)],
                                    "2017-01-01")
        self.assertEqual(len(train), 1)
        self.assertEqual(test, [])


class BenjaminiHochbergTest(unittest.TestCase):
    """The correction for the size of the search that produced the winner.

    `n_trials` has been carried on every row since the ranking was written and
    nothing ever divided by it, so an 18-way feature sweep has been reported at
    raw p throughout. That is the difference between "this feature is
    significant" and "the best of eighteen features looks significant", and on
    a book whose findings keep failing their holdout it is the likeliest
    remaining source of them.
    """

    def test_a_single_test_is_its_own_q_value(self):
        # With a family of one there is no search to correct for.
        self.assertEqual(benjamini_hochberg([0.03]), [0.03])

    def test_the_best_of_a_large_family_is_deflated_by_the_family_size(self):
        # p=0.001 found by looking at 18 features is not a p=0.001 finding.
        qs = benjamini_hochberg([0.001] + [0.5] * 17)
        self.assertAlmostEqual(qs[0], 0.018, places=6)

    def test_q_never_falls_as_p_rises(self):
        # BH is a step-UP procedure: the enforced monotonicity is what stops a
        # middling p-value scoring better than a smaller one.
        qs = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
        self.assertEqual(qs, sorted(qs))

    def test_q_is_capped_at_one(self):
        self.assertTrue(all(q <= 1.0 for q in benjamini_hochberg([0.9] * 10)))

    def test_input_order_is_preserved(self):
        qs = benjamini_hochberg([0.5, 0.001, 0.5])
        self.assertLess(qs[1], qs[0])
        self.assertEqual(qs[0], qs[2])

    def test_an_unmeasurable_feature_does_not_inflate_the_family(self):
        # A feature that could not be measured was not a trial. Counting it
        # would penalise the survivors for a test that never ran.
        self.assertEqual(benjamini_hochberg([0.02, None]), [0.02, None])

    def test_rank_features_reports_a_q_value(self):
        rows = rank_features(_monotone(noise=0.3), ["iv_rank", "other"])
        self.assertIn("q", rows[0])

    def test_the_formatted_table_shows_q(self):
        # "q(BH)" specifically — "q" alone would pass on the pre-existing
        # q_bot/q_top/q_spread columns without the correction being shown.
        text = format_ranking(rank_features(_monotone(), ["iv_rank"]))
        self.assertIn("q(BH)", text)
