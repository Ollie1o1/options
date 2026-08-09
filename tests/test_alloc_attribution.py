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

from src.alloc.attribution import (disaster_auc, feature_ic, bucket_table,
                                   format_ranking, monotonicity,
                                   rank_features, split_by_time)


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
