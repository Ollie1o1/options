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

from src.alloc.attribution import (feature_ic, bucket_table, rank_features,
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


if __name__ == "__main__":
    unittest.main()
