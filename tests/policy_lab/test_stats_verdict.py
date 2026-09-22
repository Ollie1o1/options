"""The promotion bar.

`policy_verdict` gates on a max-statistic sign-flip permutation test
(`fwer_p`), not on the deflated Sharpe (`dsr`) or PBO. `dsr` requires
`effective_n` — a count of non-overlapping holding intervals — and on this
corpus that count is 9 for every strategy regardless of row count, because it
is a function of the corpus's 26-day window and the 3-11 day holding period.
At n_eff=9, `deflated_sharpe` returns 0.917 with a single hypothesis and no
multiple-testing penalty at all, so the brief's `dsr >= 0.95` bar is
unreachable by construction and must not ship as a gate. `dsr` and `pbo` are
still carried on `PolicyResult` and printed as diagnostics so that
limitation stays visible.

The leave-one-symbol-out condition is the MU/AMD lesson encoded as a gate —
Bull Put's apparent edge collapsed from t=3.48 to t=1.61 when two
semiconductor names were dropped.
"""
import unittest

import numpy as np
import pandas as pd

from src.policy_lab.stats import (
    PolicyResult, embargo_purge, family_wise_p, leave_one_symbol_out_stable,
    policy_verdict,
)
from src.walk_forward import Trade


def _trade(rowid, entry, exit_):
    return Trade(rowid=rowid, entry_date=entry, exit_date=exit_,
                 pnl_pct=0.0, components=np.zeros(1))


def _result(**over):
    base = dict(
        policy_name="sp_tp50", n_clusters=120,
        mean_d_cross=0.03, mean_d_surface=0.02,
        ci_lo=0.01, ci_hi=0.05,
        dsr=0.97, pbo=0.10,
        fwer_p=0.01, skew_d=0.0,
        beats_all_nulls=True, corpus_b_sign_agrees=True,
        loso_stable=True, variance_ratio=0.3,
    )
    base.update(over)
    return PolicyResult(**base)


class TestEmbargo(unittest.TestCase):
    def test_embargo_drops_trades_inside_the_gap_after_the_test_window(self):
        test = [_trade(1, "2026-08-10", "2026-08-15")]
        train = [_trade(2, "2026-08-17", "2026-08-18"),   # inside 5d embargo
                 _trade(3, "2026-08-25", "2026-08-26")]   # outside
        kept = embargo_purge(train, test, embargo_days=5)
        self.assertEqual([t.rowid for t in kept], [3])

    def test_zero_embargo_matches_plain_purging(self):
        test = [_trade(1, "2026-08-10", "2026-08-15")]
        train = [_trade(2, "2026-08-16", "2026-08-17")]
        self.assertEqual(len(embargo_purge(train, test, embargo_days=0)), 1)

    def test_empty_test_block_keeps_everything(self):
        train = [_trade(2, "2026-08-16", "2026-08-17")]
        self.assertEqual(len(embargo_purge(train, [], embargo_days=5)), 1)

    def test_keeps_a_training_trade_entirely_before_the_test_window(self):
        # Nothing before the test window needs an embargo gap: only the tape
        # AFTER the test block is still autocorrelated with it.
        test = [_trade(1, "2026-08-10", "2026-08-15")]
        train = [_trade(2, "2026-08-01", "2026-08-05")]
        kept = embargo_purge(train, test, embargo_days=5)
        self.assertEqual([t.rowid for t in kept], [2])


class TestLeaveOneSymbolOut(unittest.TestCase):
    def test_stable_when_no_single_symbol_carries_the_result(self):
        df = pd.DataFrame({"symbol": ["A", "B", "C", "D"] * 5,
                           "d": [0.05, 0.04, 0.06, 0.05] * 5})
        self.assertTrue(leave_one_symbol_out_stable(df))

    def test_unstable_when_one_symbol_carries_the_sign(self):
        # Bull Put / MU+AMD in miniature.
        df = pd.DataFrame({"symbol": ["MU"] * 5 + ["B", "C", "D"] * 5,
                           "d": [2.0] * 5 + [-0.05] * 15})
        self.assertFalse(leave_one_symbol_out_stable(df))

    def test_single_symbol_is_never_stable(self):
        df = pd.DataFrame({"symbol": ["A"] * 10, "d": [0.1] * 10})
        self.assertFalse(leave_one_symbol_out_stable(df))


class TestFamilyWiseP(unittest.TestCase):
    def test_pure_noise_grid_finds_nothing(self):
        rng = np.random.default_rng(42)
        # 50 policies x 100 clusters, mean-zero noise: nothing should survive
        # the family-wise correction.
        arr = rng.normal(loc=0.0, scale=1.0, size=(50, 100))
        p = family_wise_p(arr, n_perm=500, seed=1)
        self.assertGreater(p, 0.05)

    def test_planted_effect_is_detected(self):
        rng = np.random.default_rng(42)
        arr = rng.normal(loc=0.0, scale=0.05, size=(50, 200))
        arr[0, :] += 1.0  # policy 0 has a huge, obvious effect
        p = family_wise_p(arr, n_perm=500, seed=1)
        self.assertLess(p, 0.05)

    def test_p_is_a_fraction(self):
        rng = np.random.default_rng(0)
        arr = rng.normal(size=(10, 30))
        p = family_wise_p(arr, n_perm=200, seed=2)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_deterministic_given_seed(self):
        arr = np.array([[0.01, -0.02, 0.03, 0.01], [0.5, 0.4, 0.6, 0.55]])
        p1 = family_wise_p(arr, n_perm=300, seed=7)
        p2 = family_wise_p(arr, n_perm=300, seed=7)
        self.assertEqual(p1, p2)


class TestPolicyVerdict(unittest.TestCase):
    def test_all_conditions_met_promotes(self):
        self.assertEqual(policy_verdict(_result()), "promote")

    def test_ci_containing_zero_rejects(self):
        self.assertEqual(policy_verdict(_result(ci_lo=-0.01)), "reject")

    def test_negative_under_surface_costs_rejects(self):
        self.assertEqual(policy_verdict(_result(mean_d_surface=-0.01)), "reject")

    def test_high_fwer_p_rejects(self):
        self.assertEqual(policy_verdict(_result(fwer_p=0.20)), "reject")

    def test_terrible_dsr_but_good_fwer_p_still_promotes(self):
        # The whole point of dropping the DSR gate: n_eff=9 makes dsr
        # unreachable on this corpus, so a bad dsr must not block promotion
        # when the permutation test (the real gate) passes.
        self.assertEqual(
            policy_verdict(_result(dsr=0.05, fwer_p=0.01)), "promote")

    def test_bad_pbo_no_longer_gates(self):
        # PBO is still carried as a diagnostic but is not a gate here.
        self.assertEqual(policy_verdict(_result(pbo=0.99)), "promote")

    def test_failing_a_null_rejects(self):
        self.assertEqual(policy_verdict(_result(beats_all_nulls=False)),
                         "reject")

    def test_corpus_b_sign_disagreement_rejects(self):
        self.assertEqual(policy_verdict(_result(corpus_b_sign_agrees=False)),
                         "reject")

    def test_leave_one_symbol_out_instability_rejects(self):
        self.assertEqual(policy_verdict(_result(loso_stable=False)), "reject")

    def test_too_few_clusters_is_insufficient_not_reject(self):
        self.assertEqual(policy_verdict(_result(n_clusters=12)), "insufficient")

    def test_ineffective_pairing_is_underpowered_not_reject(self):
        self.assertEqual(policy_verdict(_result(variance_ratio=0.8)),
                         "underpowered")

    def test_underpowered_outranks_other_failures(self):
        # If the pairing did not work, no other verdict is trustworthy.
        self.assertEqual(policy_verdict(_result(variance_ratio=0.9, dsr=0.1)),
                         "underpowered")


if __name__ == "__main__":
    unittest.main()
