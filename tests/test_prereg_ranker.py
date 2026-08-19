"""Tests for src/prereg_ranker.py — the pre-registered ranker statistics.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_prereg_ranker -v

Every statistic is checked against a frame with a KNOWN answer. A statistics
module nobody has pointed at a known answer is a statistics module nobody has
tested.
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src import prereg_ranker as pk


def _planted(rho, n=4000, cells=20, seed=1):
    """A frame whose within-cell rank correlation is approximately `rho`."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rho * x + np.sqrt(max(0.0, 1 - rho ** 2)) * rng.normal(size=n)
    return pd.DataFrame({
        "feature": x, "outcome": y,
        "entry_date": [f"2026-08-{1 + (i % cells):02d}" for i in range(n)],
        "strategy": ["Long Call"] * n,
        "contract_key": [f"K{i}" for i in range(n)],
    })


class TestRankIC(unittest.TestCase):
    def test_a_planted_correlation_is_recovered(self):
        ic = pk.rank_ic(_planted(0.30), "feature", "outcome",
                        ["entry_date", "strategy"])
        self.assertAlmostEqual(ic, 0.30, delta=0.05)

    def test_no_relationship_reads_near_zero(self):
        ic = pk.rank_ic(_planted(0.0), "feature", "outcome",
                        ["entry_date", "strategy"])
        self.assertAlmostEqual(ic, 0.0, delta=0.05)

    def test_a_negative_relationship_keeps_its_sign(self):
        ic = pk.rank_ic(_planted(-0.30), "feature", "outcome",
                        ["entry_date", "strategy"])
        self.assertAlmostEqual(ic, -0.30, delta=0.05)

    def test_cells_below_the_minimum_are_dropped(self):
        df = pd.DataFrame({
            "feature": [1.0, 2.0, 3.0, 4.0, 9.0, 8.0],
            "outcome": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
            "entry_date": ["d1"] * 4 + ["d2"] * 2,     # d2 has only 2 rows
            "strategy": ["Long Call"] * 6,
            "contract_key": [f"K{i}" for i in range(6)],
        })
        out = pk.demeaned_ranks(df, ["feature", "outcome"],
                                ["entry_date", "strategy"])
        self.assertEqual(len(out), 4)
        self.assertEqual(set(out["entry_date"]), {"d1"})

    def test_an_empty_frame_returns_none_rather_than_raising(self):
        empty = pd.DataFrame(columns=["feature", "outcome", "entry_date",
                                      "strategy", "contract_key"])
        self.assertIsNone(pk.rank_ic(empty, "feature", "outcome",
                                     ["entry_date", "strategy"]))


class TestSimpsonsParadox(unittest.TestCase):
    """The reason cell demeaning exists. On the live book the carry feature
    showed a whole-book Spearman of +0.104 that reversed inside strategies
    (Iron Condor -0.282). Pooling without cells reproduces that artifact."""

    def _two_strategies(self):
        rng = np.random.default_rng(7)
        rows = []
        # Both strategies have a NEGATIVE within-strategy relationship, but sit
        # at different levels on both axes, which manufactures a positive
        # pooled correlation.
        for strat, fbase, obase in (("Long Call", 0.0, 0.0),
                                    ("Iron Condor", 10.0, 10.0)):
            x = rng.normal(size=1500)
            y = -0.4 * x + np.sqrt(1 - 0.16) * rng.normal(size=1500)
            for i in range(1500):
                rows.append({"feature": fbase + x[i], "outcome": obase + y[i],
                             "entry_date": f"2026-08-{1 + (i % 20):02d}",
                             "strategy": strat,
                             "contract_key": f"{strat}-{i}"})
        return pd.DataFrame(rows)

    def test_pooling_without_cells_shows_the_artifact(self):
        from scipy.stats import spearmanr
        df = self._two_strategies()
        naive, _ = spearmanr(df["feature"], df["outcome"])
        self.assertGreater(naive, 0.2)      # the artifact, positive

    def test_cell_demeaning_recovers_the_true_negative_sign(self):
        df = self._two_strategies()
        ic = pk.rank_ic(df, "feature", "outcome", ["entry_date", "strategy"])
        self.assertLess(ic, -0.2)           # the truth, negative


class TestClusterBootstrap(unittest.TestCase):
    def _clustered(self, rho, clusters=200, per=10, seed=3):
        """Every cluster's rows share an outcome shock, so rows within a
        cluster are far from independent."""
        rng = np.random.default_rng(seed)
        rows = []
        for c in range(clusters):
            shock = rng.normal() * 2.0
            for j in range(per):
                x = rng.normal()
                rows.append({"feature": x,
                             "outcome": rho * x + shock + rng.normal(),
                             "entry_date": f"2026-08-{1 + (j % 10):02d}",
                             "strategy": "Long Call",
                             "contract_key": f"K{c}"})
        return pd.DataFrame(rows)

    def test_the_interval_brackets_a_planted_effect(self):
        df = self._clustered(0.30)
        lo, hi = pk.cluster_bootstrap_ci(df, "feature", "outcome",
                                         ["entry_date", "strategy"],
                                         "contract_key", n_boot=400, seed=1)
        ic = pk.rank_ic(df, "feature", "outcome", ["entry_date", "strategy"])
        self.assertLess(lo, ic)
        self.assertGreater(hi, ic)

    def test_a_null_frame_gives_an_interval_containing_zero(self):
        df = self._clustered(0.0)
        lo, hi = pk.cluster_bootstrap_ci(df, "feature", "outcome",
                                         ["entry_date", "strategy"],
                                         "contract_key", n_boot=400, seed=1)
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def _repeated_contracts(self, clusters=100, per=10, seed=3):
        """The shape this actually guards against: one contract recorded on
        many scans, its feature and outcome barely moving between sightings,
        so the repeats are near-duplicates rather than new information.

        A shock that moves only the outcome LEVEL is the wrong test — cell
        demeaning removes levels by construction, so it would show no effect.
        """
        rng = np.random.default_rng(seed)
        rows = []
        for c in range(clusters):
            f_c = rng.normal()
            o_c = 0.15 * f_c + rng.normal()
            for j in range(per):
                rows.append({"feature": f_c + rng.normal() * 0.05,
                             "outcome": o_c + rng.normal() * 0.05,
                             "entry_date": f"2026-08-{1 + (j % 10):02d}",
                             "strategy": "Long Call",
                             "contract_key": f"K{c}"})
        return pd.DataFrame(rows)

    def test_ignoring_clustering_manufactures_significance(self):
        # The consequence, not merely the width: treating repeats as
        # independent produces an interval that EXCLUDES zero on data where
        # the honest interval includes it.
        df = self._repeated_contracts()
        lo_c, hi_c = pk.cluster_bootstrap_ci(
            df, "feature", "outcome", ["entry_date", "strategy"],
            "contract_key", n_boot=400, seed=1)

        df2 = df.copy()
        df2["contract_key"] = [f"R{i}" for i in range(len(df2))]
        lo_i, hi_i = pk.cluster_bootstrap_ci(
            df2, "feature", "outcome", ["entry_date", "strategy"],
            "contract_key", n_boot=400, seed=1)

        self.assertGreater(hi_c - lo_c, (hi_i - lo_i) * 2)   # far wider
        self.assertGreater(lo_i, 0.0)                        # iid: "significant"
        self.assertLess(lo_c, 0.0)                           # honest: not

    def test_the_same_seed_reproduces_the_interval(self):
        df = self._clustered(0.20)
        a = pk.cluster_bootstrap_ci(df, "feature", "outcome",
                                    ["entry_date", "strategy"],
                                    "contract_key", n_boot=200, seed=42)
        b = pk.cluster_bootstrap_ci(df, "feature", "outcome",
                                    ["entry_date", "strategy"],
                                    "contract_key", n_boot=200, seed=42)
        self.assertEqual(a, b)

    def test_an_empty_frame_returns_no_interval(self):
        empty = pd.DataFrame(columns=["feature", "outcome", "entry_date",
                                      "strategy", "contract_key"])
        self.assertEqual(pk.cluster_bootstrap_ci(
            empty, "feature", "outcome", ["entry_date", "strategy"],
            "contract_key", n_boot=10, seed=1), (None, None))


class TestPowerArithmetic(unittest.TestCase):
    def test_required_n_matches_the_fisher_z_formula(self):
        # ((1.959964 + 0.8416212) / atanh(rho))**2 + 3
        self.assertAlmostEqual(pk.required_effective_n(0.08), 1224, delta=2)
        self.assertAlmostEqual(pk.required_effective_n(0.10), 783, delta=2)
        self.assertAlmostEqual(pk.required_effective_n(0.15), 347, delta=2)

    def test_a_smaller_effect_needs_more_data(self):
        self.assertGreater(pk.required_effective_n(0.05),
                           pk.required_effective_n(0.20))

    def test_an_impossible_target_is_rejected(self):
        with self.assertRaises(ValueError):
            pk.required_effective_n(0.0)
        with self.assertRaises(ValueError):
            pk.required_effective_n(1.0)

    def test_icc_is_near_zero_when_clusters_carry_no_signal(self):
        rng = np.random.default_rng(11)
        df = pd.DataFrame({"outcome": rng.normal(size=2000),
                           "contract_key": [f"K{i // 5}" for i in range(2000)]})
        self.assertLess(abs(pk.icc_oneway(df, "outcome", "contract_key")), 0.06)

    def test_icc_is_high_when_clusters_dominate(self):
        rng = np.random.default_rng(12)
        rows = []
        for c in range(400):
            shock = rng.normal() * 5.0
            for _ in range(5):
                rows.append({"outcome": shock + rng.normal() * 0.5,
                             "contract_key": f"K{c}"})
        df = pd.DataFrame(rows)
        self.assertGreater(pk.icc_oneway(df, "outcome", "contract_key"), 0.8)

    def test_design_effect_follows_the_formula(self):
        rng = np.random.default_rng(13)
        rows = []
        for c in range(400):
            shock = rng.normal() * 5.0
            for _ in range(5):
                rows.append({"outcome": shock + rng.normal() * 0.5,
                             "contract_key": f"K{c}"})
        df = pd.DataFrame(rows)
        icc = pk.icc_oneway(df, "outcome", "contract_key")
        self.assertAlmostEqual(pk.design_effect(df, "outcome", "contract_key"),
                               1 + (5 - 1) * icc, delta=0.05)

    def test_effective_n_divides_by_the_design_effect(self):
        self.assertAlmostEqual(pk.effective_n(1000, 2.0), 500.0)

    def test_singleton_clusters_give_a_design_effect_of_one(self):
        df = pd.DataFrame({"outcome": [1.0, 2.0, 3.0, 4.0],
                           "contract_key": ["a", "b", "c", "d"]})
        self.assertAlmostEqual(pk.design_effect(df, "outcome", "contract_key"),
                               1.0, delta=0.01)


class TestNegativeControl(unittest.TestCase):
    """Tests the test. A pipeline bug that manufactures signal is otherwise
    indistinguishable from a finding."""

    def test_shuffling_destroys_a_planted_effect(self):
        df = _planted(0.40, n=2000, cells=10, seed=5)
        out = pk.negative_control(df, "feature", "outcome",
                                  ["entry_date", "strategy"],
                                  n_shuffles=100, seed=1)
        self.assertGreater(abs(out["observed"]), 0.3)   # the real effect
        self.assertLess(abs(out["mean"]), 0.05)         # shuffled: nothing

    def test_the_null_band_brackets_zero(self):
        df = _planted(0.0, n=2000, cells=10, seed=6)
        out = pk.negative_control(df, "feature", "outcome",
                                  ["entry_date", "strategy"],
                                  n_shuffles=100, seed=1)
        self.assertLess(abs(out["observed"]), out["p95_abs"] + 0.05)

    def test_the_same_seed_reproduces_the_control(self):
        df = _planted(0.2, n=1000, cells=10, seed=7)
        a = pk.negative_control(df, "feature", "outcome",
                                ["entry_date", "strategy"],
                                n_shuffles=50, seed=9)
        b = pk.negative_control(df, "feature", "outcome",
                                ["entry_date", "strategy"],
                                n_shuffles=50, seed=9)
        self.assertEqual(a, b)


class TestHalfSamples(unittest.TestCase):
    def test_a_consistent_effect_keeps_its_sign_in_both_halves(self):
        df = _planted(0.35, n=3000, cells=30, seed=8)
        first, second = pk.half_sample_ics(df, "feature", "outcome",
                                           ["entry_date", "strategy"],
                                           "entry_date")
        self.assertGreater(first, 0.15)
        self.assertGreater(second, 0.15)

    def test_a_frame_with_one_date_has_no_second_half(self):
        df = _planted(0.3, n=200, cells=1, seed=9)
        first, second = pk.half_sample_ics(df, "feature", "outcome",
                                           ["entry_date", "strategy"],
                                           "entry_date")
        self.assertIsNotNone(first)
        self.assertIsNone(second)


def _seed_db(path, rows):
    """Rows shaped as candidates + candidate_positions, via the real schema."""
    from src import candidate_marks as cm
    with cm.connect(path) as conn:
        for r in rows:
            conn.execute(
                "INSERT OR REPLACE INTO candidates (scan_id, ts, board, mode,"
                " contract_key, symbol, opt_type, expiration, ev_net,"
                " quality_score, theta, premium, delta, gate_passed)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (r["scan_id"], "2026-08-19T00:00:00+00:00", "b", r["mode"],
                 r["contract_key"], "AAPL", r["opt_type"], "2026-09-18",
                 r["ev_net"], r["quality_score"], -0.05, 10.0, 0.5,
                 r["gate_passed"]))
            conn.execute(
                "INSERT OR REPLACE INTO candidate_positions (scan_id, board,"
                " contract_key, family, entry_date, entry_price, status,"
                " pnl_pct) VALUES (?,?,?,?,?,?,?,?)",
                (r["scan_id"], "b", r["contract_key"], "long_option",
                 r["entry_date"], -10.0, r["status"], r["pnl_pct"]))
        conn.commit()


def _row(**over):
    row = {"scan_id": "S1", "contract_key": "K1", "mode": "Discovery scan",
           "opt_type": "call", "ev_net": 25.0, "quality_score": 0.5,
           "gate_passed": 1, "entry_date": "2026-08-19", "status": "CLOSED",
           "pnl_pct": 0.2}
    row.update(over)
    return row


class TestLoadCohort(unittest.TestCase):
    def test_only_closed_survivors_with_a_pnl_are_loaded(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            _seed_db(path, [
                _row(scan_id="A", contract_key="K1"),                    # in
                _row(scan_id="B", contract_key="K2", status="OPEN"),     # out
                _row(scan_id="C", contract_key="K3", pnl_pct=None),      # out
                _row(scan_id="D", contract_key="K4", gate_passed=0),     # out
            ])
            df = pk.load_cohort(path)
            self.assertEqual(len(df), 1)
            self.assertEqual(df["contract_key"].iloc[0], "K1")

    def test_the_strategy_is_derived_not_the_family(self):
        # Long Call and Long Put must not collapse into `long_option`: a call
        # and a put on the same day are not exchangeable.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            _seed_db(path, [
                _row(scan_id="A", contract_key="K1", opt_type="call"),
                _row(scan_id="B", contract_key="K2", opt_type="put"),
            ])
            df = pk.load_cohort(path)
            self.assertEqual(set(df["strategy"]), {"Long Call", "Long Put"})

    def test_carry_is_computed_from_theta_and_premium(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            _seed_db(path, [_row()])
            df = pk.load_cohort(path)
            self.assertAlmostEqual(df["carry"].iloc[0], 0.05 / 10.0)

    def test_refused_rows_load_when_survivors_only_is_off(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            _seed_db(path, [_row(scan_id="A", contract_key="K1", gate_passed=0)])
            self.assertEqual(len(pk.load_cohort(path)), 0)
            self.assertEqual(len(pk.load_cohort(path, survivors_only=False)), 1)

    def test_a_missing_database_returns_an_empty_frame(self):
        with tempfile.TemporaryDirectory() as d:
            df = pk.load_cohort(os.path.join(d, "absent.db"))
            self.assertEqual(len(df), 0)
            self.assertIn("pnl_pct", df.columns)


if __name__ == "__main__":
    unittest.main()
