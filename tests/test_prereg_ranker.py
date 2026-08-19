"""Tests for src/prereg_ranker.py — the pre-registered ranker statistics.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_prereg_ranker -v

Every statistic is checked against a frame with a KNOWN answer. A statistics
module nobody has pointed at a known answer is a statistics module nobody has
tested.
"""
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


if __name__ == "__main__":
    unittest.main()
