"""Which structure the book takes next, and why it is not a fixed list.

A name-based allowlist is self-sealing. Under `allowed_strategies: ["Bull
Put"]` no other structure can enter, so no other structure can accumulate
evidence, so the rule that cited absence of evidence guarantees that absence
permanently. Bear Call, Short Put and Iron Condor last entered on 2026-07-30
and 2026-07-31; Long Put on 2026-07-13. Nothing since.

The measurement does favour Bull Put — P(highest true mean return on capital
at risk) = 99.0%, and 99.3% with 45-day decay. So exploration here is not a
statistical free lunch that the posterior asks for. It is a PURCHASE: paying
known expected return for information about structures whose evidence is
going stale. `information_cost` prices it, so the rate is set with its cost
visible rather than by taste.

The draw is deterministic in the candidate's own key, for the same reason
entry selection is: a decision that cannot be replayed cannot be audited.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src import strategy_allocation as sa

ELIGIBLE = ["Bull Put", "Bear Call", "Iron Condor", "Long Call", "Long Put",
            "Short Put"]


def _book(seed=2):
    """A book where one structure is genuinely best and the rest are not."""
    rng = np.random.default_rng(seed)
    means = {"Bull Put": 0.17, "Bear Call": -0.05, "Iron Condor": -0.05,
             "Long Call": 0.00, "Long Put": -0.05, "Short Put": 0.01}
    rows = []
    for s, m in means.items():
        for i in range(120):
            rows.append({
                "strategy": s,
                "entry_date": (pd.to_datetime("2026-05-01")
                               + pd.Timedelta(days=i // 3)).strftime("%Y-%m-%d"),
                "ret_on_risk": float(rng.normal(m, 0.5)),
            })
    return pd.DataFrame(rows)


class TestPosterior(unittest.TestCase):

    def test_the_better_structure_carries_most_of_the_posterior(self):
        p = sa.p_best(sa.posteriors(_book(), as_of="2026-08-24"))
        self.assertGreater(p["Bull Put"], 0.80)

    def test_stale_evidence_widens_rather_than_disappears(self):
        """A structure not traded for months is not more certain, it is less.
        Decay must lower the effective sample size, never the row count."""
        df = _book()
        fresh = sa.effective_n(df, as_of="2026-06-01", half_life_days=45)
        stale = sa.effective_n(df, as_of="2027-06-01", half_life_days=45)
        self.assertLess(stale["Bull Put"], fresh["Bull Put"])
        self.assertGreater(stale["Bull Put"], 0.0)


class TestAllocation(unittest.TestCase):

    def test_weights_are_a_distribution_over_eligible_structures(self):
        a = sa.allocate(_book(), ELIGIBLE, explore_rate=0.25,
                        as_of="2026-08-24")
        self.assertAlmostEqual(sum(a.weights.values()), 1.0, places=9)
        self.assertEqual(set(a.weights), set(ELIGIBLE))
        self.assertTrue(all(v >= 0 for v in a.weights.values()))

    def test_no_exploration_concentrates_on_the_posterior_best(self):
        a = sa.allocate(_book(), ELIGIBLE, explore_rate=0.0,
                        as_of="2026-08-24")
        self.assertGreater(a.weights["Bull Put"], 0.80)

    def test_full_exploration_is_uniform(self):
        a = sa.allocate(_book(), ELIGIBLE, explore_rate=1.0,
                        as_of="2026-08-24")
        for s in ELIGIBLE:
            self.assertAlmostEqual(a.weights[s], 1.0 / len(ELIGIBLE), places=9)

    def test_every_eligible_structure_keeps_a_way_back_in(self):
        """The anti-self-sealing property. With exploration on, a structure
        the posterior has written off must still reach the book."""
        a = sa.allocate(_book(), ELIGIBLE, explore_rate=0.25,
                        as_of="2026-08-24")
        for s in ELIGIBLE:
            self.assertGreater(a.weights[s], 0.0, f"{s} can never trade again")

    def test_a_structure_with_no_history_is_still_reachable(self):
        a = sa.allocate(_book(), ELIGIBLE + ["Jade Lizard"],
                        explore_rate=0.25, as_of="2026-08-24")
        self.assertGreater(a.weights["Jade Lizard"], 0.0)

    def test_an_ineligible_structure_never_gets_weight(self):
        """Exploration widens the choice; it does not remove the safety rail."""
        a = sa.allocate(_book(), ["Bull Put", "Long Call"], explore_rate=1.0,
                        as_of="2026-08-24")
        self.assertNotIn("Iron Condor", a.weights)

    def test_no_eligible_structures_yields_no_weights(self):
        a = sa.allocate(_book(), [], explore_rate=0.25, as_of="2026-08-24")
        self.assertEqual(a.weights, {})


class TestAdmission(unittest.TestCase):

    def test_the_same_candidate_always_gets_the_same_answer(self):
        a = sa.allocate(_book(), ELIGIBLE, explore_rate=0.25,
                        as_of="2026-08-24")
        first = [sa.admits(a, "Long Call", f"scan-{i}") for i in range(50)]
        again = [sa.admits(a, "Long Call", f"scan-{i}") for i in range(50)]
        self.assertEqual(first, again)

    def test_admission_rates_track_the_target_weights(self):
        a = sa.allocate(_book(), ELIGIBLE, explore_rate=0.30,
                        as_of="2026-08-24")
        for s in ("Bull Put", "Long Call"):
            hits = sum(sa.admits(a, s, f"k{i}") for i in range(4000))
            self.assertAlmostEqual(hits / 4000, a.weights[s], delta=0.03)

    def test_an_ineligible_structure_is_never_admitted(self):
        a = sa.allocate(_book(), ["Bull Put"], explore_rate=0.5,
                        as_of="2026-08-24")
        self.assertFalse(any(sa.admits(a, "Iron Condor", f"k{i}")
                             for i in range(500)))

    def test_a_different_key_can_give_a_different_answer(self):
        """Deterministic must not mean constant, or the share is never met."""
        a = sa.allocate(_book(), ELIGIBLE, explore_rate=0.30,
                        as_of="2026-08-24")
        got = {sa.admits(a, "Long Call", f"k{i}") for i in range(200)}
        self.assertEqual(got, {True, False})


class TestInformationCost(unittest.TestCase):

    def test_no_exploration_costs_almost_nothing(self):
        """Not exactly nothing: at rate 0 the policy is still Thompson, not
        greedy, so the ~0.5% of posterior mass sitting off the best structure
        is a real cost. Greedy would be free and would also never learn."""
        df = _book()
        a = sa.allocate(df, ELIGIBLE, explore_rate=0.0, as_of="2026-08-24")
        cost = sa.information_cost(a, df)
        self.assertGreaterEqual(cost, 0.0)
        self.assertLess(cost, 0.01)

    def test_exploration_costs_the_gap_to_the_best_structure(self):
        df = _book()
        a = sa.allocate(df, ELIGIBLE, explore_rate=0.25, as_of="2026-08-24")
        cost = sa.information_cost(a, df)
        self.assertGreater(cost, 0.0)
        # Cannot exceed the spread between best and worst mean return.
        means = df.groupby("strategy")["ret_on_risk"].mean()
        self.assertLess(cost, float(means.max() - means.min()))

    def test_more_exploration_costs_more(self):
        df = _book()
        a = sa.allocate(df, ELIGIBLE, explore_rate=0.10, as_of="2026-08-24")
        b = sa.allocate(df, ELIGIBLE, explore_rate=0.40, as_of="2026-08-24")
        self.assertGreater(sa.information_cost(b, df),
                           sa.information_cost(a, df))


class TestAutoLogIntegration(unittest.TestCase):
    """The allowlist is replaced, not bypassed. Every other gate still runs.

    A test that named the real config would rewrite the operator's book
    settings; each case here writes its own.
    """

    def setUp(self):
        from src import options_screener
        self.os_ = options_screener
        options_screener._allocation_cache_clear()

    def tearDown(self):
        self.os_._allocation_cache_clear()

    def _cfg(self, d, **allocation):
        import json, os
        cfg = {"auto_log": {
            "allowed_strategies": ["Bull Put"],
            "paper_only_strategies": [],
            "cohort_min_dte": 30,
        }}
        if allocation:
            cfg["auto_log"]["allocation"] = allocation
        path = os.path.join(d, "config.json")
        with open(path, "w") as fh:
            json.dump(cfg, fh)
        return path

    def _trade(self, strategy, key="k1", dte=45):
        import datetime
        exp = (datetime.date.today()
               + datetime.timedelta(days=dte)).strftime("%Y-%m-%d")
        return {"strategy_name": strategy, "expiration": exp,
                "contract_key": key, "symbol": "NVDA", "strike": 140.0}

    def test_without_allocation_the_old_allowlist_is_unchanged(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cfg = self._cfg(d)
            self.assertEqual(
                self.os_.apply_auto_log_allowlist(self._trade("Bull Put"), cfg),
                ("insert", 0))
            self.assertEqual(
                self.os_.apply_auto_log_allowlist(self._trade("Long Call"), cfg),
                ("drop", None))

    def test_with_allocation_a_long_call_can_reach_the_book(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cfg = self._cfg(d, enabled=True, explore_rate=1.0,
                            eligible_strategies=["Bull Put", "Long Call"])
            got = [self.os_.apply_auto_log_allowlist(
                       self._trade("Long Call", f"k{i}"), cfg)[0]
                   for i in range(200)]
            self.assertIn("insert", got,
                          "no Long Call ever reached the book — still an "
                          "allowlist")
            self.assertIn("drop", got, "every candidate was taken — the share "
                                       "is not being applied")

    def test_an_ineligible_structure_is_still_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cfg = self._cfg(d, enabled=True, explore_rate=1.0,
                            eligible_strategies=["Bull Put"])
            got = {self.os_.apply_auto_log_allowlist(
                       self._trade("Iron Condor", f"k{i}"), cfg)[0]
                   for i in range(200)}
            self.assertEqual(got, {"drop"})

    def test_the_long_premium_horizon_floor_still_applies(self):
        """A Long Call admitted by allocation but below the DTE floor is
        logged paper_only=1, exactly as the allowlist path does it."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cfg = self._cfg(d, enabled=True, explore_rate=1.0,
                            eligible_strategies=["Long Call"])
            flags = [self.os_.apply_auto_log_allowlist(
                         self._trade("Long Call", f"k{i}", dte=5), cfg)
                     for i in range(50)]
            inserted = [f for d_, f in flags if d_ == "insert"]
            self.assertTrue(inserted, "nothing was admitted to test")
            self.assertTrue(all(f == 1 for f in inserted),
                            "a short-dated long call skipped the floor")

    def test_the_decision_replays(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cfg = self._cfg(d, enabled=True, explore_rate=0.5,
                            eligible_strategies=["Bull Put", "Long Call"])
            first = [self.os_.apply_auto_log_allowlist(
                         self._trade("Long Call", f"k{i}"), cfg)[0]
                     for i in range(60)]
            self.os_._allocation_cache_clear()
            again = [self.os_.apply_auto_log_allowlist(
                         self._trade("Long Call", f"k{i}"), cfg)[0]
                     for i in range(60)]
            self.assertEqual(first, again)


if __name__ == "__main__":
    unittest.main()
