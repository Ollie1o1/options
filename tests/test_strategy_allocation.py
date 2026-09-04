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

import os
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

    def test_an_empty_book_allocates_NOTHING_rather_than_everything(self):
        """The bug CI caught (PR #63). With no ledger there are no posteriors,
        and the weights fell through to UNIFORM — 1/4 each across the four
        eligible structures, silently widening the book to trade everything
        equally, including structures with no evidence at all. Uniform is the
        most dangerous default available: it is maximum exposure justified by
        zero information. No evidence must mean NO allocation, so the caller
        falls back to the allowlist."""
        empty = pd.DataFrame(columns=["strategy", "entry_date", "ret_on_risk"])
        a = sa.allocate(empty, ELIGIBLE, explore_rate=0.25, as_of="2026-08-24")
        self.assertEqual(a.weights, {},
                         "an empty book produced tradeable weights")

    def test_a_book_too_thin_for_any_posterior_allocates_nothing(self):
        thin = pd.DataFrame([
            {"strategy": "Bull Put", "entry_date": "2026-05-01",
             "ret_on_risk": 0.1}] * 3)
        a = sa.allocate(thin, ELIGIBLE, explore_rate=0.25, as_of="2026-08-24")
        self.assertEqual(a.weights, {})

    def test_one_measured_structure_is_enough_to_allocate(self):
        """It must not become so cautious that it never engages: a single
        structure clearing MIN_ROWS_FOR_POSTERIOR is a real basis."""
        df = _book()
        df = df[df["strategy"].isin(["Bull Put", "Bear Call"])]
        a = sa.allocate(df, ELIGIBLE, explore_rate=0.25, as_of="2026-08-24")
        self.assertTrue(a.weights)
        self.assertGreater(a.weights["Bull Put"], 0.0)


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


class TestReplay(unittest.TestCase):
    """Replaying the policy over trades that actually happened.

    The analytic cost is only the mean of a FIXED allocation. A replay shows
    the path: posteriors update as exploration returns evidence, so the
    weights move. That is the part a closed form cannot answer, and it is what
    "what do we get for the rate" actually asks.

    It is honest only because the historical book contains all six structures
    — every entry replayed is a trade that really occurred, with its real
    return. Nothing is simulated except WHICH of them the policy takes.
    """

    def test_a_replay_only_ever_takes_trades_that_existed(self):
        df = _book()
        out = sa.replay(df, ELIGIBLE, explore_rate=0.25, warmup=120)
        self.assertGreater(len(out.taken), 0)
        real = set(zip(df["entry_date"], df["strategy"],
                       df["ret_on_risk"].round(9)))
        for _, r in out.taken.iterrows():
            self.assertIn((r["entry_date"], r["strategy"],
                           round(r["ret_on_risk"], 9)), real)

    def test_no_trade_is_taken_twice(self):
        out = sa.replay(_book(), ELIGIBLE, explore_rate=0.30, warmup=120)
        self.assertEqual(len(out.taken), len(out.taken.drop_duplicates()))

    def test_pure_exploitation_concentrates_on_the_best_structure(self):
        """`per_day` must be well below the day's supply or the policy has no
        choice to express: filling 18 slots from 18 available trades gives a
        uniform mix whatever the weights say."""
        out = sa.replay(_book(), ELIGIBLE, explore_rate=0.0, warmup=120,
                        per_day=2)
        share = (out.taken["strategy"] == "Bull Put").mean()
        self.assertGreater(share, 0.75)

    def test_taking_everything_on_offer_is_not_a_choice(self):
        """The failure mode that made the first two replays meaningless.
        When slots meet supply, availability decides the mix, not the policy —
        so a replay must be read together with its cadence."""
        greedy = sa.replay(_book(), ELIGIBLE, explore_rate=0.0, warmup=120,
                           per_day=18)
        share = (greedy.taken["strategy"] == "Bull Put").mean()
        self.assertLess(share, 0.40)

    def test_exploration_reaches_the_other_structures(self):
        out = sa.replay(_book(), ELIGIBLE, explore_rate=0.40, warmup=120,
                        per_day=2)
        self.assertGreater(out.taken["strategy"].nunique(), 2)

    def test_it_never_uses_evidence_from_the_future(self):
        """Each allocation is built only from trades entered strictly before
        the slot it is choosing for."""
        out = sa.replay(_book(), ELIGIBLE, explore_rate=0.25, warmup=120)
        self.assertTrue((out.decisions["evidence_through"]
                         < out.decisions["entry_date"]).all())

    def test_the_same_seed_replays_identically(self):
        a = sa.replay(_book(), ELIGIBLE, explore_rate=0.25, warmup=120, seed=7)
        b = sa.replay(_book(), ELIGIBLE, explore_rate=0.25, warmup=120, seed=7)
        pd.testing.assert_frame_equal(a.taken, b.taken)

    def test_a_book_shorter_than_the_warmup_replays_nothing(self):
        out = sa.replay(_book().head(10), ELIGIBLE, explore_rate=0.25,
                        warmup=120)
        self.assertEqual(len(out.taken), 0)
        self.assertEqual(out.mean_return, 0.0)


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

    def _ledger(self, d, per_strategy=40):
        """A self-contained ledger with enough closed trades per structure to
        form posteriors.

        These tests used to depend on the REAL `paper_trades.db` existing.
        That is the "tests must not name the real ledger" rule wearing a
        different hat — depending on it rather than writing to it — and it is
        why they passed locally and failed on CI, where the ledger is
        gitignored and absent. A test whose result depends on a file outside
        the repo is not a test of the code.
        """
        import sqlite3, os
        path = os.path.join(d, "ledger.db")
        conn = sqlite3.connect(path)
        conn.execute("""CREATE TABLE trades (
            entry_id INTEGER PRIMARY KEY, date TEXT, expiration TEXT,
            strategy_name TEXT, status TEXT, pnl_usd REAL, entry_delta REAL,
            entry_iv REAL, iv_rank_score REAL, net_credit REAL,
            spread_width REAL, capital_at_risk REAL)""")
        rng = np.random.default_rng(3)
        # Bull Put genuinely best, mirroring the real book's shape.
        means = {"Bull Put": 60.0, "Long Call": -5.0, "Long Put": -20.0,
                 "Short Put": 2.0, "Bear Call": -18.0, "Iron Condor": -20.0}
        i = 0
        for strat, m in means.items():
            for k in range(per_strategy):
                i += 1
                conn.execute(
                    "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (i, f"2026-06-{1 + (k % 28):02d}", "2026-12-31", strat,
                     "CLOSED", float(rng.normal(m, 40.0)), -0.3, 0.3, 0.5,
                     1.2, 5.0, 400.0))
        conn.commit(); conn.close()
        return path

    def _cfg(self, d, **allocation):
        import json, os
        cfg = {"auto_log": {
            "allowed_strategies": ["Bull Put"],
            "paper_only_strategies": [],
            "cohort_min_dte": 30,
        }}
        if allocation:
            allocation.setdefault("ledger_path", self._ledger(d))
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

    def test_two_contracts_of_one_structure_get_different_keys(self):
        """Found by reading the real call site, not by a passing suite. The
        auto-log path called this with `{"strategy_name": ...}` and nothing
        else, so every Bull Put on a given day shared one key — the draw would
        have been all-or-nothing per structure per day instead of per
        candidate, and the realised share would never approach its target."""
        a = self.os_._admission_key({"strategy_name": "Bull Put",
                                     "symbol": "NVDA", "strike": 140.0,
                                     "expiration": "2026-10-16"})
        b = self.os_._admission_key({"strategy_name": "Bull Put",
                                     "symbol": "NVDA", "strike": 145.0,
                                     "expiration": "2026-10-16"})
        self.assertIsNotNone(a)
        self.assertNotEqual(a, b)

    def test_a_trade_with_no_identity_has_no_key(self):
        self.assertIsNone(self.os_._admission_key({"strategy_name": "Bull Put"}))

    def test_an_unidentifiable_trade_falls_back_to_the_allowlist(self):
        """Fail SAFE. Without a key the draw is degenerate, so the allocation
        must stand aside rather than admit or refuse a whole structure at
        once. Falling back to the allowlist can only narrow, never widen."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cfg = self._cfg(d, enabled=True, explore_rate=1.0,
                            eligible_strategies=["Bull Put", "Long Call"])
            self.assertEqual(
                self.os_.apply_auto_log_allowlist({"strategy_name": "Long Call"},
                                                  cfg)[0],
                "drop", "an unidentifiable Long Call was admitted by a "
                        "degenerate draw")
            self.assertEqual(
                self.os_.apply_auto_log_allowlist({"strategy_name": "Bull Put"},
                                                  cfg)[0],
                "insert")

    def test_a_missing_ledger_falls_back_to_the_allowlist(self):
        """End of the same chain. CI has no paper_trades.db, so the allocation
        had no evidence and admitted Bull Put at 0.25 and Long Call at 0.25 —
        four tests failed and the real defect was that a book with no history
        would trade everything uniformly."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cfg = self._cfg(d, enabled=True, explore_rate=0.15,
                            ledger_path=os.path.join(d, "no_such_ledger.db"),
                            eligible_strategies=["Bull Put", "Long Call"])
            self.assertEqual(
                self.os_.apply_auto_log_allowlist(
                    self._trade("Bull Put", "k1"), cfg),
                ("insert", 0), "the allowlist fallback did not admit Bull Put")
            for i in range(200):
                self.assertEqual(
                    self.os_.apply_auto_log_allowlist(
                        self._trade("Long Call", f"k{i}"), cfg)[0],
                    "drop",
                    "with no ledger, Long Call reached the book anyway")

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


class TestDriftSeverity(unittest.TestCase):
    """Does the entered mix over some recent window actually track the
    allocation's target weights? `admits` gates each candidate independently
    — nothing enforces the target over any particular stretch of real
    entries, which is exactly how a downstream defect (per-symbol dedup
    picking a structure by raw row count, ahead of the weight check, fixed
    2026-09-03) went unnoticed for two weeks. This pure function is the
    comparison that should have caught it.
    """

    WEIGHTS = {"Bull Put": 0.85, "Long Call": 0.05, "Long Put": 0.05,
              "Short Put": 0.05}

    def test_a_mix_matching_target_is_ok(self):
        counts = {"Bull Put": 170, "Long Call": 10, "Long Put": 10,
                  "Short Put": 10}
        sev, lines = sa.drift_severity(self.WEIGHTS, counts)
        self.assertEqual(sev, "OK")
        self.assertEqual(lines, [])

    def test_the_dominant_structure_under_represented_is_critical(self):
        # Bull Put target 85%, entered at 25% — the failure mode this exists
        # to catch: the one measured-edge structure being crowded out.
        counts = {"Bull Put": 50, "Long Call": 50, "Long Put": 50,
                  "Short Put": 50}
        sev, lines = sa.drift_severity(self.WEIGHTS, counts)
        self.assertEqual(sev, "CRITICAL")
        self.assertTrue(any("Bull Put" in ln and "BELOW target" in ln
                            for ln in lines))

    def test_a_non_dominant_structure_over_represented_is_warn(self):
        # Bull Put exactly hits its target (its CI trivially contains it);
        # Long Call is entered at 2.5x its target share.
        counts = {"Bull Put": 170, "Long Call": 25, "Long Put": 3,
                  "Short Put": 2}
        sev, lines = sa.drift_severity(self.WEIGHTS, counts)
        self.assertEqual(sev, "WARN")
        self.assertTrue(any("Long Call" in ln and "ABOVE target" in ln
                            for ln in lines))
        self.assertFalse(any("Bull Put" in ln for ln in lines))

    def test_critical_is_not_downgraded_by_a_simultaneous_warn(self):
        counts = {"Bull Put": 50, "Long Call": 100, "Long Put": 25,
                  "Short Put": 25}
        sev, lines = sa.drift_severity(self.WEIGHTS, counts)
        self.assertEqual(sev, "CRITICAL")
        self.assertTrue(any("Bull Put" in ln for ln in lines))
        self.assertTrue(any("Long Call" in ln for ln in lines))

    def test_a_structure_absent_from_the_window_counts_as_zero(self):
        # Long Put and Short Put never appear in `counts` at all — must
        # default to 0 rather than raise, and Bull Put's real shortfall
        # (18% entered against an 85% target) must still be caught.
        counts = {"Bull Put": 20, "Long Call": 90}
        sev, lines = sa.drift_severity(self.WEIGHTS, counts)
        self.assertEqual(sev, "CRITICAL")
        self.assertTrue(any("Bull Put" in ln for ln in lines))

    def test_no_weights_is_ok_with_no_lines(self):
        sev, lines = sa.drift_severity({}, {"Bull Put": 10})
        self.assertEqual((sev, lines), ("OK", []))

    def test_zero_entries_is_ok_with_no_lines(self):
        sev, lines = sa.drift_severity(self.WEIGHTS, {})
        self.assertEqual((sev, lines), ("OK", []))


class TestDriftHealthLines(unittest.TestCase):
    """The I/O wrapper: read the live allocation and the recent entered mix
    from a real ledger, and report on them the way the other `--health`
    checks do (see `candidate_marks.health_lines` for the pattern)."""

    def _ledger(self, d, recent):
        """A book with a clear posterior (Bull Put best, as in the real
        book), plus `recent` extra rows at the END — highest entry_id, so
        `drift_health_lines` sees them as the most recent entries — carrying
        whatever mix the test wants to check.
        """
        import sqlite3, os
        path = os.path.join(d, "ledger.db")
        conn = sqlite3.connect(path)
        conn.execute("""CREATE TABLE trades (
            entry_id INTEGER PRIMARY KEY, date TEXT, expiration TEXT,
            strategy_name TEXT, status TEXT, pnl_usd REAL, entry_delta REAL,
            entry_iv REAL, iv_rank_score REAL, net_credit REAL,
            spread_width REAL, capital_at_risk REAL)""")
        rng = np.random.default_rng(3)
        means = {"Bull Put": 60.0, "Long Call": -5.0, "Long Put": -20.0,
                 "Short Put": 2.0}
        i = 0
        for strat, m in means.items():
            for k in range(40):
                i += 1
                conn.execute(
                    "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (i, f"2026-06-{1 + (k % 28):02d}", "2026-12-31", strat,
                     "CLOSED", float(rng.normal(m, 40.0)), -0.3, 0.3, 0.5,
                     1.2, 5.0, 400.0))
        for strat in recent:
            i += 1
            conn.execute(
                "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (i, "2026-09-01", "2026-12-31", strat, "OPEN", None,
                 -0.3, 0.3, 0.5, 1.2, 5.0, 400.0))
        conn.commit(); conn.close()
        return path

    def _cfg(self, d, ledger_path, window=None, **allocation):
        import json, os
        allocation.setdefault("enabled", True)
        allocation.setdefault("eligible_strategies",
                              ["Bull Put", "Long Call", "Long Put", "Short Put"])
        allocation.setdefault("ledger_path", ledger_path)
        cfg = {"auto_log": {"allocation": allocation}}
        path = os.path.join(d, "config.json")
        with open(path, "w") as fh:
            json.dump(cfg, fh)
        return path

    def test_a_healthy_recent_mix_reports_ok(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            recent = ["Bull Put"] * 26 + ["Long Call"] * 2 + \
                     ["Long Put"] * 1 + ["Short Put"] * 1
            ledger = self._ledger(d, recent)
            cfg = self._cfg(d, ledger)
            lines = sa.drift_health_lines(ledger, cfg, window=30)
            self.assertIn("[OK]", lines[0])

    def test_a_starved_dominant_structure_reports_critical(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            recent = ["Long Call"] * 15 + ["Long Put"] * 10 + \
                     ["Bull Put"] * 3 + ["Short Put"] * 2
            ledger = self._ledger(d, recent)
            cfg = self._cfg(d, ledger)
            lines = sa.drift_health_lines(ledger, cfg, window=30)
            self.assertIn("[CRITICAL]", lines[0])
            self.assertTrue(any("Bull Put" in ln for ln in lines[1:]))

    def test_disabled_allocation_reports_ok_and_says_why(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ledger = self._ledger(d, ["Bull Put"] * 5)
            cfg = self._cfg(d, ledger, enabled=False)
            lines = sa.drift_health_lines(ledger, cfg, window=30)
            self.assertIn("[OK]", lines[0])
            self.assertIn("allowlist", lines[0].lower())

    def test_a_missing_ledger_does_not_raise(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            cfg = self._cfg(d, os.path.join(d, "nope.db"))
            lines = sa.drift_health_lines(os.path.join(d, "nope.db"), cfg,
                                          window=30)
            self.assertTrue(lines)
            self.assertIn("[OK]", lines[0])

    def test_too_few_recent_entries_reports_ok_rather_than_alarming(self):
        # `window` itself smaller than MIN_WINDOW_FOR_DRIFT — a caller-chosen
        # tight window, independent of how much closed history exists for
        # the posterior. Below the floor, a Clopper-Pearson interval is too
        # wide to tell drift from noise, so this must stay quiet rather than
        # manufacture a verdict.
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ledger = self._ledger(d, ["Long Call"] * 3)
            cfg = self._cfg(d, ledger)
            lines = sa.drift_health_lines(ledger, cfg, window=5)
            self.assertIn("[OK]", lines[0])
            self.assertIn("too few", lines[0].lower())


if __name__ == "__main__":
    unittest.main()
