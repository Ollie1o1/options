"""Which strategies the auto-logger will and will not enter.

`apply_auto_log_allowlist` has three outcomes, and the third is the one that
matters here:

    strategy in allowed_strategies     -> ("insert", 0)   cohort-eligible
    strategy in paper_only_strategies  -> ("insert", 1)   logged, quarantined
    strategy in NEITHER                -> ("drop", None)  never logged

So a strategy is switched off by being absent from both lists, not by any
flag. `config.json` carried `auto_log_skip_bear_calls` and
`auto_log_skip_long_puts` for exactly that purpose and **nothing in `src/`
ever read either of them**; both were deleted 2026-08-13 rather than left to
imply a control that does not exist.

Iron Condor was removed from `paper_only_strategies` the same day, stopping
its auto-log. Measured over 408 closed credit trades, required win rate
computed from realised payoffs under the exits actually used: Iron Condor
needs 60.1% and has delivered 50.0% across 142 closed trades (-10.1pp), and
Bear Call needs 66.7% against 59.3% (-7.4pp). Bear Call had already stopped
logging on 2026-07-31 and needed no change.
"""
from __future__ import annotations

import json
import os
import unittest

import datetime as _dt

from src.options_screener import apply_auto_log_allowlist
from src.paths import repo_path


def _exp_in(days: int) -> str:
    """An expiration `days` from TODAY.

    The DTE floor measures from today, not from a row's `date` field, so a
    hardcoded expiration is a time bomb: the Bull Put case below was written
    2026-08-13 with a fixed 2026-08-31 expiry — 18 DTE then, 5 DTE by
    2026-08-26 — and started failing on a calendar roll rather than a code
    change. Anchoring to today keeps each test measuring the DTE it names.
    """
    return (_dt.date.today() + _dt.timedelta(days=days)).isoformat()


FAR_DATED = {"expiration": _exp_in(114), "date": "2026-08-13"}


def _decide(strategy):
    return apply_auto_log_allowlist(dict(FAR_DATED, strategy_name=strategy))


def _admission_rate(strategy, n=600):
    """Share of DISTINCT contracts of `strategy` the auto-logger admits.

    Since the allocation went live (2026-08-24) admission is a SHARE, not an
    invariant: the draw is deterministic per contract, so a single candidate
    is no longer representative of the structure. Asserting one contract here
    would be asserting a coin flip.
    """
    hits = 0
    for i in range(n):
        decision, _ = apply_auto_log_allowlist(
            dict(FAR_DATED, strategy_name=strategy, symbol="NVDA",
                 strike=100.0 + i, type="put"))
        hits += decision == "insert"
    return hits / n



class TestTheFamiliesThatFailTheirBreakevenAreNotLogged(unittest.TestCase):
    """Both miss their own required win rate; neither may be auto-logged."""

    def test_iron_condor_is_dropped(self):
        self.assertEqual(_decide("Iron Condor"), ("drop", None))

    def test_bear_call_is_dropped(self):
        self.assertEqual(_decide("Bear Call"), ("drop", None))

    def test_neither_appears_in_either_config_list(self):
        """Pins the mechanism, not just the outcome: a future edit that
        re-adds one to `paper_only_strategies` would silently resume logging
        it, since that still returns ("insert", 1)."""
        with open(repo_path("config.json")) as fh:
            al = json.load(fh)["auto_log"]
        both = set(al.get("allowed_strategies") or []) | set(
            al.get("paper_only_strategies") or [])
        for strat in ("Iron Condor", "Bear Call"):
            with self.subTest(strategy=strat):
                self.assertNotIn(strat, both)


class TestTheAllowlistStillWorks(unittest.TestCase):
    """Guards against 'switch it off' becoming 'switch everything off'."""

    def test_bull_put_still_takes_the_large_majority_of_entries(self):
        """Was `_decide("Bull Put") == ("insert", 0)`. The allocation made
        admission a share rather than a certainty — Bull Put holds ~88% of it,
        the rest being the priced exploration budget. The guard this test
        exists for is unchanged: 'switch it off' must not become 'switch
        everything off'."""
        self.assertGreater(_admission_rate("Bull Put"), 0.70)

    def test_an_unknown_strategy_is_dropped_rather_than_logged(self):
        self.assertEqual(_decide("Jade Lizard"), ("drop", None))


class TestLongCallIsOffOnItsEvidence(unittest.TestCase):
    """Switched off 2026-08-19, by evidence rather than by accident.

    Audited on the ledger with position sizing removed — every row is
    `quantity = 1.0`, so as-sized P&L mostly describes option premiums rather
    than picks. Equal-weighted over 295 closed Long Calls: PF 1.035, 95%
    bootstrap CI [0.783, 1.335]. The interval contains 1, so there is no
    detectable edge.

    The post-repair cohort is the same story at full volume: 25 closed,
    -$9,890, 24% win, PF 0.098. That was audited for a defect and none was
    found — exit prices reconcile against `chain_archive.db`, an archive the
    ledger never reads (QQQ recorded 6.28 against 6.43, 6.80 against 6.87),
    the -50% stop rule is implemented correctly, and the underlying moved
    against the position in 12 of 16 measurable cases. The trades simply lost.

    Bull Put stays: PF 1.911, CI [1.280, 3.081] — the only line in the book
    whose interval clears 1.

    Long Call keeps being SCANNED and recorded to `data/candidates.db`, which
    marks refused candidates forward. The strategy therefore keeps generating
    evidence without consuming book capital, which is the whole point of the
    counterfactual database.
    """

    def test_long_call_is_dropped(self):
        self.assertEqual(_decide("Long Call"), ("drop", None))

    def test_long_call_is_absent_from_BOTH_config_lists(self):
        """Pins the mechanism, not the outcome. `paper_only_strategies` still
        returns ("insert", 1) — it logs the trade and merely quarantines it out
        of the cohort. Only absence from both lists actually stops the entry,
        and this project has already shipped one 'switch it off' edit that
        moved a strategy between lists and kept logging it."""
        with open(repo_path("config.json")) as fh:
            al = json.load(fh)["auto_log"]
        both = set(al.get("allowed_strategies") or []) | set(
            al.get("paper_only_strategies") or [])
        self.assertNotIn("Long Call", both)

    def test_no_dte_makes_it_come_back(self):
        """The horizon floor quarantines; it must never resurrect. A dropped
        strategy is dropped at every DTE."""
        for exp in (_exp_in(7), _exp_in(66), None):
            with self.subTest(expiration=exp):
                trade = {"strategy_name": "Long Call", "date": "2026-08-13"}
                if exp:
                    trade["expiration"] = exp
                self.assertEqual(apply_auto_log_allowlist(trade), ("drop", None))


class TestBullPutLogsAgain(unittest.TestCase):
    """Restored 2026-08-18. It was switched off by accident, not by evidence.

    `c0cd5bc` (2026-08-01) removed Bull Put, Bear Call and Short Put from
    `paper_only_strategies` so the short-premium gate's cohort could authorise
    capital, and added none of them to `allowed_strategies`. Absence from both
    lists is exactly how a strategy is switched off, so all three stopped being
    auto-logged that day — silently, and against the intent of the change.

    Bull Put is the one line in the book that clears its own bar: it needs a
    50.9% win rate under the exits actually used and has delivered 66.4% over
    131 closed trades, +$5,315. Bear Call (-7.4pp) and Short Put (-$8,278 over
    109 closed) do not, and stay off.
    """

    def test_bull_put_is_logged_and_cohort_eligible(self):
        """Admitted candidates are cohort-eligible (flag 0), and Bull Put
        holds the large majority of the allocation. See
        `test_bull_put_still_takes_the_large_majority_of_entries`."""
        self.assertGreater(_admission_rate("Bull Put"), 0.70)
        flags = {apply_auto_log_allowlist(
                     dict(FAR_DATED, strategy_name="Bull Put", symbol="NVDA",
                          strike=100.0 + i, type="put"))
                 for i in range(200)}
        self.assertIn(("insert", 0), flags)
        self.assertNotIn(("insert", 1), flags,
                         "a far-dated Bull Put was quarantined")

    def test_bull_put_appears_in_allowed_strategies(self):
        """Pins the mechanism: `paper_only_strategies` would also make it log,
        but quarantined out of every cohort — which is the state we are
        leaving, not the one we are moving to."""
        with open(repo_path("config.json")) as fh:
            al = json.load(fh)["auto_log"]
        self.assertIn("Bull Put", al.get("allowed_strategies") or [])

    def test_bear_call_stays_off_entirely(self):
        """The fix restored one strategy on its own evidence, not the family
        that was switched off alongside it.

        Bear Call is absent from `eligible_strategies`, so the allocation can
        never draw it — 0% in every environment."""
        self.assertEqual(_admission_rate("Bear Call"), 0.0)

    def test_short_put_is_exploration_only_not_a_restoration(self):
        """SUPERSEDED ASSERTION, REWRITTEN 2026-08-26.

        This used to assert Short Put always drops. That was true under the
        name allowlist and stopped being true when allocation went live on
        2026-08-24: Short Put sits in `eligible_strategies` and draws a small
        exploration share. Measured locally: Bull Put 0.903, Short Put 0.040.

        Worse than merely stale, the old assertion was ENVIRONMENT-DEPENDENT.
        The allocation posterior reads the ledger, which is not in git, so the
        test passed on CI (no ledger, fallback drops it) and failed locally
        (ledger present, 4% admitted). A test that disagrees with itself across
        machines is worse than one that is simply wrong.

        What is invariant, and what this now asserts: Short Put is eligible for
        exploration but is NOT restored to the posterior-best rate. The bound
        holds at 0.04 locally and at 0.0 with no ledger.
        """
        self.assertLessEqual(_admission_rate("Short Put"), 0.25)

    def test_short_put_is_eligible_while_bear_call_is_not(self):
        """The config invariant behind the two tests above, asserted directly
        so the distinction survives a change in the posterior."""
        with open(repo_path("config.json")) as fh:
            alloc = (json.load(fh)["auto_log"].get("allocation") or {})
        eligible = set(alloc.get("eligible_strategies") or [])
        self.assertIn("Short Put", eligible)
        self.assertNotIn("Bear Call", eligible)


class TestTheHorizonFloorIsLongPremiumReasoning(unittest.TestCase):
    """The floor asks whether an entry has swing runway before the time-exit
    force-closes it — a question about a directional long, whose thesis needs
    time to play out. A credit spread's thesis IS decay, so short DTE is the
    point of the trade rather than a contamination of it.

    Left global, the floor would have quarantined 119 of 132 historical Bull
    Puts (median 18 DTE against a 30-DTE floor) as paper_only=1, keeping the
    strategy's evidence out of the very cohort this restoration exists to
    feed.
    """

    def test_a_short_dated_bull_put_stays_cohort_eligible(self):
        """Asserted against an EXPLICIT config, like the Long Call case below.

        Two reasons, both learned the hard way. The DTE floor measures from
        TODAY, so a hardcoded expiration is a calendar time bomb — the original
        `2026-08-31` was 18 DTE when written and 5 DTE by 2026-08-26. And since
        the 2026-08-24 allocation went live, `apply_auto_log_allowlist` runs a
        deterministic hash draw over the trade's identifying fields, so a live
        config makes this a lottery: changing the expiration changes the key,
        changes the draw, and flips the verdict for reasons that have nothing
        to do with the floor this test is about.

        Pinning the config isolates the one behaviour under test — the horizon
        floor is long-premium reasoning and must not quarantine a credit
        spread whose thesis IS decay.
        """
        import json as _json
        import tempfile as _tf
        near = {"strategy_name": "Bull Put", "expiration": _exp_in(18),
                "date": "2026-08-13"}   # 18 DTE — the historical median
        with _tf.TemporaryDirectory() as d:
            path = os.path.join(d, "cfg.json")
            with open(path, "w") as fh:
                _json.dump({"auto_log": {"allowed_strategies": ["Bull Put"],
                                         "paper_only_strategies": [],
                                         "cohort_min_dte": 30}}, fh)
            self.assertEqual(apply_auto_log_allowlist(near, cfg_path=path),
                             ("insert", 0))

    def test_the_floor_still_quarantines_a_short_dated_long_premium_entry(self):
        """The guard must narrow, not disappear.

        Asserted against an explicit config rather than the live one, because
        Long Call left `allowed_strategies` on 2026-08-19 and a dropped
        strategy never reaches the DTE floor. The floor is still live code —
        it governs Long Put, and Long Call if it is ever restored — so it keeps
        a test that does not depend on which strategies are currently enabled.
        """
        import json as _json
        import tempfile as _tf
        near = {"strategy_name": "Long Call", "expiration": _exp_in(7),
                "date": "2026-08-13"}
        with _tf.TemporaryDirectory() as d:
            path = os.path.join(d, "cfg.json")
            with open(path, "w") as fh:
                _json.dump({"auto_log": {"allowed_strategies": ["Long Call"],
                                         "paper_only_strategies": [],
                                         "cohort_min_dte": 30}}, fh)
            self.assertEqual(apply_auto_log_allowlist(near, cfg_path=path),
                             ("insert", 1))


class TestTheDeadSwitchesAreGone(unittest.TestCase):
    """They named a behaviour they did not implement.

    Someone reading `auto_log_skip_bear_calls: true` would reasonably conclude
    bear calls were already switched off by that flag. They were switched off
    by absence from both lists — the flag was inert, and had it been the only
    mechanism relied upon, bear calls would have kept logging.
    """

    def test_config_no_longer_carries_them(self):
        with open(repo_path("config.json")) as fh:
            cfg = json.load(fh)
        for dead in ("auto_log_skip_bear_calls", "auto_log_skip_long_puts"):
            with self.subTest(key=dead):
                self.assertNotIn(dead, cfg)


if __name__ == "__main__":
    unittest.main()
