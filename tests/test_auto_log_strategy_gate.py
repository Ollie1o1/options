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
import unittest

from src.options_screener import apply_auto_log_allowlist
from src.paths import repo_path

FAR_DATED = {"expiration": "2026-12-18", "date": "2026-08-13"}


def _decide(strategy):
    return apply_auto_log_allowlist(dict(FAR_DATED, strategy_name=strategy))


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

    def test_long_call_is_still_logged_and_cohort_eligible(self):
        self.assertEqual(_decide("Long Call"), ("insert", 0))

    def test_an_unknown_strategy_is_dropped_rather_than_logged(self):
        self.assertEqual(_decide("Jade Lizard"), ("drop", None))

    def test_a_short_dated_long_call_is_quarantined_not_dropped(self):
        """The DTE floor marks paper_only=1; it must not silently drop."""
        near = {"strategy_name": "Long Call", "expiration": "2026-08-20",
                "date": "2026-08-13"}
        decision, flag = apply_auto_log_allowlist(near)
        self.assertEqual(decision, "insert")
        self.assertEqual(flag, 1)


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
        self.assertEqual(_decide("Bull Put"), ("insert", 0))

    def test_bull_put_appears_in_allowed_strategies(self):
        """Pins the mechanism: `paper_only_strategies` would also make it log,
        but quarantined out of every cohort — which is the state we are
        leaving, not the one we are moving to."""
        with open(repo_path("config.json")) as fh:
            al = json.load(fh)["auto_log"]
        self.assertIn("Bull Put", al.get("allowed_strategies") or [])

    def test_the_rest_of_the_family_stays_off(self):
        """The fix restores one strategy on its own evidence, not the family
        that was switched off alongside it."""
        for strat in ("Bear Call", "Short Put"):
            with self.subTest(strategy=strat):
                self.assertEqual(_decide(strat), ("drop", None))


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
        near = {"strategy_name": "Bull Put", "expiration": "2026-08-31",
                "date": "2026-08-13"}   # 18 DTE — the historical median
        self.assertEqual(apply_auto_log_allowlist(near), ("insert", 0))

    def test_a_short_dated_long_call_is_still_quarantined(self):
        """The guard must narrow, not disappear."""
        near = {"strategy_name": "Long Call", "expiration": "2026-08-20",
                "date": "2026-08-13"}
        self.assertEqual(apply_auto_log_allowlist(near), ("insert", 1))


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
