"""A vertical must be told apart from its mirror image.

`structure_strategy_name` decided the side with
``row["type"].strip().lower() == "call"``. But `find_credit_spreads` writes
``type`` as the STRATEGY name — literally "Bull Put" (options_screener.py:3228)
and "Bear Call" (:3286) — so that comparison never matched and every vertical
came back "Bull Put", including every bear call.

What it did NOT do is corrupt the ledger: `log_spread` derives the stored
`strategy_name` from `type` itself and never consults this function. Checked on
the live book at the time of the fix — 131 Bull Puts all put-shaped, 135 Bear
Calls all call-shaped, zero mislabelled. Sizing was unaffected too: both names
take the same defined-risk branch in `capital_at_risk`.

What it did do:

* `candidate_verdict` reads the strategy name to lay out the legs, so bear
  calls were priced as put spreads — wrong side, wrong friction, feeding board
  ordering and WORTH grades;
* the auto-log allowlist saw "Bull Put" for a bear call. Harmless while
  `allowed_strategies` is `['Long Call']`, and a trap the moment "Bull Put" is
  re-enabled: Bear Call would come with it silently, and it is the family
  measured at -7.4pp against its own managed breakeven.
"""
from __future__ import annotations

import unittest

import pandas as pd

from src.options_screener import structure_strategy_name as name


class TestTheStrategyNamesTheScannerActuallyWrites(unittest.TestCase):
    """`find_credit_spreads` emits these two strings and nothing else."""

    def test_a_bear_call_is_a_bear_call(self):
        row = pd.Series({"symbol": "QQQ", "type": "Bear Call",
                         "short_strike": 744.0, "long_strike": 745.0})
        self.assertEqual(name(row), "Bear Call")

    def test_a_bull_put_is_a_bull_put(self):
        row = pd.Series({"symbol": "QQQ", "type": "Bull Put",
                         "short_strike": 750.0, "long_strike": 748.0})
        self.assertEqual(name(row), "Bull Put")

    def test_the_two_do_not_collapse_onto_one_name(self):
        """The actual defect: both answered "Bull Put"."""
        call = pd.Series({"type": "Bear Call", "short_strike": 744.0,
                          "long_strike": 745.0})
        put = pd.Series({"type": "Bull Put", "short_strike": 750.0,
                         "long_strike": 748.0})
        self.assertNotEqual(name(call), name(put))

    def test_the_scanner_still_writes_the_strings_this_relies_on(self):
        """Pinned against drift — if the emitted `type` changes, this
        function has to change with it, and nothing else would notice."""
        from src.paths import repo_path
        with open(repo_path("src/options_screener.py")) as fh:
            src = fh.read()
        self.assertIn('"type": "Bull Put"', src)
        self.assertIn('"type": "Bear Call"', src)


class TestLegacyAndAmbiguousRows(unittest.TestCase):

    def test_a_bare_call_still_reads_as_a_bear_call(self):
        """The value the old comparison expected. Kept working, because a
        fix that trades one blind spot for another is not a fix."""
        self.assertEqual(name(pd.Series({"type": "call"})), "Bear Call")

    def test_a_bare_put_still_reads_as_a_bull_put(self):
        self.assertEqual(name(pd.Series({"type": "put"})), "Bull Put")

    def test_a_condor_wins_over_the_type_column(self):
        row = pd.Series({"type": "Bear Call", "total_credit": 5.0,
                         "short_put_strike": 400.0, "short_call_strike": 440.0})
        self.assertEqual(name(row), "Iron Condor")

    def test_an_unnamed_row_falls_back_to_its_geometry(self):
        """A credit call spread sells BELOW its long leg; a credit put spread
        sells above it. Unambiguous when the label is missing."""
        self.assertEqual(
            name(pd.Series({"short_strike": 744.0, "long_strike": 745.0})),
            "Bear Call")
        self.assertEqual(
            name(pd.Series({"short_strike": 750.0, "long_strike": 748.0})),
            "Bull Put")

    def test_a_row_with_nothing_to_go_on_does_not_raise(self):
        self.assertIn(name(pd.Series({"symbol": "QQQ"})),
                      ("Bull Put", "Bear Call"))

    def test_garbage_strikes_do_not_raise(self):
        row = pd.Series({"short_strike": "nope", "long_strike": None})
        self.assertIn(name(row), ("Bull Put", "Bear Call"))

    def test_a_plain_dict_works_like_a_series(self):
        """`pick_ranking` and the auto-log path both hand it row-likes."""
        self.assertEqual(name({"type": "Bear Call"}), "Bear Call")


class TestTheConsequenceThatMattered(unittest.TestCase):
    """The trap this leaves if it regresses."""

    def test_a_bear_call_is_not_offered_to_the_allowlist_as_a_bull_put(self):
        from src.options_screener import apply_auto_log_allowlist
        row = pd.Series({"type": "Bear Call", "short_strike": 744.0,
                         "long_strike": 745.0})
        decision, _paper_only = apply_auto_log_allowlist(
            {"strategy_name": name(row)}, cfg_path="config.json")
        # Whatever config says today, the NAME it was asked about must be the
        # real one — re-enabling Bull Put must not silently re-admit Bear Call.
        self.assertEqual(name(row), "Bear Call")
        self.assertIn(decision, ("drop", "log", "paper"))


if __name__ == "__main__":
    unittest.main()
