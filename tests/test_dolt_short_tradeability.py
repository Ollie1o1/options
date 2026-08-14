"""The short-premium backtest entered positions no one could have traded.

`simulate_short_trade` sells at the bid and closes at the ask, and reports the
return as a fraction of the credit. It had a split-mismatch guard and no
tradeability guard, so it happily sold a 5-cent bid against a $4.60 ask.

Measured 2026-08-13 on the Dolt cache, entries actually picked:

    IOSP 2023-07-14   bid $0.05   ask $4.60     spread = 9,100% of credit
    CF   2020-03-13   bid $0.05   ask $4.90     spread = 9,700% of credit
    IOSP 2020-08-28   bid $0.05   ask $9.90     spread = 19,700% of credit

Closing any of those at the ask is a return of -90x to -200x the credit, and a
handful of them dominated whole sectors: XLB's mean return on short puts was
**-5.75** against a 5-95% trimmed mean of -1.97, and the five worst trades in
the sector were -199, -199, -187, -172, -99.

That is not a strategy result. It is the arithmetic of dividing by a nickel.

The live ledger already refuses these — `auto_log.max_friction_to_credit`,
tightened 0.50 -> 0.25 on 2026-08-06, which `config.json` records as "the
largest single effect measured anywhere in this system". The backtest was
measuring a universe the live system would not trade. It now applies the same
bar, read from the same config key.

Note this does NOT affect the spread backtests: `dolt_spread` reports return on
MAX RISK (width - credit), a denominator that cannot collapse, which is why
those cells looked well-formed while these did not.
"""
from __future__ import annotations

import unittest

from src.dolt_short import _entry_is_tradeable


class TestTheTradeabilityGate(unittest.TestCase):

    def test_the_measured_uncrossable_quotes_are_refused(self):
        """The three real entries from the Dolt cache, by value."""
        for bid, ask, label in ((0.05, 4.60, "IOSP 2023-07-14"),
                                (0.05, 4.90, "CF 2020-03-13"),
                                (0.05, 9.90, "IOSP 2020-08-28")):
            with self.subTest(entry=label):
                self.assertFalse(_entry_is_tradeable(bid, ask, 0.25),
                                 f"{label} is not a market")

    def test_a_normal_quote_passes(self):
        # 2.00 / 2.05 — a 2.5% round trip against the credit.
        self.assertTrue(_entry_is_tradeable(2.00, 2.05, 0.25))

    def test_the_boundary_is_the_configured_fraction(self):
        # spread exactly 25% of credit passes; a hair more does not.
        self.assertTrue(_entry_is_tradeable(1.00, 1.25, 0.25))
        self.assertFalse(_entry_is_tradeable(1.00, 1.26, 0.25))

    def test_a_missing_or_absurd_quote_is_refused_not_crashed(self):
        for bid, ask in ((None, 1.0), (1.0, None), (0.0, 1.0), (-1.0, 1.0),
                         (1.0, 0.5)):
            with self.subTest(bid=bid, ask=ask):
                self.assertFalse(_entry_is_tradeable(bid, ask, 0.25))

    def test_a_null_threshold_disables_the_gate(self):
        """`max_friction_to_credit: null` disables it live; same here, so a
        deliberate wide-quote study is still possible."""
        self.assertTrue(_entry_is_tradeable(0.05, 4.60, None))


class TestItReadsTheLiveConfigKey(unittest.TestCase):

    def test_the_default_comes_from_auto_log_max_friction_to_credit(self):
        import json

        from src.dolt_short import _friction_limit
        from src.paths import repo_path
        with open(repo_path("config.json")) as fh:
            shipped = json.load(fh)["auto_log"]["max_friction_to_credit"]
        self.assertEqual(_friction_limit({}), shipped)

    def test_an_explicit_value_overrides_config(self):
        from src.dolt_short import _friction_limit
        self.assertEqual(
            _friction_limit({"auto_log": {"max_friction_to_credit": 0.5}}), 0.5)


if __name__ == "__main__":
    unittest.main()
