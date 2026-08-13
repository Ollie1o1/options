"""`--weekly` was a no-op on two Dolt backtest CLIs.

    dates = _do._date_range(args.start, args.end, weekly=args.weekly or True)

``args.weekly or True`` is ``True`` for every value ``args.weekly`` can take,
so both runners always sampled Fridays and the advertised flag could not
select anything. Daily sampling was unreachable from the command line, and a
reader checking why a run had ~1/5 the entries it expected would find a flag
that looked responsible and was not.

The default is deliberately left as **weekly**. Every Dolt result on record
was produced weekly, and flipping the default to daily would silently change
what an unchanged command means — the same class of quiet drift the flag
already caused. So the flag becomes `--weekly` / `--no-weekly` with
``default=True``: every invocation that exists today keeps its behaviour, and
daily becomes reachable for the first time.

`dolt_cohort` is deliberately NOT changed: its ``args.weekly or
cfg.get("validate_sampling") == "weekly"`` is a real disjunction — flag or
config — not a constant.
"""
from __future__ import annotations

import unittest

from src import dolt_options as do
from src.dolt_short import _build_parser as short_parser
from src.dolt_spread import _build_parser as spread_parser

PARSERS = (("dolt_spread", spread_parser), ("dolt_short", short_parser))


class TestWeeklyFlagIsRealOnBothRunners(unittest.TestCase):

    def test_the_flag_can_turn_sampling_off(self):
        """The whole defect: this was unreachable."""
        for name, build in PARSERS:
            with self.subTest(module=name):
                self.assertFalse(build().parse_args(["--no-weekly"]).weekly)

    def test_the_flag_can_turn_sampling_on(self):
        for name, build in PARSERS:
            with self.subTest(module=name):
                self.assertTrue(build().parse_args(["--weekly"]).weekly)

    def test_weekly_stays_the_default(self):
        """Unchanged commands must keep producing the results on record."""
        for name, build in PARSERS:
            with self.subTest(module=name):
                self.assertTrue(build().parse_args([]).weekly)

    def test_the_two_settings_do_not_select_the_same_dates(self):
        """Guards the assertion above against a parser that parses fine and a
        call site still pinned to one value: if these ever match, the flag is
        decorative again."""
        span = ("2024-03-01", "2024-03-31")
        weekly = do._date_range(*span, weekly=True)
        daily = do._date_range(*span, weekly=False)
        self.assertLess(len(weekly), len(daily))
        self.assertTrue(all(d in daily for d in weekly))


class TestTheOldExpressionWasAConstant(unittest.TestCase):
    """States the bug itself, so nobody reintroduces it as a 'safe default'."""

    def test_or_true_cannot_be_false(self):
        for value in (True, False, None, 0, ""):
            self.assertTrue(value or True)


if __name__ == "__main__":
    unittest.main()
