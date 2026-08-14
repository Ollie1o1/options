"""Single-leg backtests threw away the price they entered at.

`dolt_spread` records `credit` and `max_risk` per trade, so a spread result can
be controlled for how RICH the option was at entry — the check that decides
whether an apparent effect is really the vol level restated. `dolt_short` and
`dolt_cohort` recorded neither the entry premium nor the strike, so the same
control could not be run on short puts or long calls at all.

That is the same class of gap as `run_cohort_backtest` dropping its trades
(fixed 2026-08-13): the number exists inside the simulation, is thrown away on
the way out, and its absence is invisible until someone asks the question that
needs it.

Concretely, on 2026-08-14 a sector study cleared its full pre-registered bar
for bull put spreads — including the credit-richness control, which the spread
runner could answer — and could not evaluate that control for its ten
`long_call` and five `short_put` survivors, purely because the entry terms were
not returned.

`entry_price` is the premium as transacted: the BID for a short (sold at the
bid), the ASK for a long (bought at the ask). `strike` and `spot` come with it
so richness can be expressed against notional, which is the only sensible
denominator when a structure has no width.
"""
from __future__ import annotations

import os
import unittest

from src import dolt_options as do
from src.dolt_cohort import run_cohort_backtest
from src.dolt_short import run_short_backtest
from src.dolt_spread import run_spread_backtest

DB = "data/dolt_options.db"
DATES = do._date_range("2023-01-01", "2023-06-30", weekly=True)
SYMS = ["AAPL", "MSFT"]


class TestEntryTermsAreReturned(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(DB):
            raise unittest.SkipTest(f"{DB} absent (gitignored research cache)")

    def _trades(self, run):
        out = run()
        if not out.get("n"):
            self.skipTest("no trades on this window")
        return out["trades"]

    def test_short_put_carries_its_entry_terms(self):
        for t in self._trades(
                lambda: run_short_backtest(SYMS, DATES, opt_type="put", db_path=DB)):
            for f in ("entry_price", "strike", "spot"):
                self.assertIn(f, t, f"short trade lacks {f!r}")
            self.assertGreater(t["entry_price"], 0)

    def test_long_call_carries_its_entry_terms(self):
        for t in self._trades(lambda: run_cohort_backtest(SYMS, DATES, db_path=DB)):
            for f in ("entry_price", "strike", "spot"):
                self.assertIn(f, t, f"cohort trade lacks {f!r}")
            self.assertGreater(t["entry_price"], 0)

    def test_the_spread_runner_still_carries_its_own(self):
        """It already did; pinned so all three stay answerable together."""
        for t in self._trades(
                lambda: run_spread_backtest(SYMS, DATES, db_path=DB, side="put")):
            for f in ("credit", "max_risk"):
                self.assertIn(f, t)

    def test_richness_is_computable_for_every_structure(self):
        """The point of the fix: a premium-to-notional ratio must exist for all
        three, since 'credit to width' is undefined without a width."""
        shorts = self._trades(
            lambda: run_short_backtest(SYMS, DATES, opt_type="put", db_path=DB))
        longs = self._trades(lambda: run_cohort_backtest(SYMS, DATES, db_path=DB))
        for t in shorts:
            self.assertGreater(t["entry_price"] / t["strike"], 0)
        for t in longs:
            self.assertGreater(t["entry_price"] / t["spot"], 0)


if __name__ == "__main__":
    unittest.main()
