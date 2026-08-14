"""All three Dolt backtesters must hand back their per-trade rows.

`dolt_spread.run_spread_backtest` and `dolt_short.run_short_backtest` both put
the individual trades in `out["trades"]`. `dolt_cohort.run_cohort_backtest`
returned summary statistics only — `n`, `avg_return`, `median_return`,
`profit_factor`, `exit_mix` — and dropped the rows on the floor.

That makes the long-call cohort the one strategy in this repo whose backtest
cannot be attributed. You cannot ask which symbol carried a result, split it by
sector or regime, drop the largest contributor to see whether a finding
survives, or bootstrap it — on the cohort that the Phase 1 gate is built
around. Every one of those questions has been asked of the other two.

Found 2026-08-13 while running a sector-conditioned grid: the long_call cells
came back n=0 for every sector, because the harness asked for `out["trades"]`
and got nothing. The data was there the whole time.

The summary keys are unchanged, so every existing caller keeps working.
"""
from __future__ import annotations

import unittest

from src import dolt_options as do
from src.dolt_cohort import run_cohort_backtest
from src.dolt_short import run_short_backtest
from src.dolt_spread import run_spread_backtest

DB = "data/dolt_options.db"
DATES = do._date_range("2023-01-01", "2023-06-30", weekly=True)
SYMS = ["AAPL", "MSFT"]

RUNNERS = (
    ("run_spread_backtest", lambda: run_spread_backtest(SYMS, DATES, db_path=DB, side="put")),
    ("run_short_backtest", lambda: run_short_backtest(SYMS, DATES, opt_type="put", db_path=DB)),
    ("run_cohort_backtest", lambda: run_cohort_backtest(SYMS, DATES, db_path=DB)),
)


class TestEveryRunnerReturnsItsTrades(unittest.TestCase):
    """Skips rather than fails when the cache is absent — this reads the real
    Dolt cache, which is gitignored and not present in CI."""

    @classmethod
    def setUpClass(cls):
        import os
        if not os.path.exists(DB):
            raise unittest.SkipTest(f"{DB} not present (gitignored research cache)")

    def test_trades_are_returned_and_match_the_reported_count(self):
        for name, run in RUNNERS:
            with self.subTest(runner=name):
                out = run()
                if not out.get("n"):
                    self.skipTest(f"{name} produced no trades on this window")
                self.assertIn("trades", out, f"{name} dropped its per-trade rows")
                self.assertEqual(len(out["trades"]), out["n"],
                                 f"{name}: len(trades) disagrees with n")

    def test_each_trade_carries_the_fields_attribution_needs(self):
        """`symbol` and `ret` are the minimum for per-symbol attribution —
        criterion 3 of any 'is this one name?' check."""
        for name, run in RUNNERS:
            with self.subTest(runner=name):
                out = run()
                if not out.get("n"):
                    self.skipTest(f"{name} produced no trades on this window")
                first = out["trades"][0]
                for field in ("symbol", "ret"):
                    self.assertIn(field, first, f"{name} trade lacks {field!r}")

    def test_the_cohort_summary_keys_are_unchanged(self):
        """Existing callers read these; adding `trades` must not disturb them."""
        out = run_cohort_backtest(SYMS, DATES, db_path=DB)
        if not out.get("n"):
            self.skipTest("no cohort trades on this window")
        for key in ("n", "avg_return", "median_return", "profit_factor",
                    "exit_mix", "clean_holds", "rules"):
            self.assertIn(key, out)


if __name__ == "__main__":
    unittest.main()
