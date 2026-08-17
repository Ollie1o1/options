"""The affordable cohort measures each trade against the budget IN FORCE.

`budget_at_entry` records what governed a trade at entry. Until now nothing
read it: both cohorts compared `capital_at_risk` against the CALLER's single
ceiling, so once the budget varies per scan a trade logged under a chosen
$10,000 budget fell out of the "inside budget" subset although it was inside
its own — the exact recovery the v22 migration comment says the column exists
to enable.

The rule is `capital_at_risk <= COALESCE(budget_at_entry, <caller ceiling>)`.
The COALESCE is the whole design, and the alternative was measured on the live
book before choosing:

* `budget_at_entry IS NOT NULL AND capital_at_risk <= budget_at_entry` — the
  literal reading — cut the reported Long Call subset from 130 to 31, because
  101 of the 132 cohort rows predate the cap and carry NULL. NULL means "no
  limit was in force", so under that reading the unbounded-feeder era, whose
  $27k and $83k positions the cap exists to exclude, would count as "inside
  its budget" wherever it also happened to be small.
* COALESCE reproduces 130 exactly. It changes nothing on today's data, honours
  a per-trade budget the moment one differs, and keeps the caller's ceiling
  standing in for the era that had none.

Never touches the real book — every ledger here is built in a tempfile.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from src import phase1_checkpoint as p1


COLS = ("date", "strategy_name", "status", "paper_only", "quality_score",
        "pnl_pct", "pnl_usd", "capital_at_risk", "budget_at_entry")


def _ledger(rows, with_budget_column=True):
    """A minimal `trades` table holding exactly `rows`."""
    path = os.path.join(tempfile.mkdtemp(), "ledger.db")
    cols = [c for c in COLS if with_budget_column or c != "budget_at_entry"]
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE trades ({', '.join(cols)})")
    for r in rows:
        vals = [r.get(c) for c in cols]
        conn.execute(
            f"INSERT INTO trades ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' * len(cols))})", vals)
    conn.commit()
    conn.close()
    return path


def _lc(**kw):
    row = {"date": "2026-06-01", "strategy_name": "Long Call", "status": "CLOSED",
           "paper_only": 0, "quality_score": 0.5, "pnl_pct": 1.0, "pnl_usd": 10.0,
           "capital_at_risk": 100.0, "budget_at_entry": None}
    row.update(kw)
    return row


class TestTheCohortHonoursThePerTradeBudget(unittest.TestCase):

    def _n(self, rows, cap):
        scores, _returns, _dates = p1._load_cohort(_ledger(rows), "2026-05-27", cap)
        return len(scores)

    def test_a_trade_inside_its_own_larger_budget_counts(self):
        """The finding. $9,000 risked under a chosen $10,000 budget was inside
        it, and the caller's $4,000 ceiling must not overrule that."""
        rows = [_lc(capital_at_risk=9000.0, budget_at_entry=10000.0)]
        self.assertEqual(self._n(rows, 4000.0), 1)

    def test_a_trade_outside_its_own_smaller_budget_does_not(self):
        """Symmetry: a tighter session budget must bind too. $2,000 risked
        under a chosen $500 budget is outside it, even though the caller's
        $4,000 ceiling would have admitted it."""
        rows = [_lc(capital_at_risk=2000.0, budget_at_entry=500.0)]
        self.assertEqual(self._n(rows, 4000.0), 0)

    def test_a_trade_at_exactly_its_budget_counts(self):
        rows = [_lc(capital_at_risk=4000.0, budget_at_entry=4000.0)]
        self.assertEqual(self._n(rows, 4000.0), 1)

    def test_no_budget_in_force_falls_back_to_the_callers_ceiling(self):
        """NULL is the pre-cap era, not a licence.

        Reading NULL as "no limit, so it was inside its budget" would sweep
        the unbounded feeder's own rows into the affordable subset.
        """
        rows = [_lc(capital_at_risk=100.0, budget_at_entry=None),
                _lc(capital_at_risk=27000.0, budget_at_entry=None)]
        self.assertEqual(self._n(rows, 4000.0), 1)

    def test_no_caller_ceiling_means_no_filter_at_all(self):
        """`max_capital_at_risk=None` is the gate's own read and must stay
        the whole cohort — the affordable subset is a parallel diagnostic."""
        rows = [_lc(capital_at_risk=27000.0, budget_at_entry=500.0),
                _lc(capital_at_risk=100.0)]
        self.assertEqual(self._n(rows, None), 2)

    def test_unbounded_risk_is_still_not_small_risk(self):
        """NULL capital_at_risk stays excluded, budget or no budget."""
        rows = [_lc(capital_at_risk=None, budget_at_entry=10000.0)]
        self.assertEqual(self._n(rows, 4000.0), 0)

    def test_a_ledger_without_the_column_still_reads(self):
        """Schema v22 added `budget_at_entry`; hand-built fixtures and any
        older ledger predate it. Probed, not assumed — the same idiom as
        `exclude_ruled_duplicates` and `duplicate_of`."""
        path = _ledger([_lc(capital_at_risk=100.0),
                        _lc(capital_at_risk=9000.0)],
                       with_budget_column=False)
        scores, _r, _d = p1._load_cohort(path, "2026-05-27", 4000.0)
        self.assertEqual(len(scores), 1)


class TestTheShortPremiumCohortAgrees(unittest.TestCase):
    """Both cohort readers apply one rule — two answers from one ledger is
    the failure `exclude_ruled_duplicates` was extracted to prevent."""

    def _rows(self, rows, cap):
        return p1._load_short_premium_cohort(_ledger(rows), "2026-05-27", cap)

    def _sp(self, **kw):
        return _lc(strategy_name="Short Put", **kw)

    def test_a_trade_inside_its_own_larger_budget_counts(self):
        rows = [self._sp(capital_at_risk=9000.0, budget_at_entry=10000.0)]
        self.assertEqual(len(self._rows(rows, 4000.0)), 1)

    def test_a_trade_outside_its_own_smaller_budget_does_not(self):
        rows = [self._sp(capital_at_risk=2000.0, budget_at_entry=500.0)]
        self.assertEqual(len(self._rows(rows, 4000.0)), 0)

    def test_no_budget_in_force_falls_back_to_the_callers_ceiling(self):
        rows = [self._sp(capital_at_risk=100.0),
                self._sp(capital_at_risk=27000.0)]
        self.assertEqual(len(self._rows(rows, 4000.0)), 1)

    def test_a_ledger_without_the_column_still_reads(self):
        path = _ledger([self._sp(capital_at_risk=100.0),
                        self._sp(capital_at_risk=9000.0)],
                       with_budget_column=False)
        self.assertEqual(len(p1._load_short_premium_cohort(
            path, "2026-05-27", 4000.0)), 1)


if __name__ == "__main__":
    unittest.main()
