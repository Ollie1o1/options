"""The per-contract breakeven refusal is gone, and that is not a loosening.

`verdict_for` carried a gate: refuse when `1 - credit/width` exceeds your
historical win rate on the structure ("needs a 77% win rate; your history on
this structure is 66%").

IT NEVER FIRED. Instrumented across four modes on 2026-08-17: 482 verdict
calls, `historical_win_rate` supplied on ZERO of them. `rank_by_verdict(df,
win_rates=None)` defaults it and both callers — `rank_structures_by_verdict`
and `rank_single_legs_by_verdict` — omit the argument, so the refusal branch
was unreachable in every scan path. Single legs could not reach it anyway:
they carry no `spread_width`, so `breakeven` is None for them.

WIRING IT UP WOULD HAVE BEEN WORSE THAN LEAVING IT DEAD, because the
comparison is a category error. `1 - credit/width` is the HOLD-TO-EXPIRY
requirement — what you would need if you rode every spread to expiry — and the
historical win rate it is measured against was achieved WITH management, by
taking profit early and stopping out. Comparing them is apples to oranges, and
it points the wrong way: the NVDA Bull Put 215/205 logged on 2026-08-17 needs
77.2% held to expiry against a 66.4% history, so the gate would have refused
it — while the MANAGED requirement for that family is 50.9% against the same
66.4%, a +15.5pp margin. It would have refused the one structure family this
ledger says works.

There is also a structural reason a per-contract breakeven gate cannot be
right here. Under exits managed at fixed fractions of the credit, a winner
takes tp x credit and a loser gives up sl x credit, so p* = sl / (sl + tp) —
independent of that contract's credit/width. The requirement is a property of
the STRATEGY, not the contract. That is why the measured managed rate is
per-strategy, and why the comparison belongs in the Breakeven column (which
now shows it) and in `allowed_strategies`, not in a per-candidate gate.

`Verdict.breakeven` is still computed and still reported — it is a real number
and worth seeing. It just no longer refuses anything.
"""
from __future__ import annotations

import unittest

from src.candidate_verdict import verdict_for


def _spread(credit_bid=2.20, credit_ask=2.36, width=10.0):
    """A credit spread whose hold-to-expiry p* far exceeds any real win rate.

    Mirrors the NVDA Bull Put 215/205: ~$2.28 credit on a $10 width, so
    1 - 2.28/10 = 77.2%.
    """
    return {
        "strategy_name": "Bull Put", "spread_width": width,
        "short_bid": 5.00, "short_ask": 5.10,
        "long_bid": 2.72, "long_ask": 2.80,
    }


class TestTheGateNoLongerRefuses(unittest.TestCase):

    def test_a_high_breakeven_is_not_refused(self):
        """The NVDA case: 77% needed held-to-expiry against a 66% history."""
        v = verdict_for(_spread(), historical_win_rate=0.664)
        self.assertTrue(v.priced)
        self.assertNotIn("win rate", v.reason,
                         "the breakeven refusal is back — it compares a "
                         "hold-to-expiry requirement against a managed history")

    def test_it_passes_where_it_used_to_be_refused(self):
        v = verdict_for(_spread(), historical_win_rate=0.664)
        self.assertTrue(v.passed)

    def test_supplying_no_win_rate_is_identical(self):
        """It was already unreachable without one; both paths must agree."""
        a = verdict_for(_spread(), historical_win_rate=0.664)
        b = verdict_for(_spread())
        self.assertEqual(a.passed, b.passed)
        self.assertEqual(a.reason, b.reason)


class TestTheNumberIsStillReported(unittest.TestCase):
    """Removing a refusal is not removing a measurement."""

    def test_breakeven_is_still_computed_for_a_credit_spread(self):
        v = verdict_for(_spread())
        self.assertIsNotNone(v.breakeven)
        self.assertAlmostEqual(v.breakeven, 1.0 - 2.28 / 10.0, places=2)

    def test_it_still_appears_in_the_reason_text(self):
        v = verdict_for(_spread())
        self.assertIn("breakeven", v.reason)


class TestTheOtherGatesStillBite(unittest.TestCase):
    """Only the breakeven refusal was removed."""

    def test_friction_still_refuses(self):
        wide = {"strategy_name": "Bull Put", "spread_width": 10.0,
                "short_bid": 5.00, "short_ask": 6.50,
                "long_bid": 2.00, "long_ask": 3.50}
        v = verdict_for(wide)
        self.assertFalse(v.passed)
        self.assertIn("friction", v.reason)

    def test_an_unquotable_candidate_is_still_refused(self):
        v = verdict_for({"strategy_name": "Bull Put", "spread_width": 10.0})
        self.assertFalse(v.priced)
        self.assertIn("quote", v.reason)


class TestTheDeadPlumbingIsDocumented(unittest.TestCase):

    def test_the_refusal_string_is_gone_from_the_source(self):
        from src.paths import repo_path
        with open(repo_path("src/candidate_verdict.py")) as fh:
            src = fh.read()
        self.assertNotIn("needs a {breakeven:.0%} win rate", src)


if __name__ == "__main__":
    unittest.main()
