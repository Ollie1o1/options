"""A setup carries its reasoning and its account eligibility, not just params."""
from __future__ import annotations

import unittest

from src.strategies.record import ACCOUNTS, STATUSES, StrategyRecord
from src.strategies.spec import StrategySpec


def _spec(**kw):
    base = dict(id="wheel_csp", version=1, structure="short_put",
                universe={"strata": ["liquid"], "max_price": 40},
                entry={"dte": [30, 45], "short_delta": 0.25, "iv_rank_min": 50},
                exit={"profit_target": 0.5, "hold_to_expiry": False},
                sizing={"max_capital_at_risk": 4000, "max_concurrent": 2},
                created="2026-08-06", trial_count=0)
    base.update(kw)
    return StrategySpec(**base)


def _rec(**kw):
    base = dict(
        spec=_spec(), name="Wheel: cash-secured put",
        hypothesis="Selling puts when implied vol is rich relative to its own "
                   "history collects the variance risk premium.",
        signal={"iv_rank_min": 50, "no_earnings_before_expiry": True},
        accounts=["tfsa", "taxable"],
        capital_note="Cash-secured: needs strike x 100. Sub-$40 names at $4k.",
        status="specified", evidence={}, cost_profile={}, verdict=None,
        provenance={"created": "2026-08-06", "role": "candidate"},
        links=[], amendments=[])
    base.update(kw)
    return StrategyRecord(**base)


class RecordBasicsTest(unittest.TestCase):
    def test_statuses_and_accounts_are_known(self):
        self.assertIn("dead", STATUSES)
        self.assertIn("tfsa", ACCOUNTS)

    def test_roundtrip_through_dict(self):
        r = _rec()
        back = StrategyRecord.from_dict(r.to_dict())
        self.assertEqual(back.name, r.name)
        self.assertEqual(back.signal["iv_rank_min"], 50)
        self.assertEqual(back.accounts, ["tfsa", "taxable"])

    def test_invalid_status_is_rejected(self):
        with self.assertRaises(ValueError):
            _rec(status="probably_fine")

    def test_invalid_account_is_rejected(self):
        with self.assertRaises(ValueError):
            _rec(accounts=["chequing"])

    def test_every_setup_must_declare_an_account(self):
        with self.assertRaises(ValueError):
            _rec(accounts=[])

    def test_tradeable_in(self):
        r = _rec(accounts=["taxable"])
        self.assertTrue(r.tradeable_in("taxable"))
        self.assertFalse(r.tradeable_in("tfsa"))

    def test_dead_and_retired_are_settled(self):
        self.assertTrue(_rec(status="dead").is_settled())
        self.assertFalse(_rec(status="validated").is_settled())


class AmendmentTest(unittest.TestCase):
    def test_amend_records_the_old_value_and_reason(self):
        r = _rec(status="validated").amend(
            "status", "dead", reason="died on the wider universe",
            date="2026-09-01")
        self.assertEqual(r.status, "dead")
        self.assertEqual(r.amendments[-1]["from"], "validated")
        self.assertEqual(r.amendments[-1]["reason"], "died on the wider universe")

    def test_amend_does_not_mutate_the_original(self):
        r = _rec(status="validated")
        r.amend("status", "dead", reason="x", date="2026-09-01")
        self.assertEqual(r.status, "validated")

    def test_amending_an_unknown_field_raises(self):
        with self.assertRaises(ValueError):
            _rec().amend("shoe_size", 42, reason="x", date="2026-09-01")


if __name__ == "__main__":
    unittest.main()
