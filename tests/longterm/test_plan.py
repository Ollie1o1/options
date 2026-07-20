"""Tests for the long-term plan file (longterm_plan.json)."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.longterm import plan as P


def mu_plan():
    return P.Plan(cash_pool_usd=5000.0, names=[
        P.PlanName("MU", [P.Tranche(750, 0.4), P.Tranche(650, 0.35), P.Tranche(550, 0.25)],
                   thesis="AI memory leader, HBM", allocation=0.4),
        P.PlanName("ASML", [P.Tranche(900, 0.5), P.Tranche(800, 0.5)]),
        P.PlanName("KO", [P.Tranche(60, 1.0)]),
    ])


class TestPlanRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "longterm_plan.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_autocreates_empty_plan(self):
        plan = P.load_plan(self.path)
        self.assertEqual(plan.cash_pool_usd, 0.0)
        self.assertEqual(plan.names, [])
        self.assertTrue(os.path.exists(self.path))  # auto-created on disk

    def test_round_trip_preserves_everything(self):
        P.save_plan(mu_plan(), self.path)
        back = P.load_plan(self.path)
        self.assertEqual(back, mu_plan())

    def test_ticker_uppercased_and_default_weights(self):
        with open(self.path, "w") as f:
            json.dump({"cash_pool_usd": 1000, "names": [
                {"ticker": "mu", "tranches": [{"level": 750}, {"level": 650}]}]}, f)
        plan = P.load_plan(self.path)
        self.assertEqual(plan.names[0].ticker, "MU")
        self.assertEqual([t.weight for t in plan.names[0].tranches], [0.5, 0.5])

    def test_tolerates_missing_optional_keys(self):
        with open(self.path, "w") as f:
            json.dump({"names": [{"ticker": "MU", "tranches": []}]}, f)
        plan = P.load_plan(self.path)
        self.assertEqual(plan.cash_pool_usd, 0.0)
        self.assertEqual(plan.names[0].thesis, "")
        self.assertIsNone(plan.names[0].allocation)


class TestSizing(unittest.TestCase):
    def test_allocations_explicit_plus_equal_split(self):
        # MU explicit 0.4; ASML + KO split the remaining 0.6 → 0.3 each
        alloc = P.allocations(mu_plan())
        self.assertAlmostEqual(alloc["MU"], 0.4)
        self.assertAlmostEqual(alloc["ASML"], 0.3)
        self.assertAlmostEqual(alloc["KO"], 0.3)

    def test_all_implicit_split_equally(self):
        plan = P.Plan(1000, [P.PlanName("A", [P.Tranche(1, 1.0)]),
                             P.PlanName("B", [P.Tranche(1, 1.0)])])
        self.assertAlmostEqual(P.allocations(plan)["A"], 0.5)

    def test_tranche_size_usd(self):
        plan = mu_plan()
        # 5000 × 0.4 (MU) × 0.4 (first tranche) = 800
        self.assertAlmostEqual(P.tranche_size_usd(plan, plan.names[0], plan.names[0].tranches[0]), 800.0)

    def test_empty_plan_sizes_zero(self):
        plan = P.Plan()
        self.assertEqual(P.allocations(plan), {})


if __name__ == "__main__":
    unittest.main()
