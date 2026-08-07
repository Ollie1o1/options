"""Results land on the record, and a signal must beat doing nothing clever."""
from __future__ import annotations

import unittest

from src.strategies.evidence import apply_result, beats_benchmark
from src.strategies.seed import LIBRARY


def _result(verdict="promote", dsr=0.9, pbo=0.2, tstat=3.4, sharpe=0.8):
    return {"dsr": dsr, "pbo": pbo, "tstat": tstat, "sharpe": sharpe,
            "verdict": verdict,
            "by_stratum": {"legacy": {"pnl": 100.0}, "liquid": {"pnl": 80.0},
                           "broad": {"pnl": 40.0}},
            "capacity": {"trades_per_year": 31, "cagr_on_deployed": 0.007},
            "n_trials": 14, "window": ["2022-01-07", "2026-06-12"]}


class ApplyResultTest(unittest.TestCase):
    def test_evidence_is_recorded(self):
        r = apply_result(LIBRARY[0], _result(), date="2026-09-01")
        self.assertEqual(r.evidence["dsr"], 0.9)

    def test_promote_advances_to_validated(self):
        r = apply_result(LIBRARY[0], _result("promote"), date="2026-09-01")
        self.assertEqual(r.status, "validated")

    def test_reject_marks_dead(self):
        r = apply_result(LIBRARY[0], _result("reject"), date="2026-09-01")
        self.assertEqual(r.status, "dead")

    def test_liquid_only_does_not_reach_validated(self):
        r = apply_result(LIBRARY[0], _result("liquid_only"), date="2026-09-01")
        self.assertNotEqual(r.status, "validated")

    def test_changes_are_amended_not_overwritten(self):
        r = apply_result(LIBRARY[0], _result("reject"), date="2026-09-01")
        self.assertTrue(r.amendments)

    def test_trial_count_travels_with_the_evidence(self):
        """A deflated Sharpe means nothing without the size of the search."""
        r = apply_result(LIBRARY[0], _result(), date="2026-09-01")
        self.assertEqual(r.evidence["n_trials"], 14)

    def test_capacity_is_kept_beside_edge(self):
        r = apply_result(LIBRARY[0], _result(), date="2026-09-01")
        self.assertIn("cagr_on_deployed", r.evidence["capacity"])

    def test_a_measured_cost_profile_lands_on_the_setup(self):
        """Friction measured during the backtest replaces the table estimate."""
        result = _result()
        result["cost_profile"] = {"per_share": 0.31, "credit": 1.05, "n": 412,
                                  "source": "backtest 2026-09-01"}
        r = apply_result(LIBRARY[0], result, date="2026-09-01")
        self.assertEqual(r.cost_profile["n"], 412)

    def test_no_cost_profile_leaves_the_existing_one_alone(self):
        r = apply_result(LIBRARY[0], _result(), date="2026-09-01")
        self.assertEqual(r.cost_profile, LIBRARY[0].cost_profile)


class BenchmarkTest(unittest.TestCase):
    """Selectivity made this book worse. A signal must EARN its place."""

    def test_a_signal_that_beats_the_benchmark_passes(self):
        self.assertTrue(beats_benchmark(_result(sharpe=1.2),
                                        _result(sharpe=0.8)))

    def test_a_signal_that_ties_the_benchmark_fails(self):
        self.assertFalse(beats_benchmark(_result(sharpe=0.8),
                                         _result(sharpe=0.8)))

    def test_a_signal_that_loses_to_the_benchmark_fails(self):
        self.assertFalse(beats_benchmark(_result(sharpe=0.5),
                                         _result(sharpe=0.8)))


if __name__ == "__main__":
    unittest.main()
