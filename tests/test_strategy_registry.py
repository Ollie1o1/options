"""Strategies are stored objects, and the store counts trials.

trial_count is load-bearing, not bookkeeping: Deflated Sharpe deflates by the
number of configurations tried, so an undercount silently inflates every result.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_strategy_registry -v
"""
from __future__ import annotations

import tempfile
import unittest

from src.strategies.registry import Registry
from src.strategies.spec import StrategySpec


def _spec(**kw):
    base = dict(
        id="bull_put_30d", version=1, structure="bull_put",
        universe={"strata": ["liquid"], "min_depth": 40},
        entry={"dte": [25, 45], "short_delta": 0.25},
        exit={"profit_target": 0.5, "stop": 2.0, "time_exit_dte": 21,
              "hold_to_expiry": False},
        sizing={"max_capital_at_risk": 4000, "max_concurrent": 5},
        created="2026-08-06", trial_count=0)
    base.update(kw)
    return StrategySpec(**base)


class SpecTest(unittest.TestCase):
    def test_spec_is_frozen(self):
        with self.assertRaises(Exception):
            _spec().id = "changed"

    def test_holding_days_uses_max_dte(self):
        self.assertEqual(
            _spec(exit={"hold_to_expiry": True}).holding_days(), 45)

    def test_holding_days_respects_time_exit(self):
        s = _spec(exit={"time_exit_dte": 30, "hold_to_expiry": False})
        self.assertEqual(s.holding_days(), 15)

    def test_holding_days_without_time_exit_is_full_dte(self):
        self.assertEqual(_spec(exit={"profit_target": 0.5}).holding_days(), 45)

    def test_fingerprint_ignores_trial_count(self):
        self.assertEqual(_spec(trial_count=0).fingerprint(),
                         _spec(trial_count=99).fingerprint())

    def test_fingerprint_changes_with_parameters(self):
        self.assertNotEqual(
            _spec().fingerprint(),
            _spec(entry={"dte": [7, 21], "short_delta": 0.3}).fingerprint())


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.reg = Registry(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_then_load_roundtrip(self):
        self.reg.save(_spec())
        got = self.reg.load("bull_put_30d")
        self.assertEqual(got.structure, "bull_put")
        self.assertEqual(got.entry["dte"], [25, 45])

    def test_list_returns_all_saved(self):
        self.reg.save(_spec())
        self.reg.save(_spec(id="bear_call_30d", structure="bear_call"))
        self.assertEqual({s.id for s in self.reg.list()},
                         {"bull_put_30d", "bear_call_30d"})

    def test_trial_count_starts_at_zero(self):
        self.assertEqual(self.reg.trial_count, 0)

    def test_saving_a_new_spec_records_a_trial(self):
        self.reg.save(_spec())
        self.assertEqual(self.reg.trial_count, 1)

    def test_saving_a_changed_spec_records_another_trial(self):
        self.reg.save(_spec())
        self.reg.save(_spec(version=2,
                            entry={"dte": [7, 21], "short_delta": 0.3}))
        self.assertEqual(self.reg.trial_count, 2)

    def test_resaving_an_identical_spec_does_not_record_a_trial(self):
        """Re-running an unchanged spec is not a new search."""
        self.reg.save(_spec())
        self.reg.save(_spec())
        self.assertEqual(self.reg.trial_count, 1)

    def test_trial_count_survives_a_new_registry_object(self):
        self.reg.save(_spec())
        self.assertEqual(Registry(self._tmp.name).trial_count, 1)

    def test_record_trial_counts_abandoned_configs(self):
        """Configs tried and thrown away still inflate the search."""
        self.reg.record_trial()
        self.reg.record_trial()
        self.assertEqual(self.reg.trial_count, 2)

    def test_saved_spec_carries_the_registry_trial_count(self):
        self.reg.save(_spec())
        self.assertEqual(self.reg.load("bull_put_30d").trial_count, 1)

    def test_trials_file_is_not_listed_as_a_spec(self):
        self.reg.save(_spec())
        self.assertEqual(len(self.reg.list()), 1)

    def test_corrupt_trials_file_does_not_raise(self):
        with open(self.reg._trials_path, "w") as f:
            f.write("{not json")
        self.assertEqual(self.reg.trial_count, 0)

    def test_record_trial_default_branch(self):
        """Called with no fingerprint at all."""
        self.assertEqual(self.reg.record_trial(), 1)


if __name__ == "__main__":
    unittest.main()
