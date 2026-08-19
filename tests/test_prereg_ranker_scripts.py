"""Tests for the pre-registration scripts.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_prereg_ranker_scripts -v

NOTHING here runs the real test against the real cohort. Doing so before n* or
the deadline would destroy the pre-registration this sub-project exists to
protect.
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts import prereg_ranker_power as power


def _cohort(n=500, seed=2):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "entry_date": [f"2026-08-{1 + (i % 20):02d}" for i in range(n)],
        "strategy": ["Long Call"] * n,
        "contract_key": [f"K{i // 5}" for i in range(n)],
        "pnl_pct": rng.normal(size=n),
        "ev_net": rng.normal(size=n),
        "quality_score": rng.normal(size=n),
        "carry": rng.normal(size=n),
        "delta": rng.normal(size=n),
    })


class TestBuildRegistration(unittest.TestCase):
    def _text(self, **over):
        kwargs = dict(target_ic=0.08, power=0.80, alpha=0.05,
                      deadline="2026-11-19", n_boot=10000, seed=20260819,
                      assumed_icc=0.11)
        kwargs.update(over)
        return power.build_registration(_cohort(), **kwargs)

    def test_it_states_the_hypothesis_and_the_decision_rule(self):
        text = self._text()
        for needle in ("ev_net", "PASS", "FAIL", "UNDERPOWERED", "INVERTED",
                       "2026-11-19", "contract_key"):
            self.assertIn(needle, text)

    def test_it_records_a_concrete_required_n(self):
        text = self._text()
        self.assertIn("n_star_nominal:", text)
        self.assertGreater(float(power.parse_field(text, "n_star_nominal")), 1000)

    def test_the_parameters_are_machine_readable(self):
        text = self._text()
        self.assertEqual(power.parse_field(text, "deadline"), "2026-11-19")
        self.assertEqual(power.parse_field(text, "target_ic"), "0.08")
        self.assertEqual(power.parse_field(text, "seed"), "20260819")
        self.assertEqual(power.parse_field(text, "min_cell_rows"), "3")

    def test_a_bigger_target_effect_needs_fewer_observations(self):
        small = float(power.parse_field(self._text(target_ic=0.08),
                                        "n_star_nominal"))
        big = float(power.parse_field(self._text(target_ic=0.20),
                                      "n_star_nominal"))
        self.assertGreater(small, big)

    def test_it_refuses_to_overwrite_an_existing_registration(self):
        # The registration is immutable once written. Rewriting it after
        # outcomes exist is exactly the abuse pre-registration prevents.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reg.md")
            power.write_registration("first", path)
            with self.assertRaises(FileExistsError):
                power.write_registration("second", path)
            with open(path) as fh:
                self.assertEqual(fh.read(), "first")

    def test_a_missing_field_reads_as_none(self):
        self.assertIsNone(power.parse_field(self._text(), "no_such_field"))


if __name__ == "__main__":
    unittest.main()
