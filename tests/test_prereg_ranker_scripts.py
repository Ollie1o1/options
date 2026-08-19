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


from scripts import prereg_ranker_test as look


class TestDecide(unittest.TestCase):
    def test_an_interval_above_zero_passes(self):
        self.assertEqual(look.decide(0.05, 0.20), "PASS")

    def test_an_interval_containing_zero_fails(self):
        self.assertEqual(look.decide(-0.03, 0.12), "FAIL")

    def test_an_interval_below_zero_is_inverted(self):
        # Reported separately: an ev_net that predicts backwards is real
        # information, but reversing a sign on one look is how overfitting
        # starts, so it does not PASS.
        self.assertEqual(look.decide(-0.25, -0.05), "INVERTED")

    def test_a_missing_interval_fails(self):
        self.assertEqual(look.decide(None, None), "FAIL")


class TestRefusals(unittest.TestCase):
    def _registration(self, d, n_star=1000, deadline="2099-01-01"):
        path = os.path.join(d, "reg.md")
        text = power.build_registration(
            _cohort(), target_ic=0.08, power=0.80, alpha=0.05,
            deadline=deadline, n_boot=50, seed=1, assumed_icc=0.11)
        text = text.replace(
            f"n_star_nominal: {power.parse_field(text, 'n_star_nominal')}",
            f"n_star_nominal: {n_star}")
        power.write_registration(text, path)
        return path

    def _db(self, d, n_closed):
        from src import candidate_marks as cm
        path = os.path.join(d, "c.db")
        with cm.connect(path) as conn:
            for i in range(n_closed):
                conn.execute(
                    "INSERT OR REPLACE INTO candidates (scan_id, ts, board,"
                    " mode, contract_key, symbol, opt_type, expiration,"
                    " ev_net, quality_score, theta, premium, delta,"
                    " gate_passed) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (f"S{i}", "2026-08-19T00:00:00+00:00", "b",
                     "Discovery scan", f"K{i}", "AAPL", "call", "2026-09-18",
                     float(i), 0.5, -0.05, 10.0, 0.5, 1))
                conn.execute(
                    "INSERT OR REPLACE INTO candidate_positions (scan_id,"
                    " board, contract_key, family, entry_date, entry_price,"
                    " status, pnl_pct) VALUES (?,?,?,?,?,?,?,?)",
                    (f"S{i}", "b", f"K{i}", "long_option",
                     f"2026-08-{1 + (i % 20):02d}", -10.0, "CLOSED",
                     float(i % 7) / 7.0))
            conn.commit()
        return path

    def test_it_refuses_below_n_star_and_before_the_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            reg = self._registration(d, n_star=1000, deadline="2099-01-01")
            db = self._db(d, 20)
            out = look.run(reg, db, today="2026-08-19")
            self.assertEqual(out["status"], "NOT_YET")
            self.assertNotIn("decision", out)
            with open(reg) as fh:
                self.assertIn("*Not yet run.*", fh.read())

    def test_reaching_n_star_permits_the_look(self):
        with tempfile.TemporaryDirectory() as d:
            reg = self._registration(d, n_star=10, deadline="2099-01-01")
            db = self._db(d, 40)
            out = look.run(reg, db, today="2026-08-19")
            self.assertIn(out["decision"], ("PASS", "FAIL", "INVERTED"))

    def test_passing_the_deadline_permits_the_look_and_can_underpower(self):
        with tempfile.TemporaryDirectory() as d:
            reg = self._registration(d, n_star=100000, deadline="2026-01-01")
            db = self._db(d, 40)
            out = look.run(reg, db, today="2026-08-19")
            self.assertEqual(out["decision"], "UNDERPOWERED")

    def test_a_second_invocation_returns_the_stored_result(self):
        # Mutating the data between runs must not change the answer.
        with tempfile.TemporaryDirectory() as d:
            reg = self._registration(d, n_star=10, deadline="2099-01-01")
            db = self._db(d, 40)
            first = look.run(reg, db, today="2026-08-19")

            from src import candidate_marks as cm
            with cm.connect(db) as conn:
                conn.execute("UPDATE candidate_positions SET pnl_pct = 99.0")
                conn.commit()

            second = look.run(reg, db, today="2026-08-19")
            self.assertEqual(second["status"], "ALREADY_RUN")
            self.assertEqual(second["decision"], first["decision"])

    def test_the_result_is_written_into_the_registration(self):
        with tempfile.TemporaryDirectory() as d:
            reg = self._registration(d, n_star=10, deadline="2099-01-01")
            db = self._db(d, 40)
            look.run(reg, db, today="2026-08-19")
            with open(reg) as fh:
                text = fh.read()
            self.assertNotIn("*Not yet run.*", text)
            self.assertIn("decision:", text)
            self.assertIn("rank_ic:", text)
            self.assertIn("secondary_quality_score_ic:", text)


if __name__ == "__main__":
    unittest.main()
