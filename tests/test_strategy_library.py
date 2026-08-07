"""The library, and the properties that stop it fooling us."""
from __future__ import annotations

import tempfile
import unittest

from src.strategies.record import STATUSES
from src.strategies.seed import LIBRARY, seed_library


class ShapeTest(unittest.TestCase):
    def test_every_setup_states_a_hypothesis(self):
        for r in LIBRARY:
            self.assertTrue(r.hypothesis.strip(), f"{r.spec.id}: no hypothesis")

    def test_every_setup_declares_accounts(self):
        for r in LIBRARY:
            self.assertTrue(r.accounts, f"{r.spec.id}: no account declared")

    def test_every_setup_states_its_capital_requirement(self):
        for r in LIBRARY:
            self.assertTrue(r.capital_note.strip(),
                            f"{r.spec.id}: capital requirement not stated")

    def test_ids_are_unique(self):
        ids = [r.spec.id for r in LIBRARY]
        self.assertEqual(len(ids), len(set(ids)))

    def test_statuses_are_valid(self):
        for r in LIBRARY:
            self.assertIn(r.status, STATUSES)


class AccountLegalityTest(unittest.TestCase):
    """A registered account cannot sell naked. Catch it here, not at the broker."""

    NAKED = ("naked_put", "naked_call", "short_strangle")

    def test_no_naked_setup_is_filed_under_tfsa(self):
        for r in LIBRARY:
            if r.spec.structure in self.NAKED:
                self.assertNotIn("tfsa", r.accounts,
                                 f"{r.spec.id} sells naked but is filed as TFSA")
                self.assertNotIn("both", r.accounts,
                                 f"{r.spec.id} sells naked but is filed as both")

    def test_tfsa_setups_are_secured_or_covered(self):
        allowed = ("short_put", "covered_call", "bull_put", "bear_call",
                   "iron_condor", "long_call", "long_put")
        for r in LIBRARY:
            if "tfsa" in r.accounts or "both" in r.accounts:
                self.assertIn(r.spec.structure, allowed,
                              f"{r.spec.id}: {r.spec.structure} in a TFSA")


class SignalTierTest(unittest.TestCase):
    def test_iv_rank_conditioned_setups_exist(self):
        """The genuinely open question: does timing on IV rank help?"""
        conditioned = [r for r in LIBRARY if r.signal.get("iv_rank_min")]
        self.assertGreaterEqual(len(conditioned), 3)

    def test_directional_setups_exist_for_both_directions(self):
        d = [r for r in LIBRARY if r.provenance.get("role") == "directional"]
        self.assertTrue(any(r.signal.get("above_sma50") for r in d))
        self.assertTrue(any(r.signal.get("below_sma50") for r in d))

    def test_directional_views_are_expressed_as_short_premium(self):
        """A view on direction is sold, not bought — one way to win becomes three."""
        for r in LIBRARY:
            if r.provenance.get("role") == "directional":
                self.assertIn(r.spec.structure,
                              ("bull_put", "bear_call", "short_put"),
                              f"{r.spec.id} buys premium to express a view")

    def test_an_expression_control_shares_a_signal_with_a_directional_setup(self):
        """Isolates expression from signal: same entry days, different structure."""
        ctrl = [r for r in LIBRARY
                if r.provenance.get("role") == "expression_control"]
        self.assertTrue(ctrl)
        directional = [r for r in LIBRARY
                       if r.provenance.get("role") == "directional"]
        self.assertTrue(
            any(c.signal == d.signal for c in ctrl for d in directional),
            "the expression control must share a signal with a directional setup")

    def test_an_index_and_a_single_name_probe_both_exist(self):
        roles = {r.provenance.get("role") for r in LIBRARY}
        self.assertIn("index_probe", roles)
        self.assertIn("single_name_probe", roles)


class ControlTierTest(unittest.TestCase):
    """Controls are the point. Without them a broken engine looks brilliant."""

    def test_an_unselected_benchmark_exists(self):
        """If a signal cannot beat 'every eligible day', it is decoration."""
        b = [r for r in LIBRARY if r.provenance.get("role") == "benchmark"]
        self.assertTrue(b)

    def test_the_benchmark_has_no_signal_conditions(self):
        b = [r for r in LIBRARY if r.provenance.get("role") == "benchmark"][0]
        self.assertFalse(b.signal.get("iv_rank_min"))

    def test_a_null_control_exists(self):
        self.assertTrue([r for r in LIBRARY
                         if r.provenance.get("role") == "null_control"])

    def test_a_known_negative_control_exists_and_is_dead(self):
        neg = [r for r in LIBRARY
               if r.provenance.get("role") == "known_negative"]
        self.assertTrue(neg)
        self.assertEqual(neg[0].status, "dead")


class SeedTest(unittest.TestCase):
    def test_seeding_writes_every_setup(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(len(seed_library(d)), len(LIBRARY))

    def test_seeding_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            seed_library(d)
            seed_library(d)
            from src.strategies.registry import Registry
            self.assertEqual(len(Registry(d).list()), len(LIBRARY))


if __name__ == "__main__":
    unittest.main()
