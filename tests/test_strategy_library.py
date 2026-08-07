"""The library, and the properties that stop it fooling us."""
from __future__ import annotations

import tempfile
import unittest

from src.strategies import friction as fr
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


class DteWindowTest(unittest.TestCase):
    """A window narrower than the data's expiration ladder measures nothing.

    The DoltHub backfill carries a MEDIAN of 2.2 expirations per symbol-day, so
    a narrow window contains no listed expiration on most dates. Measured over
    25 randomly sampled symbols (share of symbol-days with any expiry in range):

        [30,45]  52.4%      [7,21]   55.4%
        [25,45]  69.6%      [7,35]   95.3%
        [25,60]  99.8%      [7,45]  100.0%

    Span is what drives it, not where the window sits: every window spanning 35
    days or more cleared 99%, and every window under 30 days failed. The first
    backtest of index_put_spread_w25 returned n=3 for exactly this reason.

    SPY alone is denser (3.2 expirations/day) than the universe, so a window
    validated on SPY is NOT validated for the library. That mistake was made
    once here already.
    """

    MIN_SPAN = 35

    def test_every_window_spans_the_expiration_ladder(self):
        for r in LIBRARY:
            dte = r.spec.entry.get("dte")
            if not dte:
                continue
            lo, hi = int(dte[0]), int(dte[1])
            self.assertGreaterEqual(
                hi - lo, self.MIN_SPAN,
                f"{r.spec.id}: dte {dte} spans {hi - lo}d; under 30 days the "
                f"ladder is empty on ~30-45% of symbol-days")


class WiderIndexSpreadTest(unittest.TestCase):
    """The replacement for the killed single-name bull puts: attack the toll by
    widening the structure and tightening the universe, not by finding a signal."""

    def setUp(self):
        self.rec = [r for r in LIBRARY if r.spec.id == "index_put_spread_w25"][0]

    def test_it_is_a_25_wide_bull_put_on_the_index_only(self):
        self.assertEqual(self.rec.spec.structure, "bull_put")
        self.assertEqual(self.rec.spec.entry["width"], 25.0)
        self.assertEqual(self.rec.spec.universe["symbols"], ["SPY"])

    def test_width_is_the_only_difference_from_the_surviving_index_probe(self):
        """One variable moves, or the comparison means nothing."""
        other = [r for r in LIBRARY if r.spec.id == "csp_index_only"][0]
        a = dict(self.rec.spec.entry)
        b = dict(other.spec.entry)
        self.assertNotEqual(a.pop("width"), b.pop("width"))
        self.assertEqual(a, b)
        self.assertEqual(self.rec.spec.universe, other.spec.universe)
        self.assertEqual(self.rec.spec.exit, other.spec.exit)

    def test_the_account_caps_it_at_one_position(self):
        """~$2,250 of risk against a $4,000 book. Arithmetic, not preference."""
        self.assertEqual(self.rec.spec.sizing["max_concurrent"], 1)
        self.assertIn("56%", self.rec.capital_note)

    def test_the_hypothesis_carries_the_deflation_caveat(self):
        """The $25 row read DSR 0.921 alone and 0.432 deflated. Both belong."""
        self.assertIn("0.432", self.rec.hypothesis)

    def test_it_is_a_hypothesis_not_a_validated_setup(self):
        self.assertNotIn(self.rec.status, ("validated", "promoted", "live"))

    def test_the_replay_result_is_recorded_on_it(self):
        """A run that happened and was not written down gets quoted later
        without its caveats."""
        self.assertEqual(self.rec.evidence["n"], 28)
        self.assertEqual(self.rec.verdict, "reject")
        self.assertEqual(self.rec.status, "dead")

    def test_the_evidence_keeps_the_trial_count_and_the_capacity(self):
        self.assertEqual(self.rec.evidence["n_trials"], 35)
        self.assertIn("cagr_on_deployed", self.rec.evidence["capacity"])

    def test_the_evidence_says_it_was_in_sample(self):
        self.assertIn("IN-SAMPLE", self.rec.evidence["sample"])

    def test_the_undeflated_number_is_kept_beside_the_deflated_one(self):
        """0.83 alone and 0.25 deflated. Storing only one invites the wrong quote."""
        self.assertGreater(self.rec.evidence["dsr_undeflated"],
                           self.rec.evidence["dsr"])

    def test_its_friction_is_unmeasured_not_borrowed_from_single_names(self):
        p = fr.profile_for(self.rec, table=fr.RECORDED)
        self.assertFalse(p.measured)


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
