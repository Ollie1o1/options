"""Shape tests for the shared catalyst dataclasses."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst.models import CatalystEvent, Coverage, Trial


def a_trial(**kw):
    base = dict(
        nct_id="NCT06510816",
        sponsor_name="Annexon, Inc.",
        brief_title="A Study Investigating Vonaprument in Dry AMD With GA",
        phase="PHASE3",
        event_date="2026-10-31",
        date_precision="day",
        date_type="ESTIMATED",
        status="ACTIVE_NOT_RECRUITING",
        enrollment=400,
        allocation="RANDOMIZED",
        masking="QUADRUPLE",
        primary_outcome="Change in GA lesion area",
        conditions=("Geographic Atrophy",),
    )
    base.update(kw)
    return Trial(**base)


class TestTrial(unittest.TestCase):
    def test_is_frozen(self):
        t = a_trial()
        with self.assertRaises(Exception):
            t.nct_id = "NCT000"  # type: ignore[misc]

    def test_optional_fields_default_to_none_not_zero(self):
        t = a_trial(enrollment=None, allocation=None)
        self.assertIsNone(t.enrollment)
        self.assertIsNone(t.allocation)


class TestCatalystEvent(unittest.TestCase):
    def test_event_id_is_nct_plus_type(self):
        ev = CatalystEvent(trial=a_trial(), ticker="ANNX", mcap=976_332_558.0)
        self.assertEqual(ev.event_id, "NCT06510816:PRIMARY_COMPLETION")

    def test_exposes_trial_fields_for_convenience(self):
        ev = CatalystEvent(trial=a_trial(), ticker="ANNX", mcap=976_332_558.0)
        self.assertEqual(ev.event_date, "2026-10-31")
        self.assertEqual(ev.phase, "PHASE3")


class TestCoverage(unittest.TestCase):
    def test_counts_start_at_zero_and_accumulate(self):
        c = Coverage()
        c.swept = 599
        c.resolved = 162
        c.dropped_unresolved = 437
        self.assertEqual(c.swept - c.resolved, c.dropped_unresolved)

    def test_renders_a_one_line_summary(self):
        c = Coverage(swept=599, resolved=162, dropped_unresolved=437,
                     dropped_out_of_band=40, deep_failures=3)
        line = c.summary()
        self.assertIn("599", line)
        self.assertIn("162", line)
        self.assertIn("27.0%", line)


if __name__ == "__main__":
    unittest.main()
