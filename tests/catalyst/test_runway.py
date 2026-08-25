"""Cash runway from EDGAR XBRL. Network mocked throughout."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst import runway

CASH_PAYLOAD = {"units": {"USD": [
    {"end": "2025-12-31", "filed": "2026-02-20", "val": 500_000_000.0, "form": "10-K"},
    {"end": "2026-03-31", "filed": "2026-05-05", "val": 455_000_000.0, "form": "10-Q"},
    {"end": "2026-06-30", "filed": "2026-08-05", "val": 412_000_000.0, "form": "10-Q"},
]}}

# Operating cash flow is reported as a NEGATIVE number for a cash burner.
# 2025-07-01..2026-06-30 is a 364-day (annual) frame.
FLOW_PAYLOAD = {"units": {"USD": [
    {"start": "2025-07-01", "end": "2026-06-30", "filed": "2026-08-05",
     "val": -232_000_000.0, "form": "10-Q"},
]}}

ANNUAL_DAYS = 364
EXPECTED_BURN = 232_000_000.0 * runway.DAYS_PER_QUARTER / ANNUAL_DAYS

# Real SRPT shape, recorded 2026-08-25. The 180-day year-to-date frame is the
# most recent, and naively dividing it by 4 produced a $1.4m quarterly burn and
# 407 quarters of runway ending in 2128.
SRPT_FLOW = {"units": {"USD": [
    {"start": "2025-01-01", "end": "2025-12-31", "val": -205_500_000.0, "form": "10-K"},
    {"start": "2026-01-01", "end": "2026-03-31", "val": -202_700_000.0, "form": "10-Q"},
    {"start": "2026-01-01", "end": "2026-06-30", "val": -5_600_000.0, "form": "10-Q"},
]}}


class TestParseConcept(unittest.TestCase):
    def test_returns_points_sorted_by_end_date(self):
        pts = runway.parse_concept(CASH_PAYLOAD, "USD")
        self.assertEqual([p[1] for p in pts],
                         ["2025-12-31", "2026-03-31", "2026-06-30"])

    def test_instant_concept_has_no_start(self):
        pts = runway.parse_concept(CASH_PAYLOAD, "USD")
        self.assertTrue(all(p[0] is None for p in pts))

    def test_duration_concept_carries_its_start(self):
        pts = runway.parse_concept(FLOW_PAYLOAD, "USD")
        self.assertEqual(pts[0][0], "2025-07-01")

    def test_skips_points_missing_a_value(self):
        payload = {"units": {"USD": [{"end": "2026-06-30", "val": None}]}}
        self.assertEqual(runway.parse_concept(payload, "USD"), [])

    def test_absent_unit_is_empty(self):
        self.assertEqual(runway.parse_concept({"units": {}}, "USD"), [])

    def test_republished_duplicate_frames_are_collapsed(self):
        payload = {"units": {"USD": [
            {"start": "2025-01-01", "end": "2025-03-31", "val": -583.4, "fy": 2025},
            {"start": "2025-01-01", "end": "2025-03-31", "val": -583.4, "fy": 2026},
        ]}}
        self.assertEqual(len(runway.parse_concept(payload, "USD")), 1)


class TestQuarterlyBurn(unittest.TestCase):
    def test_normalises_an_annual_frame_by_its_real_span(self):
        burn, basis = runway.quarterly_burn(
            runway.parse_concept(FLOW_PAYLOAD, "USD"))
        self.assertAlmostEqual(burn, EXPECTED_BURN, delta=1.0)
        self.assertIn("364d", basis)

    def test_normalises_a_year_to_date_frame_by_its_own_span(self):
        ytd = {"units": {"USD": [
            {"start": "2026-01-01", "end": "2026-06-30", "val": -100_000_000.0}]}}
        burn, basis = runway.quarterly_burn(runway.parse_concept(ytd, "USD"))
        # 180 days, not 4 quarters.
        self.assertAlmostEqual(burn, 100_000_000.0 * runway.DAYS_PER_QUARTER / 180,
                               delta=1.0)
        self.assertIn("180d", basis)

    def test_prefers_the_annual_frame_over_a_fresher_short_one(self):
        burn, basis = runway.quarterly_burn(
            runway.parse_concept(SRPT_FLOW, "USD"))
        self.assertIn("2025-12-31", basis)
        self.assertAlmostEqual(
            burn, 205_500_000.0 * runway.DAYS_PER_QUARTER / 364, delta=1.0)

    def test_the_srpt_regression_does_not_recur(self):
        # The bug produced ~$1.4m/quarter. Anything near that is the defect.
        burn, _ = runway.quarterly_burn(runway.parse_concept(SRPT_FLOW, "USD"))
        self.assertGreater(burn, 40_000_000.0)

    def test_no_dated_frame_yields_none_not_zero(self):
        undated = {"units": {"USD": [{"end": "2026-06-30", "val": -50.0}]}}
        self.assertEqual(
            runway.quarterly_burn(runway.parse_concept(undated, "USD")),
            (None, None))

    def test_positive_flow_yields_a_negative_burn(self):
        positive = {"units": {"USD": [
            {"start": "2025-07-01", "end": "2026-06-30", "val": 120_000_000.0}]}}
        burn, _ = runway.quarterly_burn(runway.parse_concept(positive, "USD"))
        self.assertLess(burn, 0)


class TestFundedThrough(unittest.TestCase):
    def test_true_when_runway_outlasts_the_catalyst(self):
        self.assertTrue(runway.funded_through("2028-03-31", "2026-10-31"))

    def test_false_when_the_company_must_raise_first(self):
        self.assertFalse(runway.funded_through("2026-09-30", "2027-02-28"))

    def test_none_when_either_side_is_unknown(self):
        self.assertIsNone(runway.funded_through(None, "2026-10-31"))
        self.assertIsNone(runway.funded_through("2028-03-31", None))

    def test_month_precision_catalyst_date_is_handled(self):
        self.assertTrue(runway.funded_through("2028-03-31", "2027-03"))


class TestRunwayFor(unittest.TestCase):
    def _patched(self):
        return mock.patch.multiple(
            runway,
            _cik=mock.DEFAULT,
            _concept=mock.DEFAULT)

    def test_computes_quarters_and_funded_flag(self):
        with self._patched() as m:
            m["_cik"].return_value = 1_000_000
            m["_concept"].side_effect = [CASH_PAYLOAD, FLOW_PAYLOAD]
            r = runway.runway_for("ANNX", "2026-10-31")
        self.assertEqual(r.cash, 412_000_000.0)
        self.assertAlmostEqual(r.burn_per_quarter, EXPECTED_BURN, delta=1.0)
        self.assertAlmostEqual(r.quarters, 412_000_000.0 / EXPECTED_BURN,
                               places=4)
        self.assertTrue(r.funded_through)
        self.assertIsNotNone(r.burn_basis)

    def test_unknown_cik_yields_all_none(self):
        with mock.patch.object(runway, "_cik", return_value=None):
            r = runway.runway_for("ZZZZ", "2026-10-31")
        self.assertIsNone(r.cash)
        self.assertIsNone(r.quarters)
        self.assertIsNone(r.funded_through)

    def test_network_failure_yields_all_none_not_raise(self):
        with self._patched() as m:
            m["_cik"].return_value = 1_000_000
            m["_concept"].side_effect = OSError("boom")
            r = runway.runway_for("ANNX", "2026-10-31")
        self.assertIsNone(r.cash)

    def test_a_cash_generating_company_has_no_runway_limit(self):
        positive = {"units": {"USD": [
            {"start": "2025-07-01", "end": "2026-06-30", "filed": "2026-08-05",
             "val": 120_000_000.0, "form": "10-Q"}]}}
        with self._patched() as m:
            m["_cik"].return_value = 1_000_000
            m["_concept"].side_effect = [CASH_PAYLOAD, positive]
            r = runway.runway_for("PROF", "2026-10-31")
        self.assertIsNone(r.quarters)
        self.assertTrue(r.cash_generative)


if __name__ == "__main__":
    unittest.main()
