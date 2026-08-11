"""The DTE range the cost model was actually measured on.

Every friction threshold in this system — WORTH's 5%/10% bands, the auto-log
gate's 25%, `candidate_verdict`'s 25% — was calibrated on the Dolt universe at
DTE 10-67. Nothing declared that, so the numbers were applied at any tenor and
looked authoritative everywhere.

`docs/OPTIONSDX_RESULTS_20260811.md` measured what happens outside it, on SPY
2010-2023:

    band        fric median%   % over the 25% gate
    10-25            3.6              0.8
    25-60            4.8              1.2
    60-120           4.2              0.3
    120-250          7.3              3.5
    250-500         13.0             26.9
    500-1000        23.6             46.0

Past 250 DTE the WORTH bands become unreachable — a median candidate is capped
at THIN by cost alone, so STRONG and CLEAR cannot be earned at any quality.
That is a grade that silently stops meaning anything, which is worse than a
grade that refuses to answer.

The guard changes NO threshold. A 23.6% cost is real and still refused. It only
stops an out-of-range number being reported as though it were in range.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_cost_calibration -v
"""
from __future__ import annotations

import unittest

from src.cost_calibration import (CALIBRATED_DTE, CALIBRATED_MAX_DTE,
                                  OUT_OF_RANGE_REASON,
                                  entry_dte, in_calibration)


class EntryDteTest(unittest.TestCase):
    def test_it_is_computed_from_expiration_and_entry_date(self):
        self.assertEqual(
            entry_dte({"date": "2024-01-01", "expiration": "2024-02-15"}), 45)

    def test_an_explicit_dte_is_used_when_present(self):
        self.assertEqual(entry_dte({"dte": 33}), 33)

    def test_a_timestamped_entry_date_still_works(self):
        self.assertEqual(
            entry_dte({"date": "2024-01-01 09:30:00",
                       "expiration": "2024-02-15"}), 45)

    def test_an_unknowable_tenor_is_none_not_zero(self):
        # None means "cannot tell", and the guard must not fire on it. Zero
        # would read as an expiring contract and ungrade everything.
        self.assertIsNone(entry_dte({}))
        self.assertIsNone(entry_dte({"expiration": "not-a-date"}))


class InCalibrationTest(unittest.TestCase):
    def test_the_ceiling_is_inclusive(self):
        self.assertTrue(in_calibration(CALIBRATED_MAX_DTE))

    def test_one_day_past_the_ceiling_is_out(self):
        self.assertFalse(in_calibration(CALIBRATED_MAX_DTE + 1))

    def test_no_floor_is_enforced(self):
        """A floor was written first and was wrong.

        There is no evidence the cost model breaks at short tenors — 10-25 DTE
        carries the LOWEST measured friction of any band (3.6% median) — and
        `config.filters.min_days_to_expiration` already sets a minimum for its
        own reasons. Enforcing a second one here refused near-dated candidates
        under a cost-model heading the data does not support, and broke 31
        tests whose fixtures use near-dated expirations.
        """
        self.assertTrue(in_calibration(1))
        self.assertTrue(in_calibration(CALIBRATED_DTE[0] - 1))

    def test_an_expired_contract_is_not_this_guard_s_problem(self):
        # A negative tenor is a different defect with its own validation.
        self.assertTrue(in_calibration(-30))

    def test_an_unknown_tenor_is_treated_as_in_range(self):
        # Refusing to grade everything whose tenor cannot be read would be a
        # far larger behaviour change than this guard is entitled to make.
        self.assertTrue(in_calibration(None))

    def test_the_ceiling_clears_what_the_book_actually_trades(self):
        # All 972 logged trades are DTE 8-59; the live filters cap at 45, and
        # 60 for irons. A guard that refused ordinary activity would be a
        # regression, not a safeguard.
        for dte in (8, 45, 59, 60):
            self.assertTrue(in_calibration(dte), f"{dte} DTE must be graded")

    def test_it_excludes_the_tenors_where_the_thresholds_break(self):
        # 26.9% of 250-500 DTE candidates breach a gate that refuses 1.2%
        # inside the calibrated band. That is a different regime.
        self.assertFalse(in_calibration(300))
        self.assertFalse(in_calibration(600))

    def test_there_is_a_stated_reason_to_show_the_user(self):
        self.assertIn(str(CALIBRATED_DTE[1]), OUT_OF_RANGE_REASON)


if __name__ == "__main__":
    unittest.main()
