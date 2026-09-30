"""Before any knob is swept, the harness must reproduce what actually happened.
A lab that cannot replay the live policy has no business scoring alternatives.
"""
import unittest

from src.policy_lab.calibrate import (
    CorpusUnusable, calibration_report, reproduces,
)
from src.policy_lab.costs import CostModel
from src.policy_lab.policies import ExitPolicy
from src.policy_lab.types import PathPoint, PricePath


def _pt(date, mid, dte):
    return PathPoint(date, mid - 0.05, mid + 0.05, mid, 100.0, dte)


def _path(pid, actual_frac, points, car=100.0, strategy="Bull Put"):
    return PricePath(
        position_id=pid, symbol="AMD", strategy=strategy,
        entry_date="2026-08-18", entry_price=1.00, capital_at_risk=car,
        is_credit=True, points=tuple(points),
        actual_exit_date=points[-1].date, actual_pnl_frac=actual_frac,
        corpus="A")


NULL = ExitPolicy("null", None, None, None, None)


class TestCalibration(unittest.TestCase):
    def test_exact_reproduction_passes(self):
        # Entry 1.00, close 0.40 at mid -> 0.60 * 100 / 100 = 0.60
        p = _path("p1", 0.60, [_pt("2026-08-18", 1.00, 31),
                               _pt("2026-08-25", 0.40, 24)])
        self.assertTrue(reproduces(p, NULL, CostModel.mid()))

    def test_small_drift_inside_the_absolute_band_passes(self):
        p = _path("p1", 0.615, [_pt("2026-08-18", 1.00, 31),
                                _pt("2026-08-25", 0.40, 24)])
        self.assertTrue(reproduces(p, NULL, CostModel.mid()))

    def test_large_drift_fails(self):
        p = _path("p1", 0.20, [_pt("2026-08-18", 1.00, 31),
                               _pt("2026-08-25", 0.40, 24)])
        self.assertFalse(reproduces(p, NULL, CostModel.mid()))

    def test_report_counts_passes_and_failures(self):
        good = _path("g", 0.60, [_pt("2026-08-18", 1.00, 31),
                                 _pt("2026-08-25", 0.40, 24)])
        bad = _path("b", 0.20, [_pt("2026-08-18", 1.00, 31),
                                _pt("2026-08-25", 0.40, 24)])
        rep = calibration_report([good] * 19 + [bad], NULL, CostModel.mid())
        self.assertEqual(rep.checked, 20)
        self.assertEqual(rep.passed, 19)
        self.assertEqual(rep.failed, 1)
        self.assertAlmostEqual(rep.fail_rate, 0.05)

    def test_corpus_is_refused_above_ten_percent_failure(self):
        good = _path("g", 0.60, [_pt("2026-08-18", 1.00, 31),
                                 _pt("2026-08-25", 0.40, 24)])
        bad = _path("b", 0.20, [_pt("2026-08-18", 1.00, 31),
                                _pt("2026-08-25", 0.40, 24)])
        with self.assertRaises(CorpusUnusable):
            calibration_report([good] * 5 + [bad] * 5, NULL, CostModel.mid())

    def test_empty_corpus_is_refused(self):
        with self.assertRaises(CorpusUnusable):
            calibration_report([], NULL, CostModel.mid())

    def test_max_fail_rate_is_a_parameter_not_a_constant(self):
        # 50% failure would blow the strict default, but a corpus-specific
        # caller (Corpus A) may pass a looser bound and get a report back
        # instead of a refusal.
        good = _path("g", 0.60, [_pt("2026-08-18", 1.00, 31),
                                 _pt("2026-08-25", 0.40, 24)])
        bad = _path("b", 0.20, [_pt("2026-08-18", 1.00, 31),
                                _pt("2026-08-25", 0.40, 24)])
        with self.assertRaises(CorpusUnusable):
            calibration_report([good] * 5 + [bad] * 5, NULL, CostModel.mid())
        rep = calibration_report([good] * 5 + [bad] * 5, NULL, CostModel.mid(),
                                 max_fail_rate=0.60)
        self.assertEqual(rep.checked, 10)
        self.assertEqual(rep.failed, 5)

    def test_by_reason_sums_to_checked(self):
        good_bp = _path("g1", 0.60, [_pt("2026-08-18", 1.00, 31),
                                     _pt("2026-08-25", 0.40, 24)],
                        strategy="Bull Put")
        bad_bp = _path("b1", 0.20, [_pt("2026-08-18", 1.00, 31),
                                    _pt("2026-08-25", 0.40, 24)],
                       strategy="Bull Put")
        good_bc = _path("g2", 0.60, [_pt("2026-08-18", 1.00, 31),
                                     _pt("2026-08-25", 0.40, 24)],
                        strategy="Bear Call")
        paths = [good_bp] * 3 + [bad_bp] + [good_bc] * 2
        rep = calibration_report(paths, NULL, CostModel.mid(),
                                 max_fail_rate=0.50)
        total_checked = sum(c for c, _ in rep.by_reason.values())
        total_passed = sum(p for _, p in rep.by_reason.values())
        self.assertEqual(total_checked, rep.checked)
        self.assertEqual(total_passed, rep.passed)
        self.assertEqual(rep.by_reason["Bull Put"], (4, 3))
        self.assertEqual(rep.by_reason["Bear Call"], (2, 2))


if __name__ == "__main__":
    unittest.main()
