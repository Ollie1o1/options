"""Preregistered O1 execution verdict (docs/PREREG_O1_FILL_K_20261125.md)."""
from __future__ import annotations

import os
import tempfile
import unittest

from src.fill_experiment import is_o1_structure, log_real_fill, o1_fill_verdict, o1_progress_line


class O1FillVerdictTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self._tmp.name, "fills.db")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _log(self, k: float, *, filled: bool = True, structure: str = "O1 Bull Put SPY 645/610") -> None:
        # Spread quote 4.00 x 5.00 (mid 4.50, half 0.50); SELL at mid - k*half.
        px = 4.50 - k * 0.50
        log_real_fill(broker="ibkr", ticker="SPY", structure=structure, side="SELL", bid=4.00, ask=5.00,
                      limit_posted=px, filled=filled, fill_price=px if filled else None, db_path=self.db)

    def test_missing_db_is_collecting(self) -> None:
        v = o1_fill_verdict(os.path.join(self._tmp.name, "nope.db"))
        self.assertEqual((v["verdict"], v["filled"], v["attempts"]), ("COLLECTING", 0, 0))
        self.assertIn("0/20", o1_progress_line(v))

    def test_only_o1_tagged_rows_count(self) -> None:
        for _ in range(5):
            self._log(0.2)
        self._log(0.2, structure="Bull Put")
        self._log(0.2, structure="Long Call")
        v = o1_fill_verdict(self.db)
        self.assertEqual(v["filled"], 5)
        self.assertTrue(is_o1_structure(" o1 wide bull put"))
        self.assertFalse(is_o1_structure("Bull Put O1"))

    def test_no_verdict_before_twenty(self) -> None:
        for _ in range(19):
            self._log(0.0)
        v = o1_fill_verdict(self.db)
        self.assertEqual(v["verdict"], "COLLECTING")
        self.assertAlmostEqual(v["mean_k"], 0.0, places=9)

    def test_pass_when_upper_bound_below_target(self) -> None:
        for i in range(20):
            self._log(0.10 + 0.01 * (i % 5))
        v = o1_fill_verdict(self.db)
        self.assertEqual(v["verdict"], "PASS")
        self.assertLessEqual(v["k_hi"], 0.35)

    def test_fail_when_lower_bound_above_target(self) -> None:
        for i in range(20):
            self._log(0.70 + 0.02 * (i % 3))
        v = o1_fill_verdict(self.db)
        self.assertEqual(v["verdict"], "FAIL")
        self.assertIn("re-run", v["reason"])

    def test_inconclusive_when_bounds_straddle(self) -> None:
        for i in range(20):
            self._log(0.35 + (0.5 if i % 2 else -0.5))
        v = o1_fill_verdict(self.db)
        self.assertEqual(v["verdict"], "INCONCLUSIVE")
        self.assertIn("keep collecting", v["reason"])

    def test_unfilled_attempts_lower_fill_rate_and_flag(self) -> None:
        for _ in range(3):
            self._log(0.2)
        for _ in range(4):
            self._log(0.0, filled=False)
        v = o1_fill_verdict(self.db)
        self.assertEqual((v["attempts"], v["filled"]), (7, 3))
        self.assertTrue(v["fill_rate_flag"])
        self.assertIn("under half", o1_progress_line(v))

    def test_k_is_measured_against_the_spread_quote(self) -> None:
        self._log(0.35)
        v = o1_fill_verdict(self.db)
        self.assertAlmostEqual(v["ks"][0], 0.35, places=4)


if __name__ == "__main__":
    unittest.main()
