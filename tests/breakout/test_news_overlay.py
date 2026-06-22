"""Tests for the forward-only news overlay log (offline)."""
from __future__ import annotations
import os, tempfile, unittest
from src.breakout import news_overlay as N


class NewsOverlayTests(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "fwd.db")

    def test_tilt_clamps(self):
        self.assertLessEqual(N.tilt(0.95, 1.0), 1.0)
        self.assertGreaterEqual(N.tilt(0.02, -1.0), 0.0)

    def test_accuracy_uses_resolved_only(self):
        N.log_prediction(self.db, "2026-01-02", "AAA", "EOM", 0.20, 0.30)
        N.log_prediction(self.db, "2026-01-02", "BBB", "EOM", 0.20, 0.10)
        # nothing resolved yet
        self.assertEqual(N.forward_accuracy(self.db)["n"], 0)
        # resolve: AAA broke out (1), BBB didn't (0)
        truth = {("AAA", "2026-01-02", "EOM"): 1.0, ("BBB", "2026-01-02", "EOM"): 0.0}
        n = N.resolve_outcomes(self.db, "2099-01-01",
                               lambda tk, d, h: truth.get((tk, d, h)))
        self.assertEqual(n, 2)
        acc = N.forward_accuracy(self.db)
        self.assertEqual(acc["n"], 2)
        self.assertIn("skill", acc)


if __name__ == "__main__":
    unittest.main()
