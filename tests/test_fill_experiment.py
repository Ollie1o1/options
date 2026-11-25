"""Unit tests for src.fill_experiment (real-fill execution harness)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.fill_experiment import (
    compute_concession_pct,
    fill_rate_summary,
    list_real_fills,
    log_real_fill,
)


class TestFillExperiment(unittest.TestCase):
    def test_concession_pct_buy_and_sell(self) -> None:
        # Bid=1.00, Ask=1.20 => Mid=1.10, Half=0.10
        self.assertAlmostEqual(compute_concession_pct("BUY", 1.00, 1.20, 1.10), 0.0)
        self.assertAlmostEqual(compute_concession_pct("BUY", 1.00, 1.20, 1.20), 1.0)
        self.assertAlmostEqual(compute_concession_pct("BUY", 1.00, 1.20, 1.135), 0.35)
        self.assertAlmostEqual(compute_concession_pct("SELL", 1.00, 1.20, 1.10), 0.0)
        self.assertAlmostEqual(compute_concession_pct("SELL", 1.00, 1.20, 1.00), 1.0)
        self.assertAlmostEqual(compute_concession_pct("SELL", 1.00, 1.20, 1.075), 0.25)

    def test_log_and_summarize_fills(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "fills.db")
            fid1 = log_real_fill(
                broker="ibkr",
                ticker="spy",
                structure="Bear Call",
                side="SELL",
                bid=1.00,
                ask=1.20,
                limit_posted=1.08,
                filled=True,
                fill_price=1.08,
                wait_seconds=45,
                db_path=db,
            )
            fid2 = log_real_fill(
                broker="ibkr",
                ticker="SPY",
                structure="Bear Call",
                side="SELL",
                bid=1.00,
                ask=1.20,
                limit_posted=1.10,
                filled=False,
                wait_seconds=180,
                db_path=db,
            )
            self.assertEqual(fid1, 1)
            self.assertEqual(fid2, 2)
            rows = list_real_fills(db_path=db)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["ticker"], "SPY")

            summary = fill_rate_summary(db_path=db)
            self.assertEqual(summary["total_attempts"], 2)
            self.assertEqual(summary["total_filled"], 1)
            self.assertAlmostEqual(summary["overall_fill_rate"], 0.5)
            grp = summary["groups"][0]
            self.assertEqual(grp["broker"], "ibkr")
            self.assertEqual(grp["structure"], "Bear Call")
            self.assertAlmostEqual(grp["mean_realized_k"], 0.2)
            self.assertEqual(grp["median_wait_s"], 45)


if __name__ == "__main__":
    unittest.main()
