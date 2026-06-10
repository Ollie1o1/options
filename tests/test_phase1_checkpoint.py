"""Tests for src/phase1_checkpoint.py — Phase 1 weekly gate checkpoint.

All scenarios use _seed_cohort() to build a synthetic SQLite DB via PaperManager,
insert Long Call CLOSED rows with paper_only=0, and verify compute_checkpoint()
returns the expected gate decision.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_phase1_checkpoint -v
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _seed_cohort(
    db_path: str,
    n_trades: int,
    ic_target: float,
    start_date: str,
    seed: int = 42,
    noise_scale: float = 0.3,
) -> None:
    """Build a paper_trades DB via PaperManager then insert synthetic Long Call CLOSED rows.

    quality_score ~ Uniform(0.3, 0.9).
    pnl_pct = ic_target * quality_score + noise,  where noise ~ N(0, noise_scale).
    Dates are spread forward from start_date (one per day).
    paper_only is set to 0 (real-money cohort).
    """
    from src.paper_manager import PaperManager

    # Create schema via PaperManager (handles migrations up to v12).
    pm = PaperManager(db_path=db_path, config_path="config.json")

    rng = np.random.default_rng(seed)
    base = rng.uniform(0.3, 0.9, n_trades)
    noise = rng.normal(0, 1, n_trades)
    # Correlation formula: linear mix of signal + noise
    factor = np.sqrt(max(0.0, 1.0 - min(ic_target ** 2, 0.999)))
    returns = ic_target * base + factor * noise * noise_scale

    # Parse start_date to generate sequential trade dates
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")

    with sqlite3.connect(db_path) as conn:
        for i in range(n_trades):
            trade_date = (start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
            conn.execute(
                """
                INSERT INTO trades (
                    date, ticker, expiration, strike, type,
                    entry_price, quality_score, strategy_name,
                    status, exit_date, pnl_pct, paper_only
                ) VALUES (?, 'AAPL', '2026-12-19', 150.0, 'call',
                          2.50, ?, 'Long Call',
                          'CLOSED', ?, ?, 0)
                """,
                (
                    trade_date,
                    float(base[i]),
                    trade_date,
                    float(returns[i]),
                ),
            )
        conn.commit()


class GatheringWhenFewTrades(unittest.TestCase):
    """n < 50 always produces GATHERING regardless of IC."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = os.path.join(self.tmpdir, "paper_trades.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_gathering_when_20_trades(self):
        _seed_cohort(self.db, n_trades=20, ic_target=0.10,
                     start_date="2026-05-27", seed=42)
        from src.phase1_checkpoint import compute_checkpoint
        result = compute_checkpoint(self.db, "2026-05-27", today="2026-06-15")
        self.assertEqual(result["decision"], "GATHERING",
                         f"Expected GATHERING for n=20, got {result['decision']}")
        self.assertEqual(result["n_trades"], 20,
                         f"Expected n_trades=20, got {result['n_trades']}")


class ReadyWhenStrongIC(unittest.TestCase):
    """n >= 50, IC >= 0.08, p < 0.05 → READY."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = os.path.join(self.tmpdir, "paper_trades.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_ready_when_strong_ic(self):
        # n=80, ic_target=0.25, seed=1 → Pearson IC ~0.34, p ~0.002
        _seed_cohort(self.db, n_trades=80, ic_target=0.25,
                     start_date="2026-05-27", seed=1)
        from src.phase1_checkpoint import compute_checkpoint
        result = compute_checkpoint(self.db, "2026-05-27", today="2026-07-10")
        self.assertEqual(result["decision"], "READY",
                         f"Expected READY, got {result['decision']} "
                         f"(IC={result['ic_pearson']:.4f}, p={result['p_pearson']:.4f})")


class StopWhenWeakICAtWeek6(unittest.TestCase):
    """n >= 50, IC < 0.03, weeks >= 6 → STOP."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = os.path.join(self.tmpdir, "paper_trades.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_stop_when_weak_ic_at_week6(self):
        # n=60, ic_target=0.0, seed=2 → IC ~ -0.07 (< 0.03), weeks=7
        _seed_cohort(self.db, n_trades=60, ic_target=0.0,
                     start_date="2026-05-27", seed=2)
        from src.phase1_checkpoint import compute_checkpoint
        result = compute_checkpoint(self.db, "2026-05-27", today="2026-07-15")
        self.assertEqual(result["decision"], "STOP",
                         f"Expected STOP, got {result['decision']} "
                         f"(IC={result['ic_pearson']:.4f}, weeks={result['weeks_elapsed']})")


class ExtendWhenModerateIC(unittest.TestCase):
    """n >= 50, 0.03 <= IC < 0.08 → EXTEND."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = os.path.join(self.tmpdir, "paper_trades.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_extend_when_moderate_ic(self):
        # seed=0, ic_target=0.04, noise_scale=0.08 → IC ~0.069, in [0.03, 0.08)
        _seed_cohort(self.db, n_trades=60, ic_target=0.04,
                     start_date="2026-05-27", seed=0, noise_scale=0.08)
        from src.phase1_checkpoint import compute_checkpoint
        result = compute_checkpoint(self.db, "2026-05-27", today="2026-07-01")
        self.assertEqual(result["decision"], "EXTEND",
                         f"Expected EXTEND, got {result['decision']} "
                         f"(IC={result['ic_pearson']:.4f})")


class ExcludesPaperOnlyAndPreStart(unittest.TestCase):
    """Rows with paper_only=1 or date < phase1_start must not count in cohort."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = os.path.join(self.tmpdir, "paper_trades.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_excludes_paper_only_and_pre_start(self):
        # Seed 20 real-money post-start trades
        _seed_cohort(self.db, n_trades=20, ic_target=0.10,
                     start_date="2026-05-27", seed=42)

        # Insert 15 pre-start rows (date 2026-05-01, paper_only=0)
        # and 15 paper_only=1 rows (date after start)
        with sqlite3.connect(self.db) as conn:
            for i in range(15):
                conn.execute(
                    """
                    INSERT INTO trades (
                        date, ticker, expiration, strike, type,
                        entry_price, quality_score, strategy_name,
                        status, exit_date, pnl_pct, paper_only
                    ) VALUES ('2026-05-01', 'MSFT', '2026-12-19', 200.0, 'call',
                              3.00, 0.75, 'Long Call',
                              'CLOSED', '2026-05-20', 0.15, 0)
                    """
                )
            for i in range(15):
                conn.execute(
                    """
                    INSERT INTO trades (
                        date, ticker, expiration, strike, type,
                        entry_price, quality_score, strategy_name,
                        status, exit_date, pnl_pct, paper_only
                    ) VALUES ('2026-05-28', 'GOOGL', '2026-12-19', 180.0, 'call',
                              4.00, 0.80, 'Long Call',
                              'CLOSED', '2026-06-10', 0.20, 1)
                    """
                )
            conn.commit()

        from src.phase1_checkpoint import compute_checkpoint
        result = compute_checkpoint(self.db, "2026-05-27", today="2026-06-15")
        self.assertEqual(result["n_trades"], 20,
                         f"Expected n_trades=20 (pre-start and paper_only rows excluded), "
                         f"got {result['n_trades']}")


class WritesReportAndHistoryAndGateStatus(unittest.TestCase):
    """write_checkpoint creates checkpoint_*.md, checkpoint_history.tsv,
    and GATE_STATUS.md when decision is READY or STOP."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = os.path.join(self.tmpdir, "paper_trades.db")
        self.out = os.path.join(self.tmpdir, "reports")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_writes_report_history_and_gate_status_when_ready(self):
        # READY scenario: n=80, ic_target=0.25, seed=1 → decision=READY
        _seed_cohort(self.db, n_trades=80, ic_target=0.25,
                     start_date="2026-05-27", seed=1)
        from src.phase1_checkpoint import compute_checkpoint, write_checkpoint
        result = compute_checkpoint(self.db, "2026-05-27", today="2026-07-10")
        self.assertEqual(result["decision"], "READY",
                         f"Prerequisite failed: expected READY for this scenario, "
                         f"got {result['decision']}")

        paths = write_checkpoint(result, output_dir=self.out)

        out_dir = Path(self.out)
        # checkpoint_*.md must exist
        md_files = list(out_dir.glob("checkpoint_*.md"))
        self.assertTrue(len(md_files) >= 1,
                        "No checkpoint_*.md file found in output dir")

        # checkpoint_history.tsv must exist
        hist = out_dir / "checkpoint_history.tsv"
        self.assertTrue(hist.exists(), "checkpoint_history.tsv not found")

        # GATE_STATUS.md must exist for READY decision
        gate = out_dir / "GATE_STATUS.md"
        self.assertTrue(gate.exists(),
                        "GATE_STATUS.md not found — expected for READY decision")

        # Sanity-check history has a data row
        lines = hist.read_text().splitlines()
        self.assertGreaterEqual(len(lines), 2,
                                "checkpoint_history.tsv has no data rows")


class BayesianPosteriorTest(unittest.TestCase):
    """posterior_ic_above: P(true IC >= threshold) on the Fisher-z scale.

    Reporting only — it must not influence the gate decision.
    """

    def test_posterior_strong_edge_high(self):
        from src.phase1_checkpoint import posterior_ic_above
        p = posterior_ic_above(0.30, 60, threshold=0.08)
        self.assertGreater(p, 0.90)

    def test_posterior_no_edge_low(self):
        from src.phase1_checkpoint import posterior_ic_above
        p = posterior_ic_above(0.00, 60, threshold=0.08)
        self.assertLess(p, 0.40)

    def test_posterior_at_threshold_is_half(self):
        from src.phase1_checkpoint import posterior_ic_above
        p = posterior_ic_above(0.08, 100, threshold=0.08)
        self.assertAlmostEqual(p, 0.5, places=6)

    def test_posterior_tiny_n_returns_none(self):
        from src.phase1_checkpoint import posterior_ic_above
        self.assertIsNone(posterior_ic_above(0.30, 3, threshold=0.08))

    def test_posterior_non_finite_ic_returns_none(self):
        from src.phase1_checkpoint import posterior_ic_above
        self.assertIsNone(posterior_ic_above(float("nan"), 60, threshold=0.08))

    def test_matches_power_analysis_script(self):
        """The script must delegate to the same canonical implementation."""
        from src.phase1_checkpoint import posterior_ic_above
        from scripts.validation_power_analysis import posterior_prob_ic_above
        self.assertAlmostEqual(
            posterior_ic_above(0.15, 80), posterior_prob_ic_above(0.15, 80), places=12
        )


class CheckpointIncludesPosteriorTest(unittest.TestCase):
    """compute_checkpoint result carries posterior_ic_ge_008; decision unchanged."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = os.path.join(self.tmpdir, "trades.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_posterior_in_result_and_markdown(self):
        from src.phase1_checkpoint import compute_checkpoint, _format_markdown
        _seed_cohort(self.db, n_trades=60, ic_target=0.35,
                     start_date="2026-05-27", noise_scale=0.1)
        result = compute_checkpoint(self.db, "2026-05-27", today="2026-07-10")
        self.assertIn("posterior_ic_ge_008", result)
        self.assertIsNotNone(result["posterior_ic_ge_008"])
        self.assertGreater(result["posterior_ic_ge_008"], 0.5)
        md = _format_markdown(result)
        self.assertIn("Bayesian", md)
        self.assertIn("reporting only", md)

    def test_posterior_none_when_cohort_tiny(self):
        from src.phase1_checkpoint import compute_checkpoint, _format_markdown
        _seed_cohort(self.db, n_trades=2, ic_target=0.35, start_date="2026-05-27")
        result = compute_checkpoint(self.db, "2026-05-27", today="2026-06-10")
        self.assertIsNone(result["posterior_ic_ge_008"])
        # markdown must not crash on None
        _format_markdown(result)


if __name__ == "__main__":
    unittest.main()
