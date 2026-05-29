"""Tests for src/walk_forward.py — walk-forward OOS IC harness.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_walk_forward -v
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

import numpy as np

from src.backtest_optimizer import WEIGHT_KEYS
from src.walk_forward import build_folds, load_trades, run_walk_forward
from src.paper_manager import PaperManager

# DB column names in insertion order (matches _COMPONENT_COLS in walk_forward)
_WEIGHT_KEY_TO_COL = {
    "pop":              "pop_score",
    "em_realism":       "em_realism_score",
    "iv_mispricing":    "iv_mispricing_score",
    "rr":               "rr_score",
    "momentum":         "momentum_score",
    "iv_rank":          "iv_rank_score",
    "liquidity":        "liquidity_score",
    "catalyst":         "catalyst_score",
    "theta":            "theta_score",
    "ev":               "ev_score",
    "trader_pref":      "trader_pref_score",
    "iv_edge":          "iv_edge_score",
    "skew_align":       "skew_align_score",
    "gamma_theta":      "gamma_theta_score",
    "pcr":              "pcr_score",
    "gex":              "gex_score",
    "oi_change":        "oi_change_score",
    "sentiment":        "sentiment_score_norm",
    "option_rvol":      "option_rvol_score",
    "vrp":              "vrp_score",
    "gamma_pin":        "gamma_pin_score",
    "max_pain":         "max_pain_score",
    "iv_velocity":      "iv_velocity_score",
    "gamma_magnitude":  "gamma_magnitude_score",
    "vega_risk":        "vega_risk_score",
    "term_structure":   "term_structure_score",
    "spread":           "spread_score",
}

_COMPONENT_DB_COLS = [_WEIGHT_KEY_TO_COL[k] for k in WEIGHT_KEYS]

# Index of the signal-carrying column in the components array.
# We inject signal via pop_score (WEIGHT_KEYS index 0).
_SIGNAL_COL_IDX = WEIGHT_KEYS.index("pop")


def _seed_db(
    db_path: str,
    n_trades: int,
    ic_target: float = 0.0,
    seed: int = 42,
) -> None:
    """Create a fully migrated paper_trades.db and insert n_trades closed Long Call rows.

    Signal construction:
      - pop_score is drawn from Uniform[0,1]; all other component columns = 0.5.
      - pnl_pct = ic_target * pop_score + sqrt(1 - ic_target**2) * noise,
        where noise ~ N(0, 0.3). This gives a theoretical Pearson IC of
        approximately ic_target between pop_score and pnl_pct.
      - paper_only = 0 for all rows (eligible for validation cohort).
      - Dates span forward from 2023-01-02 in daily steps.
    """
    # Initialise schema via PaperManager (runs all migrations).
    pm = PaperManager(db_path=db_path, config_path="config.json")

    rng = np.random.default_rng(seed)
    pop_vals = rng.uniform(0.0, 1.0, n_trades)
    noise = rng.normal(0.0, 0.30, n_trades)
    coef = float(ic_target)
    noise_scale = float(np.sqrt(max(1.0 - ic_target**2, 0.0)))
    pnl_vals = coef * pop_vals + noise_scale * noise

    score_cols = ", ".join(_COMPONENT_DB_COLS)
    placeholders = ", ".join(["?"] * len(_COMPONENT_DB_COLS))

    with sqlite3.connect(db_path) as conn:
        for i in range(n_trades):
            trade_date = f"2023-01-{(i % 28) + 1:02d}" if i < 28 else f"2023-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}"
            # Simpler date: just increment from a base
            from datetime import date, timedelta
            base = date(2023, 1, 2)
            entry_date = (base + timedelta(days=i)).isoformat()

            # Build component values: signal in pop_score, rest neutral
            comp_vals = [0.5] * len(_COMPONENT_DB_COLS)
            comp_vals[_SIGNAL_COL_IDX] = float(pop_vals[i])

            sql = (
                f"INSERT INTO trades "
                f"(date, ticker, expiration, strike, type, entry_price, quality_score, "
                f"strategy_name, status, exit_price, exit_date, pnl_pct, pnl_usd, "
                f"paper_only, {score_cols}) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {placeholders})"
            )
            params = (
                entry_date,
                f"TICK{i:04d}",
                "2024-01-19",
                100.0 + i,
                "call",
                2.00,
                float(pop_vals[i]),
                "Long Call",
                "CLOSED",
                0.50,
                entry_date,
                float(pnl_vals[i]),
                float(pnl_vals[i]) * 200.0,
                0,
                *comp_vals,
            )
            conn.execute(sql, params)
        conn.commit()


class TestLeakPrevention(unittest.TestCase):
    """Fold boundaries must be strictly non-overlapping."""

    def test_five_folds_no_leak(self):
        """94 trades, train=44, test=10, step=10 => 5 folds; no train/test overlap."""
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=94)
            trades = load_trades(db, strategy="Long Call")
            folds = list(build_folds(trades, train_size=44, test_size=10, step=10))

            self.assertEqual(len(folds), 5, f"Expected 5 folds, got {len(folds)}")
            for idx, (train_ids, test_ids) in enumerate(folds):
                overlap = set(train_ids) & set(test_ids)
                self.assertSetEqual(
                    overlap,
                    set(),
                    f"Fold {idx} has {len(overlap)} leaking rowids: {overlap}",
                )


class TestRecoversKnownIC(unittest.TestCase):
    """With synthetic signal, pooled OOS IC should be positive."""

    def test_positive_pooled_ic(self):
        """200 trades with ic_target=0.15 => pooled_ic > 0 and n_folds >= 4."""
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=200, ic_target=0.15, seed=7)
            result = run_walk_forward(
                db_path=db,
                strategy="Long Call",
                train_size=80,
                test_size=20,
                step=20,
            )
            self.assertGreaterEqual(
                result["n_folds"],
                4,
                f"Expected >= 4 folds, got {result['n_folds']}",
            )
            self.assertGreater(
                result["pooled_ic"],
                0.0,
                f"Expected positive pooled_ic, got {result['pooled_ic']:.4f}",
            )


class TestPaperOnlyExclusion(unittest.TestCase):
    """Rows with paper_only=1 must not appear in load_trades output."""

    def test_paper_only_excluded(self):
        """20 trades inserted; first 10 flagged paper_only=1 => only 10 returned."""
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=20)

            # Flag the first 10 rows as paper_only
            with sqlite3.connect(db) as conn:
                rows = conn.execute(
                    "SELECT rowid FROM trades ORDER BY rowid LIMIT 10"
                ).fetchall()
                ids = [r[0] for r in rows]
                conn.execute(
                    f"UPDATE trades SET paper_only=1 WHERE rowid IN ({','.join('?' * len(ids))})",
                    ids,
                )
                conn.commit()

            trades = load_trades(db, strategy="Long Call")
            self.assertEqual(
                len(trades),
                10,
                f"Expected 10 non-paper-only trades, got {len(trades)}",
            )
            # Confirm none of the returned rowids are in the paper_only set
            returned_ids = {t.rowid for t in trades}
            self.assertFalse(
                returned_ids & set(ids),
                "Some paper_only=1 rows leaked into load_trades output",
            )


class TestWritesReportFiles(unittest.TestCase):
    """run_walk_forward must write .json and .md into output_dir."""

    def test_report_files_created(self):
        """94 trades => after run, walk_forward_*.json and walk_forward_*.md exist."""
        import glob
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            out_dir = os.path.join(tmp, "reports")
            _seed_db(db, n_trades=94)
            result = run_walk_forward(
                db_path=db,
                strategy="Long Call",
                train_size=44,
                test_size=10,
                step=10,
                output_dir=out_dir,
            )
            json_files = glob.glob(os.path.join(out_dir, "walk_forward_*.json"))
            md_files = glob.glob(os.path.join(out_dir, "walk_forward_*.md"))
            self.assertTrue(
                len(json_files) >= 1,
                f"No walk_forward_*.json found in {out_dir}",
            )
            self.assertTrue(
                len(md_files) >= 1,
                f"No walk_forward_*.md found in {out_dir}",
            )
            self.assertIn("json_path", result)
            self.assertIn("md_path", result)


if __name__ == "__main__":
    unittest.main()
