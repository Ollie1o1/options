"""Tests for src/paper_manager.py."""
import sys
import os
import json
import sqlite3
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paper_manager import PaperManager, _SCHEMA_VERSION, _is_short_position


def _write_config(path):
    """Write a minimal config.json for testing."""
    cfg = {
        "exit_rules": {"take_profit": 0.50, "stop_loss": -0.25, "time_exit_dte": 21},
        "paper_trading": {
            "commission_per_contract": 0.65,
            "slippage_per_share": 0.05,
            "default_db_path": "paper_trades.db"
        }
    }
    with open(path, "w") as f:
        json.dump(cfg, f)


def _sample_trade(**overrides):
    trade = {
        "ticker": "AAPL",
        "expiration": "2026-06-20",
        "strike": 150.0,
        "type": "call",
        "entry_price": 3.50,
        "quality_score": 0.75,
        "strategy_name": "Long Call",
    }
    trade.update(overrides)
    return trade


def test_log_trade_creates_row(tmp_path):
    """log_trade creates exactly 1 OPEN row."""
    db = str(tmp_path / "trades.db")
    cfg = str(tmp_path / "config.json")
    _write_config(cfg)
    pm = PaperManager(db_path=db, config_path=cfg)
    pm.log_trade(_sample_trade())
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT * FROM trades WHERE status='OPEN'").fetchall()
    assert len(rows) == 1


def test_log_trade_normalises_ticker_case(tmp_path):
    """'aapl' stored as 'AAPL'."""
    db = str(tmp_path / "trades.db")
    cfg = str(tmp_path / "config.json")
    _write_config(cfg)
    pm = PaperManager(db_path=db, config_path=cfg)
    pm.log_trade(_sample_trade(ticker="aapl"))
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT ticker FROM trades").fetchone()
    assert row[0] == "AAPL"


def test_get_performance_summary_empty(tmp_path):
    """Fresh DB -> total_count=0."""
    db = str(tmp_path / "trades.db")
    cfg = str(tmp_path / "config.json")
    _write_config(cfg)
    pm = PaperManager(db_path=db, config_path=cfg)
    summary = pm.get_performance_summary()
    assert int(summary["Total Trades"].iloc[0]) == 0


def test_schema_migration_runs(tmp_path):
    """After init, PRAGMA user_version should equal _SCHEMA_VERSION."""
    db = str(tmp_path / "trades.db")
    cfg = str(tmp_path / "config.json")
    _write_config(cfg)
    PaperManager(db_path=db, config_path=cfg)
    with sqlite3.connect(db) as conn:
        ver = conn.execute("PRAGMA user_version").fetchone()[0]
    assert ver == _SCHEMA_VERSION


def test_log_spread(tmp_path):
    """log_spread stores the spread name from `type` as strategy_name (1 row).

    Production passes the spread name in `type` ("Bull Put" / "Bear Call" —
    see the log_spread docstring); the DB `type` column gets the inferred
    option type ("put" / "call").
    """
    db = str(tmp_path / "trades.db")
    cfg = str(tmp_path / "config.json")
    _write_config(cfg)
    pm = PaperManager(db_path=db, config_path=cfg)
    pm.log_spread({
        "ticker": "SPY",
        "expiration": "2026-06-20",
        "short_strike": 500.0,
        "long_strike": 495.0,
        "type": "Bull Put",
        "net_credit": 1.50,
        "max_profit": 150.0,
        "max_loss": 350.0,
        "quality_score": 0.65,
    })
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT strategy_name, type FROM trades").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "Bull Put"
    assert rows[0][1] == "put"


def test_is_short_position_detection():
    """'Short Put'->True, 'Credit Spread'->True, 'Long Call'->False."""
    assert _is_short_position("Short Put") is True
    assert _is_short_position("Credit Spread") is True
    assert _is_short_position("Long Call") is False


class SchemaMigrationV12Test(unittest.TestCase):
    """Verify schema v12 adds paper_only column and idx_paper_only index."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = os.path.join(self.tmpdir, "trades.db")
        self.cfg = os.path.join(self.tmpdir, "config.json")
        _write_config(self.cfg)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_pm(self):
        return PaperManager(db_path=self.db, config_path=self.cfg)

    def test_paper_only_column_exists_with_default_zero(self):
        """After init, paper_only column must exist with a default value of 0."""
        self._make_pm()
        with sqlite3.connect(self.db) as conn:
            info = conn.execute("PRAGMA table_info(trades)").fetchall()
        col_names = [row[1] for row in info]
        self.assertIn("paper_only", col_names, "paper_only column not found in trades schema")
        # Verify default: insert a row via PaperManager then read paper_only
        pm = self._make_pm()
        pm.log_trade(_sample_trade())
        with sqlite3.connect(self.db) as conn:
            row = conn.execute("SELECT paper_only FROM trades LIMIT 1").fetchone()
        self.assertIsNotNone(row, "No row found after log_trade")
        self.assertEqual(row[0], 0, f"Expected paper_only default 0, got {row[0]}")

    def test_logged_trade_has_paper_only_zero(self):
        """A trade logged via log_trade must have paper_only=0 (cohort-eligible)."""
        pm = self._make_pm()
        pm.log_trade(_sample_trade(strategy_name="Long Call"))
        with sqlite3.connect(self.db) as conn:
            row = conn.execute(
                "SELECT paper_only FROM trades WHERE strategy_name='Long Call' LIMIT 1"
            ).fetchone()
        self.assertIsNotNone(row, "Trade row not found")
        self.assertEqual(row[0], 0, f"Long Call trade should have paper_only=0, got {row[0]}")

    def test_idx_paper_only_index_exists(self):
        """idx_paper_only index must be present after migration."""
        self._make_pm()
        with sqlite3.connect(self.db) as conn:
            indexes = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_paper_only'"
            ).fetchall()
        self.assertEqual(
            len(indexes), 1,
            "idx_paper_only index not found in sqlite_master"
        )


def _ic_row(**overrides):
    """A dict-shaped iron-condor row with every column _legs_for_row touches."""
    row = {
        "strategy_name": "Iron Condor",
        "type": "call",
        "strike": 415.0,
        "long_strike": None,
        "short_put_strike": 415.0,
        "long_put_strike": 390.0,
        "short_call_strike": 440.0,
        "long_call_strike": 465.0,
        "net_credit": 8.0,
    }
    row.update(overrides)
    return row


def test_intrinsic_value_call_and_put():
    from src.paper_manager import _intrinsic_value
    assert _intrinsic_value("call", 110.0, 100.0) == 10.0
    assert _intrinsic_value("call", 90.0, 100.0) == 0.0
    assert _intrinsic_value("put", 90.0, 100.0) == 10.0
    assert _intrinsic_value("put", 110.0, 100.0) == 0.0


def test_legs_for_row_full_iron_condor_has_four_legs():
    from src.paper_manager import _legs_for_row
    legs = _legs_for_row(_ic_row())
    assert len(legs) == 4
    assert sorted(opt for _, opt, _ in legs) == ["call", "call", "put", "put"]


def test_legs_for_row_malformed_ic_missing_call_legs_keeps_puts():
    """Legacy ICs stored without call strikes must NOT collapse to [] (the bug
    that left them OPEN forever). They degrade to the put legs present."""
    from src.paper_manager import _legs_for_row
    legs = _legs_for_row(_ic_row(short_call_strike=None, long_call_strike=None))
    assert len(legs) == 2, f"expected 2 put legs, got {legs}"
    assert all(opt == "put" for _, opt, _ in legs)


def test_legs_intrinsic_close_value_all_otm_is_zero():
    """All legs expire worthless -> nothing to pay to flatten -> seller keeps credit."""
    from src.paper_manager import _legs_for_row, _legs_intrinsic_close_value
    legs = _legs_for_row(_ic_row())  # SP425? no: 415/390 puts, 440/465 calls
    # spot inside the wings: 390 < 420 < 440 -> all OTM
    assert _legs_intrinsic_close_value(legs, 420.0) == 0.0


def test_legs_intrinsic_close_value_put_side_breached():
    """Spot below short put: short put is ITM, debit to buy it back is positive."""
    from src.paper_manager import _legs_for_row, _legs_intrinsic_close_value
    legs = _legs_for_row(_ic_row(short_call_strike=None, long_call_strike=None))
    # short put 415 / long put 390, spot 400 -> short put intrinsic 15, long put 0
    assert _legs_intrinsic_close_value(legs, 400.0) == 15.0


if __name__ == "__main__":
    unittest.main()
