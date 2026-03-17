"""Tests for src/paper_manager.py."""
import sys
import os
import json
import sqlite3
import pytest

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
    """log_spread creates 1 row with 'Spread' in strategy_name."""
    db = str(tmp_path / "trades.db")
    cfg = str(tmp_path / "config.json")
    _write_config(cfg)
    pm = PaperManager(db_path=db, config_path=cfg)
    pm.log_spread({
        "ticker": "SPY",
        "expiration": "2026-06-20",
        "short_strike": 500.0,
        "long_strike": 495.0,
        "type": "put",
        "net_credit": 1.50,
        "max_profit": 150.0,
        "max_loss": 350.0,
        "quality_score": 0.65,
    })
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT strategy_name FROM trades").fetchall()
    assert len(rows) == 1
    assert "SPREAD" in rows[0][0].upper() or "Spread" in rows[0][0]


def test_is_short_position_detection():
    """'Short Put'->True, 'Credit Spread'->True, 'Long Call'->False."""
    assert _is_short_position("Short Put") is True
    assert _is_short_position("Credit Spread") is True
    assert _is_short_position("Long Call") is False
