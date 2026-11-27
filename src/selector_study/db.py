"""Storage and frozen settings for the selector-vs-liquidity paper study.

`data/selector_study.db` is separate from `paper_trades.db` and `candidates.db`. The
settings are written once, when the study database is created, and every later run
reads them back. Changing a rule means a new study database, not an edited one.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from typing import Any, Dict, Optional

from src.paths import repo_path

DEFAULT_DB = repo_path(os.path.join("data", "selector_study.db"))
STUDY_ID = "selector_vs_liquidity_bull_put_v1"

# The 40 symbols the local archive refresh covers (EXPANDED_CHAIN_UNIVERSE[:40]), frozen
# here so a later change to that tuple cannot silently change the study's population.
AGREED_SYMBOLS = (
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU",
    "XLB", "XLRE", "XLC", "SMH", "GLD", "SLV", "TLT", "HYG", "EFA", "EEM", "GDX", "XBI",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "AMD", "NFLX", "JPM",
    "BAC", "GS", "V", "MA", "UNH",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS study_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS capture_runs (
    run_id TEXT PRIMARY KEY,
    session TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    source TEXT NOT NULL,
    status TEXT NOT NULL,          -- running | partial | complete
    counts_json TEXT
);
CREATE TABLE IF NOT EXISTS observations (
    session TEXT NOT NULL,
    symbol TEXT NOT NULL,
    status TEXT NOT NULL,          -- see capture.STATUSES
    detail TEXT,
    source TEXT,
    observed_first TEXT,           -- earliest snapshot_ts of the symbol's rows (UTC)
    observed_at TEXT,              -- latest snapshot_ts (UTC); request time, not quote time
    quote_time TEXT,               -- NULL: the provider gives no bid/ask timestamp
    spot REAL,
    expiration TEXT,
    dte INTEGER,
    two_sided INTEGER,
    selector_candidates INTEGER,
    slice_json TEXT,               -- the exact quotes both arms received
    inputs_json TEXT,              -- spot, earnings, settings hash
    attempts INTEGER NOT NULL DEFAULT 1,
    first_run TEXT NOT NULL,
    last_run TEXT NOT NULL,
    PRIMARY KEY (session, symbol)
);
CREATE TABLE IF NOT EXISTS entries (
    entry_id TEXT PRIMARY KEY,
    pair_id TEXT NOT NULL,
    arm TEXT NOT NULL,             -- selector | baseline
    session TEXT NOT NULL,
    symbol TEXT NOT NULL,
    structure TEXT NOT NULL,
    expiration TEXT NOT NULL,
    dte INTEGER NOT NULL,
    short_strike REAL NOT NULL,
    long_strike REAL NOT NULL,
    width REAL NOT NULL,
    short_delta REAL,
    credit_mid REAL NOT NULL,
    credit_cross REAL NOT NULL,
    rel_spread REAL NOT NULL,      -- round-trip bid-ask / mid credit
    frozen_json TEXT NOT NULL,     -- legs and everything known at selection
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entries_pair ON entries(pair_id);
CREATE TABLE IF NOT EXISTS pairs (
    pair_id TEXT PRIMARY KEY,
    session TEXT NOT NULL,
    symbol TEXT NOT NULL,
    expiration TEXT NOT NULL,
    status TEXT NOT NULL,          -- matched | identical | no_match
    reason TEXT,
    eligible_baseline INTEGER,
    selector_entry TEXT NOT NULL,
    baseline_entry TEXT
);
CREATE TABLE IF NOT EXISTS marks (
    entry_id TEXT NOT NULL,
    mark_session TEXT NOT NULL,
    status TEXT NOT NULL,          -- marked | unpriceable | missing_leg | missing_session | settled
    short_bid REAL, short_ask REAL, long_bid REAL, long_ask REAL,
    underlying_close REAL,
    observed_at TEXT,
    source TEXT,
    PRIMARY KEY (entry_id, mark_session)
);
"""


def default_settings(cfg_path: str = "config.json") -> Dict[str, Any]:
    """The study rules, with the book's spread exits and fee read from config."""
    exits: Dict[str, Any] = {}
    fee = 0.0
    try:
        with open(repo_path(cfg_path), encoding="utf-8") as fh:
            cfg = json.load(fh) or {}
        exits = cfg.get("exit_rules") or {}
        fee = float((cfg.get("paper_trading") or {}).get("commission_per_contract", 0.0) or 0.0)
    except (OSError, ValueError, TypeError):
        pass
    spread = exits.get("spread") or {}
    return {
        "study_id": STUDY_ID,
        "structure": "bull_put",
        "symbols": list(AGREED_SYMBOLS),
        "dte_window": {"target": 35, "min": 14, "max": 58},
        "max_friction_pct": 0.28,
        "capture_window_et": ["14:00", "20:00"],
        "baseline": {
            "rule": "lowest round-trip bid-ask / mid credit",
            "short_delta_tolerance": 0.05,
            "width_tolerance_pct": 0.25,
            "min_short_bid": 0.10,
            "min_long_bid": 0.02,
            "min_open_interest": 1,
            "min_credit_mid": 0.10,
            "tie_breaks": ["abs short-delta gap", "abs width gap", "higher short strike", "higher long strike"],
        },
        "fee_per_contract": fee,
        "exits": {
            "take_profit": float(spread.get("take_profit", 0.5)),
            "stop_loss": float(spread.get("stop_loss", -1.0)),
            "time_exit_dte": int(exits.get("time_exit_dte", 21)),
            "min_days_held": int(exits.get("min_days_held", 3)),
            "max_mark_gap_sessions": 3,
        },
        "readout": {
            "read_after": "2027-03-01",
            "min_weeks": 12,
            "min_pairs": 40,
            "cluster": "iso_week",
            "bootstrap_draws": 10000,
            "block_weeks": 3,
            "seed": 20261126,
        },
    }


def settings_hash(settings: Dict[str, Any]) -> str:
    raw = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def connect(db_path: Optional[str] = None, *, settings: Optional[Dict[str, Any]] = None,
            synthetic: bool = False) -> sqlite3.Connection:
    """Open the study database, freezing `settings` on first creation."""
    # OPTIONS_SELECTOR_STUDY_DB keeps the test suite off the real study (tests/conftest.py).
    path = db_path or os.environ.get("OPTIONS_SELECTOR_STUDY_DB") or DEFAULT_DB
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    have = conn.execute("SELECT value FROM study_meta WHERE key='settings'").fetchone()
    if have is None:
        frozen = settings or default_settings()
        conn.executemany("INSERT INTO study_meta(key, value) VALUES (?, ?)", [
            ("settings", json.dumps(frozen, sort_keys=True)),
            ("settings_hash", settings_hash(frozen)),
            ("synthetic", "1" if synthetic else "0"),
        ])
        conn.commit()
    return conn


def load_settings(conn: sqlite3.Connection) -> Dict[str, Any]:
    row = conn.execute("SELECT value FROM study_meta WHERE key='settings'").fetchone()
    return json.loads(row[0])


def is_synthetic(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT value FROM study_meta WHERE key='synthetic'").fetchone()
    return bool(row and row[0] == "1")
