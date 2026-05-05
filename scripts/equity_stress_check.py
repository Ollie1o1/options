#!/usr/bin/env python3
"""Equity portfolio stress gate.

Reads paper_trades.db, runs stress_test.run_stress_test, and prints
SAFE or UNSAFE plus diagnostic numbers based on the –20% / +10pp
scenario versus the configured threshold.

Always exits 0 — the bash wrapper greps stdout for the SAFE / UNSAFE
token and decides what to do. Cron alerts stay reserved for genuine
errors (missing modules, Python crashes), not for "book is over-
leveraged" — that is a normal outcome.

Inputs (env vars override config defaults; useful for tests):
  EQUITY_STRESS_DB             — path to paper-trades DB
  EQUITY_STRESS_THRESHOLD_PCT  — gate threshold (% of book)
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_threshold_pct() -> float:
    env = os.environ.get("EQUITY_STRESS_THRESHOLD_PCT")
    if env:
        return float(env)
    cfg_path = PROJECT_ROOT / "config.json"
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return 100.0
    return float((cfg.get("equity") or {}).get("stress_gate_pct_book", 100.0))


def _db_path() -> Path:
    env = os.environ.get("EQUITY_STRESS_DB")
    if env:
        return Path(env)
    return PROJECT_ROOT / "paper_trades.db"


def _open_trades(db_path: Path) -> list:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM trades WHERE status='OPEN'")
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def main() -> int:
    sys.path.insert(0, str(PROJECT_ROOT))

    threshold_pct = _load_threshold_pct()
    threshold = threshold_pct / 100.0

    db = _db_path()
    open_trades = _open_trades(db)
    if not open_trades:
        print(f"SAFE  (no open positions, threshold={-threshold:+.0%})")
        return 0

    try:
        from src.stress_test import run_stress_test
    except Exception as e:
        print(f"UNSAFE  stress module import failed: {type(e).__name__}: {e}")
        return 0

    df = run_stress_test(open_trades, stock_prices=None)
    if df is None or df.empty:
        print("UNSAFE  stress test returned no data")
        return 0

    target = df[(df["stock_move"] == -0.20) & (df["iv_shock"] == 0.10)]
    if target.empty:
        print("UNSAFE  -20%/+10pp scenario row missing from stress grid")
        return 0
    worst = float(target["pnl_pct_of_book"].iloc[0])

    if worst < -threshold:
        print(f"UNSAFE  worst={worst:+.2%}  threshold={-threshold:+.2%}")
    else:
        print(f"SAFE    worst={worst:+.2%}  threshold={-threshold:+.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
