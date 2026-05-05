"""Non-interactive auto-log driver for the crypto paper ledger.

Runs from cron every 4 hours. For each of BTC and ETH:
  - check off-switch + safeguards
  - run a full live scan
  - pick the single highest-scoring contract across all surfaced buckets
  - write it to paper_trades_crypto.db with weight_profile='crypto_auto_v1'

The hourly exit enforcer (scripts/enforce_exits_crypto.sh) closes
positions on TP / SL / time-exit independently — no exit logic here.
"""
from __future__ import annotations

import datetime as _dt
import os
import sqlite3
from typing import Optional, Tuple

import pandas as pd

AUTO_WEIGHT_PROFILE = "crypto_auto_v1"

# Long-premium picks live in score column "strategy_score"; spreads,
# calendars, and condors live in "score". This map keeps the rule local.
_LONG_PREMIUM_STRATS = {"Long Call", "Long Put"}


def _score_column(strategy_name: str) -> str:
    return "strategy_score" if strategy_name in _LONG_PREMIUM_STRATS else "score"


def count_open_positions(db_path: str, currency: str) -> int:
    if not os.path.exists(db_path):
        return 0
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE ticker = ? AND status = 'OPEN'",
            (currency.upper(),),
        )
        return int(cur.fetchone()[0])
    finally:
        conn.close()


def count_today_auto_logs(db_path: str, currency: str, today_utc: Optional[str] = None) -> int:
    if not os.path.exists(db_path):
        return 0
    if today_utc is None:
        today_utc = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    midnight = f"{today_utc} 00:00:00"
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            """
            SELECT COUNT(*) FROM trades
            WHERE ticker = ?
              AND weight_profile LIKE 'crypto_auto%'
              AND date >= ?
            """,
            (currency.upper(), midnight),
        )
        return int(cur.fetchone()[0])
    finally:
        conn.close()


DEFAULT_MIN_SCORE = 0.50
MAX_OPEN_PER_CURRENCY = 3
MAX_AUTO_LOGS_PER_DAY = 4


def _default_db_path() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project_root, "paper_trades_crypto.db")


def pick_winner(picks_by_strategy: dict) -> Optional[Tuple[str, pd.Series, float]]:
    """From `_scan_currency()['picks_by_strategy']`, return the single
    (strategy_name, row, score) with the highest top-of-bucket score, or
    None if no bucket has any rows.
    """
    best: Optional[Tuple[str, pd.Series, float]] = None
    for strategy_name, df in picks_by_strategy.items():
        if df is None or df.empty:
            continue
        col = _score_column(strategy_name)
        if col not in df.columns:
            continue
        top = df.iloc[0]
        try:
            score = float(top[col])
        except (TypeError, ValueError):
            continue
        if best is None or score > best[2]:
            best = (strategy_name, top, score)
    return best


def _dispatch_log(strategy_name: str, row: pd.Series, currency: str) -> None:
    from . import screener  # heavy deps deferred to scan-time

    if strategy_name in _LONG_PREMIUM_STRATS:
        screener._log_long_premium(row, currency, weight_profile=AUTO_WEIGHT_PROFILE)
    elif strategy_name.startswith("Calendar"):
        screener._log_calendar(row, currency, weight_profile=AUTO_WEIGHT_PROFILE)
    elif strategy_name == "Iron Condor":
        screener._log_iron_condor(row, currency, weight_profile=AUTO_WEIGHT_PROFILE)
    elif strategy_name in {"Bull Put", "Bear Call"}:
        screener._log_credit_spread(row, currency, weight_profile=AUTO_WEIGHT_PROFILE)
    else:
        raise ValueError(f"No log handler for strategy {strategy_name!r}")


def _load_config(config_path: str = "config.json") -> dict:
    import json
    with open(config_path) as f:
        return json.load(f)


def _crypto_cfg(config: dict) -> dict:
    return config.get("crypto") or {}


def run_currency(
    currency: str,
    config: dict,
    db_path: Optional[str] = None,
    dry_run: bool = False,
    today_utc: Optional[str] = None,
) -> str:
    """Execute one currency. Returns a single-line status string for logging."""
    db_path = db_path or _default_db_path()
    crypto_cfg = _crypto_cfg(config)

    if not bool(crypto_cfg.get("auto_log_enabled", False)):
        return f"[auto-log] {currency} skipped: off-switch (auto_log_enabled=false)"

    open_n = count_open_positions(db_path, currency)
    if open_n >= MAX_OPEN_PER_CURRENCY:
        return f"[auto-log] {currency} skipped: concentration {open_n}/{MAX_OPEN_PER_CURRENCY} open"

    today_n = count_today_auto_logs(db_path, currency, today_utc=today_utc)
    if today_n >= MAX_AUTO_LOGS_PER_DAY:
        return f"[auto-log] {currency} skipped: per-day cap {today_n}/{MAX_AUTO_LOGS_PER_DAY}"

    from . import screener
    scan = screener._scan_currency(currency)
    if scan is None:
        return f"[auto-log] {currency} skipped: scan returned no data"

    if scan.get("regime") is None:
        return f"[auto-log] {currency} skipped: regime classifier returned None (insufficient history)"

    picks = scan.get("picks_by_strategy") or {}
    winner = pick_winner(picks)
    if winner is None:
        return f"[auto-log] {currency} skipped: no surfaced strategy buckets"

    strategy_name, row, score = winner
    floor = float(crypto_cfg.get("min_auto_log_score", DEFAULT_MIN_SCORE))
    if score < floor:
        return (f"[auto-log] {currency} skipped: top score {score:.3f} below floor {floor:.2f} "
                f"(would have logged {strategy_name})")

    if dry_run:
        return (f"[auto-log] {currency} DRY-RUN would log {strategy_name} "
                f"score={score:.3f} (no DB write)")

    _dispatch_log(strategy_name, row, currency)
    return f"[auto-log] {currency} logged {strategy_name} score={score:.3f}"


def main(argv: Optional[list] = None) -> int:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(prog="auto_logger", description="Crypto auto-log driver")
    parser.add_argument("--currency", choices=["BTC", "ETH"], default=None,
                        help="Restrict to one currency (default: both)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run safeguards + scan + pick but skip the DB write")
    parser.add_argument("--config", default="config.json",
                        help="Path to config.json (default: ./config.json)")
    args = parser.parse_args(argv)

    try:
        config = _load_config(args.config)
    except FileNotFoundError:
        print(f"[auto-log] ERROR: config not found at {args.config}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"[auto-log] ERROR: config malformed: {e}", file=sys.stderr)
        return 1

    started = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{started}] auto_log_crypto starting (dry_run={args.dry_run})")

    currencies = [args.currency] if args.currency else ["BTC", "ETH"]
    for cur in currencies:
        line = run_currency(cur, config, dry_run=args.dry_run)
        print(line)

    finished = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{finished}] auto_log_crypto done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
