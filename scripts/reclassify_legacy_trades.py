"""One-shot DB cleanup: re-classify legacy single-leg trades whose strategy_name
was hardcoded to "Short {Type}" by the old log path.

Authority of truth: trades_log/entries.csv records the screener `mode` for every
interactively logged trade. Discovery / Budget / Single-stock = long buys;
Premium Selling = short premium. We match each DB row to its entries.csv row on
(date, ticker, type, strike, expiration) and flip strategy_name only when the
authoritative source says it should be a long.

For closed rows whose strategy_name flips, we also recompute pnl_pct so the
displayed P/L sign matches the (corrected) direction. The original friction
cost is preserved exactly:

    pnl_pct_old (short) = (entry-exit)/entry - friction
    friction            = (entry-exit)/entry - pnl_pct_old
    pnl_pct_new (long)  = (exit-entry)/entry - friction
                        = -pnl_pct_old - 2*friction
"""
from __future__ import annotations
import csv
import sqlite3
from pathlib import Path
from collections import Counter, defaultdict

REPO = Path("/Users/ollie/Desktop/options")
DB = REPO / "paper_trades.db"
CSV_LOG = REPO / "trades_log" / "entries.csv"

LONG_MODES = {"Discovery scan", "Budget scan", "Single-stock"}
SHORT_MODES = {"Premium Selling"}


def _norm_strike(s: str) -> str:
    try:
        return f"{float(s):.4f}"
    except (TypeError, ValueError):
        return ""


def load_csv_modes() -> dict[tuple, set[str]]:
    """Returns {(date, ticker, type, strike, expiration): set(modes)}.
    Uses set because the same spec can appear in multiple scans on the same day."""
    out: dict[tuple, set[str]] = defaultdict(set)
    with CSV_LOG.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = (row.get("timestamp") or "")[:10]
            ticker = (row.get("symbol") or "").upper().strip()
            otype = (row.get("type") or "").lower().strip()
            strike = _norm_strike(row.get("strike", ""))
            exp = (row.get("expiration") or "").strip()
            mode = (row.get("mode") or "").strip()
            if date and ticker and otype and strike and exp and mode:
                out[(date, ticker, otype, strike, exp)].add(mode)
    return out


def main() -> None:
    csv_modes = load_csv_modes()
    print(f"Loaded {len(csv_modes)} unique trade-specs from entries.csv")

    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    rows = cur.execute(
        """
        SELECT entry_id, date, ticker, type, strike, expiration,
               strategy_name, status, entry_price, exit_price, pnl_pct
          FROM trades
         WHERE strategy_name IN ('Short Call', 'Short Put')
        """
    ).fetchall()
    print(f"Candidate rows (Short Call/Put): {len(rows)}")

    flips: list[tuple] = []
    keeps_short: list[int] = []
    unmatched: list[int] = []
    ambiguous: list[int] = []

    for r in rows:
        date = (r["date"] or "")[:10]
        ticker = (r["ticker"] or "").upper()
        otype = (r["type"] or "").lower()
        strike = _norm_strike(r["strike"])
        exp = (r["expiration"] or "")
        key = (date, ticker, otype, strike, exp)
        modes = csv_modes.get(key, set())

        if not modes:
            unmatched.append(r["entry_id"])
            continue

        is_long = bool(modes & LONG_MODES)
        is_short = bool(modes & SHORT_MODES)
        if is_long and is_short:
            ambiguous.append(r["entry_id"])
            continue
        if is_short and not is_long:
            keeps_short.append(r["entry_id"])
            continue
        if is_long and not is_short:
            flips.append(r)

    print(f"  → flip to Long (matched buy-side mode): {len(flips)}")
    print(f"  → keep Short (matched Premium Selling): {len(keeps_short)}")
    print(f"  → unmatched (no entries.csv record):    {len(unmatched)}")
    print(f"  → ambiguous (both modes for same spec): {len(ambiguous)}")

    if not flips:
        print("Nothing to flip. Exiting.")
        return

    # Apply flips
    closed_recomputed = 0
    for r in flips:
        new_strategy = "Long Call" if r["type"].lower() == "call" else "Long Put"
        update_pnl_pct = None

        if (
            r["status"] == "CLOSED"
            and r["entry_price"] is not None
            and r["exit_price"] is not None
            and r["pnl_pct"] is not None
            and float(r["entry_price"]) > 0
        ):
            entry = float(r["entry_price"])
            exit_ = float(r["exit_price"])
            pnl_old = float(r["pnl_pct"])
            old_raw_short = (entry - exit_) / entry
            friction = old_raw_short - pnl_old  # >= 0 typically
            new_raw_long = (exit_ - entry) / entry
            update_pnl_pct = new_raw_long - friction
            closed_recomputed += 1

        if update_pnl_pct is None:
            cur.execute(
                "UPDATE trades SET strategy_name = ? WHERE entry_id = ?",
                (new_strategy, r["entry_id"]),
            )
        else:
            cur.execute(
                "UPDATE trades SET strategy_name = ?, pnl_pct = ? WHERE entry_id = ?",
                (new_strategy, update_pnl_pct, r["entry_id"]),
            )

    con.commit()
    con.close()
    print(f"Updated {len(flips)} strategy_name labels.")
    print(f"Recomputed pnl_pct on {closed_recomputed} closed rows (friction preserved).")

    # Diagnostic: counts by strategy after the run
    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT strategy_name, status, COUNT(*) FROM trades
         WHERE strategy_name LIKE 'Short %' OR strategy_name LIKE 'Long %'
         GROUP BY strategy_name, status
         ORDER BY strategy_name, status
        """
    ).fetchall()
    print("\nPost-update tally:")
    for sn, st, n in rows:
        print(f"  {sn:<14} {st:<8} {n}")
    con.close()


if __name__ == "__main__":
    main()
