#!/usr/bin/env python3
"""
Crypto portfolio viewer — reads paper_trades_crypto.db and prints
open and closed positions with live Deribit mark prices, in the same
layout as src.check_pnl for the equity book.

Usage:
    python -m src.crypto.check_pnl
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src import formatting as fmt
    HAS_FMT = fmt.supports_color()
except Exception:
    HAS_FMT = False
    fmt = None  # type: ignore

try:
    from src.crypto.data_fetching import get_options_chain, get_index_price
    HAS_DERIBIT = True
except Exception:
    HAS_DERIBIT = False

from src.core.pnl import realized_pnl

DB_PATH = _PROJECT_ROOT / "paper_trades_crypto.db"


def _width() -> int:
    try:
        return max(80, min(shutil.get_terminal_size(fallback=(110, 24)).columns, 130))
    except Exception:
        return 110


def _c(text: Any, color: str = "", bold: bool = False) -> str:
    if HAS_FMT and fmt and color:
        return fmt.colorize(str(text), color, bold=bold)
    return str(text)


def _dte(expiration: str) -> int:
    try:
        exp = datetime.strptime(str(expiration)[:10], "%Y-%m-%d").date()
        return (exp - date.today()).days
    except Exception:
        return 0


def _days_between(d1: str, d2: str) -> int:
    try:
        a = datetime.strptime(str(d1)[:10], "%Y-%m-%d").date()
        b = datetime.strptime(str(d2)[:10], "%Y-%m-%d").date()
        return max((b - a).days, 0)
    except Exception:
        return 0


def _short_strategy(row: Dict[str, Any]) -> str:
    s = (row.get("strategy_name") or "").lower()
    if "iron condor" in s:
        return "IC"
    if "bull put" in s or "bear call" in s or row.get("long_strike"):
        return "SPREAD"
    return (row.get("type") or "").upper()


def _unrealized(entry: float, live: float, qty, side: str,
                structure: str = "debit"):
    """Open-position unrealized P&L via core.pnl, scaled by qty."""
    r = realized_pnl(entry=entry, exit_price=live, qty=qty, side=side,
                      structure=structure)
    return r["pnl_usd"], r["pnl_pct"]


def _is_short(strategy_name: str) -> bool:
    s = (strategy_name or "").lower()
    return any(k in s for k in ("short", "credit", "bull put", "bear call", "iron condor"))


def _fetch_open_positions(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    cur = conn.execute("SELECT * FROM trades WHERE status='OPEN' ORDER BY date DESC")
    return cur.fetchall()


def _fetch_closed_positions(conn: sqlite3.Connection, limit: Optional[int] = None) -> List[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    q = "SELECT * FROM trades WHERE status='CLOSED' ORDER BY exit_date DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    cur = conn.execute(q)
    return cur.fetchall()


def _load_chains() -> Dict[str, Any]:
    """Fetch BTC + ETH Deribit chains once; return {ticker: DataFrame|None}."""
    chains: Dict[str, Any] = {"BTC": None, "ETH": None}
    if not HAS_DERIBIT:
        return chains
    for tk in ("BTC", "ETH"):
        try:
            df = get_options_chain(tk)
            chains[tk] = df if (df is not None and not df.empty) else None
        except Exception:
            chains[tk] = None
    return chains


def _live_option_price(chain, ticker: str, expiration: str, strike: float, opt_type: str) -> Optional[float]:
    """Lookup live mark price (USD) from a cached Deribit chain DataFrame."""
    if chain is None or chain.empty:
        return None
    try:
        exp_iso = str(expiration)[:10]
        rows = chain[
            (chain["expiration"].astype(str) == exp_iso)
            & (chain["strike"].astype(float) == float(strike))
            & (chain["type"].astype(str).str.lower() == opt_type.lower())
        ]
        if rows.empty:
            return None
        mark = float(rows["mark_price"].iloc[0])
        return mark if mark > 0 else None
    except Exception:
        return None


def _live_spot(ticker: str) -> Optional[float]:
    if not HAS_DERIBIT:
        return None
    try:
        v = get_index_price(ticker)
        return float(v) if v else None
    except Exception:
        return None


def _format_pnl(pnl_usd: Optional[float], pnl_pct: Optional[float]) -> tuple[str, str]:
    if pnl_usd is None or pnl_pct is None:
        return ("       —", "      —")
    color = "green" if pnl_usd > 0 else ("red" if pnl_usd < 0 else "")
    sign = "+" if pnl_usd >= 0 else ""
    s_usd = _c(f"{sign}${pnl_usd:,.2f}", color, bold=True)
    s_pct = _c(f"{sign}{pnl_pct * 100:.1f}%", color, bold=False)
    return (s_usd, s_pct)


def _print_header(width: int) -> None:
    bar = "=" * width
    now_local = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(bar)
    print(f"  CRYPTO PAPER PORTFOLIO  —  {now_local}")
    print(bar)


def _print_open(open_rows: List[sqlite3.Row], chains: Dict[str, Any], width: int) -> None:
    print()
    print(f"  {_c('OPEN POSITIONS', '', bold=True)}")
    print()
    if not open_rows:
        print("  (no open positions)")
        return

    header = (
        f"  {'Ticker':<7} {'Type':<6} {'Strike':>9} {'Expiry':<12} {'DTE':>4} "
        f"{'Opened':<12} {'Held':>5} {'Entry $':>10} {'Live $':>10} {'P/L $':>11} {'P/L %':>8}"
    )
    print(header)
    print("  " + "-" * (width - 2))

    today = date.today().isoformat()
    total_unreal = 0.0
    total_entry = 0.0
    have_live = 0

    for r in open_rows:
        tk = (r["ticker"] or "").upper()
        chain = chains.get(tk)
        opt_type = (r["type"] or "").lower()
        entry = float(r["entry_price"] or 0.0)
        strike = float(r["strike"] or 0.0)
        exp = str(r["expiration"] or "")[:10]
        opened = str(r["date"] or "")[:10]
        live = _live_option_price(chain, tk, exp, strike, opt_type)

        is_short = _is_short(r["strategy_name"] or "")
        if live is None:
            pnl_usd: Optional[float] = None
            pnl_pct: Optional[float] = None
            live_str = "       —"
        else:
            side = "short" if is_short else "long"
            try:
                qty = float(r["quantity"]) if r["quantity"] is not None else 1.0
            except (KeyError, TypeError, IndexError):
                qty = 1.0
            pnl_usd, pnl_pct = _unrealized(entry, live, qty, side)
            live_str = f"${live:>8,.2f}"
            total_unreal += pnl_usd
            have_live += 1
        total_entry += entry

        p_usd, p_pct = _format_pnl(pnl_usd, pnl_pct)
        held = _days_between(opened, today)
        dte_d = _dte(exp)
        print(
            f"  {tk:<7} {_short_strategy(dict(r)):<6} {strike:>9,.0f} {exp:<12} {dte_d:>4} "
            f"{opened:<12} {held:>3}d {entry:>10,.2f} {live_str:>10} {p_usd:>11} {p_pct:>8}"
        )

    print("  " + "-" * (width - 2))
    if have_live:
        pct = (total_unreal / total_entry * 100.0) if total_entry else 0.0
        color = "green" if total_unreal > 0 else ("red" if total_unreal < 0 else "")
        sign = "+" if total_unreal >= 0 else ""
        print(
            f"  Unrealized P/L: {_c(f'{sign}${total_unreal:,.2f}', color, bold=True)} "
            f" ({_c(f'{sign}{pct:.1f}%', color)})   [{have_live}/{len(open_rows)} live prices]"
        )
    else:
        print(f"  Unrealized P/L: —   [0/{len(open_rows)} live prices]")


def _print_closed(closed_rows: List[sqlite3.Row], width: int, limit: int = 50) -> None:
    print()
    print(f"  {_c('CLOSED POSITIONS', '', bold=True)}")
    print()
    if not closed_rows:
        print("  (no closed positions)")
        return

    rows = closed_rows[:limit]
    header = (
        f"  {'Ticker':<7} {'Type':<6} {'Strike':>9} {'Expiry':<12} {'Opened':<12} {'Closed':<19} "
        f"{'Entry $':>10} {'Exit $':>10} {'P/L $':>12} {'P/L %':>8}  {'Reason':<22}"
    )
    print(header)
    print("  " + "-" * (width - 2))

    total_usd = 0.0
    wins = 0
    losses = 0
    for r in rows:
        tk = (r["ticker"] or "").upper()
        entry = float(r["entry_price"] or 0.0)
        exit_p = float(r["exit_price"] or 0.0)
        strike = float(r["strike"] or 0.0)
        exp = str(r["expiration"] or "")[:10]
        opened = str(r["date"] or "")[:10]
        closed = str(r["exit_date"] or "")[:19]
        pnl_usd = float(r["pnl_usd"] or 0.0)
        pnl_pct = float(r["pnl_pct"] or 0.0)
        total_usd += pnl_usd
        if pnl_usd > 0:
            wins += 1
            result = _c("WIN ", "green", bold=True)
        elif pnl_usd < 0:
            losses += 1
            result = _c("LOSS", "red", bold=True)
        else:
            result = _c("FLAT", "")
        p_usd, p_pct = _format_pnl(pnl_usd, pnl_pct)
        reason = (r["exit_reason"] or "")[:22]
        opt_type = (r["type"] or "").upper()
        print(
            f"  {tk:<7} {opt_type:<6} {strike:>9,.0f} {exp:<12} {opened:<12} {closed:<19} "
            f"{entry:>10,.2f} {exit_p:>10,.2f} {p_usd:>12} {p_pct:>8}  {reason:<22}  {result}"
        )

    print("  " + "-" * (width - 2))
    total = wins + losses
    wr = (wins / total * 100.0) if total else 0.0
    color = "green" if total_usd > 0 else ("red" if total_usd < 0 else "")
    sign = "+" if total_usd >= 0 else ""
    print(
        f"  Realized P/L: {_c(f'{sign}${total_usd:,.2f}', color, bold=True)}   "
        f"Wins: {wins}  Losses: {losses}  Win Rate: {wr:.1f}%   "
        f"[{len(rows)}/{len(closed_rows)} shown]"
    )


def _print_market_context(chains: Dict[str, Any]) -> None:
    print()
    print(f"  {_c('MARKET CONTEXT', '', bold=True)}")
    for tk in ("BTC", "ETH"):
        spot = _live_spot(tk)
        if spot is None:
            print(f"  {tk}: spot unavailable")
            continue
        chain = chains.get(tk)
        n_contracts = 0 if (chain is None or chain.empty) else len(chain)
        print(f"  {tk}: spot ${spot:,.2f}   chain={n_contracts} contracts")


def main() -> int:
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found.")
        return 1

    width = _width()

    with closing(sqlite3.connect(str(DB_PATH))) as conn:
        open_rows = _fetch_open_positions(conn)
        closed_rows = _fetch_closed_positions(conn)

    chains = _load_chains() if open_rows else {"BTC": None, "ETH": None}

    _print_header(width)
    _print_market_context(chains)
    _print_open(open_rows, chains, width)
    _print_closed(closed_rows, width)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
