"""The tracked lottery sleeve: auto-log + performance reporting.

Long far-OTM tickets are negative-EV on average — the whole point of tracking a
sleeve is to find out, forward and honestly, whether the *selected* subset (the
✦-edge flag) beats a blind basket. So this module:

  autolog_lottery_sleeve  logs top-N non-trap picks into paper_trades.db under a
                          self-identifying ``strategy_name`` ("Lottery Long Call"
                          /"Lottery Long Put"), tiny size, one per ticker, capped
                          total exposure, ``paper_only=1`` (never in the real-money
                          long-call cohort), and persists the edge flag.
  compute_sleeve_stats    pure stats over closed/open rows: hit-rate (≥Nx), median
                          winner multiple, best tail, realized $, and the
                          edge-flagged-vs-unflagged split.
  print_lottery_sleeve    check_pnl segment that fetches + renders those stats.

Rows are identified by the ``strategy_name`` prefix "Lottery ", so nothing else
in the book is touched.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from statistics import median
from typing import Any, Dict, List, Optional

SLEEVE_PREFIX = "Lottery "


def _load_sleeve_cfg(config_path: str = "config.json") -> Dict[str, Any]:
    defaults = {
        "auto_log_top_n": 5,
        "max_sleeve_exposure": 500.0,  # $ cap on total OPEN lottery debit
        "hit_multiple": 3.0,           # a "hit" = closed ticket returned >= 3x debit
    }
    try:
        with open(config_path) as fh:
            user = (json.load(fh) or {}).get("lottery_sleeve") or {}
        for k in defaults:
            if k in user:
                defaults[k] = user[k]
    except Exception:
        pass
    return defaults


def _signed_delta(row) -> Optional[float]:
    try:
        d = row.get("delta")
        if d is not None:
            return float(d)
        ad = row.get("abs_delta")
        if ad is None:
            return None
        ad = float(ad)
        is_put = str(row.get("type", "call")).lower().startswith("p")
        return -ad if is_put else ad
    except Exception:
        return None


def build_sleeve_trade(row) -> Optional[Dict[str, Any]]:
    """Build a log_trade dict from a scored lottery row, or None if unloggable."""
    try:
        sym = str(row.get("symbol") or row.get("ticker") or "").upper()
        opt_type = str(row.get("type", "call")).lower()
        strike = float(row.get("strike"))
        exp = str(row.get("expiration"))
        prem = float(row.get("premium"))
    except Exception:
        return None
    if not sym or prem <= 0 or strike <= 0 or not exp:
        return None
    side = "Call" if opt_type.startswith("c") else "Put"
    return {
        "ticker": sym,
        "expiration": exp,
        "strike": strike,
        "type": opt_type,
        "entry_price": prem,
        "quality_score": float(row.get("lottery_ticket_score") or row.get("quality_score") or 0.0),
        "strategy_name": f"{SLEEVE_PREFIX}Long {side}",
        "entry_iv": row.get("impliedVolatility"),
        "entry_delta": _signed_delta(row),
        "catalyst_score": row.get("catalyst_score"),
        "iv_rank_score": row.get("iv_rank_score"),
        "momentum_score": row.get("momentum_score"),
        "lottery_edge": bool(row.get("lottery_edge", False)),
        "paper_only": 1,          # lottery is a satellite sleeve, never the LC cohort
        "era": "finalized",
    }


def autolog_lottery_sleeve(
    picks_df,
    pm,
    config_path: str = "config.json",
    top_n: Optional[int] = None,
) -> List[str]:
    """Log the top-N non-trap lottery picks into the sleeve. One per ticker, capped
    at ``max_sleeve_exposure`` of open debit. Returns a list of human-readable
    descriptions of what was logged. Never raises on a single bad row.
    """
    if picks_df is None or len(picks_df) == 0:
        return []
    cfg = _load_sleeve_cfg(config_path)
    n = int(top_n if top_n is not None else cfg["auto_log_top_n"])
    max_exposure = float(cfg["max_sleeve_exposure"])

    open_debit = _open_sleeve_debit(pm.db_path)
    logged: List[str] = []
    seen_tickers = set()

    for i in range(len(picks_df)):
        if len(logged) >= n:
            break
        row = picks_df.iloc[i]
        # Skip crush traps — shown on the board, never picked.
        if str(row.get("lottery_crush", "") or ""):
            continue
        sym = str(row.get("symbol") or row.get("ticker") or "").upper()
        if not sym or sym in seen_tickers:
            continue
        trade = build_sleeve_trade(row)
        if trade is None:
            continue
        debit = trade["entry_price"] * 100.0
        if open_debit + debit > max_exposure:
            continue  # respect the sleeve exposure cap
        try:
            if pm.log_trade_if_new(trade):
                open_debit += debit
                seen_tickers.add(sym)
                edge = "✦" if trade["lottery_edge"] else " "
                logged.append(f"{edge} {trade['strategy_name']} {sym} ${trade['strike']:.0f} @ ${trade['entry_price']:.2f}")
        except Exception:
            continue
    return logged


def _open_sleeve_debit(db_path: str) -> float:
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            rows = conn.execute(
                "SELECT entry_price, COALESCE(quantity,1.0) q FROM trades "
                "WHERE status='OPEN' AND strategy_name LIKE ?",
                (SLEEVE_PREFIX + "%",),
            ).fetchall()
        return sum(float(p or 0) * float(q or 1.0) * 100.0 for p, q in rows)
    except Exception:
        return 0.0


# ── reporting ────────────────────────────────────────────────────────────────
def _multiple(row) -> Optional[float]:
    """Closed long-premium payoff multiple = exit/entry = 1 + pnl_pct."""
    pct = row.get("pnl_pct")
    if pct is None:
        return None
    try:
        return 1.0 + float(pct)
    except (TypeError, ValueError):
        return None


def _hit_rate(closed: List[dict], hit_multiple: float) -> Optional[float]:
    mults = [m for m in (_multiple(r) for r in closed) if m is not None]
    if not mults:
        return None
    return sum(1 for m in mults if m >= hit_multiple) / len(mults)


def compute_sleeve_stats(
    open_rows: List[dict], closed_rows: List[dict], hit_multiple: float = 3.0
) -> Dict[str, Any]:
    """Pure sleeve statistics. Edge split compares hit-rate of tickets that cleared
    the ✦ bar at entry vs those that didn't — the core "does selection help" read.
    """
    open_debit = sum(float(r.get("entry_price") or 0) * float(r.get("quantity") or 1.0) * 100.0
                     for r in open_rows)
    mults = [m for m in (_multiple(r) for r in closed_rows) if m is not None]
    winners = [m for m in mults if m >= hit_multiple]
    realized = sum(float(r.get("pnl_usd") or 0.0) for r in closed_rows)

    edged = [r for r in closed_rows if r.get("lottery_edge") in (1, True)]
    unedged = [r for r in closed_rows if r.get("lottery_edge") in (0, None)]

    return {
        "n_open": len(open_rows),
        "open_debit": open_debit,
        "n_closed": len(closed_rows),
        "n_hits": len(winners),
        "hit_rate": (len(winners) / len(mults)) if mults else None,
        "median_winner_x": median(winners) if winners else None,
        "best_tail_x": max(mults) if mults else None,
        "realized_usd": realized,
        "edge_hit_rate": _hit_rate(edged, hit_multiple),
        "edge_n": len(edged),
        "noedge_hit_rate": _hit_rate(unedged, hit_multiple),
        "noedge_n": len(unedged),
        "hit_multiple": hit_multiple,
    }


def fetch_sleeve_rows(db_path: str = "paper_trades.db"):
    """Return (open_rows, closed_rows) dicts for the lottery sleeve."""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM trades WHERE strategy_name LIKE ?", (SLEEVE_PREFIX + "%",)
        ).fetchall()
    rows = [dict(r) for r in rows]
    open_rows = [r for r in rows if str(r.get("status")) == "OPEN"]
    closed_rows = [r for r in rows if str(r.get("status")) != "OPEN"]
    return open_rows, closed_rows


def print_lottery_sleeve(db_path: str = "paper_trades.db", config_path: str = "config.json",
                         width: int = 100) -> None:
    """check_pnl segment: render the lottery sleeve scorecard. No-op (silent) when
    the sleeve is empty, so it never clutters a book that isn't using it."""
    try:
        open_rows, closed_rows = fetch_sleeve_rows(db_path)
    except Exception:
        return
    if not open_rows and not closed_rows:
        return
    cfg = _load_sleeve_cfg(config_path)
    st = compute_sleeve_stats(open_rows, closed_rows, float(cfg["hit_multiple"]))

    try:
        from src import formatting as fmt
        _has = fmt.supports_color()
    except Exception:
        fmt = None
        _has = False

    def _c(s, color="", bold=False):
        return fmt.colorize(s, color, bold=bold) if (fmt and _has) else s

    hm = st["hit_multiple"]
    print()
    title = f"LOTTERY SLEEVE  (satellite — hit = ≥{hm:.0f}× debit, held to expiry)"
    if fmt and _has:
        print(_c("─" * width, fmt.Colors.DIM))
        print(_c("  " + title, fmt.Colors.BOLD))
    else:
        print("-" * width)
        print("  " + title.replace("≥", ">="))

    def _pct(v):
        return f"{v*100:.0f}%" if v is not None else "n/a"
    def _x(v):
        return f"{v:.1f}x" if v is not None else "n/a"

    print(f"  Open: {st['n_open']}  (${st['open_debit']:,.0f} at risk)   "
          f"Closed: {st['n_closed']}   Realized: ${st['realized_usd']:,.0f}")
    if st["n_closed"]:
        hr = st["hit_rate"]
        hr_c = fmt.Colors.BRIGHT_GREEN if (hr and hr > 0) else fmt.Colors.DIM
        print(f"  Hits (≥{hm:.0f}×): {st['n_hits']}/{st['n_closed']}  "
              f"({_c(_pct(hr), hr_c)})   "
              f"median winner {_x(st['median_winner_x'])}   best tail {_c(_x(st['best_tail_x']), fmt.Colors.BRIGHT_YELLOW if fmt else '')}")
        # The validation the whole sleeve exists for:
        ev, uv = st["edge_hit_rate"], st["noedge_hit_rate"]
        print("  " + _c(f"Edge-flag validation:  ✦ {_pct(ev)} hit ({st['edge_n']}n)  "
                        f"vs  ·  {_pct(uv)} hit ({st['noedge_n']}n)", fmt.Colors.BRIGHT_CYAN if fmt else ""))
        if ev is not None and uv is not None:
            verdict = "edge selection is helping" if ev > uv else ("no separation yet" if ev == uv else "edge NOT beating blind — watch")
            print(_c(f"    → {verdict}", fmt.Colors.DIM))
    else:
        print(_c("  No closed tickets yet — hit-rate/edge validation populate as tickets resolve.", fmt.Colors.DIM if fmt else ""))


__all__ = [
    "SLEEVE_PREFIX", "build_sleeve_trade", "autolog_lottery_sleeve",
    "compute_sleeve_stats", "fetch_sleeve_rows", "print_lottery_sleeve",
]
