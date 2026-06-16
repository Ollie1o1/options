"""Earnings IV-crush SELLING backtest on real DoltHub marks (P3.8).

The bigger trade behind the ~24% post-earnings IV crush we measured: SELL the
vol BEFORE the report and buy it back AFTER. Here: a short strangle (sell an OTM
call + an OTM put in the front expiry just past earnings), opened a couple days
before the announcement, closed the first data day after it. The seller keeps
the crush as long as the underlying move stays inside the strikes — the whole
risk being a large earnings gap.

Entry = real BID (sell-to-open both legs); exit = real ASK (buy-to-close both),
all-in (real spread + 4-leg commission). Return reported on CREDIT collected
(naked strangle has no defined max-risk) plus raw per-contract P&L and the
realized underlying move, so the gap risk is visible.

CLI:
    python -m src.dolt_earnings_sell --symbols AAPL,NVDA,MSFT,GOOG,AMD
"""
from __future__ import annotations

import datetime as _dt
import json
from statistics import mean, median
from typing import Any, Dict, List, Optional

from src.dolt_cohort import _stats


def _dte(asof: str, expiration: str) -> int:
    return (_dt.date.fromisoformat(expiration) - _dt.date.fromisoformat(asof)).days


def _pick_strangle(chain, asof, spot, call_delta, put_delta, min_dte, max_dte):
    """Pick (short OTM call ~call_delta, short OTM put ~put_delta) in the nearest
    expiry within [min_dte, max_dte] — the front expiry just past earnings."""
    valid_exp = sorted({c["expiration"] for c in chain
                        if c.get("expiration") and min_dte <= _dte(asof, c["expiration"]) <= max_dte})
    for exp in valid_exp:                       # nearest qualifying expiry first
        calls = [c for c in chain if c.get("type") == "call" and c["expiration"] == exp
                 and c.get("delta") is not None and c.get("strike") and c["strike"] >= spot]
        puts = [c for c in chain if c.get("type") == "put" and c["expiration"] == exp
                and c.get("delta") is not None and c.get("strike") and c["strike"] <= spot]
        if not calls or not puts:
            continue
        sc = min(calls, key=lambda c: abs(abs(c["delta"]) - call_delta))
        sp = min(puts, key=lambda c: abs(abs(c["delta"]) - put_delta))
        return sc, sp
    return None


def _leg(chain, opt_type, strike, expiration):
    return next((c for c in chain if c.get("type") == opt_type and c.get("strike") == strike
                 and c.get("expiration") == expiration), None)


def simulate_earnings_strangle(symbol, earnings_date, db_path=None,
                               call_delta=0.20, put_delta=0.20,
                               entry_offset=2, min_dte=30, max_dte=70,
                               commission_per_contract=0.65, contract_multiplier=100,
                               spots=None, exit_scan_days=10) -> Optional[Dict[str, Any]]:
    """Sell a strangle ~entry_offset days before earnings, buy it back the first
    data day after. Return dict with ret-on-credit + raw pnl + realized move.

    ``spots`` must be RAW (un-split-adjusted) closes — DoltHub strikes are raw, so
    a split-adjusted spot (e.g. yfinance) breaks the OTM moneyness pick for split
    names. Built via dolt_stocks.close_history when not supplied."""
    from src import dolt_options as _do
    if spots is None:
        from src.dolt_stocks import close_history
        spots = close_history(symbol, db_path=db_path or _do.DEFAULT_CACHE)
    # entry chain: a data day at/just before (earnings - entry_offset)
    target_entry = (_dt.date.fromisoformat(earnings_date) - _dt.timedelta(days=entry_offset)).isoformat()
    ed_in, entry_chain = _do.get_chain_near(symbol, target_entry, max_skip=4,
                                            db_path=db_path or _do.DEFAULT_CACHE, direction=-1)
    if not entry_chain or ed_in is None or ed_in >= earnings_date:
        return None
    spot_in = spots.get(ed_in) or next((spots[d] for d in sorted(spots) if d >= ed_in), None)
    if spot_in is None:
        return None
    pick = _pick_strangle(entry_chain, ed_in, spot_in, call_delta, put_delta, min_dte, max_dte)
    if not pick:
        return None
    sc, sp = pick
    if sc.get("bid") is None or sp.get("bid") is None:
        return None
    entry_credit = sc["bid"] + sp["bid"]
    if entry_credit <= 0:
        return None
    exp = sc["expiration"]
    # Exit: scan forward day-by-day from earnings+1 for the FIRST snapshot where
    # BOTH legs quote. The DoltHub archive's per-snapshot expiry sets are
    # inconsistent (weeklies churn; only standard monthlies persist), so a single
    # fixed exit day often lacks the exact contract — scanning tolerates the gaps,
    # the same way the spread/short runners do. Captures the post-earnings crush
    # (which lands on day 1) on the first quotable day.
    cc = cp = None
    ad_out = None
    base = _dt.date.fromisoformat(earnings_date)
    for off in range(1, exit_scan_days + 1):
        d = (base + _dt.timedelta(days=off)).isoformat()
        if d > _do.COVERAGE_MAX:
            break
        ch = _do.get_chain(symbol, d, db_path=db_path or _do.DEFAULT_CACHE)
        if not ch:
            continue
        _cc = _leg(ch, "call", sc["strike"], exp)
        _cp = _leg(ch, "put", sp["strike"], exp)
        if _cc and _cp and _cc.get("ask") is not None and _cp.get("ask") is not None:
            cc, cp, ad_out = _cc, _cp, d
            break
    if cc is None or cp is None or ad_out is None:
        return None
    exit_cost = cc["ask"] + cp["ask"]
    comm = 4.0 * commission_per_contract / contract_multiplier   # 2 legs x open+close
    net = (entry_credit - exit_cost) - comm
    spot_out = spots.get(ad_out) or spot_in
    return {
        "symbol": symbol, "earnings_date": earnings_date,
        "entry_date": ed_in, "exit_date": ad_out, "expiration": exp,
        "credit": round(entry_credit, 4), "exit_cost": round(exit_cost, 4),
        "net_pnl": round(net, 4), "ret": net / entry_credit,
        "call_strike": sc["strike"], "put_strike": sp["strike"],
        "realized_move": round(spot_out / spot_in - 1.0, 4) if spot_in else None,
        "iv_in": sc.get("iv"), "iv_out": cc.get("iv"),
    }


def run_earnings_sell_backtest(symbols, db_path=None, start=None, end=None,
                               call_delta=0.20, put_delta=0.20) -> Dict[str, Any]:
    from src import dolt_options as _do
    from src.dolt_earnings import earnings_dates
    from src.dolt_stocks import close_history
    spot_db = db_path or _do.DEFAULT_CACHE
    import sys
    trades: List[Dict[str, Any]] = []
    for symbol in symbols:
        symbol = symbol.upper()
        if not _do.symbol_has_data(symbol, spot_db):
            print(f"  [warning] no chain data for {symbol}; skipped", file=sys.stderr)
            continue
        spots = close_history(symbol, db_path=spot_db)   # RAW closes (split-safe), built once
        for ed in earnings_dates(symbol, db_path=spot_db):
            if ed < _do.COVERAGE_MIN or ed > _do.COVERAGE_MAX:
                continue
            if start and ed < start:
                continue
            if end and ed > end:
                continue
            try:
                t = simulate_earnings_strangle(symbol, ed, db_path=db_path,
                                               call_delta=call_delta, put_delta=put_delta,
                                               spots=spots)
            except _do.DoltRateLimited:
                return _summarize(trades, partial=True)
            except _do.DoltQueryError:
                continue
            if t:
                trades.append(t)
    return _summarize(trades, partial=False)


def _summarize(trades, partial: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(trades), "partial": partial, "trades": trades}
    if not trades:
        return out
    out.update(_stats([t["ret"] for t in trades]))   # ret on credit collected
    out["mean_credit"] = round(mean(t["credit"] for t in trades), 3)
    out["mean_net_pnl"] = round(mean(t["net_pnl"] for t in trades), 3)
    out["mean_abs_move"] = round(mean(abs(t["realized_move"]) for t in trades
                                      if t["realized_move"] is not None), 4)
    return out


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Earnings IV-crush SELLING backtest on real marks")
    ap.add_argument("--symbols", default="AAPL,NVDA,MSFT,GOOG,AMD")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--call-delta", type=float, default=0.20)
    ap.add_argument("--put-delta", type=float, default=0.20)
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    from src import dolt_options as _do
    cfg = {}
    try:
        cfg = json.load(open("config.json")).get("dolt_options", {})
    except Exception:
        pass
    db = args.db or cfg.get("cache_path")
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print(f"Earnings IV-crush SELLING (short strangle {args.call_delta}/{args.put_delta}d): {syms}")
    out = run_earnings_sell_backtest(syms, db_path=db, start=args.start, end=args.end,
                                     call_delta=args.call_delta, put_delta=args.put_delta)
    summary = {k: v for k, v in out.items() if k != "trades"}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    _cli()
