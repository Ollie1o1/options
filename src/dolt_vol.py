"""Delta-hedged short-straddle backtest on real DoltHub marks — the PURE vol test.

Every short-premium sleeve so far is contaminated by direction (a short put is
short the market). This isolates the question that actually matters: is there a
VOL edge — selling implied vol that exceeds realized — independent of getting
direction right? Sell an ATM straddle, then neutralize delta along the REAL spot
path (rehedge on each day the chain quotes fresh deltas, accrue hedge P&L over
the full daily spot series between rehedges). What's left is the vol P&L:
premium collected + decay − realized-move (gamma) cost − transaction costs.

If this has an edge AND is uncorrelated with the directional short-premium
sleeves, it's the diversifier the earnings strangle wasn't. If it's ~breakeven
too, the data has no vol edge after costs and (C) is final.

Entry = real BID (sell straddle); exit = real ASK (buy to close). Raw spot
(split-safe). Front MONTHLY expiry (min_dte≥30) so the strikes persist in the
archive. ret is on premium collected.

CLI:
    python -m src.dolt_vol --symbols AAPL,SPY,NVDA,AMZN,MSFT
"""
from __future__ import annotations

import datetime as _dt
import json
from typing import Any, Dict, List, Optional

from src.dolt_cohort import _stats


_SPLIT_GUARD = 0.40   # a >40% one-day move in RAW prices is a split, not a market move


def _dte(asof: str, expiration: str) -> int:
    return (_dt.date.fromisoformat(expiration) - _dt.date.fromisoformat(asof)).days


def _atm_straddle(chain, asof, spot, min_dte, max_dte):
    """Pick (call, put) at the strike nearest spot in the nearest qualifying
    expiry (both legs same strike + expiry)."""
    exps = sorted({c["expiration"] for c in chain
                   if c.get("expiration") and min_dte <= _dte(asof, c["expiration"]) <= max_dte})
    for exp in exps:
        calls = {c["strike"]: c for c in chain if c.get("type") == "call"
                 and c["expiration"] == exp and c.get("strike") is not None}
        puts = {c["strike"]: c for c in chain if c.get("type") == "put"
                and c["expiration"] == exp and c.get("strike") is not None}
        common = set(calls) & set(puts)
        if not common:
            continue
        k = min(common, key=lambda s: abs(s - spot))
        return calls[k], puts[k]
    return None


def _leg(chain, opt_type, strike, expiration):
    return next((c for c in chain if c.get("type") == opt_type and c.get("strike") == strike
                 and c.get("expiration") == expiration), None)


def simulate_delta_hedged_straddle(symbol, entry_date, spots, sdates, db_path=None,
                                   target_days=21, min_dte=30, max_dte=70,
                                   commission_per_contract=0.65, contract_multiplier=100
                                   ) -> Optional[Dict[str, Any]]:
    """Sell one ATM straddle, delta-hedge along the real spot path, close at
    target_days (or expiry). Returns delta-hedged (vol) P&L per contract."""
    from src import dolt_options as _do
    kw = {"db_path": db_path} if db_path else {}
    ed, chain = _do.get_chain_near(symbol, entry_date, **kw)
    if not chain or ed is None or ed not in spots:
        return None
    spot0 = spots[ed]
    pick = _atm_straddle(chain, ed, spot0, min_dte, max_dte)
    if not pick:
        return None
    call, put = pick
    if call.get("bid") is None or put.get("bid") is None or call.get("delta") is None or put.get("delta") is None:
        return None
    credit = call["bid"] + put["bid"]
    if credit <= 0:
        return None
    K, exp = call["strike"], call["expiration"]
    m = contract_multiplier
    # short straddle: position delta (shares) = -(call_delta + put_delta)*m ; hedge neutralizes it
    pos_delta_sh = -(call["delta"] + put["delta"]) * m
    hedge_sh = -pos_delta_sh            # shares of underlying held to be delta-neutral
    cum_hedge = 0.0
    prev_spot = spot0
    last_close_cost = call["ask"] + put["ask"] if (call.get("ask") and put.get("ask")) else None
    ei = sdates.index(ed) if ed in sdates else None
    if ei is None:
        return None
    exit_date = ed
    days_held = 0
    for j in range(ei + 1, len(sdates)):
        d = sdates[j]
        if d > _do.COVERAGE_MAX:
            break
        days_held = j - ei
        S = spots.get(d)
        if S is None:
            continue
        # Skip split-day discontinuities: close_history is RAW (split-safe for
        # strikes), so a split shows as a huge fake jump that is NOT a market move
        # the hedge would P&L on (the position splits too). Guard against it.
        ratio = (S / prev_spot) if prev_spot else 1.0
        if abs(ratio - 1.0) <= _SPLIT_GUARD:
            cum_hedge += hedge_sh * (S - prev_spot)   # P&L on held hedge over the real move
        prev_spot = S
        # rehedge + mark using the chain on days that quote the legs
        _, ch = _do.get_chain_near(symbol, d, **kw)
        cc, cp = _leg(ch, "call", K, exp), _leg(ch, "put", K, exp)
        if cc and cp and cc.get("delta") is not None and cp.get("delta") is not None:
            pos_delta_sh = -(cc["delta"] + cp["delta"]) * m
            hedge_sh = -pos_delta_sh
            if cc.get("ask") is not None and cp.get("ask") is not None:
                last_close_cost = cc["ask"] + cp["ask"]
        if days_held >= target_days or _dte(d, exp) <= 1:
            exit_date = d
            break
        exit_date = d
    if last_close_cost is None:
        return None
    comm = 4.0 * commission_per_contract / m        # 2 legs x open+close, per-share terms
    straddle_pnl = (credit - last_close_cost - comm) * m
    total = straddle_pnl + cum_hedge
    notional = credit * m
    return {"symbol": symbol, "entry_date": ed, "exit_date": exit_date, "expiration": exp,
            "strike": K, "credit": round(credit, 4), "days_held": days_held,
            "straddle_pnl": round(straddle_pnl, 2), "hedge_pnl": round(cum_hedge, 2),
            "net_pnl": round(total, 2), "ret": total / notional if notional else None}


def run_vol_backtest(symbols, dates, db_path=None, config_path="config.json") -> Dict[str, Any]:
    from src import dolt_options as _do
    from src.dolt_stocks import close_history
    try:
        commission = float(json.load(open(config_path)).get("paper_trading", {})
                           .get("commission_per_contract", 0.65))
    except Exception:
        commission = 0.65
    spot_db = db_path or "data/dolt_options.db"
    import sys
    trades: List[Dict[str, Any]] = []
    for symbol in symbols:
        symbol = symbol.upper()
        if not _do.symbol_has_data(symbol, spot_db):
            print(f"  [warning] no chain data for {symbol}; skipped", file=sys.stderr)
            continue
        spots = close_history(symbol, db_path=spot_db)   # RAW (split-safe)
        sdates = sorted(spots)
        for entry_date in dates:
            entry_date = _do._clamp_date(entry_date)
            if entry_date not in spots:
                continue
            try:
                t = simulate_delta_hedged_straddle(symbol, entry_date, spots, sdates,
                                                   db_path=db_path, commission_per_contract=commission)
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
    out.update(_stats([t["ret"] for t in trades]))
    out["mean_hedge_pnl"] = round(sum(t["hedge_pnl"] for t in trades) / len(trades), 2)
    out["mean_straddle_pnl"] = round(sum(t["straddle_pnl"] for t in trades) / len(trades), 2)
    return out


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Delta-hedged short-straddle (pure vol) backtest")
    ap.add_argument("--symbols", default="AAPL,SPY,NVDA,AMZN,MSFT")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2024-12-31")
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
    dates = _do._date_range(args.start, args.end, weekly=True)
    print(f"Delta-hedged short straddle (pure vol), {syms}, {args.start}..{args.end}:")
    out = run_vol_backtest(syms, dates, db_path=db)
    print(json.dumps({k: v for k, v in out.items() if k != "trades"}, indent=1))


if __name__ == "__main__":
    _cli()
