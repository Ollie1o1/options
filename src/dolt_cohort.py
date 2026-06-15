"""Backtest the ACTUAL gate strategy — the Long-Call cohort — on real DoltHub
marks. This is the strategy real money will trade once the n>=50 gate fires, so
it deserves a real-marks test before it's funded, not a BS proxy.

Rules (from config.json exit_rules.long_option): buy a ~30-delta call at the
cohort DTE floor, hold with TP +100% / SL -50% (min 3 days held) and a time exit
when DTE <= 21. Enter at real ask, exit at real bid (spread cost is real).

CLI:
    python -m src.dolt_cohort --symbols AAPL,SPY --start 2022-01-01 --end 2024-12-31 --weekly
"""
from __future__ import annotations

import datetime as _dt
import json
from statistics import mean, median
from typing import Any, Dict, List, Optional


# Single source of truth: the SAME exit evaluator the paper ledger + live
# execution use (TP, deep-ITM delta TP, time exit, SL — researched, config-driven).
from src.paper_manager import _evaluate_long_single_leg_exit, _normalize_exit_rules

_RFR = 0.045


def exit_rules(config_path: str = "config.json") -> Dict[str, Any]:
    """Normalized exit rules — delegates to the canonical paper_manager normalizer
    so the backtest uses the EXACT researched long-option rules (incl. tp_delta)."""
    try:
        cfg = json.load(open(config_path))
    except Exception:
        cfg = {}
    return _normalize_exit_rules(cfg)


def _dte(asof: str, expiration: str) -> int:
    return (_dt.date.fromisoformat(expiration) - _dt.date.fromisoformat(asof)).days


def _reason_bucket(reason: str) -> str:
    r = (reason or "").lower()
    if "delta" in r:
        return "tp_delta"
    if "take profit" in r:
        return "take_profit"
    if "time exit" in r:
        return "time_exit"
    if "stop loss" in r:
        return "stop_loss"
    return "other"


def simulate_trade(symbol, entry_date, spot, sdates, rules,
                   spots=None, target_dte=35, db_path=None) -> Optional[Dict[str, Any]]:
    """Simulate one cohort long-call trade on real marks, exiting via the canonical
    _evaluate_long_single_leg_exit (TP / deep-ITM delta TP / time exit / SL).
    Returns a result dict (ret, exit_reason, days_held) or None if unfillable."""
    from src import dolt_options as _do
    kw = {"db_path": db_path} if db_path else {}
    spots = spots or {}
    ed_actual, chain = _do.get_chain_near(symbol, entry_date, **kw)
    if not chain or ed_actual not in sdates:
        return None
    # Entry contract: ~30-delta call, DTE above the time exit so it survives the hold.
    floor_dte = rules["time_exit_dte"] + 7
    c = _do.nearest_contract(chain, "call", spot * 1.03, ed_actual,
                             target_dte=max(target_dte, floor_dte + 5), min_dte=floor_dte)
    if not c or not c.get("ask") or c["ask"] <= 0:
        return None
    if abs(c["strike"] / spot - 1.0) > 0.4:   # split-mismatch guard
        return None
    entry_cost = c["ask"]
    strike, expiration, entry_iv = c["strike"], c["expiration"], c.get("iv")
    ei = sdates.index(ed_actual)
    last_ret = None
    for j in range(ei + 1, len(sdates)):
        d = sdates[j]
        if d > _do.COVERAGE_MAX:
            break
        days_held = j - ei
        rem_dte = _dte(d, expiration)
        if rem_dte <= 0:
            break
        _, ch = _do.get_chain_near(symbol, d, **kw)
        xc = next((x for x in ch if x["strike"] == strike
                   and x["expiration"] == expiration and x["type"] == "call"), None)
        if not xc or xc.get("bid") is None:
            continue
        current_price = xc["bid"]        # long exits at the bid
        last_ret = (current_price - entry_cost) / entry_cost
        should_close, reason, pnl_raw = _evaluate_long_single_leg_exit(
            rules=rules, option_type="call", strike=strike, spot=spots.get(d),
            entry_price=entry_cost, current_price=current_price, entry_iv=entry_iv,
            dte=rem_dte, days_held=days_held, rfr=_RFR)
        if should_close:
            return {"ret": pnl_raw, "exit_reason": _reason_bucket(reason),
                    "reason_detail": reason, "days_held": days_held,
                    "entry_date": ed_actual, "exit_date": d}
    if last_ret is None:
        return None
    return {"ret": last_ret, "exit_reason": "expiry", "days_held": len(sdates) - 1 - ei,
            "entry_date": ed_actual, "exit_date": sdates[-1]}


def run_cohort_backtest(symbols, dates, target_dte=35, db_path=None,
                        config_path="config.json") -> Dict[str, Any]:
    """Backtest the long-call cohort over (symbols x dates) on real marks."""
    from src import dolt_options as _do
    from src.dolt_validate import _spot_history
    rules = exit_rules(config_path)
    trades: List[Dict[str, Any]] = []
    for symbol in symbols:
        symbol = symbol.upper()
        spots = _spot_history(symbol)
        sdates = sorted(spots)
        for entry_date in dates:
            entry_date = _do._clamp_date(entry_date)
            spot = spots.get(entry_date)
            if spot is None:
                continue
            try:
                t = simulate_trade(symbol, entry_date, spot, sdates, rules,
                                   spots=spots, target_dte=target_dte, db_path=db_path)
            except _do.DoltRateLimited:
                return _summarize(trades, rules, partial=True)
            except _do.DoltQueryError:
                continue
            if t:
                t["symbol"] = symbol
                try:
                    from src import dolt_earnings as _de
                    t["through_earnings"] = _de.holds_through_earnings(
                        symbol, t["entry_date"], t["exit_date"], db_path=db_path or _de.DEFAULT_CACHE)
                except Exception:
                    t["through_earnings"] = None
                trades.append(t)
    return _summarize(trades, rules, partial=False)


def _stats(rets) -> Dict[str, Any]:
    n = len(rets)
    if n == 0:
        return {"n": 0}
    wins = [r for r in rets if r > 0]
    gross_win = sum(wins)
    gross_loss = -sum(r for r in rets if r <= 0)
    return {
        "n": n,
        "win_rate": round(len(wins) / n, 3),
        "avg_return": round(mean(rets), 3),
        "median_return": round(median(rets), 3),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
    }


def _summarize(trades, rules, partial: bool) -> Dict[str, Any]:
    rets = [t["ret"] for t in trades]
    out: Dict[str, Any] = {"n": len(rets), "rules": rules, "partial": partial}
    if not rets:
        return out
    out.update(_stats(rets))
    out["exit_mix"] = {k: sum(1 for t in trades if t["exit_reason"] == k)
                       for k in ("take_profit", "tp_delta", "stop_loss", "time_exit", "expiry")}
    # The headline earnings comparison: clean holds vs holds through earnings.
    clean = [t["ret"] for t in trades if t.get("through_earnings") is False]
    thru = [t["ret"] for t in trades if t.get("through_earnings") is True]
    out["clean_holds"] = _stats(clean)
    out["through_earnings"] = _stats(thru)
    return out


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Backtest the long-call cohort on real DoltHub marks")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--weekly", action="store_true")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    from src import dolt_options as _do
    cfg = {}
    try:
        cfg = (json.load(open("config.json")).get("dolt_options") or {})
    except Exception:
        pass
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or cfg.get("basket", ["AAPL", "SPY"])
    dates = _do._date_range(args.start or cfg.get("validate_start"),
                            args.end or cfg.get("validate_end"),
                            weekly=args.weekly or cfg.get("validate_sampling") == "weekly")
    print(f"Cohort backtest: {len(syms)} symbols x {len(dates)} dates (real marks, long calls)...")
    out = run_cohort_backtest(syms, dates, db_path=args.db or cfg.get("cache_path"))
    print(json.dumps(out, indent=1))
    print("\nThis is the strategy real money trades once the gate fires — real bid/ask fills.")


if __name__ == "__main__":
    _cli()
