"""Defined-risk PUT CREDIT SPREAD backtest on real DoltHub marks.

Sell a ~short_delta put, buy a further-OTM put as a wing → capped loss, lower
margin, deployable. Tests whether the short-put edge SURVIVES paying for the
protective wing. Exits via the canonical paper_manager._evaluate_multileg_exit
(spread TP/SL as fractions of credit). Returns are measured on MAX RISK
(width − credit) — the honest denominator for a defined-risk trade.

Entry credit = short_bid − long_ask; close cost = short_ask − long_bid. All-in
(real spread + 4-leg commission). Raw prices so split names work.

CLI:
    python -m src.dolt_spread --symbols SPY,NVDA,AMD --start 2022-01-01 --end 2024-12-31 --weekly
"""
from __future__ import annotations

import datetime as _dt
import json
from typing import Any, Dict, List, Optional, Tuple

from src.dolt_cohort import _stats
from src.execution_costs import FALLBACK_COMMISSION_PER_CONTRACT
from src.paper_manager import (_evaluate_multileg_exit,
                               _legs_intrinsic_close_value,
                               _normalize_exit_rules)


def _dte(asof: str, expiration: str) -> int:
    return (_dt.date.fromisoformat(expiration) - _dt.date.fromisoformat(asof)).days


def _spread_width(short_strike: float, long_strike: float, side: str) -> float:
    """Distance between the strikes, positive for a valid credit spread.

    The wing sits below the short strike on a bull put and above it on a bear
    call. Getting that backwards yields a debit spread wearing a credit
    spread's name, so the direction is expressed once, here.
    """
    return (long_strike - short_strike) if side == "call" else (short_strike - long_strike)


def _marks_seen_label(marks_seen: int) -> str:
    """Was this position ever actually manageable?

    An exit rule can only fire on a day the chain quotes both legs. With one
    mark or none, the position rode to expiry unobserved and any loss it took is
    a statement about the data, not about the strategy — the stop was never
    offered a chance to fire.
    """
    return "unobserved" if marks_seen <= 1 else "managed"


def _expiry_close_cost(short_strike: float, long_strike: float, side: str,
                       spot: float) -> float:
    """Debit to flatten a credit spread at expiry, from spot alone.

    Delegates to the ledger's own settlement so a backtested expiry and a
    logged one cannot drift apart. Bounded by construction: 0 when both legs
    finish out of the money, the width when both finish in.
    """
    opt = "call" if side == "call" else "put"
    legs = [(short_strike, opt, -1), (long_strike, opt, 1)]
    return _legs_intrinsic_close_value(legs, spot)


def _pick_credit_spread(chain, asof, short_delta, long_delta, min_dte,
                        side: str = "put") -> Optional[Tuple[dict, dict]]:
    """Pick (short ~short_delta, long ~long_delta) in the SAME expiry.

    ``side="put"`` is a bull put (long wing below the short strike);
    ``side="call"`` is a bear call (long wing above). Deltas are compared on
    absolute value so the same target works for both.
    """
    legs = [c for c in chain if c.get("type") == side and c.get("delta") is not None
            and c.get("strike") is not None and _dte(asof, c["expiration"]) >= min_dte]
    if not legs:
        return None
    short = min(legs, key=lambda c: abs(abs(c["delta"]) - short_delta))
    exp = short["expiration"]
    if side == "call":
        wings = [c for c in legs if c["expiration"] == exp and c["strike"] > short["strike"]]
    else:
        wings = [c for c in legs if c["expiration"] == exp and c["strike"] < short["strike"]]
    if not wings:
        return None
    long = min(wings, key=lambda c: abs(abs(c["delta"]) - long_delta))
    return short, long


def _pick_put_spread(chain, asof, short_delta, long_delta, min_dte
                     ) -> Optional[Tuple[dict, dict]]:
    """Bull put selection. Retained entry point; delegates to _pick_credit_spread."""
    return _pick_credit_spread(chain, asof, short_delta, long_delta, min_dte, side="put")


def _leg(chain, strike, expiration, side: str = "put"):
    return next((c for c in chain if c["type"] == side and c["strike"] == strike
                 and c["expiration"] == expiration), None)


def simulate_spread(symbol, entry_date, spot, sdates, spots, rules,
                    short_delta=0.25, long_delta=0.10, target_dte=35, db_path=None,
                    commission_per_contract=FALLBACK_COMMISSION_PER_CONTRACT, contract_multiplier=100,
                    entry_filter=None, side: str = "put") -> Optional[Dict[str, Any]]:
    """Sell one credit spread, manage via canonical spread exits.

    ``side="put"`` is a bull put, ``side="call"`` a bear call. Return dict ret
    is on MAX RISK. Legs are priced by crossing the real quote — sell the short
    at the bid, buy the wing at the ask — so no slippage assumption enters at
    all, which is the point of running this against real marks.
    """
    from src import dolt_options as _do
    kw = {"db_path": db_path} if db_path else {}
    ed_actual, chain = _do.get_chain_near(symbol, entry_date, **kw)
    if not chain or ed_actual not in sdates:
        return None
    floor_dte = rules["time_exit_dte"] + 7
    pick = _pick_credit_spread(chain, ed_actual, short_delta, long_delta, floor_dte, side)
    if not pick:
        return None
    short, long = pick
    if abs(short["strike"] / spot - 1.0) > 0.5:
        return None
    if short.get("bid") is None or long.get("ask") is None:
        return None
    entry_credit = short["bid"] - long["ask"]
    width = _spread_width(short["strike"], long["strike"], side)
    max_risk = width - entry_credit
    if entry_credit <= 0 or max_risk <= 0:
        return None
    if entry_filter is not None:
        ctx = {"symbol": symbol, "date": ed_actual, "spot": spot,
               "entry_iv": short.get("iv"), "delta": short.get("delta"),
               "dte": _dte(ed_actual, short["expiration"]), "sdates": sdates, "spots": spots}
        if not entry_filter(ctx):
            return None
    comm = 4.0 * commission_per_contract / contract_multiplier   # 2 legs x open+close
    strike_s, strike_l, exp = short["strike"], long["strike"], short["expiration"]
    ei = sdates.index(ed_actual)
    last_credit = None
    last_j = ei
    marks_seen = 0
    for j in range(ei + 1, len(sdates)):
        d = sdates[j]
        if d > _do.COVERAGE_MAX:
            break
        rem_dte = _dte(d, exp)
        if rem_dte <= 0:
            break
        last_j = j
        days_held = j - ei
        _, ch = _do.get_chain_near(symbol, d, **kw)
        cs, cl = _leg(ch, strike_s, exp, side), _leg(ch, strike_l, exp, side)
        if not cs or not cl or cs.get("ask") is None or cl.get("bid") is None:
            continue
        credit_to_close = cs["ask"] - cl["bid"]
        last_credit = credit_to_close
        marks_seen += 1
        should_close, reason, _ = _evaluate_multileg_exit(
            rules, entry_credit, credit_to_close, rem_dte, days_held)
        if should_close:
            net_pnl = (entry_credit - credit_to_close) - comm
            return {"ret": net_pnl / max_risk, "exit_reason": _bucket(reason),
                    "reason_detail": reason, "days_held": days_held,
                    "entry_date": ed_actual, "exit_date": d,
                    "credit": entry_credit, "max_risk": max_risk,
                    "marks_seen": marks_seen,
                    "observability": _marks_seen_label(marks_seen)}
    # Carried to expiry. Settle at intrinsic on the expiration date rather than
    # at the last quote seen: this chain carries only a few expirations per
    # date, so "last quote" can be weeks early — exactly the window in which a
    # short strike gets breached. Fall back to the last quote only when no spot
    # is available to settle against.
    settle_spot = spots.get(exp)
    if settle_spot is None:
        prior = [d for d in sdates if d <= exp]
        settle_spot = spots.get(prior[-1]) if prior else None
    if settle_spot is not None:
        close_cost = _expiry_close_cost(strike_s, strike_l, side, settle_spot)
        exit_on = min(exp, sdates[last_j]) if last_j > ei else exp
    elif last_credit is not None:
        close_cost = last_credit
        exit_on = sdates[last_j]
    else:
        return None
    net_pnl = (entry_credit - close_cost) - comm
    return {"ret": net_pnl / max_risk, "exit_reason": "expiry",
            "days_held": max(0, (_dt.date.fromisoformat(exit_on)
                                 - _dt.date.fromisoformat(ed_actual)).days),
            "entry_date": ed_actual, "exit_date": exit_on,
            "credit": entry_credit, "max_risk": max_risk,
            "marks_seen": marks_seen,
            "observability": _marks_seen_label(marks_seen)}


def _bucket(reason: str) -> str:
    r = (reason or "").lower()
    if "take profit" in r:
        return "take_profit"
    if "time exit" in r:
        return "time_exit"
    if "stop loss" in r:
        return "stop_loss"
    return "other"


def run_spread_backtest(symbols, dates, short_delta=0.25, long_delta=0.10,
                        db_path=None, config_path="config.json", entry_filter=None,
                        side: str = "put") -> Dict[str, Any]:
    """Backtest credit spreads on real marks. ``side`` selects bull put or bear call."""
    from src import dolt_options as _do
    from src.dolt_stocks import close_history
    rules = _normalize_exit_rules(_load_cfg(config_path))
    try:
        commission = float(_load_cfg(config_path).get("paper_trading", {})
                           .get("commission_per_contract", FALLBACK_COMMISSION_PER_CONTRACT))
    except Exception:
        commission = FALLBACK_COMMISSION_PER_CONTRACT
    spot_db = db_path or "data/dolt_options.db"
    import sys
    _missing = [s.upper() for s in symbols if not _do.symbol_has_data(s, spot_db)]
    if _missing:
        print(f"  [warning] no chain data for {_missing}; skipped", file=sys.stderr)
    trades: List[Dict[str, Any]] = []
    for symbol in symbols:
        symbol = symbol.upper()
        spots = close_history(symbol, db_path=spot_db)
        sdates = sorted(spots)
        for entry_date in dates:
            entry_date = _do._clamp_date(entry_date)
            spot = spots.get(entry_date)
            if spot is None:
                continue
            try:
                t = simulate_spread(symbol, entry_date, spot, sdates, spots, rules,
                                    short_delta=short_delta, long_delta=long_delta,
                                    db_path=db_path, commission_per_contract=commission,
                                    entry_filter=entry_filter, side=side)
            except _do.DoltRateLimited:
                return _summarize(trades, partial=True)
            except _do.DoltQueryError:
                continue
            if t:
                t["symbol"] = symbol
                trades.append(t)
    return _summarize(trades, partial=False)


def _summarize(trades, partial: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(trades), "partial": partial, "trades": trades}
    if not trades:
        return out
    out.update(_stats([t["ret"] for t in trades]))   # ret is on max-risk
    out["exit_mix"] = {k: sum(1 for t in trades if t["exit_reason"] == k)
                       for k in ("take_profit", "time_exit", "stop_loss", "expiry", "other")}
    return out


def _load_cfg(path):
    try:
        return json.load(open(path))
    except Exception:
        return {}


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Put credit spread backtest on real marks")
    ap.add_argument("--symbols", default="SPY")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--weekly", action="store_true")
    ap.add_argument("--short-delta", type=float, default=0.25)
    ap.add_argument("--long-delta", type=float, default=0.10)
    ap.add_argument("--side", choices=("put", "call"), default="put",
                    help="put = bull put spread, call = bear call spread")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    from src import dolt_options as _do
    cfg = _load_cfg("config.json").get("dolt_options", {})
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    dates = _do._date_range(args.start, args.end, weekly=args.weekly or True)
    db = args.db or cfg.get("cache_path")
    label = "Bear call" if args.side == "call" else "Bull put"
    print(f"{label} credit spread {args.short_delta}/{args.long_delta}d: {syms} (real marks, ret on max-risk)")
    out = run_spread_backtest(syms, dates, short_delta=args.short_delta,
                              long_delta=args.long_delta, db_path=db, side=args.side)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    _cli()
