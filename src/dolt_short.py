"""Short-premium backtest on real DoltHub marks — the sell side.

Every prior result (leverage, long-call breakeven, BS-vs-real) says BUYING premium
fights cost + theta + the variance risk premium. This tests the other side: SELL an
OTM option, collect the spread, and exit via the CANONICAL short-exit evaluator
(paper_manager._evaluate_short_single_leg_exit — DTE-tiered TP, time exit,
strike-breach / premium-multiple / delta stops).

Entry = real BID (sell-to-open), mark/exit = real ASK (buy-to-close). All-in (real
spread + commission). Uses raw prices so split names work.

CLI:
    python -m src.dolt_short --symbols AAPL,SPY,QQQ,MSFT --start 2022-01-01 --end 2024-12-31 --weekly
    python -m src.dolt_short --type call ...
"""
from __future__ import annotations

import datetime as _dt
import json
from statistics import mean, median
from typing import Any, Dict, List, Optional

from src.dolt_cohort import _stats
from src.paper_manager import _evaluate_short_single_leg_exit, _normalize_exit_rules
from src.execution_costs import FALLBACK_COMMISSION_PER_CONTRACT

# Mirrors the live ledger's own ceiling; see `_friction_limit`.
_DEFAULT_FRICTION_LIMIT = 0.25

_RFR = 0.045


def _dte(asof: str, expiration: str) -> int:
    return (_dt.date.fromisoformat(expiration) - _dt.date.fromisoformat(asof)).days


def _pick_short(chain, opt_type, target_delta, asof, min_dte):
    """Pick the contract of opt_type whose |delta| is closest to target_delta,
    among expiries with DTE >= min_dte."""
    opt_type = opt_type.lower()
    best, best_cost = None, float("inf")
    for c in chain:
        if c.get("type") != opt_type or c.get("delta") is None or c.get("strike") is None:
            continue
        if _dte(asof, c["expiration"]) < min_dte:
            continue
        cost = abs(abs(c["delta"]) - target_delta)
        if cost < best_cost:
            best, best_cost = c, cost
    return best


def _friction_limit(cfg: Dict[str, Any]) -> Optional[float]:
    """Round-trip friction ceiling, as a fraction of the credit received.

    Deliberately the SAME key the live ledger refuses on —
    ``auto_log.max_friction_to_credit`` — so this backtest cannot count a
    position the running system would decline to log. `config.json` records
    that key being tightened 0.50 -> 0.25 on 2026-08-06 as "the largest single
    effect measured anywhere in this system"; measuring against a looser bar
    here would make every short-premium number optimistic by construction.
    """
    try:
        v = (cfg.get("auto_log") or {}).get("max_friction_to_credit",
                                            _DEFAULT_FRICTION_LIMIT)
    except AttributeError:
        return _DEFAULT_FRICTION_LIMIT
    return None if v is None else float(v)


def _entry_is_tradeable(bid, ask, limit: Optional[float]) -> bool:
    """Is the quote a market you could actually sell into and buy back out of?

    A short leg is opened at the bid and closed at the ask, so the whole
    bid-ask is the round trip and the credit is the bid. Measured on the Dolt
    cache, entries this admitted included a $0.05 bid against a $4.60 ask — a
    spread 91x the credit, which closes at -90x and is not a trade. Five such
    rows moved XLB's mean short-put return from a trimmed -1.97 to -5.75.

    `limit=None` disables the gate, matching the live `null` semantics, so a
    deliberate wide-quote study is still possible.
    """
    if limit is None:
        return True
    try:
        b, a = float(bid), float(ask)
    except (TypeError, ValueError):
        return False
    if b <= 0 or a < b:
        return False
    return (a - b) <= limit * b


def simulate_short_trade(symbol, entry_date, spot, sdates, spots, rules,
                         opt_type="put", target_delta=0.25, target_dte=35,
                         db_path=None, commission_per_contract=FALLBACK_COMMISSION_PER_CONTRACT,
                         contract_multiplier=100, entry_filter=None,
                         friction_limit: Optional[float] = _DEFAULT_FRICTION_LIMIT
                         ) -> Optional[Dict[str, Any]]:
    """Sell one ~target_delta OTM option, manage with the canonical short exits.

    ``friction_limit`` is the round-trip bid-ask ceiling as a fraction of the
    credit; see `_entry_is_tradeable`. Pass ``None`` to disable it.
    """
    from src import dolt_options as _do
    kw = {"db_path": db_path} if db_path else {}
    ed_actual, chain = _do.get_chain_near(symbol, entry_date, **kw)
    if not chain or ed_actual not in sdates:
        return None
    floor_dte = rules["time_exit_dte"] + 7
    c = _pick_short(chain, opt_type, target_delta, ed_actual, floor_dte)
    if not c or c.get("bid") is None or c["bid"] <= 0:
        return None
    if abs(c["strike"] / spot - 1.0) > 0.5:   # split-mismatch guard (puts are further OTM)
        return None
    # Tradeability: refuse a quote that is not a market. Sold at the bid and
    # bought back at the ask, so the whole spread is the round trip.
    if not _entry_is_tradeable(c.get("bid"), c.get("ask"), friction_limit):
        return None
    if entry_filter is not None:
        ctx = {"symbol": symbol, "date": ed_actual, "spot": spot, "entry_iv": c.get("iv"),
               "delta": c.get("delta"), "dte": _dte(ed_actual, c["expiration"]),
               "sdates": sdates, "spots": spots}
        if not entry_filter(ctx):
            return None
    entry_price = c["bid"]                        # sell-to-open at bid
    entry_delta, entry_iv = c.get("delta"), c.get("iv")
    strike, expiration = c["strike"], c["expiration"]
    comm_frac = (2.0 * commission_per_contract / contract_multiplier) / entry_price
    ei = sdates.index(ed_actual)
    last_pnl = None
    for j in range(ei + 1, len(sdates)):
        d = sdates[j]
        if d > _do.COVERAGE_MAX:
            break
        rem_dte = _dte(d, expiration)
        if rem_dte <= 0:
            break
        days_held = j - ei
        _, ch = _do.get_chain_near(symbol, d, **kw)
        xc = next((x for x in ch if x["strike"] == strike
                   and x["expiration"] == expiration and x["type"] == opt_type), None)
        if not xc or xc.get("ask") is None:
            continue
        current_price = xc["ask"]                 # buy-to-close at ask
        should_close, reason, pnl_raw = _evaluate_short_single_leg_exit(
            rules=rules, option_type=opt_type, strike=strike, spot=spots.get(d),
            entry_price=entry_price, current_price=current_price, entry_delta=entry_delta,
            entry_iv=entry_iv, dte=rem_dte, days_held=days_held, rfr=_RFR)
        last_pnl = pnl_raw
        if should_close:
            return {"ret": pnl_raw - comm_frac, "gross_ret": pnl_raw,
                    "exit_reason": _bucket(reason), "reason_detail": reason,
                    "days_held": days_held, "entry_date": ed_actual, "exit_date": d}
    if last_pnl is None:
        return None
    return {"ret": last_pnl - comm_frac, "gross_ret": last_pnl, "exit_reason": "expiry",
            "days_held": len(sdates) - 1 - ei, "entry_date": ed_actual, "exit_date": sdates[-1]}


def _bucket(reason: str) -> str:
    r = (reason or "").lower()
    if "take profit" in r:
        return "take_profit"
    if "time exit" in r:
        return "time_exit"
    if "stop loss" in r:
        return "stop_loss"
    return "other"


def run_short_backtest(symbols, dates, opt_type="put", target_delta=0.25,
                       db_path=None, config_path="config.json", entry_filter=None,
                       friction_limit: Any = "config") -> Dict[str, Any]:
    """Short-premium backtest over (symbols x dates) on real marks.

    ``friction_limit`` defaults to the live ledger's own
    ``auto_log.max_friction_to_credit``, so this cannot count a position the
    running system would refuse to log. Pass ``None`` to disable the gate.
    """
    from src import dolt_options as _do
    from src.dolt_stocks import close_history
    rules = _normalize_exit_rules(_load_cfg(config_path))
    if friction_limit == "config":
        friction_limit = _friction_limit(_load_cfg(config_path))
    try:
        commission = float(_load_cfg(config_path).get("paper_trading", {})
                           .get("commission_per_contract", FALLBACK_COMMISSION_PER_CONTRACT))
    except Exception:
        commission = FALLBACK_COMMISSION_PER_CONTRACT
    spot_db = db_path or "data/dolt_options.db"
    import sys
    _missing = [s.upper() for s in symbols if not _do.symbol_has_data(s, spot_db)]
    if _missing:
        print(f"  [warning] no chain data for {_missing} — absent from DoltHub or unfetched; skipped",
              file=sys.stderr)
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
                t = simulate_short_trade(symbol, entry_date, spot, sdates, spots, rules,
                                         opt_type=opt_type, target_delta=target_delta,
                                         db_path=db_path, commission_per_contract=commission,
                                         entry_filter=entry_filter,
                                         friction_limit=friction_limit)
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
    out.update(_stats([t["ret"] for t in trades]))
    out["exit_mix"] = {k: sum(1 for t in trades if t["exit_reason"] == k)
                       for k in ("take_profit", "time_exit", "stop_loss", "expiry", "other")}
    return out


def _load_cfg(path):
    try:
        return json.load(open(path))
    except Exception:
        return {}


def _build_parser():
    """Split out from `_cli` so the flags can be asserted without a backtest.

    `--weekly` used to reach `_date_range` as `args.weekly or True`, which is
    `True` for every value the flag can hold — daily sampling was unreachable
    and the flag documented a choice the code did not offer. It is a real
    negatable flag now, and it stays defaulted to weekly: every Dolt result on
    record was sampled weekly, so an unchanged command has to keep meaning
    what it meant.
    """
    import argparse
    ap = argparse.ArgumentParser(description="Short-premium backtest on real marks")
    ap.add_argument("--symbols", default="AAPL,SPY,QQQ,MSFT")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--weekly", action=argparse.BooleanOptionalAction, default=True,
                    help="sample Fridays only (default); --no-weekly for every weekday")
    ap.add_argument("--type", choices=["put", "call"], default="put")
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--db", default=None)
    return ap


def _cli():
    args = _build_parser().parse_args()
    from src import dolt_options as _do
    cfg = _load_cfg("config.json").get("dolt_options", {})
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    dates = _do._date_range(args.start, args.end, weekly=args.weekly)
    db = args.db or cfg.get("cache_path")
    print(f"Short {args.type} ~{args.delta}delta: {syms} {args.start}..{args.end} (real marks, all-in)")
    out = run_short_backtest(syms, dates, opt_type=args.type, target_delta=args.delta, db_path=db)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    _cli()
