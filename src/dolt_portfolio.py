"""Portfolio sizing + equity curve over a real-marks backtest (P1.3).

Every Dolt backtest returns PER-TRADE returns on max-risk (or premium). That is
NOT a portfolio return — it ignores how much capital each trade risks and how
trades compound over time. This module turns a trade list into a deployable
view: fixed-fractional sizing (risk f% of equity per trade) → a dated equity
curve → the numbers that actually matter for going live (total return, CAGR,
max drawdown, and the concurrency the sizing assumes).

Sizing model (documented, sequential fixed-fractional): trades are compounded in
EXIT-date order; each risks ``risk_frac`` of current equity, so a trade with
per-trade return ``ret`` on max-risk moves equity by ``ret * risk_frac``. This
ignores intra-trade concurrency (a trade is sized off equity at its close, the
standard sequential approximation); ``max_concurrent`` is reported so the real
simultaneous exposure is visible. Honest, not a substitute for a margin engine
(that's P1.5).

CLI:
    python -m src.dolt_portfolio --strategy put_spread --symbols SPY \
        --start 2022-01-01 --end 2024-12-31 --risk-frac 0.02
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional


def max_concurrent(trades: List[Dict[str, Any]]) -> int:
    """Max number of positions open at the same time, from entry/exit dates."""
    events = []
    for t in trades:
        e, x = t.get("entry_date"), t.get("exit_date")
        if not e or not x:
            continue
        events.append((e, 1))
        events.append((x, -1))   # close frees the slot at exit day
    # sort by date; process closes (-1) before opens (+1) on the same day
    events.sort(key=lambda ev: (ev[0], ev[1]))
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def _max_drawdown(equity: List[float]) -> float:
    """Largest peak-to-trough fractional drop of an equity series (<= 0)."""
    peak = equity[0] if equity else 0.0
    worst = 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            worst = min(worst, e / peak - 1.0)
    return worst


def _cagr(start: float, end: float, first_date: str, last_date: str) -> Optional[float]:
    if start <= 0 or end <= 0 or not first_date or not last_date:
        return None
    days = (_dt.date.fromisoformat(last_date) - _dt.date.fromisoformat(first_date)).days
    if days <= 0:
        return None
    return (end / start) ** (365.0 / days) - 1.0


def equity_curve(trades: List[Dict[str, Any]], start_equity: float = 100_000.0,
                 risk_frac: float = 0.01) -> Dict[str, Any]:
    """Compound per-trade max-risk returns into a portfolio equity curve.

    Each trade risks ``risk_frac`` of current equity; equity *= (1 + ret*risk_frac),
    applied in exit-date order. Returns curve + summary stats."""
    usable = [t for t in trades if t.get("ret") is not None and t.get("exit_date")]
    usable.sort(key=lambda t: t["exit_date"])
    equity = start_equity
    curve: List[Dict[str, Any]] = []
    series = [start_equity]
    for t in usable:
        equity *= (1.0 + t["ret"] * risk_frac)
        series.append(equity)
        curve.append({"date": t["exit_date"], "equity": equity})
    first_date = usable[0]["entry_date"] if usable and usable[0].get("entry_date") else (
        usable[0]["exit_date"] if usable else None)
    last_date = usable[-1]["exit_date"] if usable else None
    out: Dict[str, Any] = {
        "n": len(usable),
        "start_equity": start_equity,
        "end_equity": equity,
        "total_return": equity / start_equity - 1.0 if start_equity else None,
        "max_drawdown": _max_drawdown(series),
        "cagr": _cagr(start_equity, equity, first_date, last_date),
        "max_concurrent": max_concurrent(usable),
        "risk_frac": risk_frac,
        "curve": curve,
    }
    return out


def margin_profile(trades: List[Dict[str, Any]], contract_multiplier: int = 100
                   ) -> Dict[str, Any]:
    """Capital actually tied up (P1.5). For a DEFINED-RISK put spread, margin per
    contract = max_risk * multiplier (the broker holds the max loss). Walks the
    calendar to find PEAK simultaneous margin across overlapping trades — the real
    capital the strategy demands — plus return-on-peak-capital and an
    early-assignment-risk proxy (stop-loss exits = short leg breached ITM).

    Naked short puts are NOT defined risk; their margin is ~broker reg-T (≈20% of
    notional) and assignment leaves you long shares — handled separately, flagged."""
    usable = [t for t in trades if t.get("max_risk") and t.get("entry_date") and t.get("exit_date")]
    # day-by-day open margin via a sweep of (+open, -close) events
    events = []
    for t in usable:
        m = t["max_risk"] * contract_multiplier
        events.append((t["entry_date"], 1, m))
        events.append((t["exit_date"], -1, m))
    events.sort(key=lambda ev: (ev[0], ev[1]))   # closes before opens same day
    cur = peak = 0.0
    for _, sign, m in events:
        cur += sign * m
        peak = max(peak, cur)
    total_pnl = sum(t["ret"] * t["max_risk"] * contract_multiplier
                    for t in usable if t.get("ret") is not None)
    assignment_risk = sum(1 for t in usable if t.get("exit_reason") == "stop_loss")
    return {
        "n": len(usable),
        "peak_margin": peak,
        "total_pnl": total_pnl,
        "return_on_peak_margin": (total_pnl / peak) if peak > 0 else None,
        "assignment_risk_trades": assignment_risk,
        "assignment_risk_frac": (assignment_risk / len(usable)) if usable else None,
        "contract_multiplier": contract_multiplier,
    }


def _run_strategy(strategy, symbols, start, end, db_path=None, entry_filter=None):
    """Run a strategy backtest and return its trade list (P1.3 needs raw trades)."""
    from src import dolt_options as _do
    from src import dolt_research as dr
    dates = _do._date_range(start, end, weekly=True)
    res = dr.STRATEGIES[strategy](symbols, dates, db_path=db_path, entry_filter=entry_filter)
    return res.get("trades", []), res


def _cli():
    import argparse
    import json
    from src import dolt_research as dr
    ap = argparse.ArgumentParser(description="Portfolio sizing + equity curve over a Dolt backtest")
    ap.add_argument("--strategy", choices=list(dr.STRATEGIES), default="put_spread")
    ap.add_argument("--symbols", default="SPY")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--start-equity", type=float, default=100_000.0)
    ap.add_argument("--risk-frac", type=float, default=0.01)
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    cfg = {}
    try:
        cfg = json.load(open("config.json")).get("dolt_options", {})
    except Exception:
        pass
    db = args.db or cfg.get("cache_path")
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    trades, res = _run_strategy(args.strategy, syms, args.start, args.end, db_path=db)
    if res.get("trades") is None:
        print(f"[note] strategy '{args.strategy}' does not expose a trade list yet; "
              "only put_spread does. Add trades to its _summarize to enable.")
    pf = equity_curve(trades, start_equity=args.start_equity, risk_frac=args.risk_frac)
    print(f"Portfolio [{args.strategy}] {syms} {args.start}..{args.end}  "
          f"risk={args.risk_frac:.1%}/trade, start=${args.start_equity:,.0f}")
    if not pf["n"]:
        print("  no trades")
        return
    def pct(x):
        return f"{x:+.1%}" if x is not None else "  -  "
    print(f"  trades={pf['n']}  end=${pf['end_equity']:,.0f}  total={pct(pf['total_return'])}  "
          f"CAGR={pct(pf['cagr'])}  maxDD={pct(pf['max_drawdown'])}  "
          f"maxConcurrent={pf['max_concurrent']}")
    mp = margin_profile(trades)
    if mp["n"]:
        print(f"  [margin] peakCapital=${mp['peak_margin']:,.0f}  "
              f"return-on-peak-capital={pct(mp['return_on_peak_margin'])}  "
              f"assignment-risk={mp['assignment_risk_trades']}/{mp['n']} "
              f"({pct(mp['assignment_risk_frac'])}) stop-loss exits")


if __name__ == "__main__":
    _cli()
