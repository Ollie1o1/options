"""python -m src.leverage {signal,backtest,paper,status}

Safe by default: paper/dry-run unless --live (which is an inert stub until the
validation gate passes). Every command prints a decision-grade ticket or an
honest verdict — never a raw data dump."""
from __future__ import annotations
import argparse
import datetime as _dt
import json
import os
import sys
from typing import Optional
import numpy as np

try:
    from src import formatting as _fmt
    _C = _fmt.Colors
except Exception:  # pragma: no cover
    _fmt = None
    _C = None
from .signals import Params, generate_signals
from .ticket import render
from .risk import liquidation_price, passes_liquidation_safety
from .sizing import effective_leverage_size

_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
_GATE_N = 100
# Last walk-forward result, cached so the menu banner shows a real, dated
# verdict instead of a hardcoded (and silently-staleable) number.
_WF_CACHE = "data/leverage_ohlcv/last_walk_forward.json"


def _strategy(name: str):
    """Resolve a strategy name to (signal_fn, default_params, label)."""
    if name == "reversion":
        from .reversion import ReversionParams, generate_reversion_signals
        return generate_reversion_signals, ReversionParams(), "reversion"
    return generate_signals, Params(), "breakout"


def build_signal_ticket(df5, df15, params, equity: float,
                        signal_fn=generate_signals) -> str:
    sigs = [s for s in signal_fn(df5, df15, params) if s.ts == df5.index[-1]]
    if not sigs:
        return f"{df5.attrs.get('symbol','?')}: no actionable setup on the latest bar."
    sig = sigs[0]
    stop_dist = abs(sig.stop - sig.entry) / sig.entry
    sizing = effective_leverage_size(equity, stop_dist, price=sig.entry)
    if sizing is None:
        return (f"{sig.symbol}: {sig.side} setup, but stop {stop_dist*100:.2f}% "
                f"is too wide to reach the 3-6x band at <=2% risk — skipped.")
    liq = liquidation_price(sig.entry, sig.side, sizing.eff_leverage)
    liq_dist = abs(liq - sig.entry) / sig.entry
    safe = passes_liquidation_safety(stop_dist, liq_dist)
    return render(sig, sizing, liq_price=liq, safe=safe)


def gate_progress(closed_trades: list) -> str:
    n = len(closed_trades)
    line = f"Validation gate: {n}/{_GATE_N} paper trades closed."
    if n >= 2:
        arr = np.array([t["pnl_pct"] for t in closed_trades])
        rng = np.random.default_rng(0)
        boots = [rng.choice(arr, len(arr), replace=True).mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        verdict = "EXCLUDES 0 (edge!)" if lo > 0 else "includes 0 (not proven)"
        line += f" Expectancy 95% CI [{lo:+.4f}, {hi:+.4f}] {verdict}."
    return line


def _record_walk_forward(symbol: str, pos: int, total: int) -> None:
    """Persist the latest walk-forward result for the menu banner. Best-effort:
    a write failure must never break a backtest run."""
    try:
        os.makedirs(os.path.dirname(_WF_CACHE), exist_ok=True)
        with open(_WF_CACHE, "w") as f:
            json.dump({"symbol": symbol, "pos": pos, "total": total,
                       "date": _dt.date.today().isoformat()}, f)
    except OSError:
        pass


def _backtest_verdict_line() -> str:
    """Honest one-liner for the menu header. Reports the real last run if one is
    cached; otherwise says so plainly — never invents a number."""
    try:
        with open(_WF_CACHE) as f:
            d = json.load(f)
        pos, total = int(d["pos"]), int(d["total"])
    except (OSError, ValueError, KeyError, TypeError):
        return ("Walk-forward not run yet on this machine — choose BACKTEST to "
                "validate (gate #1 needs ALL OOS windows net-positive).")
    verdict = "EDGE (all windows positive)" if total > 0 and pos == total else "NO EDGE"
    return (f"Last walk-forward ({d.get('date', '?')}, {d.get('symbol', '?')}): "
            f"{pos}/{total} OOS windows net-positive — {verdict}.")


def _cmd_signal(args):
    from . import data as D
    signal_fn, params, label = _strategy(getattr(args, "strategy", "reversion"))
    eq = args.equity
    print(f"[strategy: {label}]")
    for key in (["BTC", "ETH"] if args.symbol == "ALL" else [args.symbol]):
        df5, df15 = D.load_history(_SYMBOLS[key])
        print(build_signal_ticket(df5, df15, params, eq, signal_fn=signal_fn))
        print()


def _cmd_backtest(args):
    from . import data as D
    from .backtest import run_backtest, walk_forward, robustness_sweep
    from .analysis import render_analysis
    signal_fn, params, label = _strategy(getattr(args, "strategy", "reversion"))
    key = "BTC" if args.symbol == "ALL" else args.symbol
    df5, df15 = D.load_history(_SYMBOLS[key])
    if df5.empty:
        print(f"{key}: no historical data returned — cannot backtest.")
        return
    funding = D.fetch_funding(_SYMBOLS[key],
                              int(df5.index[0].timestamp() * 1000),
                              int(df5.index[-1].timestamp() * 1000))
    print(f"[strategy: {label}]")
    if args.walk_forward:
        wins = walk_forward(df5, df15, params, funding, signal_fn=signal_fn)
        pos = sum(1 for _, _, r in wins if r.expectancy > 0)
        _record_walk_forward(f"{key}/{label}", pos, len(wins))
        print(f"Walk-forward: {pos}/{len(wins)} OOS windows net-positive "
              f"(gate #1 needs ALL).")
        for ts, te, r in wins:
            print(f"  {ts.date()}->{te.date()}: n={r.n} exp={r.expectancy:+.4f} "
                  f"win={r.win_rate:.0%} dd={r.max_dd:.0%}")
    elif args.robustness:
        sweep = robustness_sweep(df5, df15, params, funding, signal_fn=signal_fn)
        pos = sum(1 for _, r in sweep if r.expectancy > 0)
        print(f"Robustness: {pos}/{len(sweep)} param variants net-positive "
              f"(gate #2 needs ALL).")
    else:
        r = run_backtest(df5, df15, params, funding, signal_fn=signal_fn)
        print(render_analysis(r, label=f"{key} {label}"))


def _cmd_optimize(args):
    from . import data as D
    from .optimize import optimize_reversion
    from .analysis import render_analysis
    key = "BTC" if args.symbol == "ALL" else args.symbol
    df5, df15 = D.load_history(_SYMBOLS[key])
    if df5.empty:
        print(f"{key}: no historical data returned — cannot optimize.")
        return
    funding = D.fetch_funding(_SYMBOLS[key],
                              int(df5.index[0].timestamp() * 1000),
                              int(df5.index[-1].timestamp() * 1000))
    best, train_r, test_r = optimize_reversion(df5, df15, funding)
    print(f"{key} reversion optimization (train-select / test-report, no leakage):")
    print(f"  best params: lookback={best.lookback} z_entry={best.z_entry} "
          f"atr_stop_mult={best.atr_stop_mult}")
    print(render_analysis(train_r, label="  TRAIN"))
    print(render_analysis(test_r, label="  TEST (held out)"))


def _cmd_paper(args):
    from . import data as D
    from .paper import PaperLedger
    if args.live:
        print("--live is inert until the validation gate passes. Running paper.")
    ledger = PaperLedger()
    signal_fn, params, _ = _strategy(getattr(args, "strategy", "reversion"))
    key = "BTC" if args.symbol == "ALL" else args.symbol
    df5, df15 = D.load_history(_SYMBOLS[key])
    print(build_signal_ticket(df5, df15, params, args.equity, signal_fn=signal_fn))
    print(f"(paper) open positions: {len(ledger.open_positions())}")


def _cmd_status(args):
    from .paper import PaperLedger
    ledger = PaperLedger()
    opens = ledger.open_positions()
    closed = ledger.closed_positions()
    day_pnl = sum(t.get("pnl_usd") or 0.0 for t in closed)
    print(f"Open positions: {len(opens)}")
    print(f"Realized P&L (paper, all-time): {_money(day_pnl)}")
    print(gate_progress(closed))


# ---------------------------------------------------------------------------
# Interactive menu (launcher mode [3]). Pure navigation + the four CLI actions,
# plus a real paper-logging path. Network actions are wrapped so a fetch failure
# returns to the menu instead of crashing — you can always trust BACK/Q.
# ---------------------------------------------------------------------------
_BAR = "═" * 80


def _c(text: str, color_attr: str, bold: bool = False) -> str:
    if _fmt is None:
        return text
    return _fmt.colorize(text, getattr(_C, color_attr), bold=bold)


def _banner(text: str) -> None:
    print(_c(_BAR, "BRIGHT_CYAN"))
    print(_c(f"  {text}", "BRIGHT_CYAN", bold=True))
    print(_c(_BAR, "BRIGHT_CYAN"))


def _money(value: float) -> str:
    """Green for positive realized P&L, red for negative, dim for flat."""
    s = f"${value:,.2f}"
    if value > 0:
        return _c(s, "BRIGHT_GREEN")
    if value < 0:
        return _c(s, "BRIGHT_RED")
    return _c(s, "DIM")


def _run_guarded(action) -> None:
    """Run a network-touching menu action. The CLI lets data-layer exceptions
    propagate (so a cron job exits non-zero); the interactive menu must not —
    a fetch failure (no connectivity, rate limit, geo-blocked exchange) or empty
    data has to drop you back to the menu, never crash the app."""
    try:
        action()
    except KeyboardInterrupt:
        print("\n  Cancelled.\n")
    except Exception as exc:  # operational failure — keep navigating
        print(f"\n  Couldn't complete that — {type(exc).__name__}: {exc}")
        print("  The exchange may be unreachable, rate-limited, or geo-blocked "
              "(Binance fapi is US-restricted).")
        print("  STATUS and PORTFOLIO work offline.\n")


def _ask(label: str, default: str = "") -> str:
    try:
        raw = input(f"  {label}{' [' + default + ']' if default else ''}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return raw or default


def _ask_symbol(default: str = "BTC") -> Optional[str]:
    raw = _ask("Symbol (BTC / ETH / ALL)", default).upper()
    if raw in ("BTC", "ETH", "ALL"):
        return raw
    print(f"  Unknown symbol {raw!r} — pick BTC, ETH, or ALL.")
    return None


def _coerce_equity(raw: str, default: float) -> tuple:
    """Pure parse+validate for account equity. Returns (value, error_or_None).

    Rejects non-numbers, non-positive equity (sizing is undefined at <=0), and
    implausibly large values (almost always a typo / wrong units)."""
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return default, f"'{raw}' is not a number — using {default:,.0f}."
    if val <= 0:
        return default, f"Equity must be positive — using {default:,.0f}."
    if val > 1e9:
        return default, f"Equity {val:,.0f} is implausibly large — using {default:,.0f}."
    return val, None


def _ask_equity(default: float = 1500.0) -> float:
    raw = _ask("Account equity USD", str(int(default)))
    val, err = _coerce_equity(raw, default)
    if err:
        print(f"  {err}")
    return val


def _print_menu_header() -> None:
    from .paper import PaperLedger
    print(_c(_BAR, "BRIGHT_CYAN"))
    print("  " + _c("LEVERAGE STRATEGIST  —  BTC / ETH perp futures", "BRIGHT_CYAN", bold=True)
          + "   " + _c("[PRE-VALIDATION — NO EDGE YET]", "BRIGHT_RED", bold=True))
    print(_c(_BAR, "BRIGHT_CYAN"))
    print()
    print(f"  {gate_progress(PaperLedger().closed_positions())}")
    print(f"  {_backtest_verdict_line()}")
    print("  " + _c("Real money is OFF: --live is inert until the validation gate passes.",
                    "BRIGHT_YELLOW"))
    print()
    print("  strategy: " + _c("mean-reversion (fade >2σ dislocations)", "BRIGHT_WHITE"))
    print()
    items = [
        ("1", "SIGNAL", "latest-bar trade ticket (BTC / ETH)"),
        ("2", "PAPER LOG", "log the latest signal to the paper ledger"),
        ("3", "PORTFOLIO", "open positions + realized P&L (paper ledger)"),
        ("4", "BACKTEST", "full backtest + risk analysis"),
        ("5", "OPTIMIZE", "train/test param search (no leakage)"),
        ("6", "STATUS", "full validation-gate progress"),
    ]
    for key, label, desc in items:
        print(f"  {_c('[' + key + ']', 'BRIGHT_YELLOW', bold=True)} "
              f"{label:<11} — {desc}")
    print(f"  {_c('[Q]', 'DIM')} BACK")
    print()


def _menu_signal() -> None:
    sym = _ask_symbol("ALL")
    if sym is None:
        return
    eq = _ask_equity()
    _run_guarded(lambda: _cmd_signal(argparse.Namespace(symbol=sym, equity=eq)))


def _menu_backtest() -> None:
    sym = _ask_symbol("BTC")
    if sym is None:
        return

    def _run():
        # walk-forward (updates the banner verdict) then full-period risk analysis
        _cmd_backtest(argparse.Namespace(symbol=sym, strategy="reversion",
                                         walk_forward=True, robustness=False))
        _cmd_backtest(argparse.Namespace(symbol=sym, strategy="reversion",
                                         walk_forward=False, robustness=False))
    _run_guarded(_run)


def _menu_optimize() -> None:
    sym = _ask_symbol("BTC")
    if sym is None:
        return
    _run_guarded(lambda: _cmd_optimize(argparse.Namespace(symbol=sym)))


def _menu_status() -> None:
    _cmd_status(None)


def _menu_portfolio() -> None:
    from .paper import PaperLedger
    ledger = PaperLedger()
    opens = ledger.open_positions()
    closed = ledger.closed_positions()
    print(f"\n  {_c('Open paper positions', 'BRIGHT_CYAN', bold=True)}: {len(opens)}")
    for t in opens:
        side_color = "BRIGHT_GREEN" if t["side"] == "long" else "BRIGHT_RED"
        print(f"    #{t['id']}  {t['symbol']} {_c(t['side'].upper(), side_color)} "
              f"@ {t['entry']:,.0f}"
              f"  stop {t['stop']:,.0f}  tgt {t['target']:,.0f}"
              f"  {t['eff_leverage']:.1f}x  notional ${t['notional']:,.0f}"
              f"  liq ~{t['liq_price']:,.0f}")
    realized = sum(t.get("pnl_usd") or 0.0 for t in closed)
    wins = sum(1 for t in closed if (t.get("pnl_usd") or 0.0) > 0)
    win_rate = f"{wins}/{len(closed)}" if closed else "0/0"
    print(f"\n  Closed: {len(closed)}  |  wins: {win_rate}  |  "
          f"realized P&L (paper): {_money(realized)}")
    print(f"  {gate_progress(closed)}\n")


def _menu_paper_log() -> None:
    """Evaluate the latest bar and, only if the setup is actionable AND passes
    the liquidation-safety rule, log it to the paper ledger after confirmation.
    Unsafe or absent setups are refused — never logged."""
    from . import data as D
    from .paper import PaperLedger
    sym = _ask_symbol("BTC")
    if sym is None:
        return
    if sym == "ALL":
        print("  Pick a single symbol (BTC or ETH) to log a position.")
        return
    eq = _ask_equity()
    signal_fn, params, label = _strategy("reversion")
    try:
        df5, df15 = D.load_history(_SYMBOLS[sym])
    except Exception as exc:  # network / data layer — stay in the menu
        print(f"  Could not load {sym} data: {type(exc).__name__}: {exc}")
        return
    sigs = [s for s in signal_fn(df5, df15, params)
            if s.ts == df5.index[-1]]
    if not sigs:
        print(f"  {_SYMBOLS[sym]}: no actionable setup on the latest bar — "
              "nothing to log.")
        return
    sig = sigs[0]
    stop_dist = abs(sig.stop - sig.entry) / sig.entry
    sizing = effective_leverage_size(eq, stop_dist, price=sig.entry)
    if sizing is None:
        print(f"  {sig.symbol}: stop {stop_dist*100:.2f}% is too wide to reach the "
              "3-6x band at <=2% risk — not logged.")
        return
    liq = liquidation_price(sig.entry, sig.side, sizing.eff_leverage)
    liq_dist = abs(liq - sig.entry) / sig.entry
    safe = passes_liquidation_safety(stop_dist, liq_dist)
    print("\n" + render(sig, sizing, liq_price=liq, safe=safe) + "\n")
    if not safe:
        print("  REJECTED by the liquidation-safety rule — refusing to log.\n")
        return
    if _ask("Log this to the paper ledger? (y/N)", "n").lower() not in ("y", "yes"):
        print("  Not logged.\n")
        return
    trade_id = PaperLedger().open_position(sig, sizing, liq_price=liq)
    print(f"  Logged paper position #{trade_id} "
          f"({sig.symbol} {sig.side.upper()} @ {sig.entry:,.0f}).\n")


def menu() -> None:
    """Interactive entry point for launcher mode [3]."""
    handlers = {"1": _menu_signal, "2": _menu_paper_log, "3": _menu_portfolio,
                "4": _menu_backtest, "5": _menu_optimize, "6": _menu_status}
    while True:
        _print_menu_header()
        choice = _ask("Choice", "Q").upper()
        if choice in ("Q", "QUIT", "EXIT", "BACK", ""):
            return
        handler = handlers.get(choice)
        if handler is None:
            print(f"  Unknown choice: {choice!r} — pick 1-6 or Q.")
            continue
        handler()


def main(argv: Optional[list] = None):
    p = argparse.ArgumentParser(prog="python -m src.leverage")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("signal", "paper"):
        sp = sub.add_parser(name)
        sp.add_argument("--symbol", choices=["BTC", "ETH", "ALL"], default="ALL")
        sp.add_argument("--equity", type=float, default=1500.0)
        sp.add_argument("--strategy", choices=["reversion", "breakout"],
                        default="reversion")
        if name == "paper":
            sp.add_argument("--live", action="store_true")
    bp = sub.add_parser("backtest")
    bp.add_argument("--symbol", choices=["BTC", "ETH", "ALL"], default="BTC")
    bp.add_argument("--strategy", choices=["reversion", "breakout"],
                    default="reversion")
    bp.add_argument("--walk-forward", action="store_true")
    bp.add_argument("--robustness", action="store_true")
    op = sub.add_parser("optimize")
    op.add_argument("--symbol", choices=["BTC", "ETH", "ALL"], default="BTC")
    sub.add_parser("status")
    args = p.parse_args(argv)
    {"signal": _cmd_signal, "backtest": _cmd_backtest, "paper": _cmd_paper,
     "status": _cmd_status, "optimize": _cmd_optimize}[args.cmd](args)


if __name__ == "__main__":
    main(sys.argv[1:])
