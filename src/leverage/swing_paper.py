"""Minimal forward paper track for the daily swing-breakout.

Crypto is a satellite: the validation harness found no promotable perp signal,
and swing is a modest, decaying edge. Rather than build more, this just lets the
one edge *prove itself forward* — when a breakout fires it is logged to the perp
paper ledger, and open positions are walked forward each run against the daily
bars (chandelier trailing stop + regime-flip + max-hold), so a real out-of-sample
R-multiple track accumulates instead of relying on a one-off backtest.

Pure orchestration: the ledger and the daily-data loader are injected, so the
whole thing is unit-testable without touching the network.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import swing as S
from .sizing import effective_leverage_size
from .risk import liquidation_price

# Perp symbols for the two names swing trades (matches __main__._SYMBOLS).
_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}


def resolve_open(feat: pd.DataFrame, entry_date, side: str, entry: float,
                 initial_stop: float,
                 max_hold: int = S.MAX_HOLD) -> Optional[Tuple[float, object, str]]:
    """Replay an open swing position forward from its entry bar. Returns
    (exit_price, exit_date, reason) once a chandelier-stop breach, regime flip
    (close back through MA), or max-hold is reached, else None (still open).

    `initial_stop` fixes the risk unit (|entry - initial_stop|); the trailing
    stop rides the running extreme by that same distance — identical semantics
    to swing.backtest, so the forward track matches the backtest."""
    idx = list(feat.index)
    if entry_date not in idx:
        return None
    i = idx.index(entry_date)
    risk = abs(entry - initial_stop)
    if risk <= 0:
        return None
    c = feat["close"].values
    h = feat["high"].values
    l = feat["low"].values
    ma = feat["ma"].values
    n = len(idx)
    extreme = entry
    end = min(i + max_hold, n - 1)
    for j in range(i + 1, end + 1):
        if side == "long":
            extreme = max(extreme, c[j])
            stop = extreme - risk
            if l[j] <= stop:
                return (float(stop), idx[j], "stop")
            if np.isfinite(ma[j]) and c[j] < ma[j]:
                return (float(c[j]), idx[j], "regime")
        else:
            extreme = min(extreme, c[j])
            stop = extreme + risk
            if h[j] >= stop:
                return (float(stop), idx[j], "stop")
            if np.isfinite(ma[j]) and c[j] > ma[j]:
                return (float(c[j]), idx[j], "regime")
    if end - i >= max_hold:
        return (float(c[end]), idx[end], "max_hold")
    return None  # latest bar reached without an exit and short of max-hold


def run_swing_paper(symbol_keys: List[str], equity: float, ledger,
                    load_daily: Callable, now=None) -> Dict[str, list]:
    """For each symbol: resolve/close any open swing paper position against the
    latest daily bars, then (if flat) log today's fresh breakout if one fired.

    `ledger` is a PaperLedger; `load_daily(key)` returns a daily OHLCV frame.
    Idempotent within a day: never opens a second position while one is open."""
    summary: Dict[str, list] = {"opened": [], "closed": []}
    for key in symbol_keys:
        try:
            df = load_daily(key)
        except Exception:
            continue
        if df is None or len(df) < S.DEFAULT_MA + S.DEFAULT_LOOKBACK + 5:
            continue
        feat = S.compute_features(df)
        sym = _SYMBOLS.get(key, key)

        # 1. Resolve open positions for this symbol.
        for pos in ledger.open_positions():
            if pos.get("symbol") != sym or pos.get("session") != "swing":
                continue
            try:
                entry_date = pd.to_datetime(pos["ts"], utc=True)
            except Exception:
                continue
            res = resolve_open(feat, entry_date, pos["side"], float(pos["entry"]),
                               float(pos["stop"]))
            if res is not None:
                exit_px, _exit_date, reason = res
                ledger.close_position(pos["id"], exit_px, reason)
                summary["closed"].append((sym, reason, round(exit_px, 2)))

        # 2. If flat on this symbol, log today's breakout (if any).
        flat = not any(p.get("symbol") == sym and p.get("session") == "swing"
                       for p in ledger.open_positions())
        if not flat:
            continue
        k_long = S.calibrate_stop_k(feat, "long")
        k_short = S.calibrate_stop_k(feat, "short")
        sig = S.latest_signal(df, stop_k=k_long)
        if sig is None:
            continue
        k = k_long if sig.side == "long" else k_short
        sig = S.latest_signal(df, stop_k=k)
        if sig is None:
            continue
        sizing = effective_leverage_size(equity, sig.risk_pct, price=sig.price)
        if sizing is None:
            continue
        liq = liquidation_price(sig.price, sig.side, sizing.eff_leverage)
        ledger.open_swing_position(sym, sig.side, sig.date, sig.price, sig.stop,
                                   sizing.qty, sizing.notional, sizing.eff_leverage)
        summary["opened"].append((sym, sig.side, round(sig.price, 2),
                                  round(sizing.eff_leverage, 1), round(liq, 2)))
    return summary
