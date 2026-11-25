"""Ghost Benchmark Race — Live Equity Curve Comparison (SPY vs VM_OUTLOOK+SGOV vs Real Book).

Computes three aligned, normalized (base=100) equity curves over `lookback_days` using ONLY
local SQLite data (`data/equity_ohlcv.db` + `data/own_picks.db` + 3m T-bill cash yield):
  1. `spy`              — Passive SPY Buy & Hold
  2. `vm_outlook_sgov`  — Mechanical monthly `VM_OUTLOOK_TOP3` rotation + SGOV T-bill yield
                          earned daily on the uninvested volatility buffer
  3. `real_book`        — Realized + unrealized MTM path of your logged TFSA picks (`own_picks.db`)

Also computes Override Attribution (`real_book` vs `vm_outlook_sgov` & `spy`) so you see
honestly whether manual picks add or subtract alpha relative to the mechanical Core sleeve.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.own_picks import list_equity_picks
from src.profile import estimate_cash_yield_annual
from src.radar.scanners import load_local_ohlcv_safe


@dataclass
class GhostRaceResult:
    dates: List[str] = field(default_factory=list)
    spy: List[float] = field(default_factory=list)
    vm_outlook_sgov: List[float] = field(default_factory=list)
    real_book: List[float] = field(default_factory=list)
    spy_ret_pct: float = 0.0
    vm_ret_pct: float = 0.0
    vm_cash_only_boost_pct: float = 0.0
    real_ret_pct: Optional[float] = None
    override_vs_core_pct: Optional[float] = None
    override_vs_spy_pct: Optional[float] = None
    cash_yield_ann: float = 0.043
    n_picks_total: int = 0
    n_picks_closed: int = 0
    n_picks_open: int = 0
    win_rate_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_ghost_race(
    *,
    ohlcv_db: str = "data/equity_ohlcv.db",
    own_picks_db: str = "data/own_picks.db",
    lookback_days: int = 126,
    cash_yield_ann: Optional[float] = None,
    account_usd: float = 5000.0,
) -> GhostRaceResult:
    """Compute aligned Ghost Benchmark Race curves from local archives (exact VM_OUTLOOK_TOP3 parity)."""
    from src.outlook.vm_rotation import UNIVERSE as VM_UNIVERSE, align_closes, target_weights

    if cash_yield_ann is None:
        y_ann, _lbl = estimate_cash_yield_annual()
    else:
        y_ann = float(cash_yield_ann)

    picks = list_equity_picks(db_path=own_picks_db)
    pick_syms = {str(p.get("ticker") or "").upper() for p in picks if p.get("ticker")}
    needed = sorted(set(VM_UNIVERSE) | {"SPY"} | pick_syms)
    ohlcv = load_local_ohlcv_safe(ohlcv_db, symbols=needed, lookback_days=max(320, lookback_days + 270))
    if "SPY" not in ohlcv or len(ohlcv["SPY"][0]) < 30:
        return GhostRaceResult(cash_yield_ann=y_ann)

    dates, cols = align_closes(ohlcv, VM_UNIVERSE)
    n_total = len(dates)
    start_idx = max(0, n_total - max(20, int(lookback_days)))
    race_dates = list(dates[start_idx:])
    n_race = len(race_dates)
    if n_race < 2:
        return GhostRaceResult(cash_yield_ann=y_ann)

    # 1. SPY normalized curve
    spy_col = cols["SPY"]
    s0 = float(spy_col[start_idx])
    spy_curve = [round(100.0 * float(spy_col[start_idx + i]) / max(s0, 1e-6), 2) for i in range(n_race)]

    # Fast price lookup per ticker on race_dates
    px_map: Dict[str, Dict[str, float]] = {}
    for sym, (ds, cs, _vs) in ohlcv.items():
        px_map[sym] = {d: float(c) for d, c in zip(ds, cs)}

    # 2. Mechanical Monthly VM_OUTLOOK_TOP3 + SGOV T-bill cash yield (exact parity, zero lookahead)
    daily_rf = (1.0 + max(0.0, y_ann)) ** (1.0 / 252.0) - 1.0
    vm_val = 100.0
    vm_val_no_rf = 100.0
    vm_curve: List[float] = [100.0]
    weights: Dict[str, float] = {}
    cash_w = 1.0
    last_month = ""

    for i in range(n_race):
        t = start_idx + i
        d = dates[t]
        ym = d[:7]
        if i == 0 or ym != last_month:
            try:
                tgt = target_weights(dates, cols, t, universe=VM_UNIVERSE)
                if tgt is not None:
                    weights = dict(tgt.weights)
                    cash_w = float(tgt.cash)
                else:
                    weights = {}
                    cash_w = 1.0
            except Exception:
                weights = {}
                cash_w = 1.0
            last_month = ym

        if i > 0:
            prev_d = race_dates[i - 1]
            port_ret = 0.0
            for tk, w in weights.items():
                p1 = px_map.get(tk, {}).get(d)
                p0 = px_map.get(tk, {}).get(prev_d)
                if p1 and p0 and p0 > 0:
                    port_ret += w * (p1 / p0 - 1.0)
            vm_val *= 1.0 + port_ret + cash_w * daily_rf
            vm_val_no_rf *= 1.0 + port_ret
            vm_curve.append(round(vm_val, 2))

    # 3. Real Book curve from data/own_picks.db
    closed = [p for p in picks if str(p.get("status") or "").upper() == "CLOSED" and p.get("exit_price")]
    open_p = [p for p in picks if str(p.get("status") or "").upper() == "OPEN"]
    real_curve: List[float] = []
    real_ret_pct: Optional[float] = None
    win_rate: Optional[float] = None

    if picks and account_usd > 0:
        # Compute daily cumulative realized + unrealized PnL relative to account_usd
        for d in race_dates:
            cum_pnl = 0.0
            for p in picks:
                ed = str(p.get("date") or "")[:10]
                if not ed or d < ed:
                    continue
                tk = str(p.get("ticker") or "").upper()
                ent = float(p.get("entry_price") or 0.0)
                sh = float(p.get("shares") or 0.0)
                if ent <= 0 or sh <= 0:
                    continue
                sign = -1.0 if str(p.get("side") or "LONG").upper() == "SHORT" else 1.0
                xd = str(p.get("exit_date") or "")[:10] if p.get("exit_date") else ""
                if str(p.get("status") or "").upper() == "CLOSED" and xd and d >= xd and p.get("exit_price"):
                    cum_pnl += sign * (float(p["exit_price"]) - ent) * sh
                else:
                    cur = px_map.get(tk, {}).get(d)
                    if cur is not None:
                        cum_pnl += sign * (cur - ent) * sh
            real_curve.append(round(100.0 * (1.0 + cum_pnl / account_usd), 2))
        real_ret_pct = round(real_curve[-1] - 100.0, 2)

    if closed:
        wins = 0
        for p in closed:
            ent = float(p.get("entry_price") or 0.0)
            ex = float(p.get("exit_price") or 0.0)
            sign = -1.0 if str(p.get("side") or "LONG").upper() == "SHORT" else 1.0
            if sign * (ex - ent) > 0:
                wins += 1
        win_rate = round(100.0 * wins / len(closed), 1)

    spy_ret = round(spy_curve[-1] - 100.0, 2)
    vm_ret = round(vm_curve[-1] - 100.0, 2)
    boost = round(vm_val - vm_val_no_rf, 2)

    return GhostRaceResult(
        dates=race_dates,
        spy=spy_curve,
        vm_outlook_sgov=vm_curve,
        real_book=real_curve,
        spy_ret_pct=spy_ret,
        vm_ret_pct=vm_ret,
        vm_cash_only_boost_pct=boost,
        real_ret_pct=real_ret_pct,
        override_vs_core_pct=round(real_ret_pct - vm_ret, 2) if real_ret_pct is not None else None,
        override_vs_spy_pct=round(real_ret_pct - spy_ret, 2) if real_ret_pct is not None else None,
        cash_yield_ann=round(y_ann, 4),
        n_picks_total=len(picks),
        n_picks_closed=len(closed),
        n_picks_open=len(open_p),
        win_rate_pct=win_rate,
    )
