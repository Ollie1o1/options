"""Exit-aware option P&L model: what your ACTUAL exit rules do to a pick.

The screener's ``ev_per_contract`` is hold-to-expiry arithmetic, but nothing
in the book is held to expiry — trades exit on the config's exit_rules
(tiered/delta take-profits, stops, the 21-DTE time exit). This module
simulates those rules over Monte-Carlo paths so a pick can be ranked and read
by the P&L process you actually run.

One source of truth, no hardcoded levels: every rule parameter comes from
``paper_manager._normalize_exit_rules(config)`` and the trigger logic mirrors
``_evaluate_long_single_leg_exit`` / ``_evaluate_short_single_leg_exit``
(same priority order, same tiering, same sticky-entry-IV delta checks). A
consistency test in tests/test_exit_model.py locks the two together.

Model assumptions (stated, not hidden):
- Underlying follows GBM at the SAME realized vol the EV column uses
  (hv_ewma → hv_30d), drift r − q: no directional alpha is assumed.
- Option marks along the path are Black-Scholes at the ENTRY IV held constant
  (sticky IV) — the same convention exit enforcement uses for delta checks.
- Rules are checked once per day, like the daily enforcement scripts.
- Exits pay a half-spread + commission; expiry settles at intrinsic.

Everything is deterministic: the RNG seed derives from the contract identity,
so tearsheet sidecars stay byte-reproducible.
"""
import hashlib
import math
from typing import Any, Dict, Optional

import numpy as np

from src.paper_manager import _normalize_exit_rules
from src.utils import bs_delta, bs_price

PEAK_MULTIPLES = (1.5, 2.0, 3.0, 5.0)
_DEFAULT_PATHS = 4000
_MAX_STEPS = 260


def rules_from_config(config: Optional[dict]) -> dict:
    """The book's exit rules, normalized exactly as enforcement reads them."""
    return _normalize_exit_rules(config or {})


def seed_for(symbol, strike, expiration, opt_type) -> int:
    key = "{}|{}|{}|{}".format(symbol, strike, expiration, opt_type)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def _f(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def simulate_exits(
    spot: float,
    strike: float,
    opt_type: str,
    entry_price: float,
    t_years: float,
    sigma_real: float,
    iv_mark: Optional[float] = None,
    *,
    rules: dict,
    is_short: bool = False,
    entry_delta: Optional[float] = None,
    spread_pct: float = 0.0,
    commission: float = 0.65,
    r: float = 0.0,
    q: float = 0.0,
    n_paths: int = _DEFAULT_PATHS,
    seed: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Simulate the book's exit rules on one single-leg contract.

    Returns a plain-JSON dict (sidecar-safe) or None when inputs are unusable.
    """
    spot, strike = _f(spot), _f(strike)
    entry_price, t_years = _f(entry_price), _f(t_years)
    sigma_real = _f(sigma_real)
    iv = _f(iv_mark) or sigma_real
    if (not spot or spot <= 0 or not strike or strike <= 0
            or not entry_price or entry_price <= 0
            or not t_years or t_years <= 0
            or not sigma_real or sigma_real <= 0 or not iv or iv <= 0):
        return None
    opt = "call" if str(opt_type or "call").lower().startswith("c") else "put"
    spread = max(0.0, _f(spread_pct) or 0.0)
    half_spread = 0.5 * spread * entry_price
    comm_share = (_f(commission) or 0.0) / 100.0

    dte0 = max(1, int(round(t_years * 365.0)))
    n_steps = min(dte0, _MAX_STEPS)
    dt = t_years / n_steps
    rng = np.random.default_rng(
        seed if seed is not None else seed_for("?", strike, t_years, opt))

    # GBM at the physical (realized) vol, risk-neutral drift — no alpha.
    z = rng.standard_normal((n_paths, n_steps))
    log_paths = np.cumsum(
        (r - q - 0.5 * sigma_real ** 2) * dt
        + sigma_real * math.sqrt(dt) * z, axis=1)
    S = spot * np.exp(log_paths)                      # (paths, steps)

    lng, sht = rules["long"], rules["short"]
    time_exit_dte = int(rules["time_exit_dte"])
    min_days_held = int(rules["min_days_held"])

    alive = np.ones(n_paths, dtype=bool)
    exit_kind = np.zeros(n_paths, dtype=np.int8)      # 1 tp, 2 time, 3 sl, 4 expiry
    exit_value = np.zeros(n_paths)                    # per-share mark at exit
    exit_day = np.full(n_paths, n_steps, dtype=np.int32)
    peak_mid = np.zeros(n_paths)
    profit_touch = np.zeros(n_paths, dtype=bool)

    basis = entry_price + half_spread + comm_share    # long all-in cost/share

    days_per_step = dte0 / n_steps
    for t in range(1, n_steps + 1):
        St = S[:, t - 1]
        dte_t = max(0, int(round(dte0 - t * days_per_step)))
        T_rem = max(dte_t / 365.0, 1.0 / 365.0)
        at_expiry = t == n_steps
        if at_expiry:
            mid = np.maximum(0.0, (St - strike) if opt == "call" else (strike - St))
        else:
            mid = np.asarray(bs_price(opt, St, strike, T_rem, r, iv), dtype=float)
        peak_mid = np.maximum(peak_mid, mid)

        if is_short:
            # short: sold at entry − friction; profitable when buying back
            # (mid + friction) still leaves something
            proceeds_entry = entry_price - half_spread - comm_share
            profit_touch |= alive & ((mid + half_spread + comm_share) < proceeds_entry)
            pnl_raw = np.where(entry_price > 0, (entry_price - mid) / entry_price, 0.0)
        else:
            liq = mid - half_spread - comm_share       # long: sell to close
            profit_touch |= alive & (liq > basis)
            pnl_raw = np.where(entry_price > 0, (mid - entry_price) / entry_price, 0.0)

        if at_expiry:
            exit_value[alive] = mid[alive]
            exit_kind[alive] = 4
            exit_day[alive] = t
            alive[:] = False
            break

        # --- trigger evaluation, same priority order as paper_manager ---
        trig = np.zeros(n_paths, dtype=np.int8)
        if is_short:
            tp_target = (sht["tp_ge_21"] if dte_t >= 21
                         else sht["tp_7_21"] if dte_t >= 7 else sht["tp_lt_7"])
            trig = np.where(alive & (trig == 0) & (pnl_raw >= tp_target), 1, trig)
            if 0 < dte_t <= time_exit_dte and t >= min_days_held:
                trig = np.where(alive & (trig == 0), 2, trig)
            if sht["sl_strike"]:
                buf = sht["sl_strike_buf"]
                breach = (St >= strike * (1 + buf)) if opt == "call" else \
                         (St <= strike * (1 - buf))
                trig = np.where(alive & (trig == 0) & breach, 3, trig)
            sl_prem = -(sht["sl_prem_mult"] - 1.0)
            trig = np.where(alive & (trig == 0) & (pnl_raw <= sl_prem), 3, trig)
            ed = _f(entry_delta)
            if ed is not None and abs(ed) > 1e-4:
                cur_delta = np.abs(np.asarray(
                    bs_delta(opt, St, strike, T_rem, r, iv), dtype=float))
                trig = np.where(alive & (trig == 0)
                                & (cur_delta >= sht["sl_delta_mult"] * abs(ed)),
                                3, trig)
        else:
            trig = np.where(alive & (trig == 0) & (pnl_raw >= lng["tp"]), 1, trig)
            if 0 < dte_t <= time_exit_dte and t >= min_days_held:
                trig = np.where(alive & (trig == 0), 2, trig)
            cur_delta = np.abs(np.asarray(
                bs_delta(opt, St, strike, T_rem, r, iv), dtype=float))
            trig = np.where(alive & (trig == 0) & (cur_delta >= lng["tp_delta"]),
                            1, trig)
            trig = np.where(alive & (trig == 0) & (pnl_raw <= lng["sl"]), 3, trig)

        fired = trig > 0
        if fired.any():
            exit_kind[fired] = trig[fired]
            exit_value[fired] = mid[fired]
            exit_day[fired] = t
            alive &= ~fired

        if not alive.any():
            # finish peak tracking on the remaining horizon for "how high did
            # it go" — the ladder is explicitly if-held, not rule-truncated
            for t2 in range(t + 1, n_steps + 1):
                St2 = S[:, t2 - 1]
                dte2 = max(0, int(round(dte0 - t2 * days_per_step)))
                if t2 == n_steps:
                    mid2 = np.maximum(0.0, (St2 - strike) if opt == "call"
                                      else (strike - St2))
                else:
                    mid2 = np.asarray(bs_price(
                        opt, St2, strike, max(dte2 / 365.0, 1.0 / 365.0),
                        r, iv), dtype=float)
                peak_mid = np.maximum(peak_mid, mid2)
            break

    # Net P&L per contract under the rules (exit friction charged on sale;
    # expiry settles at intrinsic with commission only).
    friction = np.where(exit_kind == 4, comm_share, half_spread + comm_share)
    if is_short:
        pnl_share = (entry_price - half_spread - comm_share) - (exit_value + friction)
    else:
        pnl_share = (exit_value - friction) - (entry_price + half_spread + comm_share)
    ev_exit = float(np.mean(pnl_share) * 100.0)

    peak_mult = peak_mid / entry_price
    counts = {k: float(np.mean(exit_kind == v))
              for k, v in (("tp", 1), ("time", 2), ("sl", 3), ("expiry", 4))}
    return {
        "ev_exit_per_contract": round(ev_exit, 2),
        "p_tp": round(counts["tp"], 4),
        "p_time": round(counts["time"], 4),
        "p_sl": round(counts["sl"], 4),
        "p_expiry": round(counts["expiry"], 4),
        "p_profit_touch": round(float(np.mean(profit_touch)), 4),
        "med_days_to_exit": float(np.median(exit_day) * days_per_step),
        "peak": {
            "med_mult": round(float(np.median(peak_mult)), 3),
            "p90_mult": round(float(np.percentile(peak_mult, 90)), 3),
            "p_ge": {("%g" % m): round(float(np.mean(peak_mult >= m)), 4)
                     for m in PEAK_MULTIPLES},
        },
        "rules": {
            "tp": (None if is_short else lng["tp"]),
            "tp_delta": (None if is_short else lng["tp_delta"]),
            "sl": (None if is_short else lng["sl"]),
            "tiered_tp": ([sht["tp_ge_21"], sht["tp_7_21"], sht["tp_lt_7"]]
                          if is_short else None),
            "time_exit_dte": time_exit_dte,
        },
        "n_paths": int(n_paths),
        "assumptions": ("GBM at realized vol, drift r−q (no alpha); marks at "
                        "sticky entry IV; rules checked daily; exit pays "
                        "half-spread + commission"),
    }


def simulate_for_row(row: Dict[str, Any], config: Optional[dict] = None,
                     rfr: float = 0.0, n_paths: int = _DEFAULT_PATHS,
                     ) -> Optional[Dict[str, Any]]:
    """Row-level convenience: pull the same fields the EV column uses."""
    from src.utils import is_short_position
    sigma = _f(row.get("hv_ewma")) or _f(row.get("hv_30d"))
    hv_fallback = sigma is None
    if hv_fallback:
        sigma = _f(row.get("impliedVolatility"))
    out = simulate_exits(
        row.get("underlying"), row.get("strike"), row.get("type"),
        row.get("premium"), row.get("T_years"), sigma,
        iv_mark=row.get("impliedVolatility"),
        rules=rules_from_config(config),
        is_short=is_short_position(str(row.get("strategy_name") or "")),
        entry_delta=row.get("delta"),
        spread_pct=_f(row.get("spread_pct")) or 0.0,
        r=_f(rfr) or 0.0,
        n_paths=n_paths,
        seed=seed_for(row.get("symbol"), row.get("strike"),
                      row.get("expiration"), row.get("type")),
    )
    if out is not None:
        out["hv_fallback"] = bool(hv_fallback)
    return out
