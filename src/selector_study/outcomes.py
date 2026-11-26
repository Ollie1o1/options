"""Paper outcomes for one frozen entry. Pure: entry + marks + settings in, result out.

Both arms use the same rules. Two cost cases are computed independently, each walking
the marks with its own prices, so an exit can fire on a different day in each case:

  cross (primary)      sell at the bid / buy at the ask, both ends, plus fees
  mid   (sensitivity)  every leg at its midpoint, plus fees; the fill is assumed

Exits are the book's spread rules (config.json -> exit_rules): take profit at a share of
the entry credit, stop at a multiple of it, time exit at `time_exit_dte`, no exit before
`min_days_held` calendar days. A position still open at expiration settles at intrinsic
value on the underlying's close.

Statuses: closed | open | not_filled (no positive entry credit after costs) |
missing (the marks cannot support a decision: a gap longer than `max_mark_gap_sessions`
usable sessions, or expiration passed without a settlement price).
Return on capital at risk = P&L / (width - entry credit), both per share.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional, Sequence

CASES = ("cross", "mid")


def _d(s: str) -> dt.date:
    return dt.date.fromisoformat(str(s)[:10])


def _close_debit(mark: Dict[str, Any], case: str) -> Optional[float]:
    raw = [mark.get(k) for k in ("short_bid", "short_ask", "long_bid", "long_ask")]
    if any(v is None for v in raw):
        return None
    sb, sa, lb, la = (float(v) for v in raw if v is not None)
    if case == "cross":
        return sa - lb
    return 0.5 * (sb + sa) - 0.5 * (lb + la)


def intrinsic(short_strike: float, long_strike: float, close: float) -> float:
    return max(0.0, short_strike - close) - max(0.0, long_strike - close)


def resolve(entry: Dict[str, Any], marks: Sequence[Dict[str, Any]], settings: Dict[str, Any],
            case: str) -> Dict[str, Any]:
    """Outcome of `entry` under `case`.

    `marks` holds one row per trading session after entry (the marker writes missing
    sessions as rows too), so consecutive unusable rows are the gap length in sessions."""
    ex = settings["exits"]
    fee_ps = 2.0 * float(settings.get("fee_per_contract", 0.0)) / 100.0  # two legs, per share
    credit = float(entry["credit_cross"] if case == "cross" else entry["credit_mid"]) - fee_ps
    width = float(entry["width"])
    out: Dict[str, Any] = {"case": case, "entry_credit": round(credit, 6), "status": "open",
                           "exit_session": None, "exit_reason": None, "ror": None, "pnl": None,
                           "capital_at_risk": None}
    if credit <= 0:
        out.update(status="not_filled", exit_reason="no positive credit after costs")
        return out
    risk = width - credit
    if risk <= 0:
        out.update(status="not_filled", exit_reason="credit not below width")
        return out
    out["capital_at_risk"] = round(risk, 6)
    session, expiration = str(entry["session"]), str(entry["expiration"])
    gap_limit = int(ex.get("max_mark_gap_sessions", 3))
    hold = ex.get("mode") == "hold_to_expiry"   # historical companion: one crossing, settle at intrinsic
    bad_run, last_good = 0, session

    def finish(day: str, reason: str, debit: float) -> Dict[str, Any]:
        pnl = credit - debit
        out.update(status="closed", exit_session=day, exit_reason=reason,
                   pnl=round(pnl, 6), ror=round(pnl / risk, 6))
        return out

    ordered = sorted(marks, key=lambda m: str(m["mark_session"]))
    for m in ordered:
        day = str(m["mark_session"])
        if day <= session:
            continue
        if m["status"] == "settled":
            if m.get("underlying_close") is None:
                continue
            return finish(day, "expired", intrinsic(float(entry["short_strike"]), float(entry["long_strike"]),
                                                    float(m["underlying_close"])))
        if day > expiration or hold:
            continue
        debit_raw = _close_debit(m, case) if m["status"] == "marked" else None
        if debit_raw is None:
            bad_run += 1
            if bad_run > gap_limit:
                out.update(status="missing", exit_reason=f"no usable mark for {bad_run} sessions after {last_good}")
                return out
            continue
        bad_run, last_good = 0, day
        if (_d(day) - _d(session)).days < int(ex["min_days_held"]):
            continue
        debit = debit_raw + fee_ps
        pnl_frac = (credit - debit) / credit
        if pnl_frac >= float(ex["take_profit"]):
            return finish(day, "take_profit", debit)
        if pnl_frac <= float(ex["stop_loss"]):
            return finish(day, "stop_loss", debit)
        if (_d(expiration) - _d(day)).days <= int(ex["time_exit_dte"]):
            return finish(day, "time_exit", debit)
    if hold and not any(str(m["mark_session"]) >= expiration for m in ordered):
        return out
    if any(str(m["mark_session"]) >= expiration for m in ordered):
        out.update(status="missing", exit_reason="expiration passed without a settlement price")
    return out


def resolve_all(entry: Dict[str, Any], marks: Sequence[Dict[str, Any]],
                settings: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {c: resolve(entry, marks, settings, c) for c in CASES}
