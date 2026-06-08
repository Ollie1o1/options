"""Mirror-mode order ticket + the hard switch.

``render_ticket`` emits a LIVE order ticket only when BOTH the validation gate
reads READY AND the live-execution flag is on AND the sizing is non-zero. In every
other case it returns a DRY-RUN refusal. The system never places the order — the
human reads the ticket and places it in their broker, then records the fill via
``slippage.record_fill``.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from src.execution.exits import Exits
from src.execution.sizing import Sizing


def _limit_price(pick: dict) -> float:
    """Buy limit near the mid of the quoted spread; fall back to entry/mid."""
    bid, ask = pick.get("bid"), pick.get("ask")
    if bid is not None and ask is not None and ask >= bid > 0:
        return round((bid + ask) / 2.0, 2)
    return round(float(pick.get("mid") or pick.get("entry_price") or 0.0), 2)


def render_ticket(pick: dict,
                  sizing: Sizing,
                  exits: Exits,
                  gate_decision: str,
                  live_enabled: bool,
                  now: Optional[datetime] = None) -> dict:
    """Return {'mode': 'LIVE'|'DRY_RUN', 'text': str, 'limit_price': float, ...}."""
    now = now or datetime.now()
    limit = _limit_price(pick)
    sym = pick.get("ticker", "?")
    strike = pick.get("strike")
    exp = pick.get("expiration")
    otype = (pick.get("option_type") or "call").upper()
    contract = f"{sym} {strike}{otype[0]} exp {exp}"

    gate_ok = gate_decision == "READY"
    blockers = []
    if not gate_ok:
        blockers.append(f"gate={gate_decision} (need READY)")
    if not live_enabled:
        blockers.append("config.live_execution.enabled=false")
    if sizing.contracts <= 0:
        blockers.append(f"sizing -> {sizing.contracts} contracts ({sizing.notes})")

    header = f"{contract}  |  limit ${limit:.2f}"

    if blockers:
        text = (
            "──────────── DRY RUN — no live order ────────────\n"
            f"{header}\n"
            f"Blocked by: {'; '.join(blockers)}\n"
            f"Would size: {sizing.contracts} contracts "
            f"(risk ${sizing.dollar_risk:.0f}, cost ${sizing.cost_basis:.0f})\n"
            f"Exits: TP ${exits.take_profit_price:.2f} | "
            f"SL ${exits.stop_price:.2f} | time-exit {exits.time_exit_date}\n"
            "Flip BOTH the gate (READY) and config.live_execution.enabled to arm."
        )
        return {"mode": "DRY_RUN", "text": text, "limit_price": limit,
                "blockers": blockers}

    text = (
        "════════════════ LIVE ORDER TICKET ════════════════\n"
        f"BUY  {sizing.contracts}x  {contract}\n"
        f"  Limit price ....... ${limit:.2f}  (mid of {pick.get('bid')}/{pick.get('ask')})\n"
        f"  Cost basis ........ ${sizing.cost_basis:.0f}  "
        f"({sizing.risk_pct*100:.1f}% acct at risk)\n"
        f"  Take profit ....... ${exits.take_profit_price:.2f}\n"
        f"  Stop loss ......... ${exits.stop_price:.2f}\n"
        f"  Time exit ......... {exits.time_exit_date} (DTE {exits.time_exit_dte})\n"
        f"  Min days held ..... {exits.min_days_held}\n"
        f"  Ticket time ....... {now:%Y-%m-%d %H:%M}\n"
        "After filling, record the actual price with slippage.record_fill()."
    )
    return {"mode": "LIVE", "text": text, "limit_price": limit,
            "contracts": sizing.contracts, "blockers": []}
