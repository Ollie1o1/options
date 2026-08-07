"""The setup board: what each setup claims, what it costs, and what it has proved.

Four columns carry the weight. STATUS says how far a setup has got; SIGNAL says
what it is conditioned on; FRICTION says what the round trip costs as a share of
the credit — the constraint that has killed more of this book's edge than any
signal ever supplied — and CONTROL marks the rows that exist to keep the others
honest.

Display only. Nothing here places, sizes or authorises a trade.
"""
from __future__ import annotations

import textwrap
from typing import Any, Dict, Iterable, List, Optional

from .. import formatting as fmt
from .. import ui
from . import friction as fr
from .record import StrategyRecord

STATUS_STYLE = {
    "idea": "muted",
    "specified": "label",
    "backtesting": "warn",
    "validated": "good",
    "promoted": "good",
    "live": "emph",
    "retired": "muted",
    "dead": "bad",
}

CONTROL_ROLES = ("benchmark", "null_control", "known_negative")

# Tier order on the board. Signals first because they are the open question;
# controls last because they are read as the answer's guardrail.
_TIERS = (
    ("candidate", "SIGNAL — the open question: does timing help?"),
    ("directional", "DIRECTIONAL — a view on price, expressed as short premium"),
    ("expression_control", "EXPRESSION CONTROL — same signal, bought instead of sold"),
    ("index_probe", "PROBE — index versus single name"),
    ("single_name_probe", None),
    ("benchmark", "CONTROL — what makes any of the above believable"),
    ("null_control", None),
    ("known_negative", None),
)


def _signal_summary(record: StrategyRecord) -> str:
    """The signal in a handful of characters, or an explicit 'none'."""
    sig = record.signal or {}
    if not sig:
        return "none (unselected)"
    parts: List[str] = []
    if sig.get("iv_rank_min"):
        parts.append(f"IVR>{int(sig['iv_rank_min'])}")
    if sig.get("above_sma50"):
        parts.append("above SMA50")
    if sig.get("below_sma50"):
        parts.append("below SMA50")
    if sig.get("rsi_min"):
        parts.append(f"RSI>{int(sig['rsi_min'])}")
    if sig.get("rsi_max"):
        parts.append(f"RSI<{int(sig['rsi_max'])}")
    if sig.get("drop_pct_min"):
        parts.append(f"drop>{int(sig['drop_pct_min'])}%")
    if sig.get("earnings_within"):
        parts.append(f"earnings<{int(sig['earnings_within'])}d")
    if sig.get("above_cost_basis"):
        parts.append("above basis")
    return " · ".join(parts) if parts else "none (unselected)"


def _accounts(record: StrategyRecord) -> str:
    return "/".join(record.accounts)


def _role(record: StrategyRecord) -> str:
    return str((record.provenance or {}).get("role", ""))


def format_board(records: Iterable[StrategyRecord], width: int = 100,
                 account: Optional[str] = None,
                 table: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    """One line per setup, grouped by tier, truncated to `width`.

    `table` pins the friction source; by default it is measured from the ledger.
    """
    recs = list(records)
    if account:
        recs = [r for r in recs if r.tradeable_in(account)]
    friction_table = fr.load_table() if table is None else table

    # id · status · structure · friction · accounts · signal
    w_id, w_status, w_struct, w_fric, w_acct = 30, 11, 13, 9, 14
    w_signal = max(12, width - 2 - (w_id + w_status + w_struct + w_fric + w_acct + 5))

    title = "STRATEGY SETUPS" if not account else f"STRATEGY SETUPS  ·  {account.upper()}"
    out = [ui.rule(width, title)]
    header = "  " + " ".join([
        fmt.style(ui.pad("SETUP", w_id), "label", bold=True),
        fmt.style(ui.pad("STATUS", w_status), "label", bold=True),
        fmt.style(ui.pad("STRUCTURE", w_struct), "label", bold=True),
        fmt.style(ui.pad("FRICTION", w_fric, "right"), "label", bold=True),
        fmt.style(ui.pad("ACCOUNT", w_acct), "label", bold=True),
        fmt.style(ui.pad("SIGNAL", w_signal), "label", bold=True),
    ]).rstrip()
    out.append(header)
    out.append("  " + fmt.style("─" * min(width - 2, w_id + w_status + w_struct
                                          + w_fric + w_acct + w_signal + 5), "muted"))

    seen = set()
    for role, heading in _TIERS:
        tier_recs = [r for r in recs if _role(r) == role and id(r) not in seen]
        if not tier_recs:
            continue
        if heading:
            out.append("  " + fmt.style(ui.clip(heading, width - 2), "muted"))
        for r in tier_recs:
            seen.add(id(r))
            profile = fr.profile_for(r, table=friction_table)
            cells = [
                ui.pad(ui.clip(r.spec.id, w_id), w_id),
                fmt.style(ui.pad(ui.clip(r.status, w_status), w_status),
                          STATUS_STYLE.get(r.status, "value")),
                fmt.style(ui.pad(ui.clip(r.spec.structure, w_struct), w_struct),
                          "muted"),
                fmt.style(ui.pad(fr.format_cell(profile), w_fric, "right"),
                          fr.style_for(profile)),
                ui.pad(ui.clip(_accounts(r), w_acct), w_acct),
                ui.clip(_signal_summary(r)
                        + (" [CONTROL]" if role in CONTROL_ROLES else ""),
                        w_signal),
            ]
            out.append(("  " + " ".join(cells)).rstrip())

    # Anything whose role is unknown still has to appear — a setup that is
    # invisible because its role was mistyped is worse than an ugly board.
    rest = [r for r in recs if id(r) not in seen]
    if rest:
        out.append("  " + fmt.style("OTHER", "muted"))
        for r in rest:
            profile = fr.profile_for(r, table=friction_table)
            out.append(("  " + " ".join([
                ui.pad(ui.clip(r.spec.id, w_id), w_id),
                fmt.style(ui.pad(ui.clip(r.status, w_status), w_status),
                          STATUS_STYLE.get(r.status, "value")),
                fmt.style(ui.pad(ui.clip(r.spec.structure, w_struct), w_struct),
                          "muted"),
                fmt.style(ui.pad(fr.format_cell(profile), w_fric, "right"),
                          fr.style_for(profile)),
                ui.pad(ui.clip(_accounts(r), w_acct), w_acct),
                ui.clip(_signal_summary(r), w_signal),
            ])).rstrip())

    out.append(ui.rule(width))
    note = (f"FRICTION = round-trip crossing cost as a share of credit "
            f"(ceiling {fr.ceiling():.0%}).  — = unmeasured.")
    out.append("  " + fmt.style(ui.clip(note, width - 2), "muted"))
    out.append("  " + fmt.style(ui.clip(
        "Display only — this desk never places, sizes or authorises a trade.",
        width - 2), "muted"))
    return "\n".join(out)


def _wrap(text: str, width: int, indent: str = "    ") -> List[str]:
    return [indent + ln for ln in textwrap.wrap(str(text),
                                                max(20, width - len(indent)))]


def _kv(label: str, value: str, width: int) -> List[str]:
    return _wrap(f"{label}: {value}", width, indent="  ")


def format_detail(record: StrategyRecord, width: int = 100,
                  table: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    """Everything the desk knows about one setup, including what it does not."""
    profile = fr.profile_for(record, table=table)
    body: List[str] = []

    body.append("  " + fmt.style(ui.clip(record.name, width - 2), "emph"))
    body.append("  " + fmt.style(ui.clip(
        f"{record.spec.id}  ·  {record.spec.structure}  ·  "
        f"{record.status}  ·  role={_role(record) or 'unset'}", width - 2), "muted"))
    body.append("")

    body.append("  " + fmt.style("HYPOTHESIS", "label", bold=True))
    body.extend(_wrap(record.hypothesis, width))
    body.append("")

    body.append("  " + fmt.style("SIGNAL", "label", bold=True))
    if record.signal:
        for k, v in record.signal.items():
            body.append(f"    {k} = {v}")
    else:
        body.append("    none — unselected, every eligible day")
    body.append("")

    body.append("  " + fmt.style("ENTRY / EXIT", "label", bold=True))
    body.extend(_wrap(f"entry: {record.spec.entry}", width))
    body.extend(_wrap(f"exit:  {record.spec.exit}", width))
    body.extend(_wrap(f"universe: {record.spec.universe}", width))
    body.append("")

    body.append("  " + fmt.style("FRICTION", "label", bold=True))
    body.extend(_wrap(fr.describe(profile), width))
    body.append("")

    body.append("  " + fmt.style("CAPITAL", "label", bold=True))
    body.extend(_wrap(record.capital_note, width))
    body.append("")

    body.append("  " + fmt.style("ACCOUNTS", "label", bold=True))
    body.extend(_wrap(", ".join(record.accounts)
                      + "  (intent, not broker approval — confirm with the broker)",
                      width))
    body.append("")

    body.append("  " + fmt.style("EVIDENCE", "label", bold=True))
    if record.evidence:
        for k, v in record.evidence.items():
            body.extend(_wrap(f"{k} = {v}", width))
    else:
        body.append("    not yet evaluated — no backtest has landed on this setup")
    body.extend(_wrap(f"verdict: {record.verdict or 'none yet'}", width))
    body.append("")

    if record.links:
        body.append("  " + fmt.style("LINKS", "label", bold=True))
        for link in record.links:
            body.append(f"    {ui.clip(link, width - 4)}")
        body.append("")

    if record.amendments:
        body.append("  " + fmt.style("AMENDMENTS", "label", bold=True))
        for a in record.amendments:
            body.extend(_wrap(
                f"{a.get('date')}  {a.get('field')}: {a.get('from')!r} -> "
                f"{a.get('to')!r}  ({a.get('reason')})", width))

    lines = [ui.rule(width, "SETUP")] + body + [ui.rule(width)]
    return "\n".join(ui.clip(ln, width) for ln in lines)
