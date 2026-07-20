"""Desk-kit HTML holdings report. Sidecar JSON + pure render (morning pattern).

No AI calls. `build()` is the only network-touching function (via
`data.fetch_snapshots`); `render()` is pure over the sidecar dict.
"""
import datetime as _dt
import json
import os
from typing import Optional, Tuple

from src.desk_kit import charts, shell

from .fills import DEFAULT_DB, book, deployed_usd, filled_levels
from .plan import DEFAULT_PATH as PLAN_PATH
from .plan import load_plan, tranche_size_usd
from .zones import IN_ZONE, NEAR, assess

_STATE_BADGE = {IN_ZONE: "ok", NEAR: "warn"}


def build(plan_path: str = PLAN_PATH, db_path: str = DEFAULT_DB) -> dict:
    from .data import fetch_snapshots
    plan = load_plan(plan_path)
    snaps = fetch_snapshots([n.ticker for n in plan.names])
    held = book(db_path=db_path)
    deployed = deployed_usd(db_path=db_path)
    today = _dt.date.today().isoformat()
    names = []
    book_value = 0.0
    for n in plan.names:
        snap = snaps.get(n.ticker)
        filled = filled_levels(n.ticker, db_path=db_path)
        read = assess(n, snap, filled) if snap else None
        closes = snap.closes[-252:] if snap else []
        ma50 = [sum(closes[max(0, i - 49):i + 1]) / min(i + 1, 50)
                for i in range(len(closes))] if closes else []
        ma200 = [sum(closes[max(0, i - 199):i + 1]) / min(i + 1, 200)
                 for i in range(len(closes))] if closes else []
        slot = held.get(n.ticker)
        if slot and snap:
            book_value += slot["shares"] * snap.spot
        names.append({
            "ticker": n.ticker, "thesis": n.thesis,
            "spot": snap.spot if snap else None,
            "state": read.state if read else None,
            "drawdown_pct": read.drawdown_pct if read else None,
            "sigma_dist": read.sigma_dist if read else None,
            "next_level": read.next_level if read else None,
            "ladder": [{"level": t.level,
                        "size_usd": tranche_size_usd(plan, n, t),
                        "filled": any(abs(t.level - f) < 1e-6 for f in filled)}
                       for t in n.tranches],
            "held": slot,
            "closes": closes, "ma50": ma50, "ma200": ma200,
            # Real per-date labels aren't available from snapshot_from_closes
            # (yfinance index dates are dropped) — the chart is context, not
            # analysis, so blank labels beat a wrong date repeated 252 times.
            "labels": [""] * len(closes),
        })
    cost = sum((s or {}).get("cost", 0.0) for s in held.values())
    return {"meta": {"sidecar": today + ".json",
                     "generated": _dt.datetime.now().isoformat(timespec="seconds")},
            "cash": {"pool": plan.cash_pool_usd, "deployed": deployed,
                     "remaining": max(0.0, plan.cash_pool_usd - deployed)},
            "book_value": book_value,
            "unrealized_pnl": book_value - cost,
            "names": names}


def render(data: dict) -> str:
    cash = data["cash"]
    # A KPI strip is its own top-level grid (each shell.kpi() carries its own
    # c{span} class) — a sibling of the card grid below, not nested inside
    # it, matching src/research/render.py's _market_kpis convention. Nesting
    # a second .grid as a cell of the outer grid would collapse it to one
    # implicit column since the wrapper div itself carries no c{n} class.
    kpis = ("<div class='grid'>"
            + shell.kpi("Book value", "$" + shell.num(data["book_value"], "{:,.0f}"))
            + shell.kpi("Unrealized P&L",
                       "$" + shell.num(data["unrealized_pnl"], "{:+,.0f}"),
                       tone="good" if data["unrealized_pnl"] >= 0 else "bad")
            + shell.kpi("Cash deployed", "$" + shell.num(cash["deployed"], "{:,.0f}"),
                       sub="of $" + shell.num(cash["pool"], "{:,.0f}") + " pool")
            + shell.kpi("Cash remaining", "$" + shell.num(cash["remaining"], "{:,.0f}"))
            + "</div>")
    cells = []
    zone_chips = [shell.esc(n["ticker"]) + " " +
                  shell.badge(n["state"], _STATE_BADGE.get(n["state"], "warn"))
                  for n in data["names"] if n.get("state") in (IN_ZONE, NEAR)]
    if zone_chips:
        cells.append(shell.card("In / near buy zones", " ".join(zone_chips), span=12))
    if not data["names"]:
        cells.append(shell.card("Plan", shell.ph("plan is empty — add ladders from the "
                                                 "HOLDINGS desk (ADD MU 750/650/550)"),
                                span=12))
    for i, n in enumerate(data["names"]):
        body = []
        # card() escapes the title itself, so a raw badge can't live there
        # without double-escaping into visible markup — it goes in the body
        # instead, right under the heading.
        if n.get("state") in (IN_ZONE, NEAR):
            body.append("<p>" + shell.badge(n["state"],
                                            _STATE_BADGE.get(n["state"], "warn"))
                        + "</p>")
        if n.get("thesis"):
            body.append("<p class='muted'>" + shell.esc(n["thesis"]) + "</p>")
        if n.get("closes"):
            support = ({"level": n["next_level"], "label": "next"}
                       if n.get("next_level") is not None else None)
            body.append(charts.price_chart(n["closes"], n["ma50"], n["ma200"],
                                           support, None, n.get("labels") or [],
                                           uid=f"lt{i}"))
        pairs = []
        for t in n["ladder"]:
            status = "filled" if t["filled"] else "open"
            pairs.append((f"tranche @ {t['level']:g}",
                          "$" + shell.num(t["size_usd"], "{:,.0f}") + " · " + status))
        if n.get("held"):
            pairs.append(("held", shell.num(n["held"]["shares"], "{:g}") + " sh @ avg $"
                          + shell.num(n["held"]["avg_price"], "{:,.2f}")))
        if n.get("drawdown_pct") is not None:
            pairs.append(("drawdown vs 52w high", shell.num(n["drawdown_pct"], "{:+.1f}") + "%"))
        body.append(shell.rows_table(pairs))
        cells.append(shell.card(n["ticker"], "".join(body), span=12))
    mast = shell.masthead("HOLDINGS", "Long-Term Accumulation",
                          meta_html=shell.esc(data["meta"]["generated"]))
    return shell.page("Holdings — Long-Term Accumulation", mast,
                      kpis + shell.grid(cells))


def write_report(out_dir: str = "reports/holdings",
                 data: Optional[dict] = None) -> Tuple[str, str]:
    data = data if data is not None else build()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(data["meta"]["sidecar"])[0]
    json_path = os.path.join(out_dir, base + ".json")
    html_path = os.path.join(out_dir, base + ".html")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
    with open(html_path, "w") as f:
        f.write(render(data))
    from src.desk_kit import hub
    hub.refresh_latest(out_dir, html_path)
    hub.refresh(os.path.dirname(out_dir.rstrip("/")) or "reports")
    return html_path, json_path
