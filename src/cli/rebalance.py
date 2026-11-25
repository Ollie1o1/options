"""`opt rebalance` — TFSA Core Portfolio Rebalance Calculator (`VM_OUTLOOK_TOP3`)."""
from __future__ import annotations

import json
from typing import Dict, Optional

import click
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.equity_playbooks.rebalance import compute_tfsa_rebalance, parse_holdings_spec
from src.own_picks import list_equity_picks
from src.radar import load_cached_radar, run_radar
from src.radar.scanners import load_local_ohlcv_safe
from .render import make_console


def _load_open_vm_holdings(own_picks_db: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in list_equity_picks(status="OPEN", playbook_id="vol_managed_outlook", db_path=own_picks_db):
        tk = str(p.get("ticker") or "").upper()
        sh = float(p.get("shares") or 0.0)
        if tk and sh > 0:
            out[tk] = out.get(tk, 0.0) + sh
    return out


@click.command("rebalance")
@click.option("--account", type=float, required=True, help="Total TFSA core sleeve value in USD (e.g. 5000).")
@click.option(
    "--current",
    default="",
    help="Current ETF shares as TICKER:SHARES pairs (e.g. XLK:10,GLD:5). Defaults to open own_picks.",
)
@click.option("--fractional", is_flag=True, help="Allow fractional shares (Wealthsimple supports fractional ETFs).")
@click.option("--drift-threshold", type=float, default=5.0, help="Min weight drift % to recommend rebalancing.")
@click.option("--refresh", is_flag=True, help="Re-run Radar before calculating rebalance.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_context
def rebalance_cmd(
    ctx: click.Context,
    account: float,
    current: str,
    fractional: bool,
    drift_threshold: float,
    refresh: bool,
    as_json: bool,
) -> None:
    """Compute exact BUY/SELL share orders for your VALIDATED Core TFSA ETF sleeve (VM_OUTLOOK_TOP3)."""
    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    ohlcv_db = (ctx.obj or {}).get("ohlcv_db", "data/equity_ohlcv.db")
    cache_path = (ctx.obj or {}).get("cache_path", "data/radar_cache.json")
    own_picks_db = (ctx.obj or {}).get("own_picks_db", "data/own_picks.db")

    snap = None if refresh else load_cached_radar(cache_path)
    if snap is None:
        snap = run_radar(ohlcv_db=ohlcv_db, cache_path=cache_path)

    cur_shares = parse_holdings_spec(current) if current.strip() else _load_open_vm_holdings(own_picks_db)
    ohlcv = load_local_ohlcv_safe(ohlcv_db)
    spot_lookup = {sym: float(cs[-1]) for sym, (_ds, cs, _vs) in ohlcv.items() if len(cs) > 0}

    plan = compute_tfsa_rebalance(
        snap.core_allocation,
        account_usd=account,
        current_shares=cur_shares,
        spot_lookup=spot_lookup,
        fractional=fractional,
        drift_threshold_pct=drift_threshold,
    )

    if use_json:
        click.echo(json.dumps(plan.to_dict(), indent=2))
        return

    console = make_console(no_color=no_color)
    verdict = (
        f"[bold #5ec98d]REBALANCE RECOMMENDED[/bold #5ec98d] (max drift {plan.max_drift_pct:.1f}% >= {plan.drift_threshold_pct:.1f}%)"
        if plan.rebalance_recommended
        else f"[bold #82aad2]HOLD / ON TARGET[/bold #82aad2] (max drift {plan.max_drift_pct:.1f}% < {plan.drift_threshold_pct:.1f}%)"
    )
    rv_str = f" · SPY RV20 {plan.spy_rv20*100:.1f}%" if plan.spy_rv20 is not None else ""
    console.print(
        Panel(
            f"{verdict}\n"
            f"Account: ${plan.account_usd:,.2f} USD · Vol Scale: {plan.vol_scale*100:.0f}%{rv_str} · "
            f"Cash/SGOV After: ${plan.estimated_cash_after_usd:,.2f}",
            title="TFSA CORE REBALANCE PLAN (VALIDATED: VM_OUTLOOK_TOP3)",
            border_style="#5ec98d" if plan.rebalance_recommended else "#82aad2",
        )
    )

    tbl = Table(pad_edge=False)
    tbl.add_column("Action", width=8)
    tbl.add_column("Ticker", style="bold")
    tbl.add_column("Target%", justify="right")
    tbl.add_column("Now%", justify="right")
    tbl.add_column("Drift", justify="right")
    tbl.add_column("Spot", justify="right")
    tbl.add_column("Now Sh", justify="right")
    tbl.add_column("Tgt Sh", justify="right")
    tbl.add_column("Order", justify="right")
    tbl.add_column("Delta $", justify="right")

    for o in plan.orders:
        act_style = (
            "bold #5ec98d"
            if o.action == "BUY"
            else ("bold #e06c75" if o.action == "SELL" else ("dim #82aad2" if o.action == "RESERVE" else "dim"))
        )
        ord_txt = (
            f"{o.delta_shares:+.4g} sh"
            if o.action in ("BUY", "SELL")
            else ("Keep cash/SGOV" if o.action == "RESERVE" else "—")
        )
        tbl.add_row(
            Text(o.action, style=act_style),
            o.ticker,
            f"{o.target_weight_pct:.1f}%",
            f"{o.current_weight_pct:.1f}%",
            f"{o.drift_pct:+.1f}%",
            f"${o.spot:.2f}" if o.ticker != "CASH/SGOV" else "—",
            f"{o.current_shares:g}",
            f"{o.target_shares:g}",
            ord_txt,
            f"${o.delta_usd:+,.2f}" if o.action != "HOLD" else "$0.00",
        )
    console.print(tbl)
