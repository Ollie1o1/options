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
from src.profile import (
    DEFAULT_PROFILE_PATH,
    apply_rebalance_to_profile,
    estimate_cash_yield_annual,
    load_profile,
    rebalance_due_status,
)
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
@click.option("--account", type=float, default=None, help="Total TFSA core sleeve value in USD (defaults to saved profile).")
@click.option(
    "--current",
    default="",
    help="Current ETF shares as TICKER:SHARES pairs (defaults to saved profile holdings or open own_picks).",
)
@click.option("--fractional/--whole", default=None, help="Allow fractional shares (Wealthsimple default: fractional).")
@click.option("--drift-threshold", type=float, default=None, help="Min weight drift % to recommend rebalancing.")
@click.option("--apply", "apply_save", is_flag=True, help="Save resulting target shares & today's date to data/tfsa_profile.json.")
@click.option("--refresh", is_flag=True, help="Re-run Radar before calculating rebalance.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_context
def rebalance_cmd(
    ctx: click.Context,
    account: Optional[float],
    current: str,
    fractional: Optional[bool],
    drift_threshold: Optional[float],
    apply_save: bool,
    refresh: bool,
    as_json: bool,
) -> None:
    """Compute exact BUY/SELL share orders for your VALIDATED Core TFSA ETF sleeve (VM_OUTLOOK_TOP3)."""
    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    ohlcv_db = (ctx.obj or {}).get("ohlcv_db", "data/equity_ohlcv.db")
    cache_path = (ctx.obj or {}).get("cache_path", "data/radar_cache.json")
    own_picks_db = (ctx.obj or {}).get("own_picks_db", "data/own_picks.db")
    profile_path = (ctx.obj or {}).get("profile_path", DEFAULT_PROFILE_PATH)

    prof = load_profile(profile_path)
    eff_account = float(account) if account is not None else prof.tfsa_usd
    eff_frac = prof.fractional_shares if fractional is None else bool(fractional)
    eff_drift = float(drift_threshold) if drift_threshold is not None else prof.drift_threshold_pct

    snap = None if refresh else load_cached_radar(cache_path)
    if snap is None:
        snap = run_radar(ohlcv_db=ohlcv_db, cache_path=cache_path)

    if current.strip():
        cur_shares = parse_holdings_spec(current)
    elif prof.core_holdings:
        cur_shares = dict(prof.core_holdings)
    else:
        cur_shares = _load_open_vm_holdings(own_picks_db)
    ohlcv = load_local_ohlcv_safe(ohlcv_db)
    spot_lookup = {sym: float(cs[-1]) for sym, (_ds, cs, _vs) in ohlcv.items() if len(cs) > 0}

    plan = compute_tfsa_rebalance(
        snap.core_allocation,
        account_usd=eff_account,
        current_shares=cur_shares,
        spot_lookup=spot_lookup,
        fractional=eff_frac,
        drift_threshold_pct=eff_drift,
    )
    y_ann, y_lbl = estimate_cash_yield_annual()
    est_yield_yr = round(plan.estimated_cash_after_usd * y_ann, 2)
    due, due_msg = rebalance_due_status(
        prof,
        max_drift_pct=plan.max_drift_pct,
        rebalance_recommended=plan.rebalance_recommended,
    )

    if apply_save:
        prof.tfsa_usd = eff_account
        apply_rebalance_to_profile(
            prof,
            {o.ticker: o.target_shares for o in plan.orders if o.action != "RESERVE"},
            path=profile_path,
        )

    if use_json:
        payload = plan.to_dict()
        payload["cash_etf"] = prof.cash_etf
        payload["est_cash_yield_usd_yr"] = est_yield_yr
        payload["due"] = due
        payload["due_message"] = due_msg
        payload["applied"] = apply_save
        click.echo(json.dumps(payload, indent=2))
        return

    console = make_console(no_color=no_color)
    verdict = (
        f"[bold #5ec98d]REBALANCE RECOMMENDED[/bold #5ec98d] ({due_msg})"
        if plan.rebalance_recommended
        else f"[bold #82aad2]HOLD / ON TARGET[/bold #82aad2] ({due_msg})"
    )
    rv_str = f" · Basket RV20 {plan.basket_rv20*100:.1f}%" if plan.basket_rv20 is not None else ""
    if plan.spy_rv20 is not None:
        rv_str += f" (SPY {plan.spy_rv20*100:.1f}%)"
    console.print(
        Panel(
            f"{verdict}\n"
            f"Account: ${plan.account_usd:,.2f} USD · Vol Scale: {plan.vol_scale*100:.0f}%{rv_str} · "
            f"{prof.cash_etf} Reserve: ${plan.estimated_cash_after_usd:,.2f} "
            f"(~${est_yield_yr:,.0f}/yr @ {y_lbl})"
            + ("\n[bold #5ec98d]✓ Saved target holdings to data/tfsa_profile.json.[/]" if apply_save else ""),
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
    console.print(
        "[dim]Execution timing (Study 2E): any liquid 10:00-15:55 ET limit fill matches close (±1.6 bps/mo); "
        "stay invested overnight (100%+ of VM_OUTLOOK_TOP3 return accrues overnight).[/dim]"
    )
