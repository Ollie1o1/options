"""60-Second Morning Ritual Mode (`opt morning-ritual`, TUI `m`, Web Cockpit Tab 0).

Collapses the entire desk into ONE linear 5-step morning checklist so you never bounce
between 6 tabs wondering what actually matters today:
  1. Market Mood & Macro Plumbing (SPY, VIX 4-pt curve, Fed Net Liquidity)
  2. Open Book Exit Alerts (STOP HIT, TARGET HIT, EXIT >SMA5)
  3. Wealthsimple TFSA Core Check (Monthly VM_OUTLOOK_TOP3 + SGOV rebalance & CRA safety)
  4. Today's Clean Actionable Setups (PROVEN / LEAD with zero risk flags)
  5. Pre-Mortem Gate & Next Action
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import click
from rich.panel import Panel

from src.alpha.macro_plumbing import load_macro_plumbing
from src.cli.guide import plain_english_regime
from src.cli.render import accent_color, make_console
from src.equity_playbooks.rebalance import compute_tfsa_rebalance
from src.profile import estimate_cash_yield_annual, load_profile, rebalance_due_status
from src.tfsa_tax import compute_tfsa_safety_report


def build_morning_ritual(
    *,
    provider: Optional[Any] = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    """Assemble the deterministic 5-step Morning Ritual checklist (100% offline-safe)."""
    from src.tui.state import DeskDataProvider

    prov = provider or DeskDataProvider()
    snap = prov.get_radar(refresh=refresh)
    book = prov.get_book()
    prof = load_profile()
    macro = load_macro_plumbing()
    _y_ann, y_lbl = estimate_cash_yield_annual()

    headline, advice = plain_english_regime(snap)
    vc = macro.vix_curve or {}
    vix_curve_str = " / ".join(
        f"{lbl} {vc[k]:.1f}"
        for k, lbl in (("vix9d", "9d"), ("vix", "VIX"), ("vix3m", "3m"), ("vix6m", "6m"))
        if vc.get(k) is not None
    )
    fed_str = (
        f"Fed Net Liq ${macro.fed_net_liquidity_b:,.0f}B"
        + (f" (4w {macro.fed_liq_chg_4w_b:+.0f}B)" if macro.fed_liq_chg_4w_b is not None else "")
        if macro.fed_net_liquidity_b is not None
        else "Fed Net Liq: cached/pending"
    )

    # Step 2: Open Book Exits
    exit_alerts = list(book.get("exit_alerts") or [])
    open_picks = [p for p in (book.get("equity_picks") or []) if str(p.get("status") or "").upper() == "OPEN"]

    # Step 3: TFSA Core Rebalance + CRA Safety
    rebal_due = False
    rebal_msg = "Core allocation unavailable"
    orders_summary: List[str] = []
    if snap.core_allocation:
        try:
            plan = compute_tfsa_rebalance(
                snap.core_allocation,
                account_usd=prof.tfsa_usd,
                current_shares=prof.core_holdings,
                fractional=prof.fractional_shares,
                drift_threshold_pct=prof.drift_threshold_pct,
            )
            rebal_due, rebal_msg = rebalance_due_status(
                prof,
                max_drift_pct=plan.max_drift_pct,
                rebalance_recommended=plan.rebalance_recommended,
            )
            for o in plan.orders:
                if o.action in ("BUY", "SELL"):
                    orders_summary.append(f"{o.action} {o.delta_shares:+.2f} {o.ticker} (${o.delta_usd:+,.0f})")
                elif o.action == "RESERVE":
                    orders_summary.append(f"HOLD ${o.target_usd:,.0f} in {prof.cash_etf} ({y_lbl})")
        except Exception as exc:
            rebal_msg = f"Rebalance check error: {exc}"

    safety = compute_tfsa_safety_report(
        profile=prof,
        core_allocation=snap.core_allocation,
        equity_picks=book.get("equity_picks") or [],
    )

    # Step 4: Clean actionable setups (PROVEN / LEAD with zero flags)
    clean_setups = [
        o for o in snap.opportunities
        if o.evidence_tier in ("VALIDATED", "LEAD")
        and not o.risk_flags
        and (prof.account_type != "tfsa" or o.asset_class != "option" or o.playbook_id == "covered_call_holdings")
    ]
    top_clean = [
        {
            "ticker": o.ticker,
            "tier": o.evidence_tier,
            "playbook": o.playbook_label,
            "direction": o.direction,
            "summary": o.setup_summary,
        }
        for o in clean_setups[:5]
    ]

    # Step 5: Concrete Next Action
    if exit_alerts:
        next_action = f"ACT FIRST: Handle {len(exit_alerts)} open position exit alert(s) before opening anything new."
        status_grade = "ACTION_REQUIRED"
    elif rebal_due:
        next_action = f"REBALANCE DUE: Execute your monthly TFSA Core orders (`opt rebalance --apply`), then review clean setups."
        status_grade = "REBALANCE_DUE"
    elif top_clean:
        next_action = (
            f"SELECTIVE GREEN LIGHT: Run `opt alpha premortem {top_clean[0]['ticker']}` before sizing any new entry."
        )
        status_grade = "OPPORTUNITY"
    else:
        next_action = "STAND DOWN: No exits, no rebalance due, and no clean setups. Let VM_OUTLOOK_TOP3 + SGOV compound."
        status_grade = "DO_NOTHING_DAY"

    steps = [
        {
            "step": 1,
            "title": "Market Mood & Macro Plumbing",
            "status": snap.regime_posture,
            "headline": headline,
            "detail": f"{advice} · VIX Curve: {vix_curve_str or 'cached'} ({macro.vix_term_label}) · {fed_str}",
        },
        {
            "step": 2,
            "title": "Open Book Exit Checks",
            "status": "ALERT" if exit_alerts else "CLEAR",
            "headline": f"{len(exit_alerts)} exit trigger(s) across {len(open_picks)} open stock/ETF position(s)",
            "detail": " | ".join(exit_alerts) if exit_alerts else "All open positions inside stop/target bounds.",
        },
        {
            "step": 3,
            "title": "Wealthsimple TFSA Core & CRA Guard",
            "status": "DUE" if rebal_due else "ON_TARGET",
            "headline": rebal_msg,
            "detail": (
                ("Orders: " + ", ".join(orders_summary) + " · ") if orders_summary else ""
            ) + f"CRA: {safety.cra_status} ({safety.trades_ytd} YTD trades) · 15% US Div Drag ~${safety.total_tax_drag_usd_yr:.0f}/yr",
        },
        {
            "step": 4,
            "title": "Today's Clean Actionable Setups",
            "status": f"{len(top_clean)} CLEAN" if top_clean else "NONE",
            "headline": ", ".join(f"{x['ticker']} ({x['playbook']})" for x in top_clean) if top_clean else "0 clean setups firing today",
            "detail": (
                "Filtered strictly to PROVEN/LEAD playbooks with zero landmine flags and CRA TFSA compliance."
            ),
        },
        {
            "step": 5,
            "title": "Pre-Mortem Gate & Decision",
            "status": status_grade,
            "headline": next_action,
            "detail": "Rule: Never buy a satellite pick without checking its Pre-Mortem kill-switches (`:PM TICKER`) and logging a stop.",
        },
    ]
    return {
        "generated_at": snap.generated_at,
        "status_grade": status_grade,
        "next_action": next_action,
        "steps": steps,
        "clean_setups": top_clean,
        "exit_alerts": exit_alerts,
        "tfsa_safety": safety.to_dict(),
    }


def format_morning_ritual_markup(ritual: Dict[str, Any]) -> str:
    """Rich/Textual markup string for the 60-Second Morning Ritual modal."""
    lines = [
        f"[bold]60-SECOND MORNING RITUAL[/]  ·  [bold #5ec98d]{ritual.get('status_grade', '')}[/]",
        "",
    ]
    for st in ritual.get("steps") or []:
        badge = str(st.get("status") or "")
        col = "#e06c75" if badge in ("ALERT", "RISK_OFF", "ACTION_REQUIRED") else (
            "#d6a452" if badge in ("DUE", "REBALANCE_DUE", "NEUTRAL") else "#5ec98d"
        )
        lines.append(f"[bold]{st['step']}. {st['title']}[/]  [{col}][{badge}][/]")
        lines.append(f"   {st['headline']}")
        lines.append(f"   [dim]{st['detail']}[/dim]")
        lines.append("")
    return "\n".join(lines).rstrip()


@click.command("morning-ritual")
@click.option("--refresh", is_flag=True, help="Rebuild Radar before generating ritual.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON output.")
@click.pass_context
def morning_ritual_cmd(ctx: click.Context, refresh: bool, as_json: bool) -> None:
    """60-second linear Morning Ritual: Regime -> Exits -> TFSA Rebalance -> Clean Setups -> Decision."""
    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    ritual = build_morning_ritual(refresh=refresh)
    if use_json:
        click.echo(json.dumps(ritual, indent=2))
        return
    console = make_console(no_color=no_color)
    console.print(
        Panel(
            format_morning_ritual_markup(ritual),
            title="DAILY DESK // 60-SECOND MORNING RITUAL",
            border_style=accent_color(),
        )
    )
