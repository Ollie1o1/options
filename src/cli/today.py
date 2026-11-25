"""`opt today` — 10-second Morning Action Brief."""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List

import click
from rich.panel import Panel

from src.own_picks import list_equity_picks
from src.paths import repo_path
from src.radar import load_cached_radar, run_radar
from .render import make_console, render_core_allocation, render_radar_table, render_ribbon


def _open_paper_count(db_path: str = "paper_trades.db") -> int:
    full = repo_path(db_path)
    if not os.path.exists(full):
        return 0
    try:
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        try:
            row = con.execute("SELECT COUNT(*) FROM trades WHERE status = 'OPEN'").fetchone()
            return int(row[0]) if row else 0
        finally:
            con.close()
    except Exception:
        return 0


def _ops_health_summary() -> str:
    try:
        import shutil

        free_gb = shutil.disk_usage(repo_path(".")).free / (1024 ** 3)
        tag = "OK" if free_gb >= 15 else ("WARN" if free_gb >= 8 else "LOW DISK")
        return f"Disk {free_gb:.1f} GB free [{tag}] · Live execution: OFF"
    except Exception:
        return "Health check unavailable"


@click.command("today")
@click.option("--refresh", is_flag=True, help="Re-run local scanners instead of reading cache.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_context
def today_cmd(ctx: click.Context, refresh: bool, as_json: bool) -> None:
    """10-second Morning Action Brief: regime, core TFSA weights, open book & top 5 setups."""
    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    cache_path = (ctx.obj or {}).get("cache_path", "data/radar_cache.json")
    ohlcv_db = (ctx.obj or {}).get("ohlcv_db", "data/equity_ohlcv.db")
    own_picks_db = (ctx.obj or {}).get("own_picks_db", "data/own_picks.db")

    snap = None if refresh else load_cached_radar(cache_path)
    if snap is None:
        snap = run_radar(ohlcv_db=ohlcv_db, cache_path=cache_path)

    top5 = snap.opportunities[:5]
    open_eq = list_equity_picks(status="OPEN", db_path=own_picks_db)
    open_opt_n = _open_paper_count()
    health_line = _ops_health_summary()

    if use_json:
        payload: Dict[str, Any] = {
            "snapshot": snap.to_dict(),
            "top5": [o.to_dict() for o in top5],
            "open_equity_picks": len(open_eq),
            "open_paper_options": open_opt_n,
            "health": health_line,
        }
        click.echo(json.dumps(payload, indent=2))
        return

    console = make_console(no_color=no_color)
    from .guide import plain_english_action_plan

    console.print(Panel(render_ribbon(snap), title="TODAY · DAILY TRADING DESK", border_style="#82aad2"))
    console.print(
        Panel(
            "\n".join(plain_english_action_plan(snap))
            + "\n\n[dim]Need step-by-step guidance? Run `opt guide` (interactive menu) or `opt` (full dashboard).[/dim]",
            title="WHAT TO DO TODAY (Plain English)",
            border_style="#5ec98d",
        )
    )
    if snap.core_allocation:
        console.print(render_core_allocation(snap))
        console.print()
    if top5:
        console.print(render_radar_table(top5, title="Top 5 Radar Setups (Tier -> Confluence -> Lowest Friction)"))
    else:
        console.print("[dim]No active setups on Radar right now — stay in Core TFSA allocation.[/dim]")
    console.print()
    console.print(
        f"[bold]Book:[/bold] {len(open_eq)} open stock/ETF picks · {open_opt_n} open paper options  |  "
        f"[dim]{health_line}[/dim]"
    )
    if snap.degraded_scanners:
        console.print(f"[#d6a452]Degraded scanners:[/#d6a452] {', '.join(snap.degraded_scanners)}")
