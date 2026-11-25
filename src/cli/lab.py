"""`opt lab` — Quarantined Research, Ops Health, Playbook Registry & Real-Fill Tracker."""
from __future__ import annotations

import json
import os
import shutil

import click
from rich.table import Table

from src.fill_experiment import fill_rate_summary, log_real_fill
from src.paths import repo_path
from src.radar.playbooks import list_playbooks
from .render import TIER_STYLE, make_console


@click.group("lab")
def lab_group() -> None:
    """Quarantined ops, playbook audit registry, forward readouts & real-fill harness."""


@lab_group.command("health")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def lab_health(ctx: click.Context, as_json: bool) -> None:
    """Check disk space, local DB freshness, and scheduled forward readout dates."""
    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))

    free_gb = round(shutil.disk_usage(repo_path(".")).free / (1024 ** 3), 2)
    dbs = [
        "data/equity_ohlcv.db",
        "data/dolt_options.db",
        "data/estimates_archive.db",
        "data/chain_archive.db",
        "data/own_picks.db",
        "data/fills.db",
        "paper_trades.db",
    ]
    db_info = []
    for rel in dbs:
        p = repo_path(rel)
        exists = os.path.exists(p)
        mb = round(os.path.getsize(p) / (1024 ** 2), 2) if exists else 0.0
        db_info.append({"path": rel, "exists": exists, "size_mb": mb})

    readouts = [
        {"date": "2027-01-15", "name": "Estimate Revisions Arm B", "cmd": "python -m scripts.estimate_revisions_test"},
        {"date": "2027-01-15", "name": "Bull Put §2 Forward", "cmd": "python -m scripts.delta_matched_benchmark"},
        {"date": "2027-01-20", "name": "B1-F ETF Bear Call Forward", "cmd": "python -m scripts.bear_call_absolute --forward"},
        {"date": "2027-03-15", "name": "E2 Earnings Forward", "cmd": "python -m scripts.earnings_premium_dolt --forward"},
        {"date": "2027-05-25", "name": "Discover lens forward read (26 weeks)", "cmd": "opt discover --forward"},
    ]
    payload = {"disk_free_gb": free_gb, "databases": db_info, "pending_readouts": readouts}
    if use_json:
        click.echo(json.dumps(payload, indent=2))
        return

    console = make_console(no_color=no_color)
    console.print(f"[bold]Disk Free:[/bold] {free_gb:.2f} GB")
    tbl = Table(title="Local Databases", pad_edge=False)
    tbl.add_column("Database")
    tbl.add_column("Status")
    tbl.add_column("Size (MB)", justify="right")
    for d in db_info:
        tbl.add_row(d["path"], "OK" if d["exists"] else "missing", f"{d['size_mb']:.2f}")
    console.print(tbl)

    rtbl = Table(title="Scheduled Forward Readouts", pad_edge=False)
    rtbl.add_column("Unlock Date")
    rtbl.add_column("Study")
    rtbl.add_column("Command")
    for r in readouts:
        rtbl.add_row(r["date"], r["name"], r["cmd"])
    console.print(rtbl)


@lab_group.command("playbooks")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def lab_playbooks(ctx: click.Context, as_json: bool) -> None:
    """List every playbook in configs/playbooks.json including CLOSED studies."""
    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    pbs = list_playbooks(include_closed=True)
    if use_json:
        click.echo(json.dumps([p.__dict__ for p in pbs], indent=2))
        return
    console = make_console(no_color=no_color)
    tbl = Table(title="Playbook Evidence Registry (configs/playbooks.json)", pad_edge=False)
    tbl.add_column("Tier")
    tbl.add_column("ID", style="bold")
    tbl.add_column("Asset")
    tbl.add_column("Horizon")
    tbl.add_column("Result Doc")
    tbl.add_column("Notes")
    for p in pbs:
        style = TIER_STYLE.get(p.evidence_tier, "")
        tbl.add_row(f"[{style}]{p.evidence_tier}[/{style}]", p.id, p.asset_class, p.horizon_label, p.result or "—", p.notes[:68])
    console.print(tbl)


@lab_group.command("fills")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def lab_fills(ctx: click.Context, as_json: bool) -> None:
    """Summarize empirical limit-order fill rates & realized concession `k` from data/fills.db."""
    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    fills_db = (ctx.obj or {}).get("fills_db", "data/fills.db")
    from src.fill_experiment import o1_fill_verdict, o1_progress_line

    summary = fill_rate_summary(db_path=fills_db)
    o1 = o1_fill_verdict(fills_db)
    if use_json:
        click.echo(json.dumps({**summary, "o1": o1}, indent=2))
        return
    console = make_console(no_color=no_color)
    console.print(f"[bold]{o1_progress_line(o1)}[/bold]")
    console.print(f"[dim]{o1['reason']} · rule: {o1['prereg']} · log with a structure starting 'O1'[/dim]")
    console.print(f"[bold]Real-Fill Experiment (all structures):[/bold] {summary['total_filled']}/{summary['total_attempts']} filled")
    if summary["groups"]:
        tbl = Table(pad_edge=False)
        tbl.add_column("Broker")
        tbl.add_column("Structure")
        tbl.add_column("Filled/Att", justify="right")
        tbl.add_column("Fill Rate", justify="right")
        tbl.add_column("Mean Realized k", justify="right")
        for g in summary["groups"]:
            rk = f"{g['mean_realized_k']:.2f}" if g["mean_realized_k"] is not None else "—"
            tbl.add_row(g["broker"], g["structure"], f"{g['filled']}/{g['attempts']}", f"{g['fill_rate']*100:.0f}%", rk)
        console.print(tbl)


@lab_group.command("log-fill")
@click.option("--broker", type=click.Choice(["wealthsimple", "ibkr"], case_sensitive=False), required=True)
@click.option("--ticker", required=True)
@click.option("--structure", required=True)
@click.option("--side", type=click.Choice(["BUY", "SELL"], case_sensitive=False), required=True)
@click.option("--bid", type=float, required=True)
@click.option("--ask", type=float, required=True)
@click.option("--limit", "limit_posted", type=float, required=True)
@click.option("--filled/--unfilled", default=True)
@click.option("--fill-price", type=float, default=None)
@click.option("--wait-s", type=int, default=None)
@click.option("--note", default="")
@click.pass_context
def lab_log_fill(
    ctx: click.Context,
    broker: str,
    ticker: str,
    structure: str,
    side: str,
    bid: float,
    ask: float,
    limit_posted: float,
    filled: bool,
    fill_price: float,
    wait_s: int,
    note: str,
) -> None:
    """Log one real broker limit-order fill attempt into data/fills.db."""
    fills_db = (ctx.obj or {}).get("fills_db", "data/fills.db")
    fid = log_real_fill(
        broker=broker,
        ticker=ticker,
        structure=structure,
        side=side,
        bid=bid,
        ask=ask,
        limit_posted=limit_posted,
        filled=filled,
        fill_price=fill_price,
        wait_seconds=wait_s,
        notes=note,
        db_path=fills_db,
    )
    click.echo(f"✓ Logged fill attempt #{fid} ({broker} {ticker.upper()} {structure}, filled={filled})")
