"""`opt lab` — Quarantined Research, Ops Health, Playbook Registry & Real-Fill Tracker."""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
from typing import Any, TextIO

import click
from rich.table import Table

from src.fill_experiment import fill_rate_summary, log_real_fill
from src.paths import repo_path
from src.radar.playbooks import list_playbooks
from .render import TIER_STYLE, make_console


@click.group("lab")
def lab_group() -> None:
    """Quarantined ops, playbook audit registry, forward readouts & real-fill harness."""


@lab_group.command("scan-audit")
@click.option("--ticker", "tickers", multiple=True, help="Limit the scan to one or more tickers.")
@click.option("--live", is_flag=True, help="Fetch Yahoo chains. Default: read local chain snapshots.")
@click.option("--out", type=click.File("w", encoding="utf-8", atomic=True),
              help="Save this scan and its coverage as JSON for paper research.")
@click.pass_context
def lab_scan_audit(ctx: click.Context, tickers: tuple[str, ...], live: bool, out: TextIO | None) -> None:
    """Inspect contract discovery coverage and capture the screen's decisions."""
    from src.notifications import filter_good_option_alerts
    from src.tui.options_engine import scan_options_desk

    obj = ctx.obj or {}
    coverage: dict[str, Any] = {}
    capture = out is not None or obj.get("json", False)
    inputs: list[dict[str, Any]] = []
    rows, note = scan_options_desk(
        symbols=tuple(dict.fromkeys(t.strip().upper() for t in tickers if t.strip())) or None,
        live=live, ohlcv_db=obj.get("ohlcv_db", "data/equity_ohlcv.db"), pulse_out=coverage,
        research_inputs=inputs if capture else None,
    )
    passed = {id(r) for r in filter_good_option_alerts(rows)}
    payload = {
        "schema_version": 1,
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "Paper research. Screen decisions and displayed quotes only. Fills are unmeasured.",
        "source": "yahoo_request" if live else "chain_archive",
        "status": note,
        "coverage": coverage.get("coverage", []),
        "rows": [{**row.to_dict(), "passed_alert_screen": id(row) in passed} for row in rows],
        "selector_inputs": inputs,
    }
    if capture:
        raw = json.dumps(payload, indent=2, allow_nan=False)
        if out is not None:
            click.echo(raw, file=out)
        if obj.get("json") and (out is None or out.name != "<stdout>"):
            click.echo(raw)
        if obj.get("json") or (out is not None and out.name == "<stdout>"):
            return

    click.echo(note)
    table = Table("Ticker", "Coverage", "Expirations read", "Two-sided quotes", "Candidates")
    for item in payload["coverage"]:
        table.add_row(item["ticker"], item["status"], str(item.get("checked", 0)),
                      str(item.get("two_sided", 0)), str(item["candidates"]))
    make_console(no_color=bool(obj.get("no_color"))).print(table)
    click.echo(f"{len(passed)}/{len(rows)} displayed candidates passed the alert screen. Fills are unmeasured.")


@lab_group.command("scan-replay")
@click.argument("snapshot", type=click.File("r", encoding="utf-8"))
def lab_scan_replay(snapshot: TextIO) -> None:
    """Compare the current generic selector with a saved scan on identical quotes."""
    from src.tui.options_engine import replay_option_scan

    try:
        saved = json.load(snapshot)
        if saved.get("schema_version") != 1 or not saved.get("selector_inputs"):
            raise ValueError("Expected a version 1 scan audit with captured selector inputs.")
        comparison = replay_option_scan(saved)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(comparison, indent=2, allow_nan=False))


# ── Selector vs liquidity-first baseline (docs/PREREG_SELECTOR_LIQUIDITY_BASELINE_20261126.md) ──

_STUDY_DB = "data/selector_study.db"


def _study_paths(f: Any) -> Any:
    f = click.option("--study-db", default=_STUDY_DB, show_default=True, help="Study database (research only).")(f)
    f = click.option("--chain-db", default="data/chain_archive.db", show_default=True)(f)
    f = click.option("--ohlcv-db", default="data/equity_ohlcv.db", show_default=True)(f)
    f = click.option("--today", default=None, help="Override today's date (YYYY-MM-DD, ET).")(f)
    return f


def _emit_study(ctx: click.Context, result: dict, as_json: bool) -> None:
    from src.selector_study import readout as _ro

    if as_json or (ctx.obj or {}).get("json"):
        click.echo(json.dumps(result, indent=2, allow_nan=False, default=str))
    elif "verdict" in result:
        click.echo(_ro.render(result))
    else:
        click.echo(json.dumps(result, indent=2, default=str))


@lab_group.group("study")
def lab_study() -> None:
    """Paper study: generic bull-put selector vs a matched liquidity-first baseline."""


@lab_study.command("capture")
@_study_paths
@click.option("--session", "sessions", multiple=True, help="Archived session(s). Default: every archived session.")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def lab_study_capture(ctx: click.Context, study_db: str, chain_db: str, ohlcv_db: str, today: str | None,
                      sessions: tuple[str, ...], as_json: bool) -> None:
    """Capture archived sessions and build both arms. Idempotent; retries missing symbols."""
    from src.selector_study import capture, db, workflow

    today = today or workflow._et_today()
    conn = db.connect(repo_path(study_db))
    try:
        todo = list(sessions) or capture.archive_sessions(chain_db)
        earn = _earnings_today()
        out = [capture.capture_session(conn, s, chain_db=chain_db, earnings=earn if s == today else None)
               for s in todo if s <= today]
    finally:
        conn.close()
    _emit_study(ctx, {"captures": out}, as_json)


def _earnings_today() -> dict:
    try:
        from src.tui.options_engine import _load_earnings_map
        return _load_earnings_map()
    except Exception:
        return {}


@lab_study.command("mark")
@_study_paths
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def lab_study_mark(ctx: click.Context, study_db: str, chain_db: str, ohlcv_db: str, today: str | None,
                   as_json: bool) -> None:
    """Record later archived quotes for every open study entry."""
    from src.selector_study import db, marks, workflow

    conn = db.connect(repo_path(study_db))
    try:
        out = marks.mark_entries(conn, chain_db=chain_db, ohlcv_db=ohlcv_db, today=today or workflow._et_today())
    finally:
        conn.close()
    _emit_study(ctx, {"marks": out}, as_json)


@lab_study.command("readout")
@_study_paths
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def lab_study_readout(ctx: click.Context, study_db: str, chain_db: str, ohlcv_db: str, today: str | None,
                      as_json: bool) -> None:
    """Coverage, matching, outcomes and the week-clustered paired difference."""
    from src.selector_study import db, readout, workflow

    if not os.path.exists(repo_path(study_db)):
        raise click.ClickException(f"No study database at {study_db}. Run `opt lab study run` first.")
    conn = db.connect(repo_path(study_db))
    try:
        result = readout.build(conn, today=today or workflow._et_today())
    finally:
        conn.close()
    _emit_study(ctx, result, as_json)


@lab_study.command("run")
@_study_paths
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def lab_study_run(ctx: click.Context, study_db: str, chain_db: str, ohlcv_db: str, today: str | None,
                  as_json: bool) -> None:
    """Capture every archived session, mark, and print the readout."""
    from src.selector_study import workflow

    today = today or workflow._et_today()
    out = workflow.run(study_db=repo_path(study_db), chain_db=chain_db, ohlcv_db=ohlcv_db, today=today,
                       earnings=_earnings_today())
    if as_json or (ctx.obj or {}).get("json"):
        click.echo(json.dumps(out, indent=2, allow_nan=False, default=str))
        return
    caps = [c for c in out["captures"] if not c.get("skipped")]
    click.echo(f"Captured {len(caps)} session(s); {len(out['captures']) - len(caps)} already frozen. "
               f"Marks written {out['marks']['written']} over {out['marks']['entries']} entries.")
    _emit_study(ctx, out["readout"], False)


@lab_study.command("history")
@click.option("--fetch", is_flag=True, help="Download missing weekly DoltHub chains first (resumable, slow).")
@click.option("--out-dir", default="data/selector_study_dolt", show_default=True)
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def lab_study_history(ctx: click.Context, fetch: bool, out_dir: str, as_json: bool) -> None:
    """Historical hold-to-expiry companion on DoltHub EOD chains (2019-2026)."""
    from src.selector_study import dolt_history

    if fetch:
        dolt_history.fetch()
    out = dolt_history.run(out_dir=repo_path(out_dir))
    if as_json or (ctx.obj or {}).get("json"):
        click.echo(json.dumps(out, indent=2, allow_nan=False, default=str))
        return
    click.echo(f"Converted {out['convert'].get('chains')} chains · settle {out['settle']}")
    _emit_study(ctx, out["readout"], False)


@lab_study.command("demo")
@click.option("--dir", "directory", required=True, type=click.Path(file_okay=False),
              help="Empty scratch folder for the synthetic archive and study database.")
@click.option("--sessions", default=85, show_default=True)
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def lab_study_demo(ctx: click.Context, directory: str, sessions: int, as_json: bool) -> None:
    """Run the full workflow on SYNTHETIC data. Never touches data/."""
    from src.selector_study import workflow

    if os.path.abspath(directory).startswith(os.path.abspath(repo_path("data"))):
        raise click.ClickException("Choose a folder outside data/ for synthetic files.")
    out = workflow.demo(directory, sessions=sessions)
    if as_json or (ctx.obj or {}).get("json"):
        click.echo(json.dumps(out, indent=2, allow_nan=False, default=str))
        return
    click.echo(f"Synthetic archive: {out['synthetic']['rows']} quotes, planted {out['synthetic']['planted']}")
    _emit_study(ctx, out["readout"], False)


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


CLOSED_STUDY_ARTIFACTS = (
    ("data/squeeze_prices.db", "CLOSED study: squeeze_equity raw closes (2026-11-24)"),
    ("data/squeeze_backtest.db", "CLOSED study: squeeze_equity FINRA SI panel (2026-11-24)"),
    ("data/squeeze_panel.pkl", "CLOSED study: squeeze_equity panel pickle"),
    ("data/squeeze_panel_partial.pkl", "CLOSED study: squeeze_equity partial pickle"),
    ("data/catalyst_pit.db", "CLOSED study: biotech_runup ClinicalTrials PIT cache (2026-11-24)"),
    ("data/paper_trades.db", "Empty 0-byte ghost file (real ledger is ./paper_trades.db)"),
)


@lab_group.command("cleanup")
@click.option("--confirm", is_flag=True, help="Actually delete listed closed-study caches (default is dry-run).")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def lab_cleanup(ctx: click.Context, confirm: bool, as_json: bool) -> None:
    """Inspect (or with --confirm, reclaim) disk space used by CLOSED research caches."""
    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    found = []
    total_mb = 0.0
    for rel, reason in CLOSED_STUDY_ARTIFACTS:
        p = repo_path(rel)
        if os.path.exists(p):
            mb = round(os.path.getsize(p) / (1024 ** 2), 2)
            # Also count -wal / -shm sidecars if present
            sidecars = [s for s in (f"{p}-wal", f"{p}-shm") if os.path.exists(s)]
            total_mb += mb
            deleted = False
            if confirm:
                os.remove(p)
                for sc in sidecars:
                    try:
                        os.remove(sc)
                    except OSError:
                        pass
                deleted = True
            found.append({"path": rel, "size_mb": mb, "reason": reason, "deleted": deleted})

    payload = {
        "dry_run": not confirm,
        "reclaimable_mb": round(total_mb, 2),
        "files": found,
    }
    if use_json:
        click.echo(json.dumps(payload, indent=2))
        return
    console = make_console(no_color=no_color)
    if not found:
        console.print("[#5ec98d]No closed-study cache bloat found.[/]")
        return
    tbl = Table(title=f"Closed-Study Cache Inspector ({'DELETED' if confirm else 'DRY RUN'})", pad_edge=False)
    tbl.add_column("File")
    tbl.add_column("Size (MB)", justify="right")
    tbl.add_column("Why Reclaimable")
    for item in found:
        tbl.add_row(item["path"], f"{item['size_mb']:.1f}", item["reason"])
    console.print(tbl)
    if confirm:
        console.print(f"[#5ec98d]✓ Reclaimed {total_mb:.1f} MB.[/]")
    else:
        console.print(f"[bold]Total reclaimable:[/bold] {total_mb:.1f} MB — run `opt lab cleanup --confirm` to delete.")
