"""Root Click entrypoint for the `opt` Daily Trading Desk CLI."""
from __future__ import annotations

import sys

import click

from src.paths import load_project_env
from .alpha import alpha_group
from .book import book_cmd, close_cmd, log_cmd
from .check import check_cmd
from .discover import card_cmd, discover_cmd
from .express import express_cmd
from .guide import guide_cmd, refresh_cmd
from .lab import lab_group
from .morning_ritual import morning_ritual_cmd
from .notify import notify_cmd
from .profile import profile_cmd
from .radar import radar_cmd
from .rebalance import rebalance_cmd
from .today import today_cmd


@click.group(invoke_without_command=True)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON where supported.")
@click.option("--no-color", is_flag=True, help="Disable ANSI color output.")
@click.pass_context
def cli(ctx: click.Context, as_json: bool, no_color: bool) -> None:
    """opt — Daily Stock + Option Trading Desk (`opt` launches the TUI; `opt today` prints the brief)."""
    load_project_env()
    ctx.ensure_object(dict)
    ctx.obj["json"] = as_json
    ctx.obj["no_color"] = no_color
    if ctx.invoked_subcommand is None:
        if not as_json and sys.stdin.isatty() and sys.stdout.isatty():
            try:
                from src.tui.app import run_desk

                run_desk()
                return
            except ImportError:
                pass
        ctx.invoke(today_cmd, refresh=False, as_json=as_json)


@cli.command("desk")
@click.option("--tab", type=click.Choice(["today", "radar", "workbench", "book", "lab", "discover", "options"]), default="today")
@click.option("--offline", is_flag=True, help="Start in cache-only offline mode.")
def desk_cmd(tab: str, offline: bool) -> None:
    """Launch the interactive 7-tab Textual Daily Trading Cockpit."""
    try:
        from src.tui.app import run_desk

        run_desk(initial_tab=tab, offline=offline)
    except ImportError as exc:
        click.echo(f"Textual TUI unavailable ({exc}). Run `opt today` or install textual.", err=True)
        raise SystemExit(1)


@cli.command("web")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host for the local web cockpit.")
@click.option("--port", type=int, default=8000, show_default=True, help="Bind port for the local web cockpit.")
@click.option("--no-open", is_flag=True, help="Start server without opening a browser tab.")
def web_cmd(host: str, port: int, no_open: bool) -> None:
    """Launch the zero-reload Bloomberg × Amber CRT local Web Cockpit (`/desk`)."""
    url = f"http://{host}:{port}/desk"
    click.echo(f"Launching Bloomberg CRT Web Cockpit at {url}  (Ctrl+C to stop)")
    if not no_open:
        import threading
        import time
        import webbrowser

        def _open() -> None:
            time.sleep(0.7)
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=_open, daemon=True).start()

    try:
        import fastapi  # noqa: F401
        import uvicorn

        uvicorn.run("src.api:app", host=host, port=port, log_level="warning")
    except ImportError:
        from src.web_terminal import serve_web_terminal

        serve_web_terminal(host=host, port=port)


@cli.command("commands")
@click.pass_context
def commands_cmd(ctx: click.Context) -> None:
    """Print the complete categorized cheatsheet of `opt` commands, TUI keys & `:` mnemonics."""
    from rich.panel import Panel
    from rich.table import Table

    from .render import accent_color, make_console

    no_color = bool((ctx.obj or {}).get("no_color"))
    console = make_console(no_color=no_color)
    acc = accent_color()

    sections = [
        (
            "Daily Cockpits & Guides",
            [
                (
                "opt",
                "Launch 7-tab Bloomberg × Amber CRT Textual TUI (incl. [7] OPTIONS Hub)",
            ),
            (
                "opt desk --tab options",
                "Jump straight into [7] OPTIONS (Scanner · Chain Matrix · Flow/Vol Intel)",
            ),
                ("opt web", "Launch zero-reload Live Web Cockpit at http://127.0.0.1:8000/desk"),
                ("opt morning-ritual", "60-second linear 5-step Morning Checklist (or press `m` in TUI)"),
                ("opt refresh", "Update local prices + FRED credit/rates + Radar cache"),
                ("opt today", "10-second Morning Action Brief (regime, TFSA Core, top setups)"),
                ("opt guide", "Zero-memorization step-by-step plain-English coach"),
            ],
        ),
        (
            "Wealthsimple TFSA Core Loop (PROVEN: VM_OUTLOOK_TOP3 + SGOV)",
            [
                ("opt profile --tfsa 10000 --cash-etf SGOV --room-cad 7000", "Save TFSA balance, cash ETF, risk & CRA contribution room"),
                ("opt profile --watch-downloads / --import-ws <csv>", "Auto-find & idempotently import Wealthsimple Activity CSV"),
                ("opt rebalance [--apply]", "Compute exact fractional BUY/SELL orders (--apply saves shares)"),
            ],
        ),
        (
            "Discovery, Research & Trade Planning",
            [
                ("opt discover [--sector XLE | --lens opt_skew]", "Top-down ① Market → ② Sectors → ③ Labelled Lenses"),
                ("opt discover --ledger", "Inspect NULL_LEDGER of ideas tested here that failed"),
                ("opt card NVDA", "④ Evidence Card: verdict, peers, options odds & provenance"),
                ("opt radar [--clean | --live-options]", "Unified Stock + Option Opportunity Radar"),
                ("opt check AAPL", "1-screen Ticker Dossier with Braille 60d tape & levels"),
                ("opt express AAPL --dir bull --target +8%", "Shares vs Options Workbench + Walk-Away Limit Ladders"),
            ],
        ),
        (
            "Book & Position Lifecycle",
            [
                ("opt book", "Open/closed stock picks, live P&L, exit alerts & scorecard"),
                ("opt log AAPL --price 210 --shares 5", "Log a manual stock/ETF pick into data/own_picks.db"),
                ("opt close <ID|TICKER> --price 224.50 --tag PLAN_WORKED", "Close an open pick with 10s Post-Mortem tag & realized P&L"),
            ],
        ),
        (
            "Phase 3 Alpha & Alt-Data Labs (`opt alpha` — WATCHLIST / CONTEXT)",
            [
                ("opt alpha skew", "Option-Implied CW IV Spread & Put Smirk -> trade in TFSA shares"),
                ("opt alpha gex SPY", "Dealer Net GEX, Gamma Flip strike & Call/Put OI Walls"),
                ("opt alpha edgar [--refresh]", "SEC EDGAR Form 10, 13D, 8-K Guidance, 10-K Supply Chain & Buybacks"),
                ("opt alpha archive-occ / occ-uoa", "OCC daily whole-market Volume/OI archiver & UOA surge scanner"),
                ("opt alpha macro [--refresh]", "4-pt VIX curve, FRED Fed Net Liquidity, CFTC COT & Polymarket"),
                ("opt alpha premortem NVDA [--ai]", "Pre-Mortem Devil's Advocate bear case & kill-switches"),
                ("opt alpha archive-chains --limit 60", "Snapshot 60 liquid tickers from free CBOE chains"),
            ],
        ),
        (
            "Alerts & Mission Lab",
            [
                ("opt notify --morning [--dry-run]", "Push daily Morning Dispatch rich embed to Discord"),
                ("opt notify --check / --pill", "Signal-only Discord + single-slot macOS alert / menu-bar pill"),
                ("opt notify --test-macos / --test-discord", "Send a quick test alert to verify setup"),
                ("opt lab health | playbooks | fills", "Per-pipe freshness LEDs, strategy registry & Test O1 fill k"),
                ("opt lab cleanup [--confirm]", "Preview (or reclaim) disk space from CLOSED studies"),
            ],
        ),
    ]

    for title, rows in sections:
        tbl = Table(title=title, show_header=False, box=None, pad_edge=False)
        tbl.add_column("Command", style=f"bold {acc}", width=44)
        tbl.add_column("Description")
        for cmd, desc in rows:
            tbl.add_row(cmd, desc)
        console.print(tbl)
        console.print()

    console.print(
        Panel(
            "[bold]TUI Tabs:[/] [1] Today  [2] Radar  [3] Workbench  [4] Book  [5] Lab  [6] Discover  [7] Options\n"
            "[bold]TUI Keys:[/] [:] Cmd Bar · [c] Chain Matrix · [p] Paper Opt · [h] 2D Heatmap · [,/.] Time ±5d · "
            "[{/}] IV ±5% · [b] Rebalance · [l] Log Pick · [x] Close · [f] Fill · [o] Live Opts · [t] Theme · [?] Help\n"
            "[bold]: Mnemonics:[/] NVDA · CHAIN SPY · SCAN [bull_put] · FLOW · VOL · SURF NVDA · NVDA WB · "
            "REB · SKEW · GEX SPY · MACRO · PM NVDA\n"
            "[dim]Full reference: docs/COMMANDS.md  ·  Global flags: opt --json / opt --no-color <cmd>[/dim]",
            title="INTERACTIVE TUI & BLOOMBERG COMMAND BAR",
            border_style=acc,
        )
    )


cli.add_command(guide_cmd)
cli.add_command(morning_ritual_cmd)
cli.add_command(refresh_cmd)
cli.add_command(today_cmd)
cli.add_command(radar_cmd)
cli.add_command(discover_cmd)
cli.add_command(card_cmd)
cli.add_command(check_cmd)
cli.add_command(express_cmd)
cli.add_command(book_cmd)
cli.add_command(log_cmd)
cli.add_command(close_cmd)
cli.add_command(rebalance_cmd)
cli.add_command(profile_cmd)
cli.add_command(alpha_group)
cli.add_command(lab_group)
cli.add_command(notify_cmd)


if __name__ == "__main__":
    cli()
