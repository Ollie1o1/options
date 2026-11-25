"""Root Click entrypoint for the `opt` Daily Trading Desk CLI."""
from __future__ import annotations

import sys

import click

from .book import book_cmd, log_cmd
from .check import check_cmd
from .express import express_cmd
from .guide import guide_cmd, refresh_cmd
from .lab import lab_group
from .radar import radar_cmd
from .rebalance import rebalance_cmd
from .today import today_cmd


@click.group(invoke_without_command=True)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON where supported.")
@click.option("--no-color", is_flag=True, help="Disable ANSI color output.")
@click.pass_context
def cli(ctx: click.Context, as_json: bool, no_color: bool) -> None:
    """opt — Daily Stock + Option Trading Desk (`opt` launches the TUI; `opt today` prints the brief)."""
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
@click.option("--tab", type=click.Choice(["today", "radar", "workbench", "book", "lab"]), default="today")
@click.option("--offline", is_flag=True, help="Start in cache-only offline mode.")
def desk_cmd(tab: str, offline: bool) -> None:
    """Launch the interactive 5-tab Textual Daily Trading Cockpit."""
    try:
        from src.tui.app import run_desk

        run_desk(initial_tab=tab, offline=offline)
    except ImportError as exc:
        click.echo(f"Textual TUI unavailable ({exc}). Run `opt today` or install textual.", err=True)
        raise SystemExit(1)


cli.add_command(guide_cmd)
cli.add_command(refresh_cmd)
cli.add_command(today_cmd)
cli.add_command(radar_cmd)
cli.add_command(check_cmd)
cli.add_command(express_cmd)
cli.add_command(book_cmd)
cli.add_command(log_cmd)
cli.add_command(rebalance_cmd)
cli.add_command(lab_group)


if __name__ == "__main__":
    cli()
