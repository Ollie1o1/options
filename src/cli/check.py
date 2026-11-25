"""`opt check <TICKER>` — 1-Screen Ticker Dossier."""
from __future__ import annotations

import json

import click

from src.radar import build_dossier
from .render import make_console, render_dossier


@click.command("check")
@click.argument("ticker")
@click.option("--fetch-chain/--no-chain", default=False, help="Fetch live option chain IV/RND (default: instant local cache).")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON output.")
@click.pass_context
def check_cmd(ctx: click.Context, ticker: str, fetch_chain: bool, as_json: bool) -> None:
    """1-Screen Ticker Dossier: structure, levels, landmines, vol/RND & active playbooks."""
    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    ohlcv_db = (ctx.obj or {}).get("ohlcv_db", "data/equity_ohlcv.db")
    cache_path = (ctx.obj or {}).get("cache_path", "data/radar_cache.json")

    dos = build_dossier(ticker, ohlcv_db=ohlcv_db, cache_path=cache_path, fetch_chain=fetch_chain)
    if use_json:
        click.echo(json.dumps(dos.to_dict(), indent=2))
        return
    console = make_console(no_color=no_color)
    console.print(render_dossier(dos))
