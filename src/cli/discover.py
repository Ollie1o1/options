"""`opt discover` (top-down: ① Market → ② Sectors → ③ Ideas) and `opt card TICKER` (④ Evidence Card)."""
from __future__ import annotations

import json

import click

from .render import make_console


def _hub(ctx: click.Context):
    from src.topdown.hub import build_hub

    obj = ctx.obj or {}
    return build_hub(
        ohlcv_db=obj.get("ohlcv_db", "data/equity_ohlcv.db"),
        cache_path=obj.get("cache_path", "data/radar_cache.json"),
    )


@click.command("discover")
@click.option("--sector", "sector", default=None, help="Drill into one sector ETF's constituents (e.g. XLE, SMH).")
@click.option("--lens", "lens_id", default=None,
              help="Open one lens in full. Includes opt_skew, edgar_events, edgar_quality. An invalid ID lists all choices.")
@click.option("--diversify", is_flag=True,
              help="Annotate & sort lens rows by lowest 126d correlation against your current holdings / core sleeve.")
@click.option("--ledger", is_flag=True, help="Show ideas already tested here that did NOT work.")
@click.option("--refresh-macro", is_flag=True, help="Fetch FRED credit-spread / yield-curve series first (network).")
@click.option("--record", is_flag=True, help="Record today's idea lists to the forward log (maintenance does this daily).")
@click.option("--forward", is_flag=True, help="Status of the forward record; outcomes stay locked until the preregistered read date.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
@click.pass_context
def discover_cmd(ctx: click.Context, sector: str, lens_id: str, diversify: bool, ledger: bool, refresh_macro: bool,
                 record: bool, forward: bool, as_json: bool) -> None:
    """Top-down discovery: ① Market → ② Sectors (ranked by the PROVEN model) → ③ Ideas, each labelled by its evidence."""
    from src.topdown import render as R
    from src.topdown.lenses import LENS_BY_ID

    use_json = as_json or bool((ctx.obj or {}).get("json"))
    console = make_console(no_color=bool((ctx.obj or {}).get("no_color")))
    if forward:
        from src.topdown import forward_log as FL

        rep = FL.forward_report()
        if use_json:
            from dataclasses import asdict

            click.echo(json.dumps([asdict(r) for r in rep], indent=2, default=str))
            return
        console.print(R.forward_panel(rep))
        return
    if refresh_macro:
        from src.topdown.macro import refresh_fred_cache

        status = refresh_fred_cache()
        if not use_json:
            console.print("FRED refresh: " + ", ".join(f"{k} {v}" for k, v in status.items()))
    if lens_id and lens_id not in LENS_BY_ID:
        raise click.BadParameter(f"unknown lens; choose from {', '.join(LENS_BY_ID)}", param_hint="--lens")
    hub = _hub(ctx)
    if record:
        from src.topdown.forward_log import record_hub

        counts = record_hub(hub)
        if not use_json:
            console.print(f"Recorded {counts['rows_new']} new rows ({counts['rows_seen']} listed) across "
                          f"{counts['lenses']} lenses → data/lens_forward.db")

    if use_json:
        payload = hub.to_dict()
        if sector:
            from src.topdown.sectors import sector_members

            payload["sector_members"] = {sector.upper(): sector_members(sector, hub.ohlcv)}
        click.echo(json.dumps(payload, indent=2, default=str))
        return

    if ledger:
        console.print(R.null_ledger_panel())
        return
    if sector:
        console.print(R.breadcrumb(2))
        console.print(R.sector_members_panel(hub, sector))
        return
    if lens_id:
        res = hub.lens(lens_id)
        console.print(R.breadcrumb(3))
        if res is not None:
            if diversify and res.rows:
                from dataclasses import replace
                from src.topdown.portfolio_fit import rank_rows_for_diversification

                div_rows = rank_rows_for_diversification(res.rows, holdings=hub.holdings(), ohlcv=hub.ohlcv)
                res = replace(res, rows=div_rows, note=(res.note + " " if res.note else "") + "Sorted by lowest 126d correlation to your holdings/sleeve.")
            console.print(R.lens_panel(res))
        return
    console.print(R.hub_overview(hub))
    console.print("[dim]Already-tested dead ends: opt discover --ledger[/dim]")


@click.command("card")
@click.argument("ticker")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
@click.option("--no-peer-iv", is_flag=True, help="Skip archived-chain IV lookups for peers (faster).")
@click.pass_context
def card_cmd(ctx: click.Context, ticker: str, as_json: bool, no_peer_iv: bool) -> None:
    """④ Evidence Card: verdict, price structure, peers, options odds, events — every number with its source and date."""
    from src.topdown import render as R

    hub = _hub(ctx)
    card = hub.card(ticker, with_peer_iv=not no_peer_iv)
    if as_json or bool((ctx.obj or {}).get("json")):
        click.echo(json.dumps(card.to_dict(), indent=2, default=str))
        return
    console = make_console(no_color=bool((ctx.obj or {}).get("no_color")))
    console.print(R.breadcrumb(4))
    console.print(R.card_renderable(card))
