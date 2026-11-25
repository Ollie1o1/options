"""`opt express <TICKER>` — Shares-vs-Options Express Workbench."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional

import click

from src.radar import build_dossier
from src.structure.express import compare_vehicles
from src.structure.types import DetailedView
from .render import make_console, render_express_table


def _parse_pct(raw: str) -> float:
    s = str(raw).strip().rstrip("%")
    val = float(s)
    return val / 100.0 if abs(val) > 1.0 or "%" in raw else val


@click.command("express")
@click.argument("ticker")
@click.option("--dir", "direction", type=click.Choice(["bull", "bear", "neutral"], case_sensitive=False), default="bull")
@click.option("--target", default="+8%", help="Expected move over horizon (e.g. +8%, -6%).")
@click.option("--days", type=int, default=30, help="View horizon in calendar days.")
@click.option("--vol-mult", type=float, default=1.0, help="Expected vol multiplier vs market (1.0=agree, <1=crush).")
@click.option("--risk", type=float, default=500.0, help="Risk budget in USD.")
@click.option("--account", type=click.Choice(["tfsa", "taxable"], case_sensitive=False), default="tfsa")
@click.option("--owns-shares", is_flag=True, help="Include Covered Call comparison (owns >=100 shares).")
@click.option("--spot", type=float, default=None, help="Override spot price (default: local OHLCV close).")
@click.option("--iv", type=float, default=None, help="Override implied vol decimal (e.g. 0.25).")
@click.option("--live", is_flag=True, help="Fetch live option chains instead of using local chain_archive.db.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON output.")
@click.pass_context
def express_cmd(
    ctx: click.Context,
    ticker: str,
    direction: str,
    target: str,
    days: int,
    vol_mult: float,
    risk: float,
    account: str,
    owns_shares: bool,
    spot: Optional[float],
    iv: Optional[float],
    live: bool,
    as_json: bool,
) -> None:
    """Compare Shares, LEAPS, Debit Spread, Wide Credit Spread, Covered Call & Single Option side-by-side."""
    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    ohlcv_db = (ctx.obj or {}).get("ohlcv_db", "data/equity_ohlcv.db")
    cache_path = (ctx.obj or {}).get("cache_path", "data/radar_cache.json")
    chain_db = (ctx.obj or {}).get("chain_db", "data/chain_archive.db")

    sym = ticker.upper().strip()
    dos = build_dossier(sym, ohlcv_db=ohlcv_db, cache_path=cache_path, chain_db=chain_db, fetch_chain=live)
    use_spot = spot if spot is not None else (dos.spot if dos.spot is not None else 100.0)
    use_iv = (
        iv
        if iv is not None
        else (
            (dos.atm_iv / 100.0)
            if dos.atm_iv is not None
            else ((dos.rv_20_ann / 100.0) if dos.rv_20_ann else 0.25)
        )
    )

    dir_map = {"bull": "BULLISH", "bear": "BEARISH", "neutral": "NEUTRAL"}
    d_full = dir_map[direction.lower()]
    t_move = _parse_pct(target)
    if d_full == "BEARISH" and t_move > 0:
        t_move = -t_move
    if d_full == "NEUTRAL":
        t_move = 0.0

    view = DetailedView(
        symbol=sym,
        direction=d_full,
        target_move_pct=t_move,
        horizon_days=days,
        vol_mult=vol_mult,
        risk_budget_usd=risk,
        account_type=account.lower(),
        owns_100_shares=owns_shares,
    )
    rows = compare_vehicles(view, spot=use_spot, iv=use_iv, chain_db=chain_db, allow_live=live)

    if use_json:
        click.echo(
            json.dumps(
                {
                    "symbol": sym,
                    "spot": use_spot,
                    "iv": round(use_iv, 4),
                    "view": asdict(view),
                    "vehicles": [asdict(r) for r in rows],
                },
                indent=2,
            )
        )
        return

    console = make_console(no_color=no_color)
    desc = f"{d_full} {t_move*100:+.1f}% in {days}d · risk ${risk:,.0f} · {account.upper()}"
    console.print(render_express_table(rows, symbol=sym, spot=use_spot, view_desc=desc))
