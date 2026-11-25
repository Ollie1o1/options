"""`opt book` & `opt log` — Unified Portfolio, Holdings & Playbook Pick Logger."""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

import click
from rich.table import Table

from src.own_picks import (
    build_equity_scorecard,
    list_equity_picks,
    log_equity_pick,
    log_pick,
)
from src.paths import repo_path
from .render import make_console


def _load_open_paper_options(db_path: str = "paper_trades.db", limit: int = 25) -> List[Dict[str, Any]]:
    full = repo_path(db_path)
    if not os.path.exists(full):
        return []
    try:
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT entry_id, date, ticker, strategy_name, expiration, strike, type, entry_price, quantity "
                "FROM trades WHERE status = 'OPEN' ORDER BY entry_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            con.close()
    except Exception:
        return []


@click.command("book")
@click.option("--section", type=click.Choice(["all", "positions", "picks", "scorecard"], case_sensitive=False), default="all")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON output.")
@click.pass_context
def book_cmd(ctx: click.Context, section: str, as_json: bool) -> None:
    """Inspect open paper options, manual stock/ETF picks & per-playbook scorecard."""
    use_json = as_json or bool((ctx.obj or {}).get("json"))
    no_color = bool((ctx.obj or {}).get("no_color"))
    own_picks_db = (ctx.obj or {}).get("own_picks_db", "data/own_picks.db")
    ohlcv_db = (ctx.obj or {}).get("ohlcv_db", "data/equity_ohlcv.db")

    open_opts = _load_open_paper_options()
    eq_picks = list_equity_picks(db_path=own_picks_db)
    scorecard = build_equity_scorecard(db_path=own_picks_db, ohlcv_db=ohlcv_db)

    if use_json:
        click.echo(
            json.dumps(
                {
                    "open_paper_options": open_opts,
                    "equity_picks": eq_picks,
                    "equity_scorecard": scorecard,
                },
                indent=2,
            )
        )
        return

    console = make_console(no_color=no_color)
    sec = section.lower()
    if sec in ("all", "picks"):
        tbl = Table(title=f"Stock & ETF Picks ({len(eq_picks)})", pad_edge=False)
        tbl.add_column("ID", justify="right")
        tbl.add_column("Date")
        tbl.add_column("Ticker", style="bold")
        tbl.add_column("Playbook")
        tbl.add_column("Side")
        tbl.add_column("Entry", justify="right")
        tbl.add_column("Shares", justify="right")
        tbl.add_column("Status")
        for p in eq_picks[:20]:
            tbl.add_row(
                str(p["pick_id"]),
                str(p["date"])[:10],
                str(p["ticker"]),
                str(p["playbook_id"]),
                str(p["side"]),
                f"${float(p['entry_price']):.2f}",
                f"{float(p['shares']):g}",
                str(p["status"]),
            )
        console.print(tbl)

    if sec in ("all", "scorecard"):
        sc_tbl = Table(title="Playbook Scorecard (SPY-Relative 5d / 10d)", pad_edge=False)
        sc_tbl.add_column("Playbook", style="bold")
        sc_tbl.add_column("Picks", justify="right")
        sc_tbl.add_column("Open", justify="right")
        sc_tbl.add_column("5d Excess", justify="right")
        sc_tbl.add_column("10d Excess", justify="right")
        sc_tbl.add_column("Verdict")
        for pb_id, info in (scorecard.get("by_playbook") or {}).items():
            h5 = info["horizons"].get(5) or {}
            h10 = info["horizons"].get(10) or {}
            m5 = f"{h5['mean']*100:+.2f}%" if h5.get("mean") is not None else "—"
            m10 = f"{h10['mean']*100:+.2f}%" if h10.get("mean") is not None else "—"
            sc_tbl.add_row(pb_id, str(info["n_picks"]), str(info["n_open"]), m5, m10, str(h5.get("verdict", "")))
        console.print(sc_tbl)

    if sec in ("all", "positions"):
        otbl = Table(title=f"Open Paper Option Positions ({len(open_opts)})", pad_edge=False)
        otbl.add_column("ID", justify="right")
        otbl.add_column("Ticker", style="bold")
        otbl.add_column("Strategy")
        otbl.add_column("Exp")
        otbl.add_column("Strike", justify="right")
        otbl.add_column("Entry", justify="right")
        for r in open_opts[:15]:
            otbl.add_row(
                str(r["entry_id"]),
                str(r["ticker"]),
                str(r["strategy_name"]),
                str(r["expiration"]),
                f"{r['strike']}",
                f"${float(r['entry_price'] or 0):.2f}",
            )
        console.print(otbl)


@click.command("log")
@click.argument("ticker")
@click.option("--asset", type=click.Choice(["stock", "etf", "option"], case_sensitive=False), default="stock")
@click.option("--price", type=float, required=True, help="Entry fill price per share/contract.")
@click.option("--shares", type=float, default=1.0, help="Share count (stock/etf).")
@click.option("--side", type=click.Choice(["LONG", "SHORT"], case_sensitive=False), default="LONG")
@click.option("--playbook", default="discretionary", help="Playbook ID from configs/playbooks.json.")
@click.option("--stop", type=float, default=None, help="Stop-loss price.")
@click.option("--target", type=float, default=None, help="Target price.")
@click.option("--exp", default=None, help="Expiration YYYY-MM-DD (option only).")
@click.option("--strike", type=float, default=None, help="Strike price (option only).")
@click.option("--type", "opt_type", type=click.Choice(["call", "put"], case_sensitive=False), default=None)
@click.option("--note", default="", help="Short rationale note.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON output.")
@click.pass_context
def log_cmd(
    ctx: click.Context,
    ticker: str,
    asset: str,
    price: float,
    shares: float,
    side: str,
    playbook: str,
    stop: Optional[float],
    target: Optional[float],
    exp: Optional[str],
    strike: Optional[float],
    opt_type: Optional[str],
    note: str,
    as_json: bool,
) -> None:
    """Fast 1-line manual pick logger for stocks/ETFs (`own_picks.db`) or options (`paper_trades.db`)."""
    use_json = as_json or bool((ctx.obj or {}).get("json"))
    own_picks_db = (ctx.obj or {}).get("own_picks_db", "data/own_picks.db")

    if asset.lower() == "option":
        if not (exp and strike is not None and opt_type):
            raise click.UsageError("Option logging requires --exp, --strike, and --type call|put")
        ok, msg = log_pick(ticker, exp, strike, opt_type, price=price, force=True)
        if use_json:
            click.echo(json.dumps({"ok": ok, "message": msg}))
        else:
            click.echo(("✓ " if ok else "✗ ") + msg)
        if not ok:
            raise SystemExit(1)
        return

    pid = log_equity_pick(
        ticker,
        price,
        shares=shares,
        side=side,
        asset_class=asset.lower(),
        stop_price=stop,
        target_price=target,
        playbook_id=playbook,
        notes=note,
        db_path=own_picks_db,
    )
    if use_json:
        click.echo(json.dumps({"ok": True, "pick_id": pid, "ticker": ticker.upper(), "playbook": playbook}))
    else:
        click.echo(f"✓ Logged pick #{pid}: {side.upper()} {shares:g} {ticker.upper()} @ ${price:.2f} [{playbook}]")
