"""Plain-English guidance helpers and zero-memorization interactive wizard (`opt guide`)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import click
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from src.own_picks import list_equity_picks, log_equity_pick
from src.radar import RadarSnapshot, TickerDossier, build_dossier, load_cached_radar, run_radar
from src.radar.live_options import refresh_local_prices
from src.structure.express import compare_vehicles
from src.structure.types import DetailedView, VehicleComparison
from .render import make_console, render_core_allocation, render_dossier, render_express_table, render_radar_table

TIER_PLAIN = {
    "VALIDATED": "PROVEN",
    "LEAD": "PROMISING",
    "CONTEXT": "WATCHLIST",
    "RESEARCH": "TESTING",
    "CLOSED": "RETIRED",
}


def plain_english_regime(snap: RadarSnapshot) -> Tuple[str, str]:
    """Return `(headline, advice)` explaining market conditions without quant jargon."""
    rv = getattr(snap, "spy_rv20", None)
    if snap.vix is not None:
        v = f"VIX {snap.vix:.0f}"
    elif rv is not None:
        v = f"SPY realized vol {rv:.0f}%"
    else:
        v = "volatility unknown"
    if snap.regime_posture == "RISK_OFF":
        return (
            f"DEFENSIVE MARKET ({v}, SPY below 200-day trend)",
            "Play defense today: keep the suggested cash % in T-bills/CASH, lean into defensive ETFs, "
            "and avoid aggressive dip-buying until SPY reclaims its 200-day average.",
        )
    if snap.regime_posture == "RISK_ON":
        return (
            f"HEALTHY UPTREND ({v}, SPY above 200-day trend)",
            "Green light for both Core ETFs and short-term Uptrend Pullback stock buys (in shares).",
        )
    return (
        f"MIXED / NEUTRAL MARKET ({v})",
        "Be selective: stick mainly to your Core ETF allocation and only take high-conviction setups with zero earnings warnings.",
    )


def plain_english_action_plan(snap: RadarSnapshot) -> List[str]:
    """Generate a concrete 3-step plain-English daily checklist."""
    headline, advice = plain_english_regime(snap)
    steps = [
        f"1. Market Mood: {headline} — {advice}",
    ]
    if snap.core_allocation:
        mix = ", ".join(
            f"{c['ticker']} {float(c['target_weight'])*100:.0f}%"
            for c in snap.core_allocation
        )
        steps.append(
            f"2. Core Portfolio (PROVEN · TFSA $0 comm): Target mix is {mix}. "
            "Rebalance once a month; no daily trading needed."
        )
    pullbacks = [o for o in snap.opportunities if o.playbook_id == "equity_pullback"]
    if pullbacks:
        names = ", ".join(o.ticker for o in pullbacks[:4])
        steps.append(
            f"3. Active Swing Ideas (PROMISING · Shares Only): {len(pullbacks)} dip-buy setup(s) firing ({names}). "
            "Inspect one below before buying; sell when price closes above its 5-day average or after 10 days."
        )
    else:
        steps.append(
            "3. Active Swing Ideas: None firing right now (that's normal in a choppy/defensive tape). "
            "Your best move today is sticking to your Core ETFs above, or scanning 60–120d ETF option spreads if using IBKR."
        )
    return steps


def plain_english_traffic_light(dos: TickerDossier) -> Tuple[str, str]:
    """Turn a TickerDossier into a clear GREEN / YELLOW / RED summary."""
    if dos.spot is None:
        return ("YELLOW", f"{dos.ticker} is not in your local 62-symbol price database yet — run Update Prices or check your broker.")
    red_reasons = []
    yellow_reasons = []
    for f in dos.risk_flags:
        if "POST_EARN_CRASH" in f:
            red_reasons.append("recently crashed on earnings (historically lags SPY for ~20 days)")
        elif "EARN_IN_" in f:
            red_reasons.append(f"earnings coming up soon ({f}) — avoid holding options through it")
        elif "BELOW_200MA" in f:
            yellow_reasons.append("trading below its 200-day trend")
    if red_reasons:
        return ("RED — CAUTION", "; ".join(red_reasons).capitalize() + ".")
    if yellow_reasons:
        return ("YELLOW — BE SELECTIVE", "; ".join(yellow_reasons).capitalize() + ".")
    return ("GREEN — CLEAN SETUP", "Above its 200-day uptrend with no earnings crash or imminent report in cache.")


def plain_english_vehicle_verdict(rows: List[VehicleComparison], view: DetailedView) -> str:
    """Summarize the Shares-vs-Options table in 2 plain-English sentences."""
    if not rows:
        return "No valid trade structures found for this view."
    best = next((r for r in rows if r.recommended), rows[0])
    shares = next((r for r in rows if r.vehicle == "Shares"), None)
    if best.ev_view_net_usd <= 0:
        return (
            f"SKIP / WAIT: Even if {view.symbol} moves {view.target_move_pct*100:+.1f}% in {view.horizon_days} days, "
            "spread costs eat the expected gain on every option structure. Wait for a better entry or use Shares with a tight stop."
        )
    if best.vehicle == "Shares":
        return (
            f"BEST CHOICE: {best.description} — Shares cost only {best.friction_pct_of_reward:.2f}% in friction, "
            "never suffer time decay, and work cleanly in a Wealthsimple TFSA."
        )
    extra = ""
    if shares is not None:
        extra = f" (Simpler alternative: {shares.description} at just {shares.friction_pct_of_reward:.2f}% friction.)"
    return (
        f"BEST CHOICE FOR THIS VIEW: {best.vehicle} ({best.description}) — "
        f"estimated edge ${best.ev_view_net_usd:+,.0f} after {best.friction_pct_of_reward:.1f}% spread cost. "
        f"Execution rule: {best.walk_away_label}.{extra}"
    )


GLOSSARY_TEXT = """\
PLAIN-ENGLISH GUIDE TO YOUR TRADING DESK

1. THE 3 STRATEGIES THAT ACTUALLY SURVIVED TESTING (Ignore everything else!)
   • Core ETF Rotation [PROVEN]
     Holds the top 3 strongest sector/asset ETFs + Cash based on market volatility.
     Rebalance once a month in your TFSA ($0 commission). Cut past crashes by ~68%.
   • Uptrend Pullbacks [PROMISING — Shares Only]
     Buys strong uptrending stocks that dipped sharply over 3 days (RSI2 < 10).
     Buy SHARES (never wide-spread options), exit when price closes above its 5-day
     average or after 10 trading days. Skip or size tiny when Market Mood is DEFENSIVE.
   • 60–120 Day Wide ETF Spreads [PROMISING — Limit Orders Only · TAXABLE / IBKR ONLY]
     Sells a 5%-wide Bull Put on liquid ETFs (SPY/QQQ/XLE/etc.) 60–120 days out,
     takes profit at 50% of credit, and exits at 21 days left. CRA prohibits spreads
     in Canadian TFSAs — keep spreads strictly in a taxable/margin account, and
     ONLY use limit orders (NEVER market orders).

2. WHAT THE LABELS MEAN
   • PROVEN (VAL)    = Passed full out-of-sample backtest net of real costs.
   • PROMISING (LEAD)= Strong evidence, with a specific rule (shares-only or limit-only).
   • WATCHLIST (CTX) = Informational (holdings buy zones, insiders, option skew, SEC EDGAR) — not yet OOS-proven.
   • TESTING         = A registered test exists but its readout date has not arrived yet.
   • CONTEXT         = Plain description (trend, breadth, peers, macro plumbing). Never tested as a signal.
   • AVOID           = A measured negative (e.g. post-earnings breakdown) or an imminent event.
   • RETIRED (CLOSED)= Tested and failed — hidden from your daily screen automatically.
   • POST_EARN_CRASH = Stock dropped >=5% on heavy earnings volume; avoid catching the knife.

3. DISCOVER — TOP-DOWN, IN FOUR NUMBERED STEPS  (TUI [6] · `opt discover` · `opt web`)
   ① MARKET   What kind of market is this? VIX curve, trend, credit & rotation ratios. Context only.
   ② SECTORS  The 16 sector/asset ETFs in PROVEN rotation order. FAVOR = held this month.
              Press Enter on a sector to see its stocks; press `g` to toggle the 4-quadrant RRG map.
   ③ IDEAS    Lenses ("why look here?"). Each shows its label, the evidence, and how it is sorted.
              `opt discover --ledger` lists ideas already tested here that do NOT work.
   ④ CARD     One ticker: verdict, price structure, closest peers, options odds (says when they
              are a MODEL rather than market-implied), events — and where every number came from.

4. KEYBOARD SHORTCUTS & BLOOMBERG `:` COMMAND BAR
   • Tabs: 1=Today  2=Radar  3=Workbench  4=Book (Ghost Race & Options)  5=Lab  6=Discover  7=Options
   • Options Power: `7` Options Hub (Scanner · T-Shaped Chain Matrix + Leg Composer · Flow & Vol Intel)
                    `c` Chain Matrix for focused ticker · `h` 2D Risk Surface Heatmap · `{`/`}` IV Shock ±5%
                    `p` Paper-Log Option (`paper_trades.db`) · `o` Live 60-120d ETF Spreads
   • Desk Keys: `:` Cmd Bar · `b` TFSA Rebalance · `l` Log Pick · `x` Close Pick · `f` Log Fill
                `a` TFSA/Taxable · `g` RRG Map · `[`/`]` Target ±2% · `-`/`+` Days ±10 · `t` Theme
   • `:` Mnemonics: `NVDA` (Card) · `CHAIN NVDA` (Chain Matrix) · `SCAN [bull_put]` · `FLOW` · `VOL` ·
                    `SURF NVDA` (2D Heatmap) · `NVDA WB` · `REB` · `SKEW` · `GEX SPY` · `MACRO` · `PM NVDA`

5. DAILY ROUTINE (2 MINUTES)
   • Run `opt refresh` once after the close, then `opt` (TUI) or `opt web` (browser cockpit).
   • Type `opt commands` in your terminal anytime (or read `docs/COMMANDS.md`) to see every command.
"""


@click.command("refresh")
@click.option(
    "--expand",
    is_flag=True,
    help="Expand data/equity_ohlcv.db to the curated ~250+ liquid stock/ETF universe (audited Yahoo refresh).",
)
@click.pass_context
def refresh_cmd(ctx: click.Context, expand: bool) -> None:
    """Update local stock/ETF prices and refresh today's Radar ideas in one command."""
    no_color = bool((ctx.obj or {}).get("no_color"))
    ohlcv_db = (ctx.obj or {}).get("ohlcv_db", "data/equity_ohlcv.db")
    cache_path = (ctx.obj or {}).get("cache_path", "data/radar_cache.json")
    console = make_console(no_color=no_color)
    step_msg = (
        "Updating & expanding daily prices (~250+ liquid stocks/ETFs) from Yahoo Finance…"
        if expand
        else "Updating daily prices from Yahoo Finance…"
    )
    console.print(f"[bold]Step 1/2:[/bold] {step_msg}")
    try:
        res = refresh_local_prices(ohlcv_db, expand_universe=expand)
        console.print(
            f"  Done — {res['symbols']} symbols checked "
            f"({res['updated']} updated, {res['current']} already current, {res['failed']} failed)."
        )
    except Exception as exc:
        console.print(f"  [#d6a452]Price update skipped ({exc}) — using existing local prices.[/#d6a452]")
    try:
        from src.topdown.macro import refresh_fred_cache

        fred = refresh_fred_cache()
        ok = sum(1 for v in fred.values() if v == "ok")
        console.print(f"  FRED credit/curve series: {ok}/{len(fred)} refreshed.")
    except Exception as exc:
        console.print(f"  [#d6a452]FRED refresh skipped ({exc}).[/#d6a452]")
    console.print("[bold]Step 2/2:[/bold] Refreshing Unified Radar…")
    snap = run_radar(ohlcv_db=ohlcv_db, cache_path=cache_path)
    console.print(f"  [#5ec98d]Ready![/#5ec98d] {len(snap.opportunities)} active ideas cached. Run `opt` or `opt guide` to view.")


@click.command("guide")
@click.pass_context
def guide_cmd(ctx: click.Context) -> None:
    """Step-by-step plain-English guided assistant — zero commands or flags to memorize."""
    no_color = bool((ctx.obj or {}).get("no_color"))
    ohlcv_db = (ctx.obj or {}).get("ohlcv_db", "data/equity_ohlcv.db")
    cache_path = (ctx.obj or {}).get("cache_path", "data/radar_cache.json")
    own_picks_db = (ctx.obj or {}).get("own_picks_db", "data/own_picks.db")
    console = make_console(no_color=no_color)

    snap = load_cached_radar(cache_path) or run_radar(ohlcv_db=ohlcv_db, cache_path=cache_path)

    while True:
        plan_lines = plain_english_action_plan(snap)
        console.print()
        console.print(
            Panel(
                "\n".join(plan_lines),
                title="GUIDED TRADING COACH — WHAT TO DO RIGHT NOW",
                border_style="#82aad2",
            )
        )
        console.print("  [bold][1][/bold] Morning Routine        — Refresh prices, see Core ETFs & best setups")
        console.print("  [bold][2][/bold] Check Any Stock/ETF    — Traffic light + Shares vs. Options verdict + exact limit price")
        console.print("  [bold][3][/bold] Find Swing Stock Dips  — Uptrend pullbacks ready to buy in shares")
        console.print("  [bold][4][/bold] Scan Live ETF Spreads  — Live 60–120d wide ETF Bull Puts (Test O1 limit-only)")
        console.print("  [bold][5][/bold] Log Trade / My Results — Record a buy & check if your picks beat SPY")
        console.print("  [bold][6][/bold] TFSA Core Rebalancer   — Exact monthly ETF share orders for your account")
        console.print("  [bold][7][/bold] Open Full Dashboard    — Interactive 6-tab keyboard cockpit (`opt`)")
        console.print("  [bold][8][/bold] Discover (top-down)    — ① Market → ② Sectors → ③ Ideas → ④ Evidence card")
        console.print("  [bold][?][/bold] Plain-English Guide    — Explain every strategy & label simply")
        console.print("  [bold][Q][/bold] Quit")
        console.print()

        choice = Prompt.ask("What would you like to do?", default="1").strip().upper()
        if choice in ("Q", "QUIT", "EXIT"):
            return

        if choice == "1":
            if Confirm.ask("Update latest daily prices from Yahoo first?", default=False):
                try:
                    res = refresh_local_prices(ohlcv_db)
                    console.print(f"Updated {res['updated']} symbols ({res['current']} already current).")
                except Exception as exc:
                    console.print(f"[#d6a452]Could not reach Yahoo ({exc}); using local cache.[/#d6a452]")
            snap = run_radar(ohlcv_db=ohlcv_db, cache_path=cache_path)
            if snap.core_allocation:
                console.print(render_core_allocation(snap))
            if snap.opportunities:
                console.print(render_radar_table(snap.opportunities[:8], title="Today's Best Ranked Setups"))
            else:
                console.print("[dim]No extra setups firing right now — follow the Core ETF mix above.[/dim]")

        elif choice == "2":
            tk = Prompt.ask("Which stock or ETF ticker? (e.g. SPY, AAPL, NVDA, XLE)", default="SPY").strip().upper()
            dos = build_dossier(tk, ohlcv_db=ohlcv_db, cache_path=cache_path, fetch_chain=False)
            light, reason = plain_english_traffic_light(dos)
            col = "#5ec98d" if light.startswith("GREEN") else ("#e06c75" if light.startswith("RED") else "#d6a452")
            console.print(Panel(f"[bold {col}]{light}[/bold {col}]\n{reason}", title=f"{tk} QUICK VERDICT", border_style=col))
            console.print(render_dossier(dos))
            if Confirm.ask(f"Compare Shares vs. Options for {tk}?", default=True):
                d_choice = Prompt.ask("Where do you think it's going?", choices=["up", "down", "flat"], default="up")
                d_map = {"up": "BULLISH", "down": "BEARISH", "flat": "NEUTRAL"}
                tgt_s = "0%" if d_choice == "flat" else Prompt.ask("Expected move %? (e.g. 6 for 6%)", default="6")
                tgt = float(tgt_s.rstrip("%")) / 100.0
                if d_choice == "down" and tgt > 0:
                    tgt = -tgt
                days = int(Prompt.ask("Timeframe in days?", default="30"))
                risk = float(Prompt.ask("Max dollars you want to risk?", default="500"))
                owns = Confirm.ask(f"Do you already own 100+ shares of {tk}?", default=False)
                spot = dos.spot if dos.spot is not None else float(Prompt.ask(f"Current price of {tk}?"))
                iv = (
                    (dos.atm_iv / 100.0)
                    if dos.atm_iv is not None
                    else ((dos.rv_20_ann / 100.0) if dos.rv_20_ann else 0.25)
                )
                view = DetailedView(
                    symbol=tk, direction=d_map[d_choice], target_move_pct=tgt,
                    horizon_days=days, risk_budget_usd=risk, account_type="tfsa", owns_100_shares=owns,
                )
                rows = compare_vehicles(view, spot=spot, iv=iv, chain_db="data/chain_archive.db")
                console.print(Panel(plain_english_vehicle_verdict(rows, view), title="PLAIN-ENGLISH RECOMMENDATION", border_style="#5ec98d"))
                console.print(render_express_table(rows, symbol=tk, spot=spot, view_desc=f"{d_map[d_choice]} {tgt*100:+.1f}% / {days}d"))

        elif choice == "3":
            pullbacks = [o for o in snap.opportunities if o.playbook_id == "equity_pullback"]
            if not pullbacks:
                console.print(
                    Panel(
                        "No uptrend pullback setups are firing right now.\n"
                        "Why: Either stocks aren't dipped sharply enough (RSI2 < 10) or the market is defensive.\n"
                        "Rule: Never force a trade — wait for a clean setup or stay in your Core ETFs.",
                        title="SWING STOCK PULLBACKS",
                        border_style="#d6a452",
                    )
                )
            else:
                console.print(render_radar_table(pullbacks, title="Uptrend Pullbacks (Buy Shares · Sell > 5d MA or 10d)"))

        elif choice == "4":
            console.print("[bold]Scanning live 60–120 day option chains on liquid ETFs…[/bold]")
            from src.radar.live_options import scan_live_etf_wide_spreads_detailed

            live_res = scan_live_etf_wide_spreads_detailed()
            live_opps, warns = live_res.opportunities, live_res.warnings
            if live_opps:
                console.print(
                    Panel(
                        "RULE (Test O1): Only enter with a LIMIT order at or above 'Min Limit'. "
                        "Take profit at 50% of credit, or exit at 21 days left. NEVER use market orders!",
                        title="LIVE 60–120D WIDE ETF BULL PUTS (LIMIT ONLY)",
                        border_style="#5ec98d",
                    )
                )
                console.print(render_radar_table(live_opps, title="Live 60–120d ETF Spreads (Sorted by Lowest Friction)"))
            else:
                console.print("[#d6a452]No live ETF spreads cleared the <=25% spread-cost filter right now.[/#d6a452]")
            if live_res.skipped:
                console.print("[bold]Not shown, and why:[/bold]")
                for tk, why in sorted(live_res.skipped.items()):
                    console.print(f"  [dim]{tk:<5}[/dim] {why}")
            console.print(
                "[dim]Rule (Test O1, measured on SPY 2010-23): nearest 90 DTE in 60-120, short leg nearest "
                f"0.25 delta, long leg 5% of spot further out · deltas from mid-implied IV, r = {live_res.rate_source}[/dim]"
            )
            if warns:
                console.print(f"[dim]Notes: {', '.join(warns[:3])}[/dim]")

        elif choice == "5":
            sub = Prompt.ask("Log a new trade or view your results?", choices=["log", "results"], default="results")
            if sub == "log":
                tk = Prompt.ask("Ticker").strip().upper()
                px = float(Prompt.ask("Fill price per share"))
                sh = float(Prompt.ask("How many shares?", default="10"))
                pb = Prompt.ask(
                    "Which strategy?",
                    choices=["vol_managed_outlook", "equity_pullback", "etf_wide_spread", "discretionary"],
                    default="vol_managed_outlook",
                )
                note = Prompt.ask("Short note (optional)", default="")
                pid = log_equity_pick(tk, px, shares=sh, playbook_id=pb, notes=note, db_path=own_picks_db)
                console.print(f"[#5ec98d]✓ Saved pick #{pid}: {sh:g} {tk} @ ${px:.2f} [{pb}][/#5ec98d]")
            else:
                picks = list_equity_picks(db_path=own_picks_db)
                if not picks:
                    console.print("[dim]No stock/ETF picks logged yet. Choose 'log' whenever you take a trade.[/dim]")
                else:
                    for p in picks[:10]:
                        console.print(
                            f"  #{p['pick_id']} {p['date'][:10]}  {p['ticker']:6s}  "
                            f"{p['shares']:g} sh @ ${float(p['entry_price']):.2f}  [{p['playbook_id']}]  {p['status']}"
                        )

        elif choice == "6":
            from src.equity_playbooks.rebalance import compute_tfsa_rebalance, parse_holdings_spec

            acct_s = Prompt.ask("Total TFSA / core account value ($ USD)", default="5000")
            hold_s = Prompt.ask("Current ETF shares held (e.g. XLE:10,GLD:5 or blank for all cash)", default="")
            frac = Confirm.ask("Allow fractional shares (Wealthsimple)?", default=True)
            try:
                acct_v = float(acct_s.replace("$", "").replace(",", ""))
                plan = compute_tfsa_rebalance(
                    snap.core_allocation,
                    account_usd=acct_v,
                    current_shares=parse_holdings_spec(hold_s),
                    fractional=frac,
                )
                lines = [
                    f"Account: ${plan.account_usd:,.2f}  |  Vol Scale: {plan.vol_scale*100:.0f}%  |  Max Drift: {plan.max_drift_pct:.1f}%",
                    f"Estimated Cash After Orders: ${plan.estimated_cash_after_usd:,.2f}",
                    "",
                ]
                for o in plan.orders:
                    lines.append(
                        f"  {o.action:<7} {o.delta_shares:+7.2f} sh {o.ticker:<4} @ ${o.spot:7.2f} "
                        f"({o.delta_usd:+9,.2f} · target {o.target_weight_pct:.0f}%)"
                    )
                console.print(Panel("\n".join(lines), title="TFSA MONTHLY REBALANCE ORDERS", border_style="#5ec98d"))
            except Exception as exc:
                console.print(f"[#e06c75]Could not compute rebalance: {exc}[/#e06c75]")

        elif choice == "7":
            from src.tui.app import run_desk

            run_desk()

        elif choice == "8":
            from src.topdown import render as td_render
            from src.topdown.hub import build_hub

            hub = build_hub(ohlcv_db=ohlcv_db, cache_path=cache_path)
            console.print(td_render.hub_overview(hub))
            tk = Prompt.ask("④ Evidence card for which ticker? (Enter to skip)", default="").strip().upper()
            if tk:
                console.print(td_render.card_renderable(hub.card(tk)))

        elif choice == "?":
            console.print(Panel(GLOSSARY_TEXT, title="PLAIN-ENGLISH CHEAT SHEET", border_style="#82aad2"))
        else:
            console.print("[#d6a452]Please enter 1-8, ?, or Q.[/#d6a452]")
