"""Reusable Textual widgets, QuickLogModal, LogFillModal, Visual Charts, and Command Palette provider."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from textual.app import ComposeResult
from textual.command import Hit, Hits, Provider
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static, Switch

from src.radar.schema import RadarSnapshot, TickerDossier
from src.structure.types import DetailedView, VehicleComparison
from .charts import braille_payoff_and_rnd, braille_price_chart
from .themes import THEME_ORDER


def vol_label(snap: RadarSnapshot) -> str:
    """'VIX 26.6 (inverted) · SPY RV20 31.8%' — the real VIX and SPY's realized vol, never conflated."""
    parts = []
    if snap.vix is not None:
        term = " (inverted)" if snap.vix_term == "BACKWARDATION" else ""
        parts.append(f"[bold]VIX[/] {snap.vix:.1f}{term}")
    rv = getattr(snap, "spy_rv20", None)
    if rv is not None:
        parts.append(f"[bold]SPY RV20[/] {rv:.1f}%")
    return " · ".join(parts) or "[bold]VIX[/] —"


def format_ribbon_markup(
    snap: Optional[RadarSnapshot],
    health: Optional[Dict[str, Any]] = None,
    *,
    status_msg: str = "",
) -> str:
    if snap is None:
        return status_msg or "Loading Daily Trading Desk…"
    spy_s = f"{snap.spy_last:.2f}" if snap.spy_last is not None else "—"
    spy_c = f"{snap.spy_chg_1d_pct:+.1f}%" if snap.spy_chg_1d_pct is not None else ""
    qqq_c = f"{snap.qqq_chg_1d_pct:+.1f}%" if snap.qqq_chg_1d_pct is not None else "—"
    vix_s = vol_label(snap)
    p_col = "#5ec98d" if snap.regime_posture == "RISK_ON" else ("#e06c75" if snap.regime_posture == "RISK_OFF" else "#d6a452")
    lab_dot = ""
    if health and health.get("worst") in ("WARN", "STALE"):
        dot_col = "#e06c75" if health["worst"] == "STALE" else "#d6a452"
        lab_dot = f"  |  [bold {dot_col}]● LAB {health['worst']}[/]"
    cache_tag = " [dim](cached)[/]" if snap.from_cache else ""
    kalshi_tag = ""
    if getattr(snap, "kalshi_macro", None):
        k0 = snap.kalshi_macro[0]
        kalshi_tag = f"  |  [#82aad2]Kalshi {k0['label']} {k0['prob_pct']:.0f}%[/]"
    status_tag = f"  |  [bold #d6a452]{status_msg}[/]" if status_msg else ""
    return (
        f"[bold]SPY[/] {spy_s} {spy_c}  |  [bold]QQQ[/] {qqq_c}  |  "
        f"{vix_s} [bold {p_col}][{snap.regime_posture}][/]  |  "
        f"Next: {snap.next_macro_event}{kalshi_tag}{lab_dot}{cache_tag}{status_tag}"
    )


def format_dossier_markup(dos: Optional[TickerDossier]) -> str:
    if dos is None:
        return "[dim]Select a ticker on RADAR or enter one in WORKBENCH.[/dim]"
    spot_s = f"${dos.spot:.2f}" if dos.spot is not None else "N/A"
    c1 = f"{dos.chg_1d_pct:+.2f}%" if dos.chg_1d_pct is not None else "—"
    c5 = f"{dos.chg_5d_pct:+.2f}%" if dos.chg_5d_pct is not None else "—"
    m12 = f"{dos.mom_12_1_pct:+.1f}%" if dos.mom_12_1_pct is not None else "—"
    d200 = f"{dos.dist_sma200_pct:+.1f}%" if dos.dist_sma200_pct is not None else "—"
    r2 = f"{dos.rsi_2:.1f}" if dos.rsi_2 is not None else "—"
    r14 = f"{dos.rsi_14:.1f}" if dos.rsi_14 is not None else "—"
    sup = ", ".join(f"${s['level']:.2f} ({s['pct']*100:+.1f}%)" for s in dos.supports) or "—"
    res = ", ".join(f"${r['level']:.2f} ({r['pct']*100:+.1f}%)" for r in dos.resistances) or "—"
    earn = f"in {dos.days_to_earnings}d ({dos.earnings_date})" if dos.days_to_earnings is not None else "none in cache"
    rv_s = f"{dos.rv_20_ann:.1f}%" if dos.rv_20_ann is not None else "—"
    iv_s = f"{dos.atm_iv:.1f}%" if dos.atm_iv is not None else "—"
    iv_rv_s = f" ({dos.iv_vs_rv_ratio:.2f}x)" if getattr(dos, "iv_vs_rv_ratio", None) is not None else ""
    sk_s = f" · 25Δ {dos.skew_25d_vp:+.1f}vp" if getattr(dos, "skew_25d_vp", None) is not None else ""
    im_s = f"±{dos.implied_move_1s_pct:.1f}%" if dos.implied_move_1s_pct is not None else "—"
    src_s = f" [{dos.chain_source}/{dos.rnd_source}]" if getattr(dos, "chain_source", None) else ""

    lines = [
        f"[bold]TICKER DOSSIER: {dos.ticker}[/]   {spot_s}  (1d {c1} · 5d {c5})",
        "─" * 44,
        f"[bold]Trend[/]      vs 200MA {d200} · 12-1m {m12}",
        f"[bold]Momentum[/]   RSI(2) {r2} · RSI(14) {r14} · RV20 {rv_s}",
        f"[bold]Options[/]    ATM IV {iv_s}{iv_rv_s}{sk_s}{src_s}",
        f"[bold]Supports[/]   {sup}",
        f"[bold]Resist[/]     {res}",
        f"[bold]Sector[/]     {dos.sector_etf or '—'} [{dos.sector_posture or 'NEUTRAL'}]"
        + (f" · CDR {dos.cdr_ticker}" if dos.cdr_ticker else ""),
        f"[bold]Landmines[/]  Earnings: {earn} · Macro: {dos.next_macro_event}",
        f"[bold]30d RND[/]    1σ {im_s}"
        + (
            f" · P(>+5%)={dos.rnd_prob_up_5pct*100:.0f}% / P(<-5%)={dos.rnd_prob_dn_5pct*100:.0f}%"
            if dos.rnd_prob_up_5pct is not None and dos.rnd_prob_dn_5pct is not None
            else ""
        ),
    ]
    if getattr(dos, "open_position_note", None):
        lines.append(f"[bold #d6a452]Exposure[/]   {dos.open_position_note}")
    if dos.rev_breadth_30d is not None:
        lines.append(f"[bold]Revisions[/]  30d breadth {dos.rev_breadth_30d:+.2f}")
    if dos.risk_flags:
        lines.append(f"[bold #e06c75]Flags[/]      {', '.join(dos.risk_flags)}")
    if dos.active_signals:
        sigs = " · ".join(f"[{s.evidence_tier}] {s.playbook_label}" for s in dos.active_signals)
        lines.append(f"[bold #5ec98d]Signals[/]    {sigs}")
    if dos.headlines:
        lines.append(f"[bold]News[/]       {dos.headlines[0]['title'][:58]}")
    if dos.closes_60d:
        sma200_lvl = (
            dos.spot / (1.0 + dos.dist_sma200_pct / 100.0)
            if (dos.spot is not None and dos.dist_sma200_pct is not None and (1.0 + dos.dist_sma200_pct / 100.0) > 0.1)
            else None
        )
        chart = braille_price_chart(
            dos.closes_60d[-60:],
            supports=dos.supports,
            resistances=dos.resistances,
            sma200_level=sma200_lvl,
            width=34,
            height=5,
            rich=True,
        )
        if chart:
            lines.append("[bold]60d Tape (Braille · green=Sup / red=Res / dim=200MA):[/]")
            lines.append(chart)
    lines.append("")
    lines.append("[dim][:] Cmd  [Enter] Workbench  [l] Log  [b] Rebal  [f] Fill  [o] Live Opts[/dim]")
    return "\n".join(lines)


def format_workbench_visual_markup(
    vehicles: List[VehicleComparison],
    dos: Optional[TickerDossier],
    view: DetailedView,
) -> str:
    """High-res Braille RND bell curve + Unicode EV vs Cost visual for Workbench."""
    lines: List[str] = []
    spot = dos.spot if (dos is not None and dos.spot is not None) else None
    iv = (
        (dos.atm_iv / 100.0)
        if (dos is not None and dos.atm_iv is not None)
        else ((dos.rv_20_ann / 100.0) if (dos is not None and dos.rv_20_ann) else 0.25)
    )
    if spot is not None and spot > 0:
        rec = next((v for v in vehicles if v.recommended), vehicles[0] if vehicles else None)
        rnd_art = braille_payoff_and_rnd(spot, iv, view, rec, width=44, height=4, rich=True)
        if rnd_art:
            lines.append(rnd_art)

    if dos is not None and dos.rnd_prob_up_5pct is not None and dos.rnd_prob_dn_5pct is not None:
        up = int(round(dos.rnd_prob_up_5pct * 100))
        dn = int(round(dos.rnd_prob_dn_5pct * 100))
        mid = max(0, 100 - up - dn)
        bar_dn = "▓" * max(1, dn // 5)
        bar_mid = "░" * max(1, mid // 5)
        bar_up = "█" * max(1, up // 5)
        lines.append(
            f"[bold]Market RND (30d ±5%):[/] [#e06c75]{bar_dn} {dn}% dn[/] · "
            f"[dim]{bar_mid} {mid}% flat[/] · [#5ec98d]{bar_up} {up}% up[/]"
        )

    if vehicles:
        max_abs_ev = max(max(abs(v.ev_view_net_usd) for v in vehicles), 1.0)
        lines.append(f"[bold]Edge vs Cost (Risk Budget ${view.risk_budget_usd:,.0f} · {view.account_type.upper()}):[/]")
        for v in vehicles[:5]:
            bar_w = max(1, int(round(abs(v.ev_view_net_usd) / max_abs_ev * 14)))
            col = "#5ec98d" if v.ev_view_net_usd > 0 else "#e06c75"
            star = "★" if v.recommended else " "
            lock = "" if v.tfsa_legal else " [#d6a452][TAX][/]"
            fit = "[#5ec98d]FITS[/]" if v.capital_at_risk_usd <= view.risk_budget_usd * 1.15 else "[#d6a452]OVER[/]"
            lines.append(
                f"  {star} {v.vehicle:<18} [{col}]{'█' * bar_w:<14}[/] "
                f"[{col}]${v.ev_view_net_usd:+7,.0f}[/] · cost {v.friction_pct_of_reward:4.1f}% · {fit}{lock}"
            )
    return "\n".join(lines) if lines else "[dim]Run Workbench to visualize vehicle edge & friction.[/dim]"


class HelpModal(ModalScreen[None]):
    """Press `?` anywhere in the TUI for a plain-English guide to strategies & controls."""

    def compose(self) -> ComposeResult:
        from src.cli.guide import GLOSSARY_TEXT

        with Vertical(id="help-dialog"):
            with VerticalScroll(id="help-scroll"):
                yield Static(GLOSSARY_TEXT, id="help-body")
            with Horizontal(id="help-buttons"):
                yield Button("Got it (Esc)", variant="primary", id="help-close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "help-close":
            self.dismiss(None)

    def on_key(self, event) -> None:
        if event.key in ("escape", "q", "question_mark"):
            self.dismiss(None)


class QuickLogModal(ModalScreen[Optional[Dict[str, Any]]]):
    """1-keystroke (`l`) modal to log a stock/ETF pick into data/own_picks.db."""

    def __init__(
        self,
        *,
        ticker: str = "AAPL",
        price: float = 100.0,
        playbook_id: str = "equity_pullback",
    ) -> None:
        super().__init__()
        self._ticker = ticker.upper()
        self._price = price
        self._playbook_id = playbook_id

    def compose(self) -> ComposeResult:
        with Vertical(id="quicklog-dialog"):
            yield Label(f"Quick Log Stock/ETF Pick — {self._ticker}")
            yield Input(value=self._ticker, placeholder="Ticker", id="ql-ticker")
            yield Input(value=f"{self._price:.2f}", placeholder="Entry Price", id="ql-price")
            yield Input(value="10", placeholder="Shares", id="ql-shares")
            yield Input(value="", placeholder="Stop Price (optional)", id="ql-stop")
            yield Input(value="", placeholder="Target Price (optional)", id="ql-target")
            yield Input(value=self._playbook_id, placeholder="Playbook ID", id="ql-playbook")
            yield Input(value="", placeholder="Note (optional)", id="ql-note")
            yield Static("", id="ql-err", classes="modal-error")
            with Horizontal(id="quicklog-buttons"):
                yield Button("Cancel (Esc)", id="ql-cancel")
                yield Button("Log Pick", variant="primary", id="ql-confirm")

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)

    def _submit(self) -> None:
        try:
            tk = self.query_one("#ql-ticker", Input).value.strip().upper()
            px = float(self.query_one("#ql-price", Input).value.strip())
            sh = float(self.query_one("#ql-shares", Input).value.strip())
            raw_stop = self.query_one("#ql-stop", Input).value.strip()
            raw_tgt = self.query_one("#ql-target", Input).value.strip()
            stop_px = float(raw_stop) if raw_stop else None
            tgt_px = float(raw_tgt) if raw_tgt else None
            pb = self.query_one("#ql-playbook", Input).value.strip() or "discretionary"
            note = self.query_one("#ql-note", Input).value.strip()
            if not tk or px <= 0 or sh <= 0:
                raise ValueError("Ticker, positive price, and positive shares required.")
            if stop_px is not None and stop_px <= 0:
                raise ValueError("Stop price must be positive.")
            if tgt_px is not None and tgt_px <= 0:
                raise ValueError("Target price must be positive.")
        except ValueError as exc:
            self.query_one("#ql-err", Static).update(f"[#e06c75]{exc}[/]")
            return
        self.dismiss(
            {
                "ticker": tk,
                "price": px,
                "shares": sh,
                "stop_price": stop_px,
                "target_price": tgt_px,
                "playbook_id": pb,
                "notes": note,
            }
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ql-cancel":
            self.dismiss(None)
            return
        if event.button.id == "ql-confirm":
            self._submit()


class ClosePickModal(ModalScreen[Optional[Dict[str, Any]]]):
    """1-keystroke (`x`) modal to close an open stock/ETF pick in data/own_picks.db."""

    def __init__(
        self,
        *,
        pick_id: int,
        ticker: str,
        entry_price: float,
        current_price: Optional[float] = None,
        shares: float = 1.0,
        exit_hint: str = "",
    ) -> None:
        super().__init__()
        self._pick_id = int(pick_id)
        self._ticker = ticker.upper()
        self._entry_price = float(entry_price)
        self._exit_default = float(current_price) if current_price and current_price > 0 else float(entry_price)
        self._shares = float(shares)
        self._exit_hint = exit_hint

    def compose(self) -> ComposeResult:
        with Vertical(id="closepick-dialog"):
            yield Label(
                f"Close Pick #{self._pick_id} — {self._ticker} ({self._shares:g} sh @ ${self._entry_price:.2f})"
            )
            if self._exit_hint:
                yield Static(f"[#d6a452]Signal: {self._exit_hint}[/]")
            yield Input(value=f"{self._exit_default:.2f}", placeholder="Exit Fill Price", id="cp-price")
            yield Select(
                [
                    ("Plan Worked", "PLAN_WORKED"),
                    ("Bad Fill / Slippage", "BAD_FILL"),
                    ("Thesis Wrong", "THESIS_WRONG"),
                    ("Regime Shift", "REGIME_SHIFT"),
                    ("Broke Rules", "BROKE_RULES"),
                    ("Time Exit", "TIME_EXIT"),
                ],
                value="PLAN_WORKED",
                id="cp-tag",
            )
            yield Input(value=self._exit_hint, placeholder="Exit Note / Reason (optional)", id="cp-note")
            yield Static("", id="cp-err", classes="modal-error")
            with Horizontal(id="closepick-buttons"):
                yield Button("Cancel (Esc)", id="cp-cancel")
                yield Button("Close Pick", variant="primary", id="cp-confirm")

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cp-cancel":
            self.dismiss(None)
            return
        if event.button.id == "cp-confirm":
            try:
                exit_px = float(self.query_one("#cp-price", Input).value.strip())
                if exit_px <= 0:
                    raise ValueError("Exit price must be positive.")
                note = self.query_one("#cp-note", Input).value.strip()
                raw_tag = self.query_one("#cp-tag", Select).value
                tag = str(raw_tag).upper() if raw_tag and str(raw_tag) != "Select.BLANK" else "PLAN_WORKED"
            except ValueError as exc:
                self.query_one("#cp-err", Static).update(f"[#e06c75]{exc}[/]")
                return
            self.dismiss({"pick_id": self._pick_id, "exit_price": exit_px, "notes": note or None, "close_tag": tag})


class LogFillModal(ModalScreen[Optional[Dict[str, Any]]]):
    """1-keystroke (`f`) modal to record a real limit-order fill attempt into data/fills.db (Test O1 k<=0.35)."""

    def __init__(
        self,
        *,
        ticker: str = "SPY",
        structure: str = "Bull Put",
        side: str = "SELL",
        limit_price: float = 1.00,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
    ) -> None:
        super().__init__()
        self._ticker = ticker.upper()
        self._structure = structure
        self._side = side.upper()
        self._limit = limit_price
        self._bid = max(0.05, limit_price - 0.10) if bid is None else float(bid)
        self._ask = (limit_price + 0.10) if ask is None else float(ask)

    def compose(self) -> ComposeResult:
        with Vertical(id="logfill-dialog"):
            yield Label(f"Log Limit-Order Fill — {self._ticker} (structure starting 'O1' counts toward the Test O1 k <= 0.35 check)")
            yield Select(
                [("IBKR", "ibkr"), ("Wealthsimple", "wealthsimple")],
                value="ibkr",
                id="lf-broker",
            )
            yield Input(value=self._ticker, placeholder="Ticker", id="lf-ticker")
            yield Input(value=self._structure, placeholder="Structure (e.g. Bull Put)", id="lf-structure")
            yield Select([("SELL (Credit)", "SELL"), ("BUY (Debit)", "BUY")], value=self._side, id="lf-side")
            yield Input(value=f"{self._bid:.2f}", placeholder="Bid (spread: natural credit)", id="lf-bid")
            yield Input(value=f"{self._ask:.2f}", placeholder="Ask (spread: far side)", id="lf-ask")
            yield Input(value=f"{self._limit:.2f}", placeholder="Limit Posted", id="lf-limit")
            with Horizontal(id="lf-filled-row"):
                yield Label("Order Filled?")
                yield Switch(value=True, id="lf-filled")
            yield Input(value="", placeholder="Note (optional)", id="lf-note")
            yield Static("", id="lf-err", classes="modal-error")
            with Horizontal(id="logfill-buttons"):
                yield Button("Cancel (Esc)", id="lf-cancel")
                yield Button("Save Fill", variant="primary", id="lf-confirm")

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "lf-cancel":
            self.dismiss(None)
            return
        if event.button.id == "lf-confirm":
            try:
                broker = str(self.query_one("#lf-broker", Select).value or "ibkr")
                tk = self.query_one("#lf-ticker", Input).value.strip().upper()
                struct = self.query_one("#lf-structure", Input).value.strip() or "Bull Put"
                side = str(self.query_one("#lf-side", Select).value or "SELL")
                bid = float(self.query_one("#lf-bid", Input).value.strip())
                ask = float(self.query_one("#lf-ask", Input).value.strip())
                lim = float(self.query_one("#lf-limit", Input).value.strip())
                filled = bool(self.query_one("#lf-filled", Switch).value)
                note = self.query_one("#lf-note", Input).value.strip()
                if not tk or bid <= 0 or ask < bid or lim <= 0:
                    raise ValueError("Require ticker, positive bid <= ask, and positive limit.")
            except ValueError as exc:
                self.query_one("#lf-err", Static).update(f"[#e06c75]{exc}[/]")
                return
            self.dismiss(
                {
                    "broker": broker,
                    "ticker": tk,
                    "structure": struct,
                    "side": side,
                    "bid": bid,
                    "ask": ask,
                    "limit_price": lim,
                    "filled": filled,
                    "fill_price": lim if filled else None,
                    "notes": note,
                }
            )


class DeskCommandProvider(Provider):
    """Command Palette (`Ctrl+P`) provider: jump to `:TICKER`, live scans, pick/fill logger, themes."""

    async def search(self, query: str) -> Hits:
        q = query.strip()
        ql = q.lower()
        app = self.app

        if q and q.replace(".", "").isalnum() and len(q) <= 6:
            tk = q.upper()
            yield Hit(
                0.9,
                f"Open Workbench: {tk}",
                lambda t=tk: app.open_workbench(t),  # type: ignore[attr-defined]
                help=f"Inspect {tk} Dossier & Shares-vs-Options Express",
            )

        if not ql or "rebal" in ql or "tfsa" in ql or "sgov" in ql:
            yield Hit(
                0.86,
                "TFSA Core Rebalance (VM_OUTLOOK_TOP3 + SGOV)",
                lambda: app.action_rebalance_modal(),  # type: ignore[attr-defined]
                help="Compute Wealthsimple USD TFSA fractional orders & SGOV reserve",
            )

        if not ql or "skew" in ql or "alpha" in ql or "gex" in ql or "edgar" in ql or "macro" in ql or "premortem" in ql:
            for lbl, cmd, desc in (
                ("Alpha Lab: Option-Implied Skew -> Shares", "SKEW", "Cremers-Weinbaum IV spread & Put Smirk [WATCHLIST]"),
                ("Alpha Lab: Dealer GEX (Focused Ticker)", "GEX", "Net dealer gamma, Gamma Flip & Call/Put OI Walls"),
                ("Alpha Lab: SEC EDGAR Forced-Flow Events", "EDGAR", "Form 10 spinoffs, 5d 13D activists & XBRL buybacks"),
                ("Alpha Lab: Macro Plumbing", "MACRO", "4-pt VIX curve, FRED Fed Net Liquidity, COT & Polymarket"),
                ("Pre-Mortem Devil's Advocate (Focused Ticker)", "PM", "Bear case, local landmines & objective kill-switches"),
            ):
                if not ql or ql in lbl.lower() or ql in cmd.lower() or "alpha" in ql:
                    yield Hit(
                        0.83,
                        lbl,
                        lambda c=cmd: app._open_alpha_modal(c, getattr(app, "focused_ticker", "SPY")),  # type: ignore[attr-defined]
                        help=desc,
                    )

        if not ql or "log" in ql or "pick" in ql:
            yield Hit(
                0.85,
                "Quick Log Stock/ETF Pick",
                lambda: app.action_quick_log(),  # type: ignore[attr-defined]
                help="Log a manual stock/ETF pick to data/own_picks.db",
            )

        if not ql or "close" in ql or "exit" in ql or "pick" in ql:
            yield Hit(
                0.84,
                "Close Open Stock/ETF Pick",
                lambda: app.action_close_pick(),  # type: ignore[attr-defined]
                help="Close an open pick in data/own_picks.db and record exit price",
            )

        if not ql or "fill" in ql or "limit" in ql:
            yield Hit(
                0.82,
                "Log Real Limit-Order Fill Attempt",
                lambda: app.action_log_fill(),  # type: ignore[attr-defined]
                help="Record bid/ask/limit fill into data/fills.db (Test O1 k<=0.35)",
            )

        if not ql or "live" in ql or "option" in ql or "etf" in ql:
            yield Hit(
                0.80,
                "Scan Live 60-120d Wide ETF Options",
                lambda: app.action_live_options(),  # type: ignore[attr-defined]
                help="Fetch live 60-120d 5%-wide ETF Bull Puts (Test O1) into Radar",
            )

        if not ql or "refresh" in ql:
            yield Hit(
                0.75,
                "Refresh Radar Scanners",
                lambda: app.action_refresh(),  # type: ignore[attr-defined]
                help="Re-run local scanners in background and update radar_cache.json",
            )

        if not ql or "price" in ql or "update" in ql:
            yield Hit(
                0.70,
                "Update Daily Stock/ETF Prices",
                lambda: app.action_update_prices(),  # type: ignore[attr-defined]
                help="Run audited Yahoo daily price refresh into data/equity_ohlcv.db",
            )

        for name in THEME_ORDER:
            if not ql or "theme" in ql or name in ql:
                yield Hit(
                    0.6,
                    f"Theme: {name}",
                    lambda n=name: app.apply_desk_theme(n),  # type: ignore[attr-defined]
                    help=f"Switch desk theme to {name}",
                )


class BloombergCommandModal(ModalScreen[Optional[str]]):
    """Bloomberg-style `:` mnemonic command bar (`NVDA`, `NVDA WB`, `REB`, `SKEW`, `GEX SPY`, `EDGAR`, `MACRO`, `PM AAPL`)."""

    def compose(self) -> ComposeResult:
        with Vertical(id="bcmd-dialog"):
            yield Label("BLOOMBERG COMMAND BAR  [dim](Enter to run · Esc to cancel)[/]")
            yield Input(
                value="",
                placeholder="NVDA | NVDA WB | REB | SKEW | GEX SPY | EDGAR | MACRO | PM AAPL | CLOSE | THEME amber_crt",
                id="bcmd-input",
            )
            yield Static(
                "[dim]Mnemonics:[/] [b]<TICKER>[/] Card · [b]<TICKER> WB[/] Workbench · [b]REB[/] TFSA Rebal · "
                "[b]SKEW[/] Opt-Skew Shares · [b]GEX <TK>[/] Dealer Gamma · [b]EDGAR[/] 13D/Spinoffs · "
                "[b]MACRO[/] Plumbing · [b]PM <TK>[/] Pre-Mortem · [b]CLOSE[/] Close Pick",
            )
            with Horizontal(id="bcmd-buttons"):
                yield Button("Cancel (Esc)", id="bcmd-cancel")
                yield Button("Execute", variant="primary", id="bcmd-run")

    def on_mount(self) -> None:
        self.query_one("#bcmd-input", Input).focus()

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "bcmd-input":
            self.dismiss(event.value.strip())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "bcmd-cancel":
            self.dismiss(None)
        elif event.button.id == "bcmd-run":
            self.dismiss(self.query_one("#bcmd-input", Input).value.strip())


class RebalanceModal(ModalScreen[Optional[Dict[str, Any]]]):
    """In-TUI Wealthsimple TFSA Core Rebalance Calculator (VM_OUTLOOK_TOP3 + SGOV cash reserve)."""

    def __init__(self, *, snap: RadarSnapshot, plan_text: str, target_shares: Dict[str, float], account_usd: float) -> None:
        super().__init__()
        self._snap = snap
        self._plan_text = plan_text
        self._target_shares = dict(target_shares)
        self._account_usd = float(account_usd)

    def compose(self) -> ComposeResult:
        with Vertical(id="reb-dialog"):
            yield Label(f"TFSA CORE REBALANCE (VM_OUTLOOK_TOP3 · Wealthsimple USD · ${self._account_usd:,.0f})")
            with VerticalScroll(id="reb-scroll"):
                yield Static(self._plan_text, id="reb-body")
            with Horizontal(id="reb-buttons"):
                yield Button("Close (Esc)", id="reb-close")
                yield Button("Apply & Save Target Shares", variant="primary", id="reb-apply")

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "reb-close":
            self.dismiss(None)
        elif event.button.id == "reb-apply":
            self.dismiss({"apply": True, "target_shares": self._target_shares, "account_usd": self._account_usd})


class AlphaViewerModal(ModalScreen[None]):
    """Scrollable Bloomberg telemetry viewer for Phase 3 Alpha / GEX / EDGAR / Macro / Pre-Mortem screens."""

    def __init__(self, *, title: str, body: str) -> None:
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        with Vertical(id="alpha-dialog"):
            yield Label(self._title)
            with VerticalScroll(id="alpha-scroll"):
                yield Static(self._body, id="alpha-body")
            with Horizontal(id="alpha-buttons"):
                yield Button("Close (Esc)", variant="primary", id="alpha-close")

    def on_key(self, event) -> None:
        if event.key in ("escape", "q"):
            self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "alpha-close":
            self.dismiss(None)
