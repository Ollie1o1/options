"""Reusable Textual widgets, QuickLogModal, and Command Palette provider."""
from __future__ import annotations

from typing import Any, Dict, Optional

from textual.app import ComposeResult
from textual.command import Hit, Hits, Provider
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from src.radar.schema import RadarSnapshot, TickerDossier
from .themes import THEME_ORDER


def format_ribbon_markup(snap: Optional[RadarSnapshot], health: Optional[Dict[str, Any]] = None) -> str:
    if snap is None:
        return "Loading Daily Trading Desk…"
    spy_s = f"{snap.spy_last:.2f}" if snap.spy_last is not None else "—"
    spy_c = f"{snap.spy_chg_1d_pct:+.1f}%" if snap.spy_chg_1d_pct is not None else ""
    qqq_c = f"{snap.qqq_chg_1d_pct:+.1f}%" if snap.qqq_chg_1d_pct is not None else "—"
    vix_s = f"{snap.vix:.1f}" if snap.vix is not None else "—"
    p_col = "#5ec98d" if snap.regime_posture == "RISK_ON" else ("#e06c75" if snap.regime_posture == "RISK_OFF" else "#d6a452")
    lab_dot = ""
    if health and health.get("worst") in ("WARN", "STALE"):
        dot_col = "#e06c75" if health["worst"] == "STALE" else "#d6a452"
        lab_dot = f"  |  [bold {dot_col}]● LAB {health['worst']}[/]"
    cache_tag = " [dim](cached)[/]" if snap.from_cache else ""
    return (
        f"[bold]SPY[/] {spy_s} {spy_c}  |  [bold]QQQ[/] {qqq_c}  |  "
        f"[bold]RV/VIX[/] {vix_s} [bold {p_col}][{snap.regime_posture}][/]  |  "
        f"Next: {snap.next_macro_event}{lab_dot}{cache_tag}"
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
    im_s = f"±{dos.implied_move_1s_pct:.1f}%" if dos.implied_move_1s_pct is not None else "—"

    lines = [
        f"[bold]TICKER DOSSIER: {dos.ticker}[/]   {spot_s}  (1d {c1} · 5d {c5})",
        "─" * 44,
        f"[bold]Trend[/]      vs 200MA {d200} · 12-1m {m12}",
        f"[bold]Momentum[/]   RSI(2) {r2} · RSI(14) {r14} · RV20 {rv_s}",
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
    if dos.rev_breadth_30d is not None:
        lines.append(f"[bold]Revisions[/]  30d breadth {dos.rev_breadth_30d:+.2f}")
    if dos.risk_flags:
        lines.append(f"[bold #e06c75]Flags[/]      {', '.join(dos.risk_flags)}")
    if dos.active_signals:
        sigs = " · ".join(f"[{s.evidence_tier}] {s.playbook_label}" for s in dos.active_signals)
        lines.append(f"[bold #5ec98d]Signals[/]    {sigs}")
    if dos.closes_60d:
        # Compact sparkline inside dossier
        blocks = "▁▂▃▄▅▆▇█"
        lo, hi = min(dos.closes_60d), max(dos.closes_60d)
        span = max(hi - lo, 1e-6)
        spark = "".join(blocks[min(7, int((v - lo) / span * 7.99))] for v in dos.closes_60d[-36:])
        lines.append(f"[bold]60d Tape[/]   {spark}")
    lines.append("")
    lines.append("[dim][Enter] Workbench   [l] Quick Log   [r] Refresh[/dim]")
    return "\n".join(lines)


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
            yield Input(value=self._playbook_id, placeholder="Playbook ID", id="ql-playbook")
            yield Input(value="", placeholder="Note (optional)", id="ql-note")
            with Horizontal(id="quicklog-buttons"):
                yield Button("Cancel", id="ql-cancel")
                yield Button("Log Pick", variant="primary", id="ql-confirm")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ql-cancel":
            self.dismiss(None)
            return
        if event.button.id == "ql-confirm":
            try:
                tk = self.query_one("#ql-ticker", Input).value.strip().upper()
                px = float(self.query_one("#ql-price", Input).value.strip())
                sh = float(self.query_one("#ql-shares", Input).value.strip())
                pb = self.query_one("#ql-playbook", Input).value.strip() or "discretionary"
                note = self.query_one("#ql-note", Input).value.strip()
            except ValueError:
                return
            self.dismiss(
                {"ticker": tk, "price": px, "shares": sh, "playbook_id": pb, "notes": note}
            )


class DeskCommandProvider(Provider):
    """Command Palette (`Ctrl+K`) provider: jump to `:TICKER`, switch themes, open Quick Log."""

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

        if not ql or "log" in ql:
            yield Hit(
                0.8,
                "Quick Log Pick",
                lambda: app.action_quick_log(),  # type: ignore[attr-defined]
                help="Log a manual stock/ETF pick to data/own_picks.db",
            )

        if not ql or "refresh" in ql:
            yield Hit(
                0.7,
                "Refresh Radar Scanners",
                lambda: app.action_refresh(),  # type: ignore[attr-defined]
                help="Re-run local scanners and update radar_cache.json",
            )

        for name in THEME_ORDER:
            if not ql or "theme" in ql or name in ql:
                yield Hit(
                    0.6,
                    f"Theme: {name}",
                    lambda n=name: app.apply_desk_theme(n),  # type: ignore[attr-defined]
                    help=f"Switch desk theme to {name}",
                )
