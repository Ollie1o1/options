"""[6] DISCOVER tab pieces for the Textual desk: table fillers and the full-screen Evidence Card.

Screen map (always the same, so the reader knows where they are):

  top     ① MARKET  — one compact panel, context only
  left    ② SECTORS — the 16 sector/asset ETFs in PROVEN Outlook order   (Enter = drill into a sector)
  right   ③ IDEAS   — pick a lens; its evidence label and caveat sit above the rows
  Enter on a row → ④ EVIDENCE CARD (full screen; Esc returns, w = Workbench)
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Static

from src.topdown.lenses import LENSES, LensResult, tags_by_ticker
from src.topdown.render import (
    POSTURE_STYLE,
    breadcrumb,
    card_renderable,
    chip,
    freshness_line,
    lens_header,
    signed,
    tag_chips,
)

LENS_OPTIONS: List[Tuple[str, str]] = [(f"[{lz.label}] {lz.title}", lz.id) for lz in LENSES]

BOARD_COLUMNS = ("#", "ETF", "Sector", "Call", "Sleeve", "RS 1m", "RS 3m", ">200d", "n")
LENS_COLUMNS = ("Ticker", "Sector", "Listed in", "Warnings", "Why it is listed", "As of")
SHORT_NAME = {
    "Consumer Staples": "Staples", "Consumer Discretionary": "Discretionary",
    "Communication Services": "Communication", "Long Treasuries": "Long Treasury",
    "Semiconductors": "Semis",
}
SECTOR_COLUMNS = ("Ticker", "12-1m", "1m", "3m", "vs 200d", "Vol 20d", "Listed in")
DTE_COLUMNS = ("DTE Bucket", "Verdict", "Best Setup", "Fric%", "Calibrated P(Win) [95% CI]", "PF", "AvgLoss", "Δ-Adj")


def fill_dte_strip(tbl: DataTable, bundle: Any) -> List[str]:
    """Fill the ③ table with Options-by-Expiry (0DTE -> 240D) & calibrated probabilities."""
    tbl.clear(columns=True)
    tbl.add_columns(*DTE_COLUMNS)
    out: List[str] = []
    for i, r in enumerate(bundle.dte_rows):
        setup = f"{r.ticker} {r.strikes} ({r.dte}d)" if r.ticker else "— (archive)"
        fric_s = f"{r.friction_pct*100:.1f}%" if r.friction_pct is not None else "—"
        if r.p_win is not None and r.n_backtest > 0:
            ci = f"[{r.p_win_ci95[0]*100:.0f}-{r.p_win_ci95[1]*100:.0f}%]" if r.p_win_ci95 else ""
            pw_s = f"{r.p_win*100:.0f}% {ci} n={r.n_backtest}"
        else:
            pw_s = "—"
        pf_s = f"{r.pf:.2f}" if r.pf is not None else "—"
        al_s = f"{r.avg_loss*100:.1f}%" if r.avg_loss is not None else "—"
        ex_s = signed(r.excess_limit * 100.0, "{:+.2f}%") if r.excess_limit is not None else Text("—", style="dim")
        verdict = Text(" CLOSED ", style="bold #0e1117 on #e06c75") if r.evidence_label == "CLOSED" else chip(r.evidence_label)
        tk_key = r.ticker or "SPY"
        tbl.add_row(
            Text(r.bucket_label, style="bold"), verdict, setup, fric_s, pw_s, pf_s, al_s, ex_s,
            key=f"{i}|{tk_key}",
        )
        out.append(tk_key)
    return out


def dte_header(bundle: Any, *, sort_by: str = "dte") -> Text:
    sort_lbl = "0DTE → 6M+ (shortest first)" if sort_by == "dte" else "Best evidence first"
    t = Text("OPTIONS BY EXPIRY [TAXABLE_ONLY] — ", style="bold")
    t.append(f"Sorted: {sort_lbl}\n", style="#d6a452")
    t.append(f"Clock: {bundle.execution_window} · {bundle.execution_note}", style="dim")
    return t


def fill_board(tbl: DataTable, hub: Any) -> None:
    tbl.clear()
    for r in hub.board.rows:
        call = Text(r.posture, style=POSTURE_STYLE.get(r.posture, ""))
        tbl.add_row(
            str(r.rank or "—"), Text(r.etf, style="bold"), SHORT_NAME.get(r.name, r.name)[:14], call,
            Text(f"{r.weight*100:.0f}%", style="bold #5ec98d") if r.weight > 0 else Text("—", style="dim"),
            signed(r.rs_21d_pct, "{:+.0f}%"), signed(r.rs_63d_pct, "{:+.0f}%"),
            "—" if r.pct_above_200d is None else f"{r.pct_above_200d:.0f}%",
            str(r.n_members or "—"),
            key=r.etf,
        )


def fill_lens(tbl: DataTable, res: LensResult, all_results: List[LensResult]) -> List[str]:
    """Fill the ③ table with one lens; returns the tickers in row order."""
    tbl.clear(columns=True)
    tbl.add_columns(*LENS_COLUMNS)
    tags = tags_by_ticker(all_results)
    out: List[str] = []
    for i, r in enumerate(res.rows):
        tbl.add_row(
            Text(r.ticker, style="bold"), r.sector_etf or "—", tag_chips(tags.get(r.ticker, [])),
            Text(", ".join(r.flags), style="#e06c75"), (f"{r.evidence_label} · " + r.why if r.evidence_label else r.why)[:70], r.as_of or "—",
            key=f"{i}|{r.ticker}",
        )
        out.append(r.ticker)
    return out


def fill_sector(tbl: DataTable, hub: Any, etf: str) -> List[str]:
    """Fill the ③ table with a sector's constituents (descriptive, 12-1m order)."""
    from src.topdown.peers import _metrics
    from src.topdown.sectors import sector_members

    tbl.clear(columns=True)
    tbl.add_columns(*SECTOR_COLUMNS)
    tags = tags_by_ticker(hub.lenses)
    ms = [(m, _metrics(hub.ohlcv, m)) for m in sector_members(etf, hub.ohlcv)]
    ms.sort(key=lambda kv: -(kv[1].get("ret_12_1_pct") or -1e9))
    out: List[str] = []
    for i, (sym, m) in enumerate(ms):
        tbl.add_row(
            Text(sym, style="bold"), signed(m.get("ret_12_1_pct"), "{:+.0f}%"), signed(m.get("ret_21d_pct")),
            signed(m.get("ret_63d_pct")), signed(m.get("vs_200d_pct")),
            "—" if m.get("rv20_pct") is None else f"{m['rv20_pct']:.0f}%",
            tag_chips(tags.get(sym, [])),
            key=f"{i}|{sym}",
        )
        out.append(sym)
    return out


def sector_header(hub: Any, etf: str) -> Text:
    row = hub.board.row(etf)
    t = Text("② SECTORS › ", style="bold")
    t.append(f"{etf} {row.name if row else ''} constituents", style="bold")
    if row:
        t.append(f" · Outlook #{row.rank} ")
        t.append(row.posture, style=POSTURE_STYLE.get(row.posture, ""))
        t.append(f" · {row.pct_above_200d or 0:.0f}% of {row.n_members} above 200d")
    t.append("\nSorted by 12-1 month return — description, not a forecast. ", style="dim")
    t.append_text(chip("CONTEXT"))
    t.append("  Pick a lens in the ③ IDEAS box to go back.", style="dim")
    return t


def board_header() -> Text:
    t = Text("② SECTORS — ranked by the ", style="bold")
    t.append_text(chip("PROVEN"))
    t.append(" rotation model", style="bold")
    t.append("\nFAVOR = held by the TFSA sleeve · AVOID = regime-gated bearish · Enter = see the stocks", style="dim")
    return t


def lens_tag_legend() -> Text:
    from src.topdown.render import tag_legend

    return tag_legend()


def crumb_and_freshness(hub: Optional[Any], active: int) -> Text:
    t = breadcrumb(active)
    if hub is not None:
        t.append("   ")
        t.append_text(freshness_line(hub))
    return t


class EvidenceCardScreen(ModalScreen[Optional[str]]):
    """④ Full-screen Evidence Card. Esc = back; w = open Workbench; c = open Chain Matrix."""

    BINDINGS = [
        ("escape", "close", "Back"),
        ("q", "close", "Back"),
        ("w", "workbench", "Workbench"),
        ("c", "chain", "Chain"),
    ]

    def __init__(self, ticker: str, builder: Callable[[str], Any]) -> None:
        super().__init__()
        self.ticker = ticker.upper()
        self._builder = builder

    def compose(self) -> ComposeResult:
        with Vertical(id="card-dialog"):
            yield Static(breadcrumb(4), id="card-crumb")
            with VerticalScroll(id="card-scroll"):
                yield Static(f"Building the Evidence Card for {self.ticker}…", id="card-body")
            yield Static("[b]Esc[/b] back to ③ IDEAS   [b]w[/b] Workbench   [b]c[/b] Options Chain   "
                         "[b]↑↓ PgUp PgDn[/b] scroll", id="card-keys")

    def on_mount(self) -> None:
        self._build()

    @work(thread=True, exclusive=True, group="evidence_card")
    def _build(self) -> None:
        try:
            card = self._builder(self.ticker)
            renderable: Any = card_renderable(card)
        except Exception as exc:  # show the failure in place, never crash the desk
            renderable = Text(f"Could not build the card for {self.ticker}: {exc}", style="#e06c75")
        self.app.call_from_thread(self._show, renderable)

    def _show(self, renderable: Any) -> None:
        self.query_one("#card-body", Static).update(renderable)
        self.query_one("#card-scroll", VerticalScroll).focus()

    def action_close(self) -> None:
        self.dismiss(None)

    def action_workbench(self) -> None:
        self.dismiss(self.ticker)

    def action_chain(self) -> None:
        self.dismiss(f"CHAIN:{self.ticker}")


__all__ = [
    "LENS_OPTIONS", "BOARD_COLUMNS", "LENS_COLUMNS", "SECTOR_COLUMNS", "DTE_COLUMNS",
    "fill_board", "fill_lens", "fill_sector", "fill_dte_strip",
    "sector_header", "board_header", "dte_header", "lens_tag_legend",
    "crumb_and_freshness", "EvidenceCardScreen", "lens_header",
]
