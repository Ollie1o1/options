"""Interactive 5-tab Textual Daily Trading Cockpit (`src/tui/app.py`).

Tabs:
  [1] TODAY      — Core TFSA Allocation (VM_OUTLOOK_TOP3) + Top 5 Radar Setups
  [2] RADAR      — Filterable (presets + query), sortable unified Opportunity Radar + live DossierPane
  [3] WORKBENCH  — Full Ticker Dossier + Visual RND/Edge Bars + Shares-vs-Options + Walk-Away Ladder
  [4] BOOK       — Stock/ETF Picks (live P&L + exit alerts), Paper Options, Holdings & Scorecard
  [5] LAB        — Real Ops Health, OHLCV Freshness, Fill Concession `k`, Forward Readouts & Playbooks
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Input,
    Label,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)

from src.cli.guide import TIER_PLAIN, plain_english_action_plan, plain_english_vehicle_verdict
from src.radar.engine import merge_confluence
from src.radar.schema import Opportunity, RadarSnapshot, TickerDossier
from src.structure.types import DetailedView, VehicleComparison
from .state import DeskDataProvider
from .themes import THEME_ORDER, THEMES, persist_theme_name, saved_theme_name
from .widgets import (
    DeskCommandProvider,
    HelpModal,
    LogFillModal,
    QuickLogModal,
    format_dossier_markup,
    format_ribbon_markup,
    format_workbench_visual_markup,
)

SORT_MODES = ("tier", "confluence", "friction", "ticker")

FILTER_PRESETS = (
    ("preset-all", "All", ""),
    ("preset-act", "Proven+Lead", "actionable"),
    ("preset-stk", "Stocks", "stock"),
    ("preset-opt", "Options", "option"),
    ("preset-cln", "Clean", "clean"),
    ("preset-cnf", "Conf>=2", "conf>=2"),
)


def filter_opportunities(opps: List[Opportunity], query: str) -> List[Opportunity]:
    """Client-side micro-query filter (`stock`, `option`, `validated`, `actionable`, `clean`, `conf>=2`, `fric<10%`, or substring)."""
    q = query.strip().lower()
    if not q:
        return list(opps)
    tokens = q.split()
    out: List[Opportunity] = []
    for o in opps:
        ok = True
        for tok in tokens:
            if tok in ("stock", "etf", "option", "crypto"):
                ok = o.asset_class.lower() == tok
            elif tok in ("validated", "lead", "context", "research"):
                ok = o.evidence_tier.lower() == tok
            elif tok == "actionable":
                ok = o.evidence_tier.upper() in ("VALIDATED", "LEAD")
            elif tok == "clean":
                ok = len(o.risk_flags) == 0
            elif tok.startswith("conf>="):
                try:
                    ok = o.confluence_count >= int(tok[6:])
                except ValueError:
                    ok = False
            elif tok.startswith("fric<"):
                try:
                    lim = float(tok[5:].rstrip("%")) / (100.0 if "%" in tok else 1.0)
                    ok = o.friction_value <= lim
                except ValueError:
                    ok = False
            else:
                hay = f"{o.ticker} {o.playbook_id} {o.playbook_label} {o.setup_summary} {' '.join(o.badges)}".lower()
                ok = tok in hay
            if not ok:
                break
        if ok:
            out.append(o)
    return out


class DeskApp(App[None]):
    """5-Tab Daily Stock + Option Trading Desk."""

    CSS_PATH = "styles.tcss"
    COMMANDS = App.COMMANDS | {DeskCommandProvider}

    BINDINGS = [
        Binding("1", "switch_tab('today')", "Today", show=True),
        Binding("2", "switch_tab('radar')", "Radar", show=True),
        Binding("3", "switch_tab('workbench')", "Workbench", show=True),
        Binding("4", "switch_tab('book')", "Book", show=True),
        Binding("5", "switch_tab('lab')", "Lab", show=True),
        Binding("slash", "focus_filter", "Filter", show=True),
        Binding("s", "cycle_sort", "Sort", show=True),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("l", "quick_log", "Log Pick", show=True),
        Binding("f", "log_fill", "Log Fill", show=True),
        Binding("o", "live_options", "Live Opts", show=True),
        Binding("u", "update_prices", "Upd Prices", show=False),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("t", "cycle_theme", "Theme", show=True),
        Binding("question_mark", "show_help", "Guide/Help", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(
        self,
        *,
        provider: Optional[DeskDataProvider] = None,
        initial_tab: str = "today",
        offline: bool = False,
    ) -> None:
        super().__init__()
        self.provider = provider or DeskDataProvider()
        self.initial_tab = initial_tab
        self.offline = offline
        self.snapshot: Optional[RadarSnapshot] = None
        self.health: Dict[str, Any] = {}
        self.filtered_opps: List[Opportunity] = []
        self.sort_idx: int = 0
        self.focused_ticker: str = "SPY"
        self.focused_spot: float = 100.0
        self.focused_playbook: str = "equity_pullback"
        self._last_vehicles: List[VehicleComparison] = []
        self._status_msg: str = ""
        self._wb_loaded_ticker: Optional[str] = None
        self._last_dossier_ticker: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Static("Loading Daily Trading Desk…", id="market-ribbon")
        with TabbedContent(initial=self.initial_tab, id="main-tabs"):
            with TabPane("[1] TODAY", id="today"):
                with Vertical(id="today-grid"):
                    yield Static("Loading your plain-English daily action plan…", id="today-core-card")
                    with Horizontal(id="today-actions"):
                        yield Button("Refresh Ideas [r]", id="act-refresh", variant="primary")
                        yield Button("Live ETF Spreads [o]", id="act-live-opts")
                        yield Button("Analyze a Stock", id="act-workbench")
                        yield Button("Log Pick [l]", id="act-log")
                        yield Button("Log Fill [f]", id="act-fill")
                        yield Button("? Guide", id="act-help")
                    yield DataTable(id="today-top5-table", zebra_stripes=True, cursor_type="row")

            with TabPane("[2] RADAR", id="radar"):
                with Horizontal(id="radar-toolbar"):
                    yield Input(placeholder="Filter (e.g. pullback, clean, conf>=2)…", id="radar-filter-input")
                    for pid, plabel, _pq in FILTER_PRESETS:
                        yield Button(plabel, id=pid, classes="preset-btn")
                    yield Static("Sort: tier [s]", id="radar-pills")
                with Horizontal(id="radar-split"):
                    yield DataTable(id="radar-table", zebra_stripes=True, cursor_type="row")
                    yield Static("Select a row to inspect Ticker Dossier.", id="radar-dossier-pane")

            with TabPane("[3] WORKBENCH", id="workbench"):
                with Horizontal(id="wb-header"):
                    yield Input(value="SPY", placeholder="Ticker", id="wb-ticker-input")
                    yield Select(
                        [("BULLISH", "BULLISH"), ("BEARISH", "BEARISH"), ("NEUTRAL", "NEUTRAL")],
                        value="BULLISH",
                        id="wb-dir-select",
                    )
                    yield Input(value="+8%", placeholder="Target", id="wb-target-input")
                    yield Input(value="30", placeholder="Days", id="wb-days-input")
                    yield Input(value="500", placeholder="Risk $", id="wb-risk-input")
                    yield Label("100sh:")
                    yield Switch(value=False, id="wb-owns-switch")
                    yield Button("Run", id="wb-run-btn", variant="primary")
                    yield Button("Log Fill [f]", id="wb-fill-btn")
                with Horizontal(id="wb-split"):
                    yield Static("Enter a ticker to load Dossier.", id="wb-left")
                    with Vertical(id="wb-right"):
                        yield Static("Plain-English recommendation will appear here.", id="wb-verdict-box")
                        yield Static("", id="wb-visual-box")
                        yield DataTable(id="wb-vehicles-table", zebra_stripes=True, cursor_type="row")
                        yield Static("Select a vehicle row to inspect Walk-Away Limit Ladder.", id="wb-ladder-box")

            with TabPane("[4] BOOK", id="book"):
                with Vertical(id="book-container"):
                    yield Static("Playbook Scorecard loading…", id="book-scorecard-box")
                    yield DataTable(id="book-picks-table", zebra_stripes=True, cursor_type="row")
                    yield Static("", id="book-extra-box")

            with TabPane("[5] LAB", id="lab"):
                with Vertical(id="lab-container"):
                    yield Static("Ops Health loading…", id="lab-health-box")
                    yield DataTable(id="lab-playbooks-table", zebra_stripes=True, cursor_type="row")

        yield Footer()

    def on_mount(self) -> None:
        for th in THEMES.values():
            self.register_theme(th)
        self.theme = saved_theme_name()
        self._apply_responsive_class(self.size.width)
        self._setup_tables()
        self._load_all_data(refresh=False)

    def on_resize(self, event) -> None:
        self._apply_responsive_class(event.size.width)

    def _apply_responsive_class(self, width: int) -> None:
        if width < 110:
            self.screen.add_class("compact")
        else:
            self.screen.remove_class("compact")

    def _setup_tables(self) -> None:
        t5 = self.query_one("#today-top5-table", DataTable)
        t5.add_columns("Evidence", "Ticker", "Strategy", "Direction", "Match", "Cost", "Why It Triggered (Press Enter to Plan)")

        rt = self.query_one("#radar-table", DataTable)
        rt.add_columns("Evidence", "Ticker", "Strategy", "Direction", "Match", "Cost", "Warnings", "Why It Triggered")

        vt = self.query_one("#wb-vehicles-table", DataTable)
        vt.add_columns("Vehicle", "Trade Plan", "TFSA?", "Max Risk", "Spread%", "Mkt EV", "Your Edge", "Win Odds")

        bt = self.query_one("#book-picks-table", DataTable)
        bt.add_columns("ID", "Date", "Ticker", "Strategy", "Entry", "Now", "P&L%", "Shares", "Exit / Status")

        lt = self.query_one("#lab-playbooks-table", DataTable)
        lt.add_columns("Evidence", "Strategy ID", "Asset", "Horizon", "Notes")

    def _set_status(self, msg: str) -> None:
        self._status_msg = msg
        self._render_ribbon()

    def _apply_snapshot(self, snap: RadarSnapshot, health: Dict[str, Any]) -> None:
        self.snapshot = snap
        self.health = health
        self._status_msg = ""
        self._last_dossier_ticker = None
        self._render_ribbon()
        self._render_today()
        self._render_radar()
        self._render_book()
        self._render_lab()
        if self.snapshot.opportunities:
            first = self.snapshot.opportunities[0]
            self.focused_ticker = first.ticker
            self.focused_spot = first.spot
            self.focused_playbook = first.playbook_id
            self._update_radar_dossier(first.ticker)
        self._wb_loaded_ticker = None
        if self.query_one("#main-tabs", TabbedContent).active == "workbench":
            self._compute_workbench(self.focused_ticker)

    def _load_all_data(self, *, refresh: bool = False) -> None:
        snap = self.provider.get_radar(refresh=refresh)
        health = self.provider.get_health()
        self._apply_snapshot(snap, health)
        try:
            from src.cache_release import release_yfinance_caches

            release_yfinance_caches()
        except Exception:
            pass

    def _render_ribbon(self) -> None:
        self.query_one("#market-ribbon", Static).update(
            format_ribbon_markup(self.snapshot, self.health, status_msg=self._status_msg)
        )

    def _render_today(self) -> None:
        snap = self.snapshot
        if snap is None:
            return
        lines = ["[bold]YOUR DAILY ACTION PLAN (Plain English — press [?] for Cheat Sheet)[/]"]
        for step in plain_english_action_plan(snap):
            lines.append("  " + step)
        self.query_one("#today-core-card", Static).update("\n".join(lines))

        tbl = self.query_one("#today-top5-table", DataTable)
        tbl.clear()
        for o in snap.opportunities[:5]:
            tbl.add_row(
                TIER_PLAIN.get(o.evidence_tier, o.evidence_tier),
                o.ticker,
                o.playbook_label[:22],
                "▲ BUY" if o.direction == "LONG" else ("▼ SELL" if o.direction == "SHORT" else "• FLAT"),
                str(o.confluence_count),
                o.friction_label or f"{o.friction_value*100:.2f}%",
                o.setup_summary[:55],
                key=f"{o.ticker}|{o.playbook_id}",
            )

    def _render_radar(self) -> None:
        snap = self.snapshot
        if snap is None:
            return
        q = self.query_one("#radar-filter-input", Input).value
        rows = filter_opportunities(snap.opportunities, q)
        mode = SORT_MODES[self.sort_idx]
        if mode == "confluence":
            rows.sort(key=lambda o: (-o.confluence_count, o.friction_value, o.ticker))
        elif mode == "friction":
            rows.sort(key=lambda o: (o.friction_value, o.ticker))
        elif mode == "ticker":
            rows.sort(key=lambda o: (o.ticker, o.playbook_id))
        self.filtered_opps = rows

        self.query_one("#radar-pills", Static).update(
            f"Sort: [bold]{mode}[/] ([s]) · {len(rows)}/{len(snap.opportunities)}"
        )
        tbl = self.query_one("#radar-table", DataTable)
        tbl.clear()
        for idx, o in enumerate(rows):
            tbl.add_row(
                TIER_PLAIN.get(o.evidence_tier, o.evidence_tier),
                o.ticker,
                o.playbook_label[:20],
                "▲ BUY" if o.direction == "LONG" else ("▼ SELL" if o.direction == "SHORT" else "• FLAT"),
                str(o.confluence_count),
                o.friction_label or f"{o.friction_value*100:.2f}%",
                ",".join(o.risk_flags[:1]) or "—",
                o.setup_summary[:45],
                key=str(idx),
            )

    def _update_radar_dossier(self, ticker: str) -> None:
        sym = str(ticker).upper().strip()
        cached = self.provider.get_cached_dossier(sym)
        if cached is not None:
            self._last_dossier_ticker = sym
            self.query_one("#radar-dossier-pane", Static).update(format_dossier_markup(cached))
            return
        if self._last_dossier_ticker == sym:
            return
        self._last_dossier_ticker = sym
        self.query_one("#radar-dossier-pane", Static).update(
            f"[bold]TICKER DOSSIER: {sym}[/]\n[dim]Loading price structure, chain IV & news…[/dim]"
        )
        self._worker_load_dossier(sym)

    @work(thread=True, exclusive=True, group="radar_dossier")
    def _worker_load_dossier(self, ticker: str) -> None:
        try:
            dos = self.provider.get_dossier(ticker, fetch_chain=False)

            def _apply() -> None:
                if self.focused_ticker == ticker:
                    self.query_one("#radar-dossier-pane", Static).update(format_dossier_markup(dos))

            self.call_from_thread(_apply)
        except Exception:
            pass

    def _compute_workbench(self, ticker: str) -> None:
        sym = ticker.upper().strip() or "SPY"
        self.focused_ticker = sym
        self._wb_loaded_ticker = sym
        self.query_one("#wb-ticker-input", Input).value = sym
        dos: TickerDossier = self.provider.get_dossier(sym, fetch_chain=False)
        self.query_one("#wb-left", Static).update(format_dossier_markup(dos))

        spot = dos.spot if dos.spot is not None else 100.0
        self.focused_spot = spot
        iv = (
            (dos.atm_iv / 100.0)
            if dos.atm_iv is not None
            else ((dos.rv_20_ann / 100.0) if dos.rv_20_ann else 0.25)
        )

        dir_val = str(self.query_one("#wb-dir-select", Select).value or "BULLISH")
        raw_t = self.query_one("#wb-target-input", Input).value.strip().rstrip("%")
        try:
            t_num = float(raw_t)
            t_move = t_num / 100.0 if abs(t_num) > 1.0 or "%" in self.query_one("#wb-target-input", Input).value else t_num
        except ValueError:
            t_move = 0.08
        if dir_val == "BEARISH" and t_move > 0:
            t_move = -t_move
        if dir_val == "NEUTRAL":
            t_move = 0.0
        try:
            days = max(5, int(self.query_one("#wb-days-input", Input).value.strip()))
        except ValueError:
            days = 30
        try:
            risk = max(50.0, float(self.query_one("#wb-risk-input", Input).value.strip()))
        except ValueError:
            risk = 500.0
        owns_100 = bool(self.query_one("#wb-owns-switch", Switch).value)

        view = DetailedView(
            symbol=sym,
            direction=dir_val,
            target_move_pct=t_move,
            horizon_days=days,
            risk_budget_usd=risk,
            account_type="tfsa",
            owns_100_shares=owns_100,
        )
        vehicles = self.provider.get_express(view, spot=spot, iv=iv)
        self._last_vehicles = vehicles
        self.query_one("#wb-verdict-box", Static).update(
            f"[bold]{plain_english_vehicle_verdict(vehicles, view)}[/]"
        )
        self.query_one("#wb-visual-box", Static).update(
            format_workbench_visual_markup(vehicles, dos, view)
        )

        vt = self.query_one("#wb-vehicles-table", DataTable)
        vt.clear()
        for idx, v in enumerate(vehicles):
            vt.add_row(
                ("★ " if v.recommended else "") + v.vehicle,
                v.description[:34],
                "Yes" if v.tfsa_legal else "No",
                f"${v.capital_at_risk_usd:,.0f}",
                f"{v.friction_pct_of_reward:.1f}%",
                f"${v.ev_mkt_net_usd:+,.0f}",
                f"${v.ev_view_net_usd:+,.0f}",
                f"{v.pop_view*100:.0f}%",
                key=str(idx),
            )
        if vehicles:
            v0 = next((v for v in vehicles if v.recommended), vehicles[0])
            self.query_one("#wb-ladder-box", Static).update(
                f"[bold]Selected Vehicle Execution Ticket ({v0.vehicle}):[/]\n"
                f"  {v0.walk_away_label}\n"
                f"  Delta $: ${v0.delta_dollars:+,.0f} · Theta/day: ${v0.theta_per_day_usd:+.2f}"
                + (f" · [#d6a452]{v0.caution}[/]" if v0.caution else "")
            )

    def _render_book(self) -> None:
        book = self.provider.get_book()
        picks = book.get("equity_picks") or []
        sc = book.get("scorecard") or {}
        by_pb = sc.get("by_playbook") or {}
        alerts = book.get("exit_alerts") or []
        paper_opts = book.get("paper_options") or []
        lt_holdings = book.get("longterm_holdings") or []

        lines = [f"[bold]OWN PICKS SCORECARD[/]  (Total stock/ETF picks: {sc.get('total_picks', 0)})"]
        if alerts:
            lines.append("[bold #d6a452]EXIT ALERTS:[/] " + "  |  ".join(alerts[:4]))
        if by_pb:
            for pb_id, info in by_pb.items():
                h5 = (info.get("horizons") or {}).get(5) or {}
                m5 = f"{h5['mean']*100:+.2f}%" if h5.get("mean") is not None else "—"
                lines.append(
                    f"  • [bold]{pb_id}[/]: n={info['n_picks']} (open {info['n_open']}) · "
                    f"5d excess {m5} · {h5.get('verdict', '')}"
                )
        else:
            lines.append("  [dim]Press [l] anywhere to log your first stock/ETF pick · [f] to log a limit fill.[/dim]")
        self.query_one("#book-scorecard-box", Static).update("\n".join(lines))

        bt = self.query_one("#book-picks-table", DataTable)
        bt.clear()
        for p in picks[:40]:
            cur_p = p.get("current_price")
            unrl = p.get("unrealized_pct")
            exit_s = p.get("exit_signal") or str(p.get("status", ""))
            bt.add_row(
                str(p["pick_id"]),
                str(p["date"])[:10],
                str(p["ticker"]),
                str(p["playbook_id"]),
                f"${float(p['entry_price']):.2f}",
                f"${float(cur_p):.2f}" if cur_p is not None else "—",
                f"{float(unrl):+.1f}%" if unrl is not None else "—",
                f"{float(p['shares']):g}",
                str(exit_s),
            )

        extra_lines: List[str] = []
        if paper_opts:
            opt_str = " · ".join(
                f"{r['ticker']} {r['strategy_name']} ${r.get('strike') or ''} exp {str(r.get('expiration') or '')[:10]}"
                for r in paper_opts[:5]
            )
            extra_lines.append(f"[bold]OPEN PAPER OPTIONS ({len(paper_opts)}):[/] {opt_str}")
        else:
            extra_lines.append("[bold]OPEN PAPER OPTIONS:[/] [dim]0 open in paper_trades.db[/dim]")
        if lt_holdings:
            lt_str = " · ".join(
                f"{h['ticker']} ({h['tranches_filled']}/{h['tranches_total']} tranches"
                + (f", next ${h['next_level']:g}" if h.get("next_level") is not None else "")
                + ")"
                for h in lt_holdings[:6]
            )
            extra_lines.append(f"[bold]LONG-TERM PLAN:[/] {lt_str}")
        self.query_one("#book-extra-box", Static).update("\n".join(extra_lines))

    def _render_lab(self) -> None:
        h = self.health
        fills = h.get("fills") or {}
        maint = h.get("maintenance_summary") or ""
        ohlcv_n = h.get("ohlcv_symbols")
        ohlcv_last = h.get("ohlcv_last")
        ohlcv_s = f" · OHLCV: {ohlcv_n} syms ({ohlcv_last})" if ohlcv_n else ""
        maint_s = f" · {maint}" if maint else ""
        k_groups = fills.get("groups") or []
        k_str = ""
        if k_groups:
            med_k = k_groups[0].get("median_k")
            if med_k is not None:
                k_str = f" (med k={med_k:.2f}, goal <=0.35)"
        lines = [
            f"[bold]OPS HEALTH:[/] Disk {h.get('disk_free_gb', '?')} GB free [{h.get('worst', 'OK')}]{ohlcv_s}{maint_s}",
            f"[bold]EXECUTION:[/] Live OFF · Real Limit Fills Logged: {fills.get('total_filled', 0)}/{fills.get('total_attempts', 0)}{k_str} (press [f] to log)",
            "[bold]PENDING FORWARD READOUTS:[/] " + "  |  ".join(h.get("readouts") or []),
        ]
        self.query_one("#lab-health-box", Static).update("\n".join(lines))

        lt = self.query_one("#lab-playbooks-table", DataTable)
        lt.clear()
        for pb in self.provider.get_playbooks():
            lt.add_row(pb.evidence_tier, pb.id, pb.asset_class, pb.horizon_label, pb.notes[:72])

    # ── Background Workers (non-blocking UI) ─────────────────────────────────

    @work(thread=True, exclusive=True, group="radar_refresh")
    def _worker_refresh_radar(self) -> None:
        try:
            snap = self.provider.get_radar(refresh=True)
            health = self.provider.get_health()
            self.call_from_thread(self._apply_snapshot, snap, health)
        except Exception as exc:
            self.call_from_thread(self._set_status, f"Refresh err: {str(exc)[:30]}")

    @work(thread=True, exclusive=True, group="live_options")
    def _worker_live_options(self) -> None:
        try:
            live_opps, _warns = self.provider.get_live_etf_options()

            def _merge() -> None:
                if self.snapshot is not None:
                    base = [o for o in self.snapshot.opportunities if "LIVE_CHAIN" not in o.badges]
                    self.snapshot.opportunities = merge_confluence(base + live_opps)
                    self._render_today()
                    self._render_radar()
                self._set_status(f"Live ETF opts: {len(live_opps)} found")

            self.call_from_thread(_merge)
        except Exception as exc:
            self.call_from_thread(self._set_status, f"Live opts err: {str(exc)[:30]}")

    @work(thread=True, exclusive=True, group="price_update")
    def _worker_update_prices(self) -> None:
        try:
            res = self.provider.update_prices(expand_universe=False)
            snap = self.provider.get_radar(refresh=True)
            health = self.provider.get_health()

            def _done() -> None:
                self._apply_snapshot(snap, health)
                self._set_status(f"Prices updated ({res.get('updated', 0)}/{res.get('symbols', 0)})")

            self.call_from_thread(_done)
        except Exception as exc:
            self.call_from_thread(self._set_status, f"Price update err: {str(exc)[:30]}")

    # ── Events ───────────────────────────────────────────────────────────────

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        pane_id = event.pane.id if event.pane is not None else ""
        if pane_id == "workbench" and self._wb_loaded_ticker != self.focused_ticker:
            self._compute_workbench(self.focused_ticker)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "radar-filter-input":
            self._render_radar()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id in ("wb-ticker-input", "wb-target-input", "wb-days-input", "wb-risk-input"):
            self._compute_workbench(self.query_one("#wb-ticker-input", Input).value)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "wb-dir-select":
            self._compute_workbench(self.query_one("#wb-ticker-input", Input).value)

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "wb-owns-switch":
            self._compute_workbench(self.query_one("#wb-ticker-input", Input).value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "wb-run-btn":
            self._compute_workbench(self.query_one("#wb-ticker-input", Input).value)
        elif bid == "wb-fill-btn" or bid == "act-fill":
            self.action_log_fill()
        elif bid == "act-refresh":
            self.action_refresh()
        elif bid == "act-live-opts":
            self.action_live_options()
        elif bid == "act-workbench":
            self.open_workbench(self.focused_ticker)
        elif bid == "act-log":
            self.action_quick_log()
        elif bid == "act-help":
            self.action_show_help()
        else:
            for pid, _label, pq in FILTER_PRESETS:
                if bid == pid:
                    self.query_one("#radar-filter-input", Input).value = pq
                    self._render_radar()
                    break

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        tid = event.data_table.id
        if tid == "radar-table" and event.row_key is not None:
            try:
                idx = int(str(event.row_key.value))
                opp = self.filtered_opps[idx]
                self.focused_ticker = opp.ticker
                self.focused_spot = opp.spot
                self.focused_playbook = opp.playbook_id
                self._update_radar_dossier(opp.ticker)
            except (ValueError, IndexError):
                pass
        elif tid == "wb-vehicles-table" and event.row_key is not None:
            try:
                idx = int(str(event.row_key.value))
                v = self._last_vehicles[idx]
                self.query_one("#wb-ladder-box", Static).update(
                    f"[bold]Selected Vehicle Execution Ticket ({v.vehicle}):[/]\n"
                    f"  {v.walk_away_label}\n"
                    f"  Delta $: ${v.delta_dollars:+,.0f} · Theta/day: ${v.theta_per_day_usd:+.2f}"
                    + (f" · [#d6a452]{v.caution}[/]" if v.caution else "")
                )
            except (ValueError, IndexError):
                pass

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        tid = event.data_table.id
        if tid == "radar-table" and event.row_key is not None:
            try:
                opp = self.filtered_opps[int(str(event.row_key.value))]
                self.open_workbench(opp.ticker)
            except (ValueError, IndexError):
                pass
        elif tid == "today-top5-table" and event.row_key is not None:
            tk = str(event.row_key.value).split("|")[0]
            self.open_workbench(tk)

    # ── Actions ──────────────────────────────────────────────────────────────

    def open_workbench(self, ticker: str) -> None:
        self.query_one("#main-tabs", TabbedContent).active = "workbench"
        self._compute_workbench(ticker)
        try:
            self.query_one("#wb-vehicles-table", DataTable).focus()
        except Exception:
            pass

    def action_switch_tab(self, tab_id: str) -> None:
        self.query_one("#main-tabs", TabbedContent).active = tab_id

    def action_focus_filter(self) -> None:
        self.query_one("#main-tabs", TabbedContent).active = "radar"
        self.query_one("#radar-filter-input", Input).focus()

    def action_cycle_sort(self) -> None:
        self.sort_idx = (self.sort_idx + 1) % len(SORT_MODES)
        self._render_radar()

    def action_cursor_down(self) -> None:
        if isinstance(self.focused, DataTable):
            self.focused.action_cursor_down()

    def action_cursor_up(self) -> None:
        if isinstance(self.focused, DataTable):
            self.focused.action_cursor_up()

    def action_refresh(self) -> None:
        self._set_status("Refreshing Radar…")
        self._worker_refresh_radar()

    def action_live_options(self) -> None:
        if self.offline:
            self._set_status("Offline mode — live options skipped")
            return
        self.query_one("#main-tabs", TabbedContent).active = "radar"
        self._set_status("Scanning live 60-120d ETF spreads…")
        self._worker_live_options()

    def action_update_prices(self) -> None:
        if self.offline:
            self._set_status("Offline mode — price update skipped")
            return
        self._set_status("Updating daily OHLCV prices…")
        self._worker_update_prices()

    def apply_desk_theme(self, name: str) -> None:
        if name in THEMES:
            self.theme = name
            persist_theme_name(name)

    def action_cycle_theme(self) -> None:
        cur = self.theme if self.theme in THEME_ORDER else THEME_ORDER[0]
        nxt = THEME_ORDER[(THEME_ORDER.index(cur) + 1) % len(THEME_ORDER)]
        self.apply_desk_theme(nxt)

    def action_show_help(self) -> None:
        self.push_screen(HelpModal())

    def action_quick_log(self) -> None:
        def _after_modal(res: Optional[Dict[str, Any]]) -> None:
            if not res:
                return
            self.provider.log_stock_pick(
                ticker=res["ticker"],
                price=res["price"],
                shares=res["shares"],
                playbook_id=res["playbook_id"],
                notes=res["notes"],
            )
            self._render_book()

        self.push_screen(
            QuickLogModal(
                ticker=self.focused_ticker,
                price=self.focused_spot,
                playbook_id=self.focused_playbook,
            ),
            _after_modal,
        )

    def action_log_fill(self) -> None:
        def _after_fill(res: Optional[Dict[str, Any]]) -> None:
            if not res:
                return
            self.provider.log_fill_attempt(**res)
            self.health = self.provider.get_health()
            self._render_lab()
            self._set_status(f"Fill logged for {res['ticker']}")

        self.push_screen(
            LogFillModal(ticker=self.focused_ticker, structure="Bull Put", side="SELL"),
            _after_fill,
        )


def run_desk(*, initial_tab: str = "today", offline: bool = False) -> None:
    app = DeskApp(initial_tab=initial_tab, offline=offline)
    app.run()
