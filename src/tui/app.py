"""Interactive 5-tab Textual Daily Trading Cockpit (`src/tui/app.py`).

Tabs:
  [1] TODAY      — Core TFSA Allocation (VM_OUTLOOK_TOP3) + Top 5 Radar Setups
  [2] RADAR      — Filterable, sortable unified Opportunity Radar + live DossierPane
  [3] WORKBENCH  — Full Ticker Dossier + Shares-vs-Options Express + Walk-Away Ladder
  [4] BOOK       — Stock/ETF Picks + Per-Playbook Scorecard
  [5] LAB        — Ops Health, Forward Readout Countdowns, Playbook Evidence Registry
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

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
from src.radar.schema import Opportunity, RadarSnapshot, TickerDossier
from src.structure.types import DetailedView, VehicleComparison
from .state import DeskDataProvider
from .themes import THEME_ORDER, THEMES, persist_theme_name, saved_theme_name
from .widgets import (
    DeskCommandProvider,
    HelpModal,
    QuickLogModal,
    format_dossier_markup,
    format_ribbon_markup,
)

SORT_MODES = ("tier", "confluence", "friction", "ticker")


def filter_opportunities(opps: List[Opportunity], query: str) -> List[Opportunity]:
    """Client-side micro-query filter (`stock`, `option`, `validated`, `conf>=2`, or substring)."""
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
        Binding("l", "quick_log", "Log Trade", show=True),
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

    def compose(self) -> ComposeResult:
        yield Static("Loading Daily Trading Desk…", id="market-ribbon")
        with TabbedContent(initial=self.initial_tab, id="main-tabs"):
            with TabPane("[1] TODAY", id="today"):
                with Vertical(id="today-grid"):
                    yield Static("Loading your plain-English daily action plan…", id="today-core-card")
                    with Horizontal(id="today-actions"):
                        yield Button("Refresh Ideas", id="act-refresh", variant="primary")
                        yield Button("Analyze a Stock", id="act-workbench")
                        yield Button("Log a Trade", id="act-log")
                        yield Button("? Plain-English Guide", id="act-help")
                    yield DataTable(id="today-top5-table", zebra_stripes=True, cursor_type="row")

            with TabPane("[2] RADAR", id="radar"):
                with Horizontal(id="radar-toolbar"):
                    yield Input(placeholder="Filter (e.g. pullback, stock, conf>=2)…", id="radar-filter-input")
                    yield Static("Sort: tier  |  Enter: Workbench  |  l: Quick Log", id="radar-pills")
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
                with Horizontal(id="wb-split"):
                    yield Static("Enter a ticker to load Dossier.", id="wb-left")
                    with Vertical(id="wb-right"):
                        yield Static("Plain-English recommendation will appear here.", id="wb-verdict-box")
                        yield DataTable(id="wb-vehicles-table", zebra_stripes=True, cursor_type="row")
                        yield Static("Select a vehicle row to inspect Walk-Away Limit Ladder.", id="wb-ladder-box")

            with TabPane("[4] BOOK", id="book"):
                with Vertical(id="book-container"):
                    yield Static("Playbook Scorecard loading…", id="book-scorecard-box")
                    yield DataTable(id="book-picks-table", zebra_stripes=True, cursor_type="row")

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
        bt.add_columns("ID", "Date", "Ticker", "Strategy", "Side", "Entry", "Shares", "Status")

        lt = self.query_one("#lab-playbooks-table", DataTable)
        lt.add_columns("Evidence", "Strategy ID", "Asset", "Horizon", "Notes")

    def _load_all_data(self, *, refresh: bool = False) -> None:
        self.snapshot = self.provider.get_radar(refresh=refresh)
        self.health = self.provider.get_health()
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
        self._compute_workbench(self.focused_ticker)
        try:
            from src.cache_release import release_yfinance_caches

            release_yfinance_caches()
        except Exception:
            pass

    def _render_ribbon(self) -> None:
        self.query_one("#market-ribbon", Static).update(
            format_ribbon_markup(self.snapshot, self.health)
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
            f"Sort: [bold]{mode}[/] ([s]) · Showing {len(rows)}/{len(snap.opportunities)} · Enter: Workbench"
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
        dos = self.provider.get_dossier(ticker, fetch_chain=False)
        self.query_one("#radar-dossier-pane", Static).update(format_dossier_markup(dos))

    def _compute_workbench(self, ticker: str) -> None:
        sym = ticker.upper().strip() or "SPY"
        self.focused_ticker = sym
        self.query_one("#wb-ticker-input", Input).value = sym
        dos: TickerDossier = self.provider.get_dossier(sym, fetch_chain=False)
        self.query_one("#wb-left", Static).update(format_dossier_markup(dos))

        spot = dos.spot if dos.spot is not None else 100.0
        self.focused_spot = spot
        iv = (dos.rv_20_ann / 100.0) if dos.rv_20_ann else 0.25

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
            v0 = vehicles[0]
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
        lines = [f"[bold]OWN PICKS SCORECARD[/]  (Total stock/ETF picks: {sc.get('total_picks', 0)})"]
        if by_pb:
            for pb_id, info in by_pb.items():
                h5 = (info.get("horizons") or {}).get(5) or {}
                m5 = f"{h5['mean']*100:+.2f}%" if h5.get("mean") is not None else "—"
                lines.append(
                    f"  • [bold]{pb_id}[/]: n={info['n_picks']} (open {info['n_open']}) · "
                    f"5d excess {m5} · {h5.get('verdict', '')}"
                )
        else:
            lines.append("  [dim]Press [l] anywhere to log your first stock/ETF pick.[/dim]")
        self.query_one("#book-scorecard-box", Static).update("\n".join(lines))

        bt = self.query_one("#book-picks-table", DataTable)
        bt.clear()
        for p in picks[:40]:
            bt.add_row(
                str(p["pick_id"]),
                str(p["date"])[:10],
                str(p["ticker"]),
                str(p["playbook_id"]),
                str(p["side"]),
                f"${float(p['entry_price']):.2f}",
                f"{float(p['shares']):g}",
                str(p["status"]),
            )

    def _render_lab(self) -> None:
        h = self.health
        fills = h.get("fills") or {}
        lines = [
            f"[bold]OPS HEALTH:[/] Disk {h.get('disk_free_gb', '?')} GB free [{h.get('worst', 'OK')}] · "
            f"Live Execution: OFF · Real Fills Logged: {fills.get('total_filled', 0)}/{fills.get('total_attempts', 0)}",
            "[bold]PENDING FORWARD READOUTS:[/] " + "  |  ".join(h.get("readouts") or []),
        ]
        self.query_one("#lab-health-box", Static).update("\n".join(lines))

        lt = self.query_one("#lab-playbooks-table", DataTable)
        lt.clear()
        for pb in self.provider.get_playbooks():
            lt.add_row(pb.evidence_tier, pb.id, pb.asset_class, pb.horizon_label, pb.notes[:72])

    # ── Events ───────────────────────────────────────────────────────────────

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
        elif bid == "act-refresh":
            self.action_refresh()
        elif bid == "act-workbench":
            self.open_workbench(self.focused_ticker)
        elif bid == "act-log":
            self.action_quick_log()
        elif bid == "act-help":
            self.action_show_help()

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
        self._load_all_data(refresh=True)

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


def run_desk(*, initial_tab: str = "today", offline: bool = False) -> None:
    app = DeskApp(initial_tab=initial_tab, offline=offline)
    app.run()
