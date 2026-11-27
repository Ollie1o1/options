"""Interactive 7-tab Textual Daily Trading Cockpit (`src/tui/app.py`).

Tabs:
  [1] TODAY      — Core TFSA Allocation (VM_OUTLOOK_TOP3) + Top 5 Radar Setups
  [2] RADAR      — Filterable (presets + query), sortable unified Opportunity Radar + live DossierPane
  [3] WORKBENCH  — Full Ticker Dossier + Visual RND/2D Risk Heatmap + Shares-vs-Options + Walk-Away Ladder
  [4] BOOK       — Stock/ETF Picks (live P&L + exit alerts), Open Paper Options Table, Holdings & Scorecard
  [5] LAB        — Real Ops Health, OHLCV Freshness, Fill Concession `k`, Forward Readouts & Playbooks
  [6] DISCOVER   — Top-down: ① Market → ② Sectors (PROVEN rank) → ③ Ideas (labelled lenses) → ④ Evidence Card
  [7] OPTIONS    — Multi-Strategy Scanner · Interactive T-Shaped Chain Matrix + Leg Composer · Flow & Vol Intel
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
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
from .discover import (
    LENS_OPTIONS,
    BOARD_COLUMNS,
    EvidenceCardScreen,
    board_header,
    crumb_and_freshness,
    fill_board,
    fill_lens,
    fill_sector,
    lens_header,
    lens_tag_legend,
    sector_header,
)
from .charts import braille_ghost_race, braille_sector_rrg
from .options_engine import (
    SCAN_STRATEGIES,
    ChainMatrix,
    OptionLeg,
    OptionScanRow,
    OptionsIntelBundle,
    classify_legs,
    filter_option_rows,
)
from .options_tab import (
    OPT_CHAIN_COLUMNS,
    OPT_FILTER_PRESETS,
    OPT_FLOW_COLUMNS,
    OPT_SCAN_COLUMNS,
    fill_chain_matrix_table,
    fill_options_intel_table,
    fill_options_scan_table,
    format_chain_inspector,
    format_intel_inspector,
    format_scan_inspector,
)
from .state import DeskDataProvider
from .themes import THEME_ORDER, THEMES, persist_theme_name, saved_theme_name
from .widgets import (
    AlphaViewerModal,
    BloombergCommandModal,
    ClosePickModal,
    DeskCommandProvider,
    HelpModal,
    LogFillModal,
    PaperOptionModal,
    QuickLogModal,
    RebalanceModal,
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
    """6-Tab Daily Stock + Option Trading Desk."""

    CSS_PATH = "styles.tcss"
    COMMANDS = App.COMMANDS | {DeskCommandProvider}

    BINDINGS = [
        Binding("1", "switch_tab('today')", "Today", show=True),
        Binding("2", "switch_tab('radar')", "Radar", show=True),
        Binding("3", "switch_tab('workbench')", "Workbench", show=True),
        Binding("4", "switch_tab('book')", "Book", show=True),
        Binding("5", "switch_tab('lab')", "Lab", show=True),
        Binding("6", "switch_tab('discover')", "Discover", show=True),
        Binding("7", "switch_tab('options')", "Options", show=True),
        Binding("d", "switch_tab('discover')", "Discover", show=False),
        Binding("D", "discord_push", "Discord", show=True),
        Binding("c", "open_chain", "Chain", show=True),
        Binding("slash", "focus_filter", "Filter", show=True),
        Binding("s", "cycle_sort", "Sort", show=True),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("colon", "bloomberg_cmd", "Cmd :", show=True),
        Binding("m", "morning_ritual", "Ritual", show=True),
        Binding("b", "rebalance_modal", "Rebal", show=True),
        Binding("l", "quick_log", "Log Pick", show=True),
        Binding("p", "paper_log_option", "Paper Opt", show=True),
        Binding("x", "close_pick", "Close Pick", show=True),
        Binding("f", "log_fill", "Log Fill", show=True),
        Binding("o", "live_options", "Live Opts", show=True),
        Binding("h", "toggle_heatmap", "Heatmap", show=False),
        Binding("left_curly_bracket", "nudge_iv(-5)", "IV -5%", show=False),
        Binding("right_curly_bracket", "nudge_iv(5)", "IV +5%", show=False),
        Binding("comma", "nudge_elapsed(-5)", "T -5d", show=False),
        Binding("full_stop", "nudge_elapsed(5)", "T +5d", show=False),
        Binding("u", "update_prices", "Upd Prices", show=False),
        Binding("a", "toggle_account", "TFSA/Tax", show=False),
        Binding("left_square_bracket", "nudge_target(-2)", "Tgt -2%", show=False),
        Binding("right_square_bracket", "nudge_target(2)", "Tgt +2%", show=False),
        Binding("minus", "nudge_days(-10)", "Days -10", show=False),
        Binding("equals_sign", "nudge_days(10)", "Days +10", show=False),
        Binding("plus", "nudge_days(10)", "Days +10", show=False),
        Binding("g", "toggle_rrg", "RRG Map", show=False),
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
        self.focused_opp: Optional[Opportunity] = None
        self._last_vehicles: List[VehicleComparison] = []
        self._selected_vehicle_idx: int = 0
        self._book_picks: List[Dict[str, Any]] = []
        self._focused_pick_id: Optional[int] = None
        self._status_msg: str = ""
        self._wb_loaded_ticker: Optional[str] = None
        self._last_dossier_ticker: Optional[str] = None
        self.hub: Optional[Any] = None
        self._hub_loading: bool = False
        self._disc_mode: str = "lens"          # "lens" | "sector"
        self._disc_lens: str = "core_rotation"
        self._disc_sector: str = ""
        self._disc_tickers: List[str] = []
        self._show_rrg: bool = True
        # Workbench & Options state
        self._wb_show_heatmap: bool = False
        self._wb_last_dos: Optional[TickerDossier] = None
        self._wb_last_view: Optional[DetailedView] = None
        self._iv_shock_pct: float = 0.0
        self._days_elapsed: int = 0
        self._book_paper_opts: List[Dict[str, Any]] = []
        # [7] OPTIONS state (lazy loaded)
        self._opt_view: str = "scanner"        # "scanner" | "chain" | "intel"
        self._opt_scan_all: List[OptionScanRow] = []
        self._opt_scan_filtered: List[OptionScanRow] = []
        self._opt_scan_loaded: bool = False
        self._opt_selected_scan: Optional[OptionScanRow] = None
        self._opt_chain_matrix: Optional[ChainMatrix] = None
        self._opt_chain_row_idx: int = 0
        self._opt_staged_legs: List[OptionLeg] = []
        self._opt_intel: Optional[OptionsIntelBundle] = None
        self._opt_intel_mode: str = "all"
        self._opt_intel_items: List[Dict[str, Any]] = []
        self._opt_chain_syncing_exp: bool = False

    def compose(self) -> ComposeResult:
        yield Static("Loading Daily Trading Desk…", id="market-ribbon")
        with TabbedContent(initial=self.initial_tab, id="main-tabs"):
            with TabPane("[1] TODAY", id="today"):
                with Vertical(id="today-grid"):
                    yield Static("Loading your plain-English daily action plan…", id="today-core-card")
                    with Horizontal(id="today-actions"):
                        yield Button("Ritual [m]", id="act-ritual", variant="primary")
                        yield Button("Refresh [r]", id="act-refresh")
                        yield Button("Rebalance [b]", id="act-rebal")
                        yield Button("Cmd [:]", id="act-bcmd")
                        yield Button("Plan [3]", id="act-workbench")
                        yield Button("Options [7]", id="act-options")
                        yield Button("Discord [D]", id="act-discord")
                        yield Button("Live Opts [o]", id="act-live-opts")
                        yield Button("Discover [6]", id="act-discover")
                        yield Button("Log [l]", id="act-log")
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
                    yield Select(
                        [("TFSA", "tfsa"), ("Taxable", "taxable")],
                        value="tfsa",
                        id="wb-acct-select",
                    )
                    yield Label("100sh:")
                    yield Switch(value=False, id="wb-owns-switch")
                    yield Button("Run", id="wb-run-btn", variant="primary")
                    yield Button("Chain [c]", id="wb-chain-btn")
                    yield Button("Heatmap [h]", id="wb-heat-btn")
                    yield Button("Paper [p]", id="wb-paper-btn")
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
                    yield Static("Ghost Benchmark Race loading…", id="book-race-box")
                    yield Static("Playbook Scorecard loading…", id="book-scorecard-box")
                    with Horizontal(id="book-tables-split"):
                        yield DataTable(id="book-picks-table", zebra_stripes=True, cursor_type="row")
                        with Vertical(id="book-options-pane"):
                            yield Static("OPEN PAPER OPTIONS (Enter → Chain)", id="book-options-head")
                            yield DataTable(id="book-options-table", zebra_stripes=True, cursor_type="row")
                    yield Static("", id="book-extra-box")

            with TabPane("[5] LAB", id="lab"):
                with Vertical(id="lab-container"):
                    yield Static("Ops Health loading…", id="lab-health-box")
                    yield DataTable(id="lab-playbooks-table", zebra_stripes=True, cursor_type="row")

            with TabPane("[6] DISCOVER", id="discover"):
                with Vertical(id="disc-root"):
                    yield Static("① MARKET › ② SECTORS › ③ IDEAS › ④ EVIDENCE", id="disc-crumb")
                    yield Static("① MARKET — loading…", id="disc-market")
                    with Horizontal(id="disc-split"):
                        with Vertical(id="disc-left"):
                            yield Static(board_header(), id="disc-board-head")
                            yield DataTable(id="disc-board", zebra_stripes=True, cursor_type="row")
                            yield Static("", id="disc-rrg-box")
                        with Vertical(id="disc-right"):
                            with Horizontal(id="disc-lens-bar"):
                                yield Label("③ IDEAS:", id="disc-lens-label")
                                yield Select(LENS_OPTIONS, value="core_rotation", allow_blank=True,
                                             prompt="Pick a lens…", id="disc-lens-select")
                            yield Static("", id="disc-lens-head")
                            yield DataTable(id="disc-lens-table", zebra_stripes=True, cursor_type="row")
                            yield Static("", id="disc-hint")

            with TabPane("[7] OPTIONS", id="options"):
                with Vertical(id="opt-root"):
                    with Horizontal(id="opt-subnav"):
                        yield Button("Scanner", id="opt-nav-scanner", variant="primary")
                        yield Button("Chain Matrix", id="opt-nav-chain")
                        yield Button("Flow & Vol Intel", id="opt-nav-intel")
                        yield Button("Send Discord [D]", id="opt-discord-btn")
                        yield Static("Archive-first · instant", id="opt-status")

                    with Vertical(id="opt-view-scanner", classes="opt-view"):
                        with Horizontal(id="opt-scan-bar"):
                            yield Select(list(SCAN_STRATEGIES), value="all", id="opt-strat-select")
                            yield Input(
                                placeholder="Filter (bull_put, tfsa, clean, fric<12%, dte>=30)…",
                                id="opt-filter-input",
                            )
                            for pid, plabel, _pq in OPT_FILTER_PRESETS:
                                yield Button(plabel, id=pid, classes="preset-btn")
                            yield Button("Live", id="opt-live-scan-btn", classes="preset-btn")
                        with Horizontal(id="opt-scan-split"):
                            yield DataTable(id="opt-scan-table", zebra_stripes=True, cursor_type="row")
                            yield Static("Loading archived option chains…", id="opt-scan-inspector")

                    with Vertical(id="opt-view-chain", classes="opt-view"):
                        with Horizontal(id="opt-chain-bar"):
                            yield Input(value="SPY", placeholder="Ticker", id="opt-chain-ticker")
                            yield Select([("Default (~35d)", "__default__")], value="__default__", id="opt-chain-exp")
                            yield Select(
                                [("±12%", "0.12"), ("±20%", "0.20"), ("All", "0.0")],
                                value="0.12",
                                id="opt-chain-band",
                            )
                            yield Button("Load", id="opt-chain-load-btn", variant="primary", classes="preset-btn")
                            yield Button("Live", id="opt-chain-live-btn", classes="preset-btn")
                        with Horizontal(id="opt-chain-legbar"):
                            yield Label("Stage:", id="opt-leg-label")
                            yield Button("+Buy C", id="opt-leg-bc", classes="preset-btn")
                            yield Button("+Sell C", id="opt-leg-sc", classes="preset-btn")
                            yield Button("+Buy P", id="opt-leg-bp", classes="preset-btn")
                            yield Button("+Sell P", id="opt-leg-sp", classes="preset-btn")
                            yield Button("BullPut 5%", id="opt-leg-bp5", classes="preset-btn")
                            yield Button("Clear", id="opt-leg-clear", classes="preset-btn")
                        with Horizontal(id="opt-chain-split"):
                            yield DataTable(id="opt-chain-table", zebra_stripes=True, cursor_type="row")
                            yield Static("Select a ticker to load T-shaped Chain Matrix.", id="opt-chain-inspector")

                    with Vertical(id="opt-view-intel", classes="opt-view"):
                        with Horizontal(id="opt-intel-bar"):
                            yield Button("All Intel", id="opt-intel-all", variant="primary")
                            yield Button("Flow (UOA/OCC)", id="opt-intel-flow")
                            yield Button("Vol / VRP", id="opt-intel-vol")
                            yield Button("Skew / GEX", id="opt-intel-skew")
                            yield Button("Refresh Intel", id="opt-intel-refresh")
                        with Horizontal(id="opt-intel-split"):
                            yield DataTable(id="opt-intel-table", zebra_stripes=True, cursor_type="row")
                            yield Static("Loading UOA, OCC, Vol-Intel & Skew…", id="opt-intel-inspector")

        yield Footer()

    def on_mount(self) -> None:
        for th in THEMES.values():
            self.register_theme(th)
        self.theme = saved_theme_name()
        self._apply_responsive_class(self.size.width)
        self._setup_tables()
        self._sync_options_subview()
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

        bot = self.query_one("#book-options-table", DataTable)
        bot.add_columns("ID", "Ticker", "Strategy", "Exp (DTE)", "Strike", "Entry", "Qty")

        lt = self.query_one("#lab-playbooks-table", DataTable)
        lt.add_columns("Evidence", "Strategy ID", "Asset", "Horizon", "Notes")

        self.query_one("#disc-board", DataTable).add_columns(*BOARD_COLUMNS)

        self.query_one("#opt-scan-table", DataTable).add_columns(*OPT_SCAN_COLUMNS)
        self.query_one("#opt-chain-table", DataTable).add_columns(*OPT_CHAIN_COLUMNS)
        self.query_one("#opt-intel-table", DataTable).add_columns(*OPT_FLOW_COLUMNS)

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
            self.focused_opp = first
            self._update_radar_dossier(first.ticker)
        self._wb_loaded_ticker = None
        active = self.query_one("#main-tabs", TabbedContent).active
        if active == "workbench":
            self._compute_workbench(self.focused_ticker)
        elif active == "discover":
            self._ensure_hub()
        elif active == "options":
            self._ensure_options_loaded()

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
        for idx, o in enumerate(snap.opportunities[:5]):
            tbl.add_row(
                TIER_PLAIN.get(o.evidence_tier, o.evidence_tier),
                o.ticker,
                o.playbook_label[:22],
                "▲ BUY" if o.direction == "LONG" else ("▼ SELL" if o.direction == "SHORT" else "• FLAT"),
                str(o.confluence_count),
                o.friction_label or f"{o.friction_value*100:.2f}%",
                o.setup_summary[:55],
                key=f"{idx}|{o.ticker}|{o.playbook_id}",
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
        if dos.spot is None and not self.offline:
            try:
                dos = self.provider.get_dossier(sym, fetch_chain=True, allow_live_spot=True)
            except TypeError:
                dos = self.provider.get_dossier(sym, fetch_chain=True)
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
        acct_val = str(self.query_one("#wb-acct-select", Select).value or "tfsa").lower()

        view = DetailedView(
            symbol=sym,
            direction=dir_val,
            target_move_pct=t_move,
            horizon_days=days,
            risk_budget_usd=risk,
            account_type=acct_val,
            owns_100_shares=owns_100,
        )
        allow_live_chain = (not self.offline) and (dos.atm_iv is None)
        try:
            vehicles = self.provider.get_express(view, spot=spot, iv=iv, allow_live=allow_live_chain)
        except TypeError:
            vehicles = self.provider.get_express(view, spot=spot, iv=iv)
        self._last_vehicles = vehicles
        self._selected_vehicle_idx = 0
        self._wb_last_dos = dos
        self._wb_last_view = view
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
            rec_i = next((i for i, v in enumerate(vehicles) if v.recommended), 0)
            self._selected_vehicle_idx = rec_i
            self._render_wb_ladder(vehicles[rec_i])
        self._render_wb_visual()

    def _render_wb_visual(self) -> None:
        if self._wb_last_view is None or not self.is_mounted:
            return
        try:
            self.query_one("#wb-visual-box", Static).update(
                format_workbench_visual_markup(
                    self._last_vehicles,
                    self._wb_last_dos,
                    self._wb_last_view,
                    show_heatmap=self._wb_show_heatmap,
                    iv_shock_pct=self._iv_shock_pct,
                    selected_idx=self._selected_vehicle_idx,
                )
            )
        except NoMatches:
            pass

    def _render_wb_ladder(self, v: VehicleComparison) -> None:
        if not self.is_mounted:
            return
        steps = list(v.walk_away_steps) if v.walk_away_steps else [v.walk_away_label]
        step_lines = "\n".join(f"  {s}" for s in steps)
        tfsa_tag = "[#5ec98d]TFSA OK[/]" if v.tfsa_legal else "[#d6a452]TAXABLE / IBKR ONLY (CRA TFSA ban)[/]"
        try:
            self.query_one("#wb-ladder-box", Static).update(
                f"[bold]Execution Ticket & 3-Step Walk-Away Ladder ({v.vehicle})[/] · {tfsa_tag}\n"
                f"{step_lines}\n"
                f"  Delta $: ${v.delta_dollars:+,.0f} · Theta/day: ${v.theta_per_day_usd:+.2f}"
                + (f" · [#d6a452]{v.caution}[/]" if v.caution else "")
            )
        except NoMatches:
            pass

    def _render_book(self) -> None:
        book = self.provider.get_book()
        race = self.provider.get_ghost_race() if hasattr(self.provider, "get_ghost_race") else None
        if race is not None and getattr(race, "spy", None):
            self.query_one("#book-race-box", Static).update(braille_ghost_race(race, width=50, height=5, rich=True))
        else:
            self.query_one("#book-race-box", Static).update(
                "[bold]GHOST BENCHMARK RACE[/] [dim]— Log picks [l] or update OHLCV [u] to race SPY & VM_OUTLOOK+SGOV.[/]"
            )
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

        # CRA TFSA Safety & 15% US Dividend Tax-Drag Meter
        try:
            from src.profile import load_profile
            from src.tfsa_tax import compute_tfsa_safety_report

            prof = load_profile()
            safety = compute_tfsa_safety_report(
                profile=prof,
                core_allocation=(self.snapshot.core_allocation if self.snapshot else ()),
                equity_picks=picks,
            )
            cra_col = "#5ec98d" if safety.cra_status == "SAFE_INVESTOR" else (
                "#d6a452" if safety.cra_status == "WATCH_CADENCE" else "#e06c75"
            )
            room_part = (
                f" · Room Left: ${safety.room_remaining_cad:,.0f} CAD"
                if safety.room_remaining_cad is not None
                else ""
            )
            lines.append(
                f"[bold]CRA TFSA GUARD:[/] [{cra_col}]{safety.cra_status}[/] ({safety.trades_ytd} YTD trades"
                + (f", avg hold {safety.avg_hold_days}d" if safety.avg_hold_days is not None else "")
                + f") · 15% US Div Drag: ~${safety.total_tax_drag_usd_yr:,.0f}/yr ({safety.weighted_tax_drag_bps:.0f} bps){room_part}"
            )
        except Exception:
            pass

        # 10-Second Post-Mortem Mistake Cost Summary
        mc = book.get("mistake_costs") or {}
        if mc.get("by_tag"):
            tag_parts = [
                f"{r['tag']}: ${r['pnl_usd']:+,.0f} ({r['count']}x, {r['win_rate_pct']:.0f}%W)"
                for r in mc["by_tag"][:4]
            ]
            lines.append("[bold]POST-MORTEM TAGS:[/] " + " · ".join(tag_parts))

        self.query_one("#book-scorecard-box", Static).update("\n".join(lines))

        bt = self.query_one("#book-picks-table", DataTable)
        bt.clear()
        self._book_picks = list(picks[:40])
        for p in self._book_picks:
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
                key=str(p["pick_id"]),
            )

        bot = self.query_one("#book-options-table", DataTable)
        bot.clear()
        self._book_paper_opts = list(paper_opts[:30])
        self.query_one("#book-options-head", Static).update(
            f"OPEN PAPER OPTIONS ({len(paper_opts)}) · [dim]Enter → Chain · [p] Paper Log[/]"
        )
        for r in self._book_paper_opts:
            exp_s = str(r.get("expiration") or "")[:10]
            dte_v = r.get("dte")
            exp_lbl = f"{exp_s[5:]} ({dte_v}d)" if (len(exp_s) >= 10 and dte_v is not None) else exp_s
            k_main = r.get("strike")
            k_long = r.get("long_strike")
            if k_main is not None and k_long:
                k_lbl = f"{float(k_main):g}/{float(k_long):g}"
            elif k_main is not None:
                k_lbl = f"${float(k_main):g}"
            else:
                k_lbl = "—"
            bot.add_row(
                str(r.get("entry_id") or ""),
                str(r.get("ticker") or ""),
                str(r.get("strategy_name") or "")[:14],
                exp_lbl,
                k_lbl,
                f"${float(r.get('entry_price') or 0):.2f}",
                f"{float(r.get('quantity') or 1):g}",
                key=f"{r.get('entry_id')}|{r.get('ticker')}|{exp_s}",
            )

        extra_lines: List[str] = []
        if paper_opts:
            extra_lines.append(
                f"[bold]OPEN PAPER OPTIONS ({len(paper_opts)}):[/] [dim]Inspect in table above · Enter → Chain · [p] Paper Log[/dim]"
            )
        else:
            extra_lines.append("[bold]OPEN PAPER OPTIONS:[/] [dim]0 open in paper_trades.db · press [p] to paper-log[/dim]")
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
        o1 = h.get("o1_fills") or {}
        if o1:
            from src.fill_experiment import o1_progress_line

            o1_str = o1_progress_line(o1)
        else:
            o1_str = "TEST O1 FILLS — unavailable"
        pipes = h.get("pipes") or {}
        led_parts: List[str] = []
        for pname, pinfo in pipes.items():
            pst = str(pinfo.get("status") or "OK")
            pcol = "#5ec98d" if pst == "OK" else ("#d6a452" if pst == "WARN" else "#e06c75")
            led_parts.append(f"[{pcol}]● {pname.upper()}[/] [dim]({pinfo.get('detail', '')})[/]")
        pipe_line = ("[bold]DATA PIPES:[/] " + "  ".join(led_parts)) if led_parts else ""
        mc_line = ""
        try:
            bk = self.provider.get_book()
            mc = (bk.get("mistake_costs") or {}).get("by_tag") or []
            if mc:
                mc_line = "[bold]POST-MORTEM MISTAKE COSTS:[/] " + " · ".join(
                    f"{r['tag']} ${r['pnl_usd']:+,.0f} ({r['count']}x)" for r in mc[:4]
                )
        except Exception:
            pass
        lines = [
            f"[bold]OPS HEALTH:[/] Disk {h.get('disk_free_gb', '?')} GB free [{h.get('worst', 'OK')}]{ohlcv_s}{maint_s}",
            *([pipe_line] if pipe_line else []),
            *([mc_line] if mc_line else []),
            f"[bold]EXECUTION:[/] Live OFF · all real limit fills logged: {fills.get('total_filled', 0)}/{fills.get('total_attempts', 0)} (press \\[f] to log)",
            f"[bold]{o1_str}[/] · rule: docs/PREREG_O1_FILL_K_20261125.md",
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
            live_opps, warns = self.provider.get_live_etf_options()
            skipped: Dict[str, str] = dict(getattr(self.provider, "last_live_skipped", {}) or {})

            def _merge() -> None:
                if warns:
                    self.notify(
                        "\n".join(warns[:6]),
                        title="Live ETF feed warnings",
                        severity="warning",
                        timeout=12,
                    )
                if skipped:
                    # Say which ETFs produced nothing and why (thin Yahoo quotes vs. the rule).
                    lines = [f"{tk}: {why}" for tk, why in sorted(skipped.items())]
                    self.notify("\n".join(lines[:11]), title="Live ETF spreads — not shown, and why",
                                severity="information", timeout=15)
                if self.snapshot is not None and live_opps:
                    # Replace both prior LIVE_CHAIN and archived etf_wide_spread rows so
                    # fresh live quotes supersede archived ones cleanly without duplicates.
                    live_tickers = {o.ticker for o in live_opps}
                    base = [
                        o
                        for o in self.snapshot.opportunities
                        if "LIVE_CHAIN" not in o.badges
                        and not (o.playbook_id == "etf_wide_spread" and o.ticker in live_tickers)
                    ]
                    self.snapshot.opportunities = merge_confluence(base + live_opps)
                    inp = self.query_one("#radar-filter-input", Input)
                    if inp.value != "option":
                        inp.value = "option"
                    self._render_today()
                    self._render_radar()
                    if self.filtered_opps:
                        first = self.filtered_opps[0]
                        self.focused_ticker = first.ticker
                        self.focused_spot = first.spot
                        self.focused_playbook = first.playbook_id
                        self.focused_opp = first
                        self._update_radar_dossier(first.ticker)
                    self._set_status(f"Live ETF spreads: {len(live_opps)} found (showing Options)")
                elif live_opps:
                    self._set_status(f"Live ETF spreads: {len(live_opps)} found")
                elif warns:
                    self._set_status(f"Live ETF spreads: 0 ({warns[0][:40]})")
                else:
                    self._set_status("Live ETF spreads: 0 cleared <=25% cost (kept cached)")

            self.call_from_thread(_merge)
        except Exception as exc:
            self.call_from_thread(self._set_status, f"Live opts err: {str(exc)[:40]}")

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

    # ── [6] DISCOVER ─────────────────────────────────────────────────────────

    def _ensure_hub(self, *, refresh: bool = False) -> None:
        if self._hub_loading:
            return
        if self.hub is not None and not refresh:
            self._render_discover()
            return
        self._hub_loading = True
        self.query_one("#disc-market", Static).update("① MARKET — building the top-down view from local archives…")
        self._worker_build_hub(refresh)

    @work(thread=True, exclusive=True, group="discover_hub")
    def _worker_build_hub(self, refresh: bool) -> None:
        try:
            hub = self.provider.get_hub(refresh=refresh)

            def _apply() -> None:
                self.hub = hub
                self._hub_loading = False
                self._render_discover()

            self.call_from_thread(_apply)
        except Exception as exc:
            def _fail() -> None:
                self._hub_loading = False
                self.query_one("#disc-market", Static).update(f"[#e06c75]Discover unavailable: {exc}[/]")

            self.call_from_thread(_fail)

    def _render_discover(self) -> None:
        hub = self.hub
        if hub is None:
            return
        from src.topdown.render import market_panel

        self.query_one("#disc-crumb", Static).update(crumb_and_freshness(hub, 3 if self._disc_mode == "lens" else 2))
        self.query_one("#disc-market", Static).update(market_panel(hub, compact=True))
        fill_board(self.query_one("#disc-board", DataTable), hub)
        rrg_box = self.query_one("#disc-rrg-box", Static)
        if self._show_rrg:
            rrg_box.display = True
            rrg_box.update(
                "[bold]SECTOR ROTATION MAP (RRG · press \\[g] to toggle)[/]\n"
                + braille_sector_rrg(hub.board.rows, width=42, height=7, rich=True)
            )
        else:
            rrg_box.display = False
        self._render_disc_right()

    def _render_disc_right(self) -> None:
        hub = self.hub
        if hub is None:
            return
        tbl = self.query_one("#disc-lens-table", DataTable)
        head = self.query_one("#disc-lens-head", Static)
        if self._disc_mode == "sector" and self._disc_sector:
            head.update(sector_header(hub, self._disc_sector))
            self._disc_tickers = fill_sector(tbl, hub, self._disc_sector)
        else:
            res = hub.lens(self._disc_lens)
            if res is None:
                return
            head.update(lens_header(res))
            self._disc_tickers = fill_lens(tbl, res, hub.lenses)
        self._update_disc_hint()

    def _update_disc_hint(self, ticker: Optional[str] = None) -> None:
        tk = ticker or (self._disc_tickers[0] if self._disc_tickers else None)
        if tk:
            msg = f"[b]Enter[/b] = ④ Evidence Card for [b]{tk}[/b] · [b]l[/b] log pick · [b]r[/b] refresh"
        else:
            msg = "Nothing listed here right now — pick another lens or sector."
        legend = lens_tag_legend()
        self.query_one("#disc-hint", Static).update(f"{msg}\n[dim]{legend.plain}[/dim]")

    def open_evidence_card(self, ticker: str) -> None:
        sym = str(ticker).upper().strip()
        self.focused_ticker = sym

        def _after(res: Optional[str]) -> None:
            if not res:
                return
            if res.startswith("CHAIN:"):
                self.open_options_chain(res.split(":", 1)[1])
            else:
                self.open_workbench(res)

        self.push_screen(EvidenceCardScreen(sym, self.provider.get_card), _after)

    # ── [7] OPTIONS (Scanner · Chain Matrix · Flow & Vol Intel) ──────────────

    def _sync_options_subview(self) -> None:
        v = self._opt_view
        self.query_one("#opt-view-scanner").display = (v == "scanner")
        self.query_one("#opt-view-chain").display = (v == "chain")
        self.query_one("#opt-view-intel").display = (v == "intel")
        for bid, key in (
            ("#opt-nav-scanner", "scanner"),
            ("#opt-nav-chain", "chain"),
            ("#opt-nav-intel", "intel"),
        ):
            self.query_one(bid, Button).variant = "primary" if v == key else "default"

    def open_options_view(self, view: str = "scanner", *, strategy: Optional[str] = None) -> None:
        self._opt_view = view if view in ("scanner", "chain", "intel") else "scanner"
        self.query_one("#main-tabs", TabbedContent).active = "options"
        self._sync_options_subview()
        if strategy and self._opt_view == "scanner":
            sel = self.query_one("#opt-strat-select", Select)
            valid = {val for _lbl, val in SCAN_STRATEGIES}
            if strategy.lower() in valid:
                sel.value = strategy.lower()
                self._opt_scan_loaded = False
        self._ensure_options_loaded()

    def open_options_chain(
        self,
        ticker: str,
        *,
        expiration: Optional[str] = None,
        live: bool = False,
    ) -> None:
        sym = str(ticker or self.focused_ticker or "SPY").upper().strip() or "SPY"
        self.focused_ticker = sym
        self._opt_view = "chain"
        self._opt_staged_legs = []
        self.query_one("#main-tabs", TabbedContent).active = "options"
        self._sync_options_subview()
        self.query_one("#opt-chain-ticker", Input).value = sym
        self._load_chain(sym, expiration=expiration, live=live)

    def _ensure_options_loaded(self) -> None:
        if self._opt_view == "scanner":
            if not self._opt_scan_loaded:
                self._run_options_scan(live=False)
            else:
                self._render_options_scanner()
        elif self._opt_view == "chain":
            tk = self.query_one("#opt-chain-ticker", Input).value.strip().upper() or self.focused_ticker or "SPY"
            if self._opt_chain_matrix is None or self._opt_chain_matrix.ticker != tk:
                self._load_chain(tk, live=False)
        elif self._opt_view == "intel":
            if self._opt_intel is None:
                self._load_options_intel()
            else:
                self._render_options_intel()

    def _run_options_scan(self, *, live: bool = False) -> None:
        if live and self.offline:
            self._set_opt_status("Offline mode — live scan skipped")
            return
        strat = str(self.query_one("#opt-strat-select", Select).value or "all")
        self._set_opt_status(
            f"Scanning {'live Yahoo' if live else 'archived'} chains ({strat})…"
        )
        self._worker_scan_options(strat, live)

    def _set_opt_status(self, text: str) -> None:
        """Update the Options status line; a no-op once the tab is unmounted.

        Workers report back after the screen may be gone (app exit, test teardown).
        A missing widget must not turn a finished worker into a WorkerFailed."""
        try:
            self.query_one("#opt-status", Static).update(text)
        except NoMatches:
            pass

    @work(thread=True, exclusive=True, group="options_scan")
    def _worker_scan_options(self, strategy: str, live: bool) -> None:
        try:
            rows, note = self.provider.scan_options(strategy=strategy, live=live)

            def _apply() -> None:
                self._opt_scan_all = list(rows)
                self._opt_scan_loaded = True
                self._set_opt_status(note)
                self._render_options_scanner()

            self.call_from_thread(_apply)
        except Exception as exc:
            self.call_from_thread(
                self._set_opt_status,
                f"[#e06c75]Scan error: {str(exc)[:48]}[/]",
            )

    def _render_options_scanner(self) -> None:
        q = self.query_one("#opt-filter-input", Input).value
        rows = filter_option_rows(self._opt_scan_all, q)
        self._opt_scan_filtered = rows
        tbl = self.query_one("#opt-scan-table", DataTable)
        fill_options_scan_table(tbl, rows)
        if rows:
            self._opt_selected_scan = rows[0]
            self.focused_ticker = rows[0].ticker
            self.focused_spot = rows[0].spot
            self.query_one("#opt-scan-inspector", Static).update(
                format_scan_inspector(rows[0], days_elapsed=self._days_elapsed, iv_shock_pct=self._iv_shock_pct)
            )
        else:
            self._opt_selected_scan = None
            self.query_one("#opt-scan-inspector", Static).update(
                format_scan_inspector(None)
                + "\n\n[dim]0 candidates matched this filter — clear filter or click Live Scan.[/]"
            )

    def _load_chain(
        self,
        ticker: str,
        *,
        expiration: Optional[str] = None,
        live: bool = False,
    ) -> None:
        if live and self.offline:
            self._set_opt_status("Offline mode — live chain skipped")
            return
        sym = str(ticker or "SPY").upper().strip() or "SPY"
        try:
            band = float(str(self.query_one("#opt-chain-band", Select).value or "0.12"))
        except ValueError:
            band = 0.12
        self._set_opt_status(
            f"Loading {sym} {'live Yahoo' if live else 'archived'} option chain…"
        )
        self._worker_load_chain(sym, expiration, band, live)

    @work(thread=True, exclusive=True, group="options_chain")
    def _worker_load_chain(
        self,
        ticker: str,
        expiration: Optional[str],
        band_pct: float,
        live: bool,
    ) -> None:
        try:
            mat = self.provider.get_chain_matrix(
                ticker,
                expiration=expiration,
                band_pct=band_pct,
                allow_live=live,
            )

            def _apply() -> None:
                self._opt_chain_matrix = mat
                self._opt_chain_row_idx = 0
                if mat is None:
                    self.query_one("#opt-chain-table", DataTable).clear()
                    self._set_opt_status(
                        f"No archived chain for {ticker} — click Live to fetch from Yahoo"
                    )
                    self.query_one("#opt-chain-inspector", Static).update(
                        f"[bold]{ticker}[/] has no local chain in `data/chain_archive.db`.\n"
                        "[dim]Click [b]Live[/b] above to pull live Yahoo quotes, or run `opt alpha archive-chains`.[/]"
                    )
                    return
                self.focused_ticker = mat.ticker
                self.focused_spot = mat.spot
                # Populate expiry selector without triggering a reload loop
                self._opt_chain_syncing_exp = True
                try:
                    exp_sel = self.query_one("#opt-chain-exp", Select)
                    opts = [(f"{e} ({d}d)", e) for e, d in mat.available_expirations]
                    exp_sel.set_options(opts)
                    exp_sel.value = mat.expiration
                finally:
                    self._opt_chain_syncing_exp = False
                self._render_chain_matrix()
                self._set_opt_status(
                    f"{mat.ticker} ${mat.spot:.2f} · {mat.expiration} ({mat.dte}d) · {len(mat.rows)} strikes [{mat.source}]"
                )

            self.call_from_thread(_apply)
        except Exception as exc:
            self.call_from_thread(
                self._set_opt_status,
                f"[#e06c75]Chain err: {str(exc)[:48]}[/]",
            )

    def _render_chain_matrix(self) -> None:
        mat = self._opt_chain_matrix
        if mat is None:
            return
        tbl = self.query_one("#opt-chain-table", DataTable)
        fill_chain_matrix_table(tbl, mat)
        # Auto-focus cursor on the ATM SPOT strike row for immediate visual ergonomics
        spot_idx = next((i for i, r in enumerate(mat.rows) if "SPOT" in r.tags), len(mat.rows) // 2)
        if mat.rows:
            self._opt_chain_row_idx = spot_idx
            try:
                tbl.move_cursor(row=spot_idx)
            except Exception:
                pass
        self.query_one("#opt-chain-inspector", Static).update(
            format_chain_inspector(
                mat,
                self._opt_staged_legs,
                days_elapsed=self._days_elapsed,
                iv_shock_pct=self._iv_shock_pct,
            )
        )

    def _stage_chain_leg(self, opt_type: str, side: str) -> None:
        mat = self._opt_chain_matrix
        if mat is None or not mat.rows:
            return
        idx = max(0, min(self._opt_chain_row_idx, len(mat.rows) - 1))
        pair = mat.rows[idx]
        contract = pair.call if opt_type == "call" else pair.put
        if contract is None:
            self._set_opt_status(
                f"No {opt_type} quoted at strike ${pair.strike:g}"
            )
            return
        if len(self._opt_staged_legs) >= 4:
            self._set_opt_status("Max 4 legs staged — click Clear to reset")
            return
        iv = contract.iv if (contract.iv is not None and 0.02 <= contract.iv <= 4.0) else ((mat.atm_iv_pct or 25.0) / 100.0)
        leg = OptionLeg(
            opt_type=opt_type,
            side=side.upper(),
            strike=contract.strike,
            expiration=mat.expiration,
            dte=mat.dte,
            bid=contract.bid,
            ask=contract.ask,
            mid=contract.mid,
            iv=iv,
            delta=float(contract.delta or 0.0),
            theta=float(contract.theta or 0.0),
            open_interest=contract.open_interest,
            volume=contract.volume,
        )
        self._opt_staged_legs.append(leg)
        sname, _ = classify_legs(self._opt_staged_legs)
        self._set_opt_status(
            f"Staged {side.upper()} {opt_type.upper()} ${contract.strike:g} → {sname} ({len(self._opt_staged_legs)} legs)"
        )
        self.query_one("#opt-chain-inspector", Static).update(
            format_chain_inspector(
                mat,
                self._opt_staged_legs,
                days_elapsed=self._days_elapsed,
                iv_shock_pct=self._iv_shock_pct,
            )
        )

    def _stage_bull_put_template(self) -> None:
        """1-click 5%-wide Bull Put spread anchored at the highlighted strike on the Chain Matrix."""
        mat = self._opt_chain_matrix
        if mat is None or not mat.rows:
            return
        idx = max(0, min(self._opt_chain_row_idx, len(mat.rows) - 1))
        short_pair = mat.rows[idx]
        sp = short_pair.put
        if sp is None:
            self._set_opt_status(f"No put quoted at ${short_pair.strike:g}")
            return
        target_long_k = sp.strike - max(1.0, mat.spot * 0.05)
        put_candidates = [r.put for r in mat.rows if r.put is not None and r.strike < sp.strike]
        if not put_candidates:
            self._set_opt_status("No lower put strike available for wing")
            return
        lp = min(put_candidates, key=lambda c: abs(c.strike - target_long_k))
        fallback_iv = (mat.atm_iv_pct or 25.0) / 100.0
        self._opt_staged_legs = [
            OptionLeg(
                "put", "SELL", sp.strike, mat.expiration, mat.dte,
                sp.bid, sp.ask, sp.mid,
                sp.iv if (sp.iv and 0.02 <= sp.iv <= 4.0) else fallback_iv,
                float(sp.delta or 0.0), float(sp.theta or 0.0), sp.open_interest, sp.volume,
            ),
            OptionLeg(
                "put", "BUY", lp.strike, mat.expiration, mat.dte,
                lp.bid, lp.ask, lp.mid,
                lp.iv if (lp.iv and 0.02 <= lp.iv <= 4.0) else fallback_iv,
                float(lp.delta or 0.0), float(lp.theta or 0.0), lp.open_interest, lp.volume,
            ),
        ]
        self._set_opt_status(
            f"Staged 5%-wide Bull Put {sp.strike:g}/{lp.strike:g}P ({mat.dte}d)"
        )
        self.query_one("#opt-chain-inspector", Static).update(
            format_chain_inspector(
                mat,
                self._opt_staged_legs,
                days_elapsed=self._days_elapsed,
                iv_shock_pct=self._iv_shock_pct,
            )
        )

    def _load_options_intel(self) -> None:
        self._set_opt_status("Loading UOA, OCC, Vol-Intel & Skew…")
        self._worker_load_intel()

    @work(thread=True, exclusive=True, group="options_intel")
    def _worker_load_intel(self) -> None:
        try:
            intel = self.provider.get_options_intel()

            def _apply() -> None:
                self._opt_intel = intel
                self._render_options_intel()

            self.call_from_thread(_apply)
        except Exception as exc:
            self.call_from_thread(
                self._set_opt_status,
                f"[#e06c75]Intel err: {str(exc)[:48]}[/]",
            )

    def _render_options_intel(self) -> None:
        intel = self._opt_intel
        if intel is None:
            return
        for bid, key in (
            ("#opt-intel-all", "all"),
            ("#opt-intel-flow", "flow"),
            ("#opt-intel-vol", "vol"),
            ("#opt-intel-skew", "skew"),
        ):
            self.query_one(bid, Button).variant = "primary" if self._opt_intel_mode == key else "default"
        tbl = self.query_one("#opt-intel-table", DataTable)
        self._opt_intel_items = fill_options_intel_table(tbl, intel, mode=self._opt_intel_mode)
        self._set_opt_status(
            f"Intel Hub: {len(self._opt_intel_items)} rows · UOA {len(intel.uoa_rows)} · "
            f"OCC {len(intel.occ_rows)} · VRP {len(intel.vol_rows)} · Skew {len(intel.skew_rows)}"
        )
        first = self._opt_intel_items[0] if self._opt_intel_items else None
        self.query_one("#opt-intel-inspector", Static).update(format_intel_inspector(first))

    # ── Events ───────────────────────────────────────────────────────────────

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        pane_id = event.pane.id if event.pane is not None else ""
        if pane_id == "workbench" and self._wb_loaded_ticker != self.focused_ticker:
            self._compute_workbench(self.focused_ticker)
        elif pane_id == "discover":
            self._ensure_hub()
        elif pane_id == "options":
            self._ensure_options_loaded()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "radar-filter-input":
            self._render_radar()
        elif event.input.id == "opt-filter-input":
            self._render_options_scanner()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id in ("wb-ticker-input", "wb-target-input", "wb-days-input", "wb-risk-input"):
            self._compute_workbench(self.query_one("#wb-ticker-input", Input).value)
        elif event.input.id == "opt-chain-ticker":
            self._opt_staged_legs = []
            self._load_chain(event.value.strip().upper(), live=False)

    def on_select_changed(self, event: Select.Changed) -> None:
        if not self.is_mounted:
            return
        try:
            if event.select.id in ("wb-dir-select", "wb-acct-select"):
                self._compute_workbench(self.query_one("#wb-ticker-input", Input).value)
            elif event.select.id == "disc-lens-select" and isinstance(event.value, str):
                self._disc_mode = "lens"
                self._disc_lens = event.value
                if self.hub is not None:
                    self.query_one("#disc-crumb", Static).update(crumb_and_freshness(self.hub, 3))
                self._render_disc_right()
            elif event.select.id == "opt-strat-select" and isinstance(event.value, str):
                self._opt_scan_loaded = False
                if self.query_one("#main-tabs", TabbedContent).active == "options":
                    self._run_options_scan(live=False)
            elif event.select.id == "opt-chain-exp" and isinstance(event.value, str):
                if not self._opt_chain_syncing_exp and event.value != "__default__":
                    tk = self.query_one("#opt-chain-ticker", Input).value.strip().upper() or self.focused_ticker
                    is_live = bool(self._opt_chain_matrix and self._opt_chain_matrix.source == "live")
                    self._load_chain(tk, expiration=event.value, live=is_live)
            elif event.select.id == "opt-chain-band" and isinstance(event.value, str):
                if self._opt_chain_matrix is not None:
                    self._load_chain(
                        self._opt_chain_matrix.ticker,
                        expiration=self._opt_chain_matrix.expiration,
                        live=(self._opt_chain_matrix.source == "live"),
                    )
        except NoMatches:
            pass

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "wb-owns-switch":
            self._compute_workbench(self.query_one("#wb-ticker-input", Input).value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "wb-run-btn":
            self._compute_workbench(self.query_one("#wb-ticker-input", Input).value)
        elif bid == "wb-chain-btn":
            self.open_options_chain(self.query_one("#wb-ticker-input", Input).value)
        elif bid == "wb-heat-btn":
            self.action_toggle_heatmap()
        elif bid == "wb-paper-btn":
            self.action_paper_log_option()
        elif bid == "wb-fill-btn" or bid == "act-fill":
            self.action_log_fill()
        elif bid == "act-ritual":
            self.action_morning_ritual()
        elif bid == "act-refresh":
            self.action_refresh()
        elif bid == "act-rebal":
            self.action_rebalance_modal()
        elif bid == "act-bcmd":
            self.action_bloomberg_cmd()
        elif bid == "act-options":
            self.open_options_view("scanner")
        elif bid == "act-discord":
            self._dispatch_discord("morning")
        elif bid == "opt-discord-btn":
            self.action_discord_push()
        elif bid == "act-live-opts":
            self.action_live_options()
        elif bid == "act-workbench":
            self.open_workbench(self.focused_ticker)
        elif bid == "act-discover":
            self.action_switch_tab("discover")
        elif bid == "act-log":
            self.action_quick_log()
        elif bid == "act-help":
            self.action_show_help()
        elif bid == "opt-nav-scanner":
            self.open_options_view("scanner")
        elif bid == "opt-nav-chain":
            self.open_options_view("chain")
        elif bid == "opt-nav-intel":
            self.open_options_view("intel")
        elif bid == "opt-live-scan-btn":
            self._run_options_scan(live=True)
        elif bid == "opt-chain-load-btn":
            self._opt_staged_legs = []
            self._load_chain(self.query_one("#opt-chain-ticker", Input).value, live=False)
        elif bid == "opt-chain-live-btn":
            self._opt_staged_legs = []
            self._load_chain(self.query_one("#opt-chain-ticker", Input).value, live=True)
        elif bid == "opt-leg-bc":
            self._stage_chain_leg("call", "BUY")
        elif bid == "opt-leg-sc":
            self._stage_chain_leg("call", "SELL")
        elif bid == "opt-leg-bp":
            self._stage_chain_leg("put", "BUY")
        elif bid == "opt-leg-sp":
            self._stage_chain_leg("put", "SELL")
        elif bid == "opt-leg-bp5":
            self._stage_bull_put_template()
        elif bid == "opt-leg-clear":
            self._opt_staged_legs = []
            if self._opt_chain_matrix is not None:
                self.query_one("#opt-chain-inspector", Static).update(
                    format_chain_inspector(
                        self._opt_chain_matrix,
                        [],
                        days_elapsed=self._days_elapsed,
                        iv_shock_pct=self._iv_shock_pct,
                    )
                )
            self._set_opt_status("Cleared staged legs")
        elif bid in ("opt-intel-all", "opt-intel-flow", "opt-intel-vol", "opt-intel-skew"):
            self._opt_intel_mode = bid.replace("opt-intel-", "")
            self._render_options_intel()
        elif bid == "opt-intel-refresh":
            self._load_options_intel()
        else:
            matched = False
            for pid, _label, pq in FILTER_PRESETS:
                if bid == pid:
                    self.query_one("#radar-filter-input", Input).value = pq
                    self._render_radar()
                    matched = True
                    break
            if not matched:
                for pid, _label, pq in OPT_FILTER_PRESETS:
                    if bid == pid:
                        self.query_one("#opt-filter-input", Input).value = pq
                        self._render_options_scanner()
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
                self.focused_opp = opp
                self._update_radar_dossier(opp.ticker)
            except (ValueError, IndexError):
                pass
        elif tid == "disc-lens-table" and event.row_key is not None:
            parts = str(event.row_key.value).split("|")
            if len(parts) == 2:
                tk = parts[1]
                self.focused_ticker = tk
                self.focused_opp = None
                if self.hub is not None:
                    ohlcv = getattr(self.hub, "ohlcv", None) or {}
                    if tk in ohlcv and len(ohlcv[tk][1]):
                        self.focused_spot = round(float(ohlcv[tk][1][-1]), 2)
                    if self._disc_mode == "lens":
                        res = self.hub.lens(self._disc_lens)
                        if res is not None:
                            row = next((r for r in res.rows if r.ticker == tk), None)
                            if row is not None and row.playbook_id:
                                self.focused_playbook = row.playbook_id
                            else:
                                self.focused_playbook = self._disc_lens
                    else:
                        self.focused_playbook = "discretionary"
                self._update_disc_hint(tk)
        elif tid == "book-picks-table" and event.row_key is not None:
            try:
                self._focused_pick_id = int(str(event.row_key.value))
            except ValueError:
                self._focused_pick_id = None
        elif tid == "wb-vehicles-table" and event.row_key is not None:
            try:
                idx = int(str(event.row_key.value))
                self._selected_vehicle_idx = idx
                self._render_wb_ladder(self._last_vehicles[idx])
                if self._wb_show_heatmap:
                    self._render_wb_visual()
            except (ValueError, IndexError):
                pass
        elif tid == "opt-scan-table" and event.row_key is not None:
            try:
                idx = int(str(event.row_key.value))
                scan_row = self._opt_scan_filtered[idx]
                self._opt_selected_scan = scan_row
                self.focused_ticker = scan_row.ticker
                self.focused_spot = scan_row.spot
                self.query_one("#opt-scan-inspector", Static).update(
                    format_scan_inspector(scan_row, days_elapsed=self._days_elapsed, iv_shock_pct=self._iv_shock_pct)
                )
            except (ValueError, IndexError, NoMatches):
                pass
        elif tid == "opt-chain-table" and event.row_key is not None:
            try:
                self._opt_chain_row_idx = int(str(event.row_key.value))
            except ValueError:
                pass
        elif tid == "opt-intel-table" and event.row_key is not None:
            parts = str(event.row_key.value).split("|")
            try:
                idx = int(parts[0])
                item = self._opt_intel_items[idx]
                self.focused_ticker = str(item.get("ticker") or self.focused_ticker)
                self.query_one("#opt-intel-inspector", Static).update(format_intel_inspector(item))
            except (ValueError, IndexError, NoMatches):
                pass

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        tid = event.data_table.id
        if tid == "radar-table" and event.row_key is not None:
            try:
                opp = self.filtered_opps[int(str(event.row_key.value))]
                self.open_workbench(opp.ticker)
            except (ValueError, IndexError):
                pass
        elif tid == "book-picks-table" and event.row_key is not None:
            try:
                self._focused_pick_id = int(str(event.row_key.value))
                self.action_close_pick()
            except ValueError:
                pass
        elif tid == "today-top5-table" and event.row_key is not None:
            parts = str(event.row_key.value).split("|")
            tk = parts[1] if len(parts) >= 2 else parts[0]
            self.open_workbench(tk)
        elif tid == "disc-board" and event.row_key is not None:
            self._disc_mode = "sector"
            self._disc_sector = str(event.row_key.value)
            # blank the lens picker so choosing ANY lens (even the previous one) fires a change
            self.query_one("#disc-lens-select", Select).clear()
            if self.hub is not None:
                self.query_one("#disc-crumb", Static).update(crumb_and_freshness(self.hub, 2))
            self._render_disc_right()
            self.query_one("#disc-lens-table", DataTable).focus()
        elif tid == "disc-lens-table" and event.row_key is not None:
            parts = str(event.row_key.value).split("|")
            if len(parts) == 2:
                self.open_evidence_card(parts[1])
        elif tid == "book-options-table" and event.row_key is not None:
            parts = str(event.row_key.value).split("|")
            if len(parts) >= 2 and parts[1]:
                exp = parts[2] if len(parts) >= 3 and parts[2] else None
                self.open_options_chain(parts[1], expiration=exp)
        elif tid == "opt-scan-table" and event.row_key is not None:
            try:
                scan_row = self._opt_scan_filtered[int(str(event.row_key.value))]
                exp = scan_row.expiration if len(scan_row.expiration) == 10 else None
                self.open_options_chain(scan_row.ticker, expiration=exp, live=(scan_row.source == "live"))
            except (ValueError, IndexError):
                pass
        elif tid == "opt-intel-table" and event.row_key is not None:
            parts = str(event.row_key.value).split("|")
            if len(parts) >= 2 and parts[1]:
                self.open_options_chain(parts[1])

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
        active = self.query_one("#main-tabs", TabbedContent).active
        if active == "options" and self._opt_view == "scanner":
            self.query_one("#opt-filter-input", Input).focus()
            return
        self.query_one("#main-tabs", TabbedContent).active = "radar"
        self.query_one("#radar-filter-input", Input).focus()

    def action_open_chain(self) -> None:
        self.open_options_chain(self.focused_ticker or "SPY")

    def action_discord_push(self) -> None:
        """Context-aware 1-key Discord push (staged/selected option ticket on Options tab, else Morning Dispatch)."""
        active = self.query_one("#main-tabs", TabbedContent).active
        if active == "options":
            if (self._opt_view == "chain" and self._opt_staged_legs) or (
                self._opt_view == "scanner" and self._opt_selected_scan is not None
            ):
                self._dispatch_discord("ticket")
                return
            self._dispatch_discord("options")
            return
        self._dispatch_discord("morning")

    def _dispatch_discord(self, mode: str = "morning") -> None:
        if self.offline:
            self._set_status("Offline mode — Discord push skipped")
            return
        self._set_status(f"Sending {mode} to Discord…")
        self._worker_discord_push(mode)

    @work(thread=True, exclusive=True, group="discord_push")
    def _worker_discord_push(self, mode: str) -> None:
        try:
            from src.notifications import (
                build_morning_embed,
                build_option_ticket_embed,
                build_options_embed,
                send_discord_embed,
            )
            from src.tui.options_engine import analyze_custom_legs

            embed: Optional[Dict[str, Any]] = None
            label = mode
            if mode == "ticket":
                if self._opt_view == "chain" and self._opt_staged_legs and self._opt_chain_matrix is not None:
                    mat = self._opt_chain_matrix
                    an = analyze_custom_legs(
                        mat.ticker,
                        mat.spot,
                        self._opt_staged_legs,
                        days_elapsed=self._days_elapsed,
                        iv_shock_pct=self._iv_shock_pct,
                    )
                    legs_s = "\n".join(
                        f"• `{lg.side}` {lg.opt_type.upper()} `${lg.strike:g}` ({lg.expiration} · {lg.dte}d) @ `${lg.mid:.2f}`"
                        for lg in self._opt_staged_legs
                    )
                    be_s = ", ".join(f"${b:.2f}" for b in an.breakevens) or "—"
                    embed = build_option_ticket_embed(
                        mat.ticker,
                        an.structure_name,
                        spot=mat.spot,
                        legs_summary=legs_s,
                        net_label=f"{'Credit' if an.is_credit else 'Debit'} `${an.net_mid:.2f}`",
                        friction_pct=an.friction_pct,
                        max_risk_usd=an.max_risk_usd,
                        tfsa_legal=an.tfsa_legal,
                        greeks_line=f"Δ$ `{an.delta_dollars:+,.0f}` · Θ/d `${an.theta_per_day_usd:+.2f}` · Mkt PoP `{an.pop_mkt*100:.0f}%` · BE `{be_s}`",
                        ladder_lines=(an.ladder.step1_label, an.ladder.step2_label, an.ladder.abort_label),
                    )
                    label = f"{mat.ticker} {an.structure_name}"
                elif self._opt_selected_scan is not None:
                    r = self._opt_selected_scan
                    greeks_s = ""
                    lad_lines: Tuple[str, ...] = (f"Screen reference price: ${r.walk_away_limit:.2f}",)
                    if r.legs and r.spot > 0:
                        an = analyze_custom_legs(r.ticker, r.spot, r.legs, days_elapsed=self._days_elapsed, iv_shock_pct=self._iv_shock_pct)
                        be_s = ", ".join(f"${b:.2f}" for b in an.breakevens) or "—"
                        greeks_s = f"Δ$ `{an.delta_dollars:+,.0f}` · Θ/d `${an.theta_per_day_usd:+.2f}` · Mkt PoP `{an.pop_mkt*100:.0f}%` · BE `{be_s}`"
                    embed = build_option_ticket_embed(
                        r.ticker,
                        r.strategy,
                        spot=r.spot,
                        legs_summary=f"`{r.strikes_label}` · Exp `{r.expiration}` ({r.dte}d)\n{r.summary}",
                        net_label=f"{'Credit' if r.is_credit else 'Debit'} `${r.mid_price:.2f}`",
                        friction_pct=r.friction_pct,
                        max_risk_usd=r.max_risk_usd,
                        tfsa_legal=r.tfsa_legal,
                        greeks_line=greeks_s,
                        ladder_lines=lad_lines,
                        flags=r.risk_flags,
                    )
                    label = f"{r.ticker} {r.strategy}"
            elif mode == "options":
                rows = list(self._opt_scan_filtered or self._opt_scan_all)
                note = f"{len(rows)} setups from TUI [7] OPTIONS"
                if not rows:
                    rows, note = self.provider.scan_options(strategy="all", live=False)
                embed = build_options_embed(rows, note=note, strategy="all")
                label = f"Options board ({len(rows)})"
            else:
                snap = self.snapshot or self.provider.get_radar(refresh=False)
                book = self.provider.get_book()
                health = self.health or self.provider.get_health()
                embed = build_morning_embed(snap, book, health)
                label = "Morning Dispatch"

            if embed is None:
                self.call_from_thread(self._set_status, "Nothing selected to send to Discord")
                return
            ok = send_discord_embed(embed)
            msg = f"Discord sent: {label}" if ok else "Discord failed (check DISCORD_WEBHOOK_URL in .env)"
            self.call_from_thread(self._set_status, msg)
        except Exception as exc:
            self.call_from_thread(self._set_status, f"Discord err: {str(exc)[:36]}")

    def action_toggle_heatmap(self) -> None:
        self._wb_show_heatmap = not self._wb_show_heatmap
        if self.query_one("#main-tabs", TabbedContent).active == "workbench":
            self._render_wb_visual()
            self._set_status("Workbench: 2D Risk Surface" if self._wb_show_heatmap else "Workbench: RND & Edge Bars")

    def action_nudge_iv(self, delta_pct: int) -> None:
        self._iv_shock_pct = max(-40.0, min(50.0, self._iv_shock_pct + float(delta_pct)))
        self._refresh_shock_views()
        self._set_status(f"Scenario: T+{self._days_elapsed}d · IV {self._iv_shock_pct:+.0f}%")

    def action_nudge_elapsed(self, delta_days: int) -> None:
        self._days_elapsed = max(0, min(90, self._days_elapsed + int(delta_days)))
        self._refresh_shock_views()
        self._set_status(f"Scenario: T+{self._days_elapsed}d · IV {self._iv_shock_pct:+.0f}%")

    def _refresh_shock_views(self) -> None:
        active = self.query_one("#main-tabs", TabbedContent).active
        if active == "workbench":
            self._render_wb_visual()
        elif active == "options":
            if self._opt_view == "scanner" and self._opt_selected_scan is not None:
                self.query_one("#opt-scan-inspector", Static).update(
                    format_scan_inspector(
                        self._opt_selected_scan,
                        days_elapsed=self._days_elapsed,
                        iv_shock_pct=self._iv_shock_pct,
                    )
                )
            elif self._opt_view == "chain" and self._opt_chain_matrix is not None:
                self.query_one("#opt-chain-inspector", Static).update(
                    format_chain_inspector(
                        self._opt_chain_matrix,
                        self._opt_staged_legs,
                        days_elapsed=self._days_elapsed,
                        iv_shock_pct=self._iv_shock_pct,
                    )
                )

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
        self.hub = None
        self._opt_scan_loaded = False
        self._opt_intel = None
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

    def action_toggle_account(self) -> None:
        sel = self.query_one("#wb-acct-select", Select)
        sel.value = "taxable" if str(sel.value) == "tfsa" else "tfsa"

    def action_nudge_target(self, delta_pct: int) -> None:
        if self.query_one("#main-tabs", TabbedContent).active != "workbench":
            return
        inp = self.query_one("#wb-target-input", Input)
        raw = inp.value.strip().rstrip("%")
        try:
            cur = int(round(float(raw)))
        except ValueError:
            cur = 6
        nxt = max(-30, min(30, cur + int(delta_pct)))
        inp.value = f"{nxt:+d}%"
        self._compute_workbench(self.query_one("#wb-ticker-input", Input).value)

    def action_nudge_days(self, delta_days: int) -> None:
        if self.query_one("#main-tabs", TabbedContent).active != "workbench":
            return
        inp = self.query_one("#wb-days-input", Input)
        try:
            cur = int(inp.value.strip())
        except ValueError:
            cur = 30
        nxt = max(5, min(180, cur + int(delta_days)))
        inp.value = str(nxt)
        self._compute_workbench(self.query_one("#wb-ticker-input", Input).value)

    def action_toggle_rrg(self) -> None:
        self._show_rrg = not self._show_rrg
        if self.hub is not None:
            self._render_discover()

    def action_rebalance_modal(self) -> None:
        snap = self.snapshot
        if snap is None or not snap.core_allocation:
            self._set_status("Core allocation unavailable — refresh first")
            return
        from src.equity_playbooks.rebalance import compute_tfsa_rebalance
        from src.profile import (
            apply_rebalance_to_profile,
            estimate_cash_yield_annual,
            load_profile,
            rebalance_due_status,
        )

        prof = load_profile()
        plan = compute_tfsa_rebalance(
            snap.core_allocation,
            account_usd=prof.tfsa_usd,
            current_shares=prof.core_holdings,
            fractional=prof.fractional_shares,
            drift_threshold_pct=prof.drift_threshold_pct,
        )
        y_ann, y_lbl = estimate_cash_yield_annual()
        est_y = plan.estimated_cash_after_usd * y_ann
        due, due_msg = rebalance_due_status(
            prof,
            max_drift_pct=plan.max_drift_pct,
            rebalance_recommended=plan.rebalance_recommended,
        )
        lines = [
            f"[bold]{'REBALANCE DUE' if due else 'ON TARGET'}[/]: {due_msg}",
            f"Broker: {prof.broker} · Account: ${plan.account_usd:,.2f} USD · Vol Scale: {plan.vol_scale*100:.0f}%",
            f"Defensive Buffer ({prof.cash_etf}): ${plan.estimated_cash_after_usd:,.2f} (~${est_y:,.0f}/yr @ {y_lbl})",
            "─" * 74,
            f"{'ACTION':<8} {'ETF':<10} {'TGT%':>6} {'NOW%':>6} {'DRIFT':>7} {'ORDER SH':>10} {'DELTA $':>11}",
        ]
        target_map: Dict[str, float] = {}
        for o in plan.orders:
            if o.action != "RESERVE":
                target_map[o.ticker] = o.target_shares
            ord_s = f"{o.delta_shares:+.4g}" if o.action in ("BUY", "SELL") else ("SGOV" if o.action == "RESERVE" else "—")
            lines.append(
                f"{o.action:<8} {o.ticker:<10} {o.target_weight_pct:5.1f}% {o.current_weight_pct:5.1f}% "
                f"{o.drift_pct:+6.1f}% {ord_s:>10} ${o.delta_usd:+10,.2f}"
            )

        def _after_reb(res: Optional[Dict[str, Any]]) -> None:
            if res and res.get("apply"):
                apply_rebalance_to_profile(prof, res["target_shares"])
                self._set_status("Saved TFSA Core target shares to data/tfsa_profile.json")
                self._render_today()
                self._render_book()

        self.push_screen(
            RebalanceModal(
                snap=snap,
                plan_text="\n".join(lines),
                target_shares=target_map,
                account_usd=prof.tfsa_usd,
            ),
            _after_reb,
        )

    def action_morning_ritual(self) -> None:
        try:
            from src.cli.morning_ritual import build_morning_ritual, format_morning_ritual_markup

            ritual = build_morning_ritual(provider=self.provider, refresh=False)
            body = format_morning_ritual_markup(ritual)
        except Exception as exc:
            body = f"[#e06c75]Morning Ritual unavailable: {exc}[/]"
        self.push_screen(AlphaViewerModal(title="60-SECOND MORNING RITUAL", body=body))

    def action_bloomberg_cmd(self) -> None:
        def _run_cmd(raw: Optional[str]) -> None:
            if not raw:
                return
            self._execute_bloomberg_mnemonic(raw)

        self.push_screen(BloombergCommandModal(), _run_cmd)

    def _execute_bloomberg_mnemonic(self, raw: str) -> None:
        parts = raw.strip().split()
        if not parts:
            return
        cmd = parts[0].upper()
        arg = parts[1].upper() if len(parts) > 1 else ""
        if cmd in ("RITUAL", "MORNING"):
            self.action_morning_ritual()
            return
        if cmd in ("REB", "REBAL", "REBALANCE"):
            self.action_rebalance_modal()
            return
        if cmd == "LOG":
            self.action_quick_log()
            return
        if cmd == "CLOSE":
            self.action_close_pick()
            return
        if cmd == "FILL":
            self.action_log_fill()
            return
        if cmd == "THEME" and arg:
            self.apply_desk_theme(arg.lower())
            self._set_status(f"Theme: {arg.lower()}")
            return
        if cmd in ("TODAY", "RADAR", "WORKBENCH", "BOOK", "LAB", "DISCOVER", "OPTIONS"):
            self.action_switch_tab(cmd.lower())
            return
        if cmd in ("OPT", "SCAN"):
            self.open_options_view("scanner", strategy=arg.lower() if arg else None)
            return
        if cmd == "CHAIN":
            self.open_options_chain(arg or self.focused_ticker or "SPY")
            return
        if cmd in ("FLOW", "UOA"):
            self._opt_intel_mode = "flow"
            self.open_options_view("intel")
            return
        if cmd in ("VOL", "VRP"):
            self._opt_intel_mode = "vol"
            self.open_options_view("intel")
            return
        if cmd == "SURF":
            self._wb_show_heatmap = True
            self.open_workbench(arg or self.focused_ticker or "SPY")
            return
        if cmd in ("DISCORD", "DISC", "SEND"):
            sub = "options" if arg in ("OPT", "OPTIONS", "SCAN") else ("ticket" if arg in ("THIS", "TICKET", "LEG") else "morning")
            self._dispatch_discord(sub)
            return
        if cmd in ("SKEW", "GEX", "EDGAR", "13D", "SPIN", "BUYBACK", "GUIDANCE", "SUPPLY", "OCC", "CAL", "CALENDAR", "MACRO", "PM", "PREMORTEM"):
            self._open_alpha_mnemonic(cmd, arg or self.focused_ticker)
            return
        if arg in ("CHAIN", "OPT"):
            self.open_options_chain(cmd)
            return
        if arg in ("WB", "WORKBENCH", "EXPRESS"):
            self.open_workbench(cmd)
            return
        if arg in ("CARD", "C") or len(parts) == 1:
            self.open_evidence_card(cmd)
            return
        self.open_workbench(cmd)

    def _open_alpha_mnemonic(self, cmd: str, ticker: str) -> None:
        try:
            from src.alpha.formatters import format_alpha_screen

            title, body = format_alpha_screen(cmd, ticker=ticker)
        except Exception as exc:
            title = f"ALPHA TELEMETRY: {cmd}"
            body = f"[#d6a452]Telemetry unavailable:[/] {exc}"
        self.push_screen(AlphaViewerModal(title=title, body=body))

    def action_quick_log(self) -> None:
        def _after_modal(res: Optional[Dict[str, Any]]) -> None:
            if not res:
                return
            self.provider.log_stock_pick(
                ticker=res["ticker"],
                price=res["price"],
                shares=res["shares"],
                playbook_id=res["playbook_id"],
                stop_price=res.get("stop_price"),
                target_price=res.get("target_price"),
                notes=res["notes"],
            )
            self._render_book()
            self._set_status(f"Logged pick: {res['ticker']}")

        self.push_screen(
            QuickLogModal(
                ticker=self.focused_ticker,
                price=self.focused_spot,
                playbook_id=self.focused_playbook,
            ),
            _after_modal,
        )

    def action_paper_log_option(self) -> None:
        """Open PaperOptionModal prefilled from Options Scanner, Chain Composer, or Workbench."""
        active = self.query_one("#main-tabs", TabbedContent).active
        tk = self.focused_ticker or "SPY"
        strat = "Bull Put"
        exp = ""
        strike = round(self.focused_spot or 100.0, 1)
        long_strike: Optional[float] = None
        opt_type = "put"
        price = 1.00
        max_loss: Optional[float] = None
        iv: Optional[float] = None
        delta: Optional[float] = None
        warn = ""

        if active == "options" and self._opt_view == "scanner" and self._opt_selected_scan is not None:
            r = self._opt_selected_scan
            tk = r.ticker
            strat = r.strategy
            exp = r.expiration[:10]
            price = r.mid_price
            max_loss = r.max_risk_usd
            iv = (r.iv_pct / 100.0) if r.iv_pct else None
            delta = r.short_delta
            if r.legs:
                primary = next((lg for lg in r.legs if lg.side == "SELL"), r.legs[0])
                wing = next((lg for lg in r.legs if lg is not primary), None)
                strike = primary.strike
                opt_type = primary.opt_type
                long_strike = wing.strike if wing is not None else None
            if not r.tfsa_legal:
                warn = "Multi-leg / short option — TAXABLE account only (not CRA TFSA-legal)."
            if r.friction_pct > 0.20:
                warn = (warn + " " if warn else "") + f"Spread friction is {r.friction_pct*100:.1f}% — use limit order."
        elif active == "options" and self._opt_view == "chain" and self._opt_staged_legs:
            sname, tfsa_ok = classify_legs(self._opt_staged_legs)
            mat = self._opt_chain_matrix
            tk = mat.ticker if mat else tk
            strat = sname
            primary = next((lg for lg in self._opt_staged_legs if lg.side == "SELL"), self._opt_staged_legs[0])
            wing = next((lg for lg in self._opt_staged_legs if lg is not primary), None)
            exp = primary.expiration[:10]
            strike = primary.strike
            long_strike = wing.strike if wing is not None else None
            opt_type = primary.opt_type
            price = max(0.05, abs(sum(lg.signed_qty * lg.mid for lg in self._opt_staged_legs)))
            iv = primary.iv
            delta = primary.delta
            if not tfsa_ok:
                warn = "TAXABLE account only (not CRA TFSA-legal)."
        elif active == "workbench" and self._last_vehicles and 0 <= self._selected_vehicle_idx < len(self._last_vehicles):
            v = self._last_vehicles[self._selected_vehicle_idx]
            strat = "Bull Put" if "Credit" in v.vehicle else ("Long Call" if (self._wb_last_view and self._wb_last_view.direction != "BEARISH") else "Long Put")
            opt_type = "put" if "Put" in strat else "call"
            dte = self._wb_last_view.horizon_days if self._wb_last_view else 35
            exp = (dt.date.today() + dt.timedelta(days=dte)).isoformat()
            price = max(0.05, round((v.max_profit_usd if "Credit" in v.vehicle else v.capital_at_risk_usd) / 100.0, 2))
            max_loss = v.capital_at_risk_usd
            if not v.tfsa_legal:
                warn = "TAXABLE account only (CRA TFSA ban on spreads)."

        if not exp:
            exp = (dt.date.today() + dt.timedelta(days=45)).isoformat()

        def _after_paper(res: Optional[Dict[str, Any]]) -> None:
            if not res:
                return
            try:
                ok = self.provider.log_paper_option(**res)
                if ok:
                    self._render_book()
                    self._set_status(f"Paper-logged {res['ticker']} {res['strategy_name']} @ ${res['entry_price']:.2f}")
                else:
                    self._set_status("Paper option refused by guard")
            except Exception as exc:
                self._set_status(f"Paper log err: {str(exc)[:36]}")

        self.push_screen(
            PaperOptionModal(
                ticker=tk,
                strategy_name=strat,
                expiration=exp,
                strike=strike,
                opt_type=opt_type,
                entry_price=price,
                long_strike=long_strike,
                max_loss_usd=max_loss,
                entry_iv=iv,
                entry_delta=delta,
                warning_text=warn,
            ),
            _after_paper,
        )

    def action_close_pick(self) -> None:
        open_picks = [p for p in self._book_picks if str(p.get("status") or "").upper() == "OPEN"]
        if not open_picks:
            book = self.provider.get_book()
            self._book_picks = list(book.get("equity_picks") or [])
            open_picks = [p for p in self._book_picks if str(p.get("status") or "").upper() == "OPEN"]
        if not open_picks:
            self._set_status("No OPEN stock/ETF picks to close")
            return
        pick = next(
            (p for p in open_picks if self._focused_pick_id is not None and int(p["pick_id"]) == self._focused_pick_id),
            open_picks[0],
        )

        def _after_close(res: Optional[Dict[str, Any]]) -> None:
            if not res:
                return
            try:
                ok = self.provider.close_stock_pick(
                    pick_id=res["pick_id"],
                    exit_price=res["exit_price"],
                    notes=res.get("notes"),
                    close_tag=res.get("close_tag"),
                )
            except TypeError:
                ok = self.provider.close_stock_pick(
                    pick_id=res["pick_id"],
                    exit_price=res["exit_price"],
                    notes=res.get("notes"),
                )
            self._render_book()
            if ok:
                tag_s = f" [{res['close_tag']}]" if res.get("close_tag") else ""
                self._set_status(f"Closed pick #{res['pick_id']} @ ${res['exit_price']:.2f}{tag_s}")

        self.push_screen(
            ClosePickModal(
                pick_id=int(pick["pick_id"]),
                ticker=str(pick["ticker"]),
                entry_price=float(pick["entry_price"]),
                current_price=pick.get("current_price"),
                shares=float(pick.get("shares") or 1.0),
                exit_hint=str(pick.get("exit_signal") or ""),
            ),
            _after_close,
        )

    def action_log_fill(self) -> None:
        def _after_fill(res: Optional[Dict[str, Any]]) -> None:
            if not res:
                return
            self.provider.log_fill_attempt(**res)
            self.health = self.provider.get_health()
            self._render_lab()
            self._set_status(f"Fill logged for {res['ticker']}")

        opp = self.focused_opp
        ex = (opp.extra or {}) if opp is not None else {}
        active_tab = self.query_one("#main-tabs", TabbedContent).active
        if (opp is not None and opp.playbook_id == "etf_wide_spread" and opp.ticker == self.focused_ticker
                and ex.get("credit_cross") is not None and ex.get("credit_mid") is not None):
            # Pre-fill the SPREAD's own quote, so k is measured the way Test O1
            # assumed it: bid = natural credit, ask = the far side.
            nat, mid = float(ex["credit_cross"]), float(ex["credit_mid"])
            kind = "Bull Put" if opp.direction == "LONG" else "Bear Call"
            modal = LogFillModal(
                ticker=opp.ticker,
                structure=f"O1 {kind} {ex.get('short_strike', 0):g}/{ex.get('long_strike', 0):g} {ex.get('expiration', '')}",
                side="SELL",
                limit_price=float(ex.get("walk_away_limit") or mid),
                bid=nat,
                ask=round(2.0 * mid - nat, 2),
            )
        elif active_tab == "workbench" and self._last_vehicles and 0 <= self._selected_vehicle_idx < len(self._last_vehicles):
            v = self._last_vehicles[self._selected_vehicle_idx]
            is_credit = v.vehicle in ("Wide Credit Spread", "Covered Call")
            est_px = max(0.05, round((v.max_profit_usd if is_credit else v.capital_at_risk_usd) / 100.0, 2))
            est_half = max(0.02, round(v.friction_usd / 200.0, 2))
            struct_lbl = f"O1 {v.description}" if v.vehicle == "Wide Credit Spread" else v.vehicle
            modal = LogFillModal(
                ticker=self.focused_ticker,
                structure=struct_lbl[:48],
                side="SELL" if is_credit else "BUY",
                limit_price=est_px,
                bid=max(0.01, round(est_px - est_half, 2)),
                ask=round(est_px + est_half, 2),
            )
        else:
            modal = LogFillModal(ticker=self.focused_ticker, structure="Bull Put", side="SELL")
        self.push_screen(modal, _after_fill)


def run_desk(*, initial_tab: str = "today", offline: bool = False) -> None:
    app = DeskApp(initial_tab=initial_tab, offline=offline)
    app.run()
