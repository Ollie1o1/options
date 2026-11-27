"""Rich renderables for the top-down Discover view — shared by `opt discover`/`opt card` and the TUI.

Layout contract (the same on every surface, so a reader always knows where they are):

  ① MARKET      what kind of market is this?            (context only)
  ② SECTORS     where is the strength? ranked by the PROVEN Outlook model
  ③ IDEAS       lenses — each says what tested it and how it is sorted
  ④ EVIDENCE    one ticker: verdict, why, structure, peers, options, events, sources

Evidence chips always use the same words and colours:
  PROVEN · PROMISING · TESTING · WATCHLIST · CONTEXT · AVOID
Green/red on numbers only ever means up/down — never buy/sell.
Every panel ends with a dim "Source · as of" line.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .card import EvidenceCard
from .hub import DiscoveryHub
from .lenses import NULL_LEDGER, LensResult, tags_by_ticker
from .sectors import SECTOR_NAMES

GREEN, BLUE, AMBER, RED, PURPLE, GREY = "#5ec98d", "#82aad2", "#d6a452", "#e06c75", "#c678dd", "#8b949e"

LABEL_STYLE = {
    "PROVEN": f"bold #0e1117 on {GREEN}",
    "PROMISING": f"bold #0e1117 on {BLUE}",
    "TESTING": f"bold #0e1117 on {PURPLE}",
    "WATCHLIST": f"bold #0e1117 on {AMBER}",
    "CONTEXT": f"bold #0e1117 on {GREY}",
    "AVOID": f"bold #0e1117 on {RED}",
    "RETIRED": f"bold #0e1117 on {GREY}",
}
LABEL_MEANING = {
    "PROVEN": "passed a preregistered out-of-sample test",
    "PROMISING": "passed part of one — read the caveat",
    "TESTING": "registered test, readout not due yet",
    "WATCHLIST": "descriptive overlay, not tested yet",
    "CONTEXT": "description only, never tested as a signal",
    "AVOID": "measured negative or imminent event",
}
POSTURE_STYLE = {"FAVOR": f"bold {GREEN}", "AVOID": f"bold {RED}", "NEUTRAL": GREY}
ACTION_WORD = {"main": "maintains", "up": "upgrades", "down": "downgrades", "init": "initiates", "reit": "reiterates"}
VERDICT_COLOR = {"green": GREEN, "amber": AMBER, "red": RED, "grey": GREY}

STEP_TITLES = {
    1: "① MARKET — what kind of market is this?",
    2: "② SECTORS — where is the strength?",
    3: "③ IDEAS — why look here?",
    4: "④ EVIDENCE CARD",
}


def chip(label: str) -> Text:
    return Text(f" {label} ", style=LABEL_STYLE.get(label, "bold reverse"))


def _num(x: Optional[float], fmt: str = "{:+.1f}%", none: str = "—") -> str:
    return none if x is None else fmt.format(x)


def signed(x: Optional[float], fmt: str = "{:+.1f}%") -> Text:
    if x is None:
        return Text("—", style="dim")
    return Text(fmt.format(x), style=GREEN if x > 0 else (RED if x < 0 else ""))


def source_line(*parts: str) -> Text:
    return Text("Source · " + " · ".join(p for p in parts if p), style="dim italic")


def legend() -> Text:
    t = Text("Evidence labels: ", style="dim")
    for lab in ("PROVEN", "PROMISING", "TESTING", "WATCHLIST", "CONTEXT", "AVOID"):
        t.append_text(chip(lab))
        t.append(" ")
    return t


def tag_chips(tags: Any) -> Text:
    t = Text()
    for i, (tag, lab) in enumerate(tags):
        if i:
            t.append(" ")
        t.append(tag, style=LABEL_STYLE.get(lab, "reverse"))
    return t


def tag_legend() -> Text:
    from .lenses import LENS_TAG_MEANING

    t = Text("Tags: ", style="dim")
    t.append(" · ".join(f"{k} {v}" for k, v in LENS_TAG_MEANING.items()), style="dim")
    return t


def freshness_line(hub: DiscoveryHub) -> Text:
    f = hub.freshness()
    t = Text("Data: ", style="bold")
    t.append(f"{f['symbols']} symbols, SPY through {f['spy_last'] or '—'}")
    if f["behind"]:
        t.append(f" · {f['behind']} symbols a session or more behind (e.g. {', '.join(f['behind_sample'])})",
                 style=AMBER)
        t.append(" → run `opt refresh`", style="bold")
    else:
        t.append(" · all current", style=GREEN)
    t.append(f" · built {hub.built_at}", style="dim")
    return t


def breadcrumb(active: int) -> Text:
    t = Text()
    for i in (1, 2, 3, 4):
        name = {1: "MARKET", 2: "SECTORS", 3: "IDEAS", 4: "EVIDENCE"}[i]
        sym = "①②③④"[i - 1]
        t.append(f" {sym} {name} ", style="bold reverse" if i == active else "dim")
        if i < 4:
            t.append("›", style="dim")
    return t


# ── ① MARKET ────────────────────────────────────────────────────────────────


def regime_read(hub: DiscoveryHub) -> tuple:
    """One plain-English description of the tape from trend + the VIX curve."""
    s, v = hub.macro.spy, hub.macro.vol
    above = s.get("above_200d")
    inverted = v.get("term") == "BACKWARDATION"
    hot = (v.get("vix_pctile_1y") or 0) >= 80
    if above is None:
        return "unknown (no SPY history)", "dim"
    if above and not inverted and not hot:
        return "Calm uptrend — SPY above its 200-day, VIX curve upward-sloping", f"bold {GREEN}"
    if above:
        return "Uptrend under stress — SPY above its 200-day but volatility is elevated", f"bold {AMBER}"
    if inverted:
        return "Stressed downtrend — SPY below its 200-day and the VIX curve is inverted", f"bold {RED}"
    return "Downtrend — SPY below its 200-day average", f"bold {AMBER}"



def market_panel(hub: DiscoveryHub, *, compact: bool = False) -> Panel:
    v, s = hub.macro.vol, hub.macro.spy
    top = Text()
    if s.get("as_of"):
        top.append("SPY ", style="bold")
        top.append(f"{s['last']:.2f} ")
        top.append_text(signed(s.get("ret_21d_pct")))
        top.append(" 1m · ", style="dim")
        top.append("vs 200d ", style="dim")
        top.append_text(signed(s.get("vs_200d_pct")))
        top.append(" · from 52w high ", style="dim")
        top.append_text(signed(s.get("from_52w_high_pct")))
        top.append(f" · realized vol (20d) {_num(s.get('rv20_pct'), '{:.0f}%')}", style="dim")
    vol = Text()
    if v.get("vix") is not None:
        vol.append("VIX ", style="bold")
        vol.append(f"{v['vix']:.1f}")
        vol.append(f" ({_num(v.get('vix_pctile_1y'), '{:.0f}')}th pct of 1y)", style="dim")
        vol.append("  VIX3M ", style="bold")
        vol.append(_num(v.get("vix3m"), "{:.1f}"))
        term = v.get("term", "UNKNOWN")
        vol.append("  term ", style="dim")
        vol.append(f"{term}", style=f"bold {RED}" if term == "BACKWARDATION" else f"bold {GREEN}")
        vol.append(f" ({_num(v.get('term_ratio'), '{:.2f}')})", style="dim")
        if v.get("vvix") is not None:
            vol.append(f"  VVIX {v['vvix']:.0f} · SKEW {_num(v.get('skew'), '{:.0f}')}", style="dim")
        if v.get("stale"):
            vol.append("  STALE", style=f"bold {AMBER}")
    else:
        vol.append("VIX history unavailable (data/discovery/VIX_History.csv)", style="dim")

    up, n = hub.macro.ratio_tally()
    read, read_style = regime_read(hub)
    headline = Text("Read: ", style="bold")
    headline.append(read, style=read_style)
    headline.append("  (a description of today, not a forecast)", style="dim")
    rows: List[Any] = [headline, top, vol]
    if not compact:
        rt = Table(box=box.SIMPLE_HEAD, pad_edge=False, expand=False, show_edge=False)
        rt.add_column("Rotation ratio", style="bold")
        rt.add_column("A rise means", style="dim")
        rt.add_column("1m", justify="right")
        rt.add_column("3m", justify="right")
        rt.add_column("vs 200d", justify="right")
        rt.add_column("1y pct", justify="right")
        missing = []
        for r in hub.macro.ratios:
            if r.missing:
                missing.append(r)
                continue
            rt.add_row(r.name, r.rise_means, signed(r.chg_21d_pct), signed(r.chg_63d_pct),
                       signed(r.vs_200d_pct), _num(r.pctile_1y, "{:.0f}"))
        rows.append(rt)
        if missing:
            rows.append(Text("Not available: " + "; ".join(f"{r.name} ({r.missing})" for r in missing), style="dim"))
        fred = [f for f in hub.macro.fred if f.get("value") is not None]
        if fred:
            ft = Text("FRED: ", style="bold")
            for f in fred:
                ft.append(f"{f['label']} {f['value']:.2f}{'' if f['unit'] == 'idx' else f['unit']}")
                if f.get("chg_1m") is not None:
                    ft.append(f" ({f['chg_1m']:+.2f} 1m)", style="dim")
                ft.append(f" [{f['as_of']}]  ", style="dim")
            rows.append(ft)
        else:
            rows.append(Text("FRED credit/curve series not cached — `opt discover --refresh-macro` to fetch.", style="dim"))
    if compact:
        fred = [f for f in hub.macro.fred if f.get("value") is not None]
        if fred:
            fl = Text("FRED: ", style="bold")
            short = {"BAMLH0A0HYM2": "HY spread", "T10Y2Y": "10y-2y", "T10Y3M": "10y-3m", "NFCI": "NFCI"}
            for f in fred:
                unit = "" if f["unit"] == "idx" else "%"
                fl.append(f"{short.get(f['series'], f['series'])} {f['value']:.2f}{unit}")
                if f.get("pctile_1y") is not None:
                    fl.append(f" ({f['pctile_1y']:.0f}th pct 1y)", style="dim")
                fl.append("  ")
            rows.append(fl)
        rl = Text("Ratios 3m: ", style="bold")
        for r in hub.macro.ratios:
            if r.chg_63d_pct is None:
                continue
            rl.append(f"{r.name} ")
            rl.append_text(signed(r.chg_63d_pct, "{:+.0f}%"))
            rl.append("  ")
        rows.append(rl)
    tally = Text(f"Risk-appetite ratios rising over 3m: {up}/{n}", style="bold")
    tally.append("  (a descriptive count, not a tested signal)", style="dim")
    rows.append(tally)
    rows.append(source_line(
        f"SPY {s.get('as_of') or '—'} (data/equity_ohlcv.db)",
        f"Cboe VIX/VIX3M {v.get('as_of') or '—'}",
        "VIX→direction tested NOT_ESTABLISHED; ratios untested",
    ))
    return Panel(Group(*rows), title=STEP_TITLES[1], title_align="left", border_style=BLUE, subtitle="context only",
                 subtitle_align="right")


# ── ② SECTORS ───────────────────────────────────────────────────────────────


def board_table(hub: DiscoveryHub) -> Table:
    b = hub.board
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False, expand=False)
    t.add_column("#", justify="right", style="dim")
    t.add_column("ETF", style="bold")
    t.add_column("Sector / asset")
    t.add_column("Outlook call")
    t.add_column("Sleeve", justify="right")
    t.add_column("RS vs SPY 1m", justify="right")
    t.add_column("RS 3m", justify="right")
    t.add_column("vs 200d", justify="right")
    t.add_column("Stocks >50d", justify="right")
    t.add_column(">200d", justify="right")
    t.add_column("n", justify="right", style="dim")
    for r in b.rows:
        post = Text(r.posture, style=POSTURE_STYLE.get(r.posture, ""))
        if r.direction in ("BULLISH", "BEARISH"):
            post.append(f" {r.direction.lower()}", style="dim")
        t.add_row(
            str(r.rank or "—"), r.etf, r.name, post,
            Text(f"{r.weight*100:.0f}%", style=f"bold {GREEN}") if r.weight > 0 else Text("—", style="dim"),
            signed(r.rs_21d_pct), signed(r.rs_63d_pct), signed(r.vs_200d_pct),
            _num(r.pct_above_50d, "{:.0f}%"), _num(r.pct_above_200d, "{:.0f}%"),
            str(r.n_members or "—"),
        )
    return t


def board_panel(hub: DiscoveryHub) -> Panel:
    b = hub.board
    head = Text()
    head.append_text(chip("PROVEN"))
    head.append(" Rank and FAVOR/AVOID come from the validated rotation model.")
    head2 = Text()
    head2.append_text(chip("CONTEXT"))
    head2.append(" RS, vs 200d and breadth columns are description only.", style="dim")
    key = Text("FAVOR = held by the sleeve (top 3 with a positive score) · AVOID = BEARISH call, which only fires "
               "when SPY itself is below trend · NEUTRAL = neither", style="dim")
    sleeve = Text()
    if b.sleeve_cash is not None:
        held = ", ".join(f"{r.etf} {r.weight*100:.0f}%" for r in b.rows if r.weight > 0) or "nothing"
        sleeve.append("This month's TFSA core sleeve: ", style="bold")
        sleeve.append(f"{held} · cash/T-bills {b.sleeve_cash*100:.0f}%")
        sleeve.append(f"  (scaled {b.sleeve_scale:.2f}× because the basket's 20d vol is "
                      f"{(b.basket_rv20 or 0)*100:.1f}% vs the 14% target)", style="dim")
    parts: List[Any] = [head, head2, sleeve, board_table(hub), key]
    if b.excluded:
        parts.append(Text("Not ranked: " + "; ".join(f"{k} ({v})" for k, v in b.excluded.items()), style=f"{AMBER}"))
    parts.append(source_line(f"prices as of {b.as_of}", "breadth = archived GICS constituents above their own 50d/200d",
                             "docs/RESULT_VOL_MANAGED_OUTLOOK_20261124.md"))
    return Panel(Group(*parts), title=STEP_TITLES[2], title_align="left", border_style=GREEN)


# ── ③ IDEAS ─────────────────────────────────────────────────────────────────


def lens_header(res: LensResult) -> Group:
    lz = res.lens
    t1 = Text()
    t1.append_text(chip(lz.label))
    t1.append(f" {lz.title}", style="bold")
    t1.append(f" — {lz.question}", style="")
    t2 = Text("Evidence: ", style="bold")
    t2.append(lz.evidence)
    t3 = Text(f"Sorted by: {lz.sort_note} · {len(res.rows)} row(s)", style="dim")
    parts: List[Any] = [t1, t2, t3]
    if res.note:
        parts.append(Text(res.note, style=AMBER if lz.label != "PROVEN" else "dim"))
    return Group(*parts)


def lens_table(res: LensResult, *, limit: Optional[int] = None, hub: Optional[DiscoveryHub] = None) -> Table:
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    t.add_column("Ticker", style="bold")
    t.add_column("Sector", style="dim")
    t.add_column("Evidence")
    t.add_column("Why it is listed")
    t.add_column("Warnings", style=RED)
    t.add_column("As of", style="dim")
    rows = res.rows if limit is None else res.rows[:limit]
    for r in rows:
        t.add_row(r.ticker, r.sector_etf or "—", r.evidence_label or res.lens.label, r.why, ", ".join(r.flags) or "", r.as_of or "—")
    return t


def lens_panel(res: LensResult, *, limit: Optional[int] = None) -> Panel:
    body: List[Any] = [lens_header(res)]
    if res.rows:
        body.append(lens_table(res, limit=limit))
        if limit is not None and len(res.rows) > limit:
            body.append(Text(f"… {len(res.rows) - limit} more — `opt discover --lens {res.lens.id}`", style="dim"))
    else:
        body.append(Text("Nothing listed right now.", style="dim"))
    body.append(source_line(res.lens.source))
    return Panel(Group(*body), title=f"③ IDEAS · {res.lens.title}", title_align="left",
                 border_style=_label_border(res.lens.label))


def _label_border(label: str) -> str:
    return {"PROVEN": GREEN, "PROMISING": BLUE, "TESTING": PURPLE, "WATCHLIST": AMBER, "AVOID": RED}.get(label, GREY)


def ideas_overview(hub: DiscoveryHub) -> Panel:
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    t.add_column("Label")
    t.add_column("Lens", style="bold")
    t.add_column("id", style="dim")
    t.add_column("Rows", justify="right")
    t.add_column("First names listed", no_wrap=True, overflow="ellipsis", max_width=58)
    for r in hub.lenses:
        names = ", ".join(dict.fromkeys(x.ticker for x in r.rows))
        t.add_row(chip(r.lens.label), r.lens.title, r.lens.id, str(len(r.rows)),
                  names or Text("—", style="dim"))
    foot = Text("Open one: ", style="dim")
    foot.append("opt discover --lens leaders", style="bold")
    foot.append(" · drill into a sector: ", style="dim")
    foot.append("opt discover --sector XLE", style="bold")
    foot.append(" · one ticker: ", style="dim")
    foot.append("opt card NVDA", style="bold")
    return Panel(Group(t, foot, legend()), title=STEP_TITLES[3], title_align="left", border_style=PURPLE)


def null_ledger_panel() -> Panel:
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    t.add_column("Idea (tested here)", style="bold")
    t.add_column("Result")
    t.add_column("Date", style="dim")
    t.add_column("Source", style="dim")
    for idea, result, date, src in NULL_LEDGER:
        t.add_row(idea, result, date, src)
    return Panel(t, title="Already tested — does NOT work here (don't re-chase)", title_align="left",
                 border_style=GREY)


def forward_panel(rep: Sequence[Any]) -> Panel:
    """Forward record of the lenses: counts always; outcomes only after the preregistered read date."""
    from .forward_log import MIN_WEEKS, PREREG, READ_NOT_BEFORE, status_line

    any_open = any(not r.locked for r in rep)
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    t.add_column("Lens", style="bold", no_wrap=True)
    t.add_column("Label", no_wrap=True)
    t.add_column("Hold", justify="right", no_wrap=True)
    t.add_column("Weeks", justify="right", no_wrap=True)
    t.add_column("Rows", justify="right", no_wrap=True)
    t.add_column("Since", no_wrap=True)
    if any_open:
        t.add_column("Excess vs SPY", justify="right", no_wrap=True)
        t.add_column("99.17% CI", justify="right", no_wrap=True)
        t.add_column("Read")
    for r in rep:
        cells = [r.lens_id, chip(r.label), f"{r.horizon}d", f"{r.weeks_recorded}/{MIN_WEEKS}", str(r.rows),
                 r.first or "—"]
        if any_open:
            if r.locked:
                cells += ["—", "—", r.lock_reason]
            else:
                cells += [f"{r.mean_excess * 100:+.2f}%",
                          f"[{r.ci_bonf[0] * 100:+.2f}, {r.ci_bonf[1] * 100:+.2f}]", r.proposal]
        t.add_row(*cells)
    head = Text(status_line(rep), style="bold")
    lock = (Text("Outcomes are not computed yet: the read unlocks on " + READ_NOT_BEFORE
                 + f" with {MIN_WEEKS} complete weeks per lens (preregistered, so nobody can peek and re-tune).",
                 style=AMBER) if not any_open else Text(""))
    rule = Text(f"Rule: {PREREG}. One observation = one lens-week; entry after both the data date and "
                "first record; 10 bps cost; Bonferroni 99.17% CI across 6 lenses.", style="dim")
    return Panel(Group(head, lock, t, rule), title="Discover lenses — forward record", title_align="left",
                 border_style=GREY)


def sector_members_panel(hub: DiscoveryHub, etf: str) -> Panel:
    from .peers import _metrics
    from .sectors import sector_members

    etf = etf.upper()
    row = hub.board.row(etf)
    members = sector_members(etf, hub.ohlcv)
    hits = tags_by_ticker(hub.lenses)
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    t.add_column("Ticker", style="bold")
    t.add_column("12-1m", justify="right")
    t.add_column("1m", justify="right")
    t.add_column("3m", justify="right")
    t.add_column("vs 200d", justify="right")
    t.add_column("Vol 20d", justify="right")
    t.add_column("From 52w high", justify="right")
    t.add_column("Listed in")
    ms = [(m, _metrics(hub.ohlcv, m)) for m in members]
    ms.sort(key=lambda kv: -(kv[1].get("ret_12_1_pct") or -1e9))
    for sym, m in ms:
        tags = tag_chips(hits.get(sym, []))
        t.add_row(sym, signed(m.get("ret_12_1_pct"), "{:+.0f}%"), signed(m.get("ret_21d_pct")), signed(m.get("ret_63d_pct")),
                  signed(m.get("vs_200d_pct")), _num(m.get("rv20_pct"), "{:.0f}%"), signed(m.get("from_52w_high_pct")), tags)
    head = Text()
    if row:
        head.append(f"{etf} {row.name}: Outlook #{row.rank} ", style="bold")
        head.append(row.posture, style=POSTURE_STYLE.get(row.posture, ""))
        head.append(f" · {row.pct_above_200d or 0:.0f}% of {row.n_members} stocks above their 200d", style="dim")
    body: List[Any] = [head, Text("Sorted by 12-1 month return (descriptive). 'Listed in' = the lenses in ③ that "
                                  "list the stock, coloured by their evidence label.", style="dim"), t, tag_legend(),
                       source_line(f"prices as of {hub.board.as_of}", "GICS membership (src/radar/scanners.py)")]
    return Panel(Group(*body), title=f"② SECTORS › {etf} {SECTOR_NAMES.get(etf, '')} — constituents",
                 title_align="left", border_style=GREEN)


# ── ④ EVIDENCE CARD ─────────────────────────────────────────────────────────


def _kv(t: Table, k: str, v: Any) -> None:
    t.add_row(Text(k, style="bold"), v if isinstance(v, Text) else Text(str(v)))


def card_renderable(card: EvidenceCard) -> Group:
    d = card.dossier
    col = VERDICT_COLOR.get(card.verdict_style, GREY)
    # 1. verdict
    v = Text()
    v.append(f" {card.verdict} ", style=f"bold #0e1117 on {col}")
    v.append(f"  {card.ticker}", style="bold")
    v.append(f"  ${d.spot:.2f}" if d.spot is not None else "  price n/a")
    if d.chg_1d_pct is not None:
        v.append("  1d ")
        v.append_text(signed(d.chg_1d_pct, "{:+.2f}%"))
    vb: List[Any] = [v]
    for reason in card.reasons:
        vb.append(Text("• " + reason))
    for w in card.warnings:
        vb.append(Text("⚠ " + w, style=AMBER))
    if card.lens_hits:
        lh = Text("Listed in: ", style="dim")
        for title, lab in card.lens_hits:
            lh.append_text(chip(lab))
            lh.append(f" {title}  ")
        vb.append(lh)
    verdict_panel = Panel(Group(*vb), title="④ EVIDENCE CARD · 1 Verdict", title_align="left", border_style=col)

    # 2. price structure
    ps = Table.grid(padding=(0, 2))
    ps.add_column(min_width=14)
    ps.add_column()
    trend = Text()
    for lab, val in (("vs 20d ", d.dist_sma20_pct), ("  50d ", d.dist_sma50_pct), ("  200d ", d.dist_sma200_pct)):
        trend.append(lab, style="dim")
        trend.append_text(signed(val))
    _kv(ps, "Trend", trend)
    mom = Text()
    mom.append("1m ", style="dim")
    mom.append_text(signed(d.chg_21d_pct))
    mom.append("  12-1m ", style="dim")
    mom.append_text(signed(d.mom_12_1_pct))
    mom.append(f"  RSI(2) {_num(d.rsi_2, '{:.0f}')}  RSI(14) {_num(d.rsi_14, '{:.0f}')}", style="dim")
    _kv(ps, "Momentum", mom)
    _kv(ps, "Supports", ", ".join(f"${s['level']:.2f} ({s['pct']*100:+.1f}%)" for s in d.supports) or "—")
    _kv(ps, "Resistances", ", ".join(f"${r['level']:.2f} ({r['pct']*100:+.1f}%)" for r in d.resistances) or "—")
    if card.sector is not None:
        sx = Text(f"{card.sector.etf} {card.sector.name} · Outlook #{card.sector.rank} ")
        sx.append(card.sector.posture, style=POSTURE_STYLE.get(card.sector.posture, ""))
        _kv(ps, "Sector", sx)
    if d.closes_60d:
        blocks = "▁▂▃▄▅▆▇█"
        lo, hi = min(d.closes_60d), max(d.closes_60d)
        span = max(hi - lo, 1e-6)
        _kv(ps, "60d tape", "".join(blocks[min(7, int((x - lo) / span * 7.99))] for x in d.closes_60d[-48:]))
    price_panel = Panel(ps, title="2 Price structure", title_align="left", border_style=GREY)

    # 3. peers
    pt = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    for c, j in (("Ticker", "left"), (f"Corr w/ {card.ticker}", "right"), ("Beta vs SPY", "right"), ("1m", "right"), ("3m", "right"),
                 ("12-1m", "right"), ("Vol 20d", "right"), ("vs 200d", "right"), ("IV/RV", "right")):
        pt.add_column(c, justify=j)  # type: ignore[arg-type]
    for pr in card.peers.rows:
        name = Text(("▶ " if pr.is_subject else "  ") + pr.ticker + ("" if pr.in_sector else " ·"),
                    style="bold" if pr.is_subject else "")
        ivrv = _num(pr.iv_rv, "{:.2f}") + (f" ({pr.iv_as_of[5:]})" if pr.iv_rv is not None and pr.iv_as_of else "")
        pt.add_row(name, _num(pr.corr_126d, "{:.2f}"), _num(pr.beta_126d, "{:.2f}"), signed(pr.ret_21d_pct),
                   signed(pr.ret_63d_pct), signed(pr.ret_12_1_pct, "{:+.0f}%"), _num(pr.rv20_pct, "{:.0f}%"),
                   signed(pr.vs_200d_pct), ivrv)
    standing = Text("Rank among these (1 = highest): ", style="dim")
    names = {"ret_21d_pct": "1m", "ret_63d_pct": "3m", "ret_12_1_pct": "12-1m", "rv20_pct": "vol", "vs_200d_pct": "vs 200d"}
    standing.append("  ".join(f"{names[k]} {v}" for k, v in card.peers.standing.items() if k in names) or "—")
    peer_parts: List[Any] = [Text(card.peers.method, style="dim"), pt, standing]
    if card.peers.note:
        peer_parts.append(Text(card.peers.note, style=AMBER))
    peers_panel = Panel(Group(*peer_parts),
                        title="3 Peers", title_align="left", border_style=GREY)

    # 4. options & probabilities
    ot = Table.grid(padding=(0, 2))
    ot.add_column(min_width=14)
    ot.add_column()
    iv = Text(_num(d.atm_iv, "{:.1f}%"))
    iv.append(f" vs realized 20d {_num(d.rv_20_ann, '{:.1f}%')}", style="dim")
    if d.iv_vs_rv_ratio is not None:
        iv.append(f"  → IV/RV {d.iv_vs_rv_ratio:.2f} ({'rich' if d.iv_vs_rv_ratio > 1.15 else ('cheap' if d.iv_vs_rv_ratio < 0.85 else 'fair')})")
    _kv(ot, "ATM IV (30d)", iv)
    if d.skew_25d_vp is not None:
        _kv(ot, "25Δ skew", f"{d.skew_25d_vp:+.1f} vol pts (put minus call)")
    _kv(ot, "1σ move (30d)", _num(d.implied_move_1s_pct, "±{:.1f}%"))
    odds = Text("—")
    if d.rnd_prob_up_5pct is not None and d.rnd_prob_dn_5pct is not None:
        odds = Text(f"P(>+5%) {d.rnd_prob_up_5pct*100:.0f}%  ·  P(<−5%) {d.rnd_prob_dn_5pct*100:.0f}%")
        odds.append(f"   [{card.prob_label}]", style=AMBER if card.prob_label.startswith("MODEL") else "dim")
    _kv(ot, "30d odds", odds)
    opt_panel = Panel(ot, title="4 Options & market-implied odds", title_align="left", border_style=GREY)

    # 5. events, revisions, smart money, news
    et = Table.grid(padding=(0, 2))
    et.add_column(min_width=14)
    et.add_column()
    earn = (f"in {d.days_to_earnings}d ({d.earnings_date})" if d.days_to_earnings is not None
            else "no upcoming date on file")
    _kv(et, "Earnings", earn)
    _kv(et, "Next macro", d.next_macro_event)
    if d.kalshi_macro:
        _kv(et, "Kalshi odds", " · ".join(f"{k.get('label')} {float(k.get('prob_pct') or 0):.0f}%" for k in d.kalshi_macro[:3]))
    rev = Text(_num(d.rev_breadth_30d, "{:+.2f}"))
    rev.append("  ")
    rev.append_text(chip("TESTING"))
    rev.append(" readout ≥ 2027-01-15", style="dim")
    _kv(et, "EPS revisions", rev)
    if d.analyst_actions:
        acts = "; ".join(
            f"{a['date']} {a['firm']} {ACTION_WORD.get(str(a['action']).lower(), a['action'])} {a.get('to_grade') or ''}"
            + (f" PT {a['prior_pt']:.0f}→{a['current_pt']:.0f}" if a.get("prior_pt") and a.get("current_pt") else "")
            for a in d.analyst_actions[:3]
        )
        _kv(et, "Analysts", Text(acts + "  (context — tested NULL)", style=""))
    if d.open_position_note:
        _kv(et, "Your exposure", Text(d.open_position_note, style=AMBER))
    for h in d.headlines[:3]:
        _kv(et, "News", Text(f"{h.get('date', '')} {h['title'][:90]}"))
    if d.cdr_ticker:
        _kv(et, "TFSA", f"Canadian CDR available: {d.cdr_ticker}")
    ev_panel = Panel(et, title="5 Events, revisions & news", title_align="left", border_style=GREY)

    # 6. sources
    st = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    st.add_column("Section", style="bold", no_wrap=True)
    st.add_column("Source", style="dim", no_wrap=True, overflow="ellipsis", max_width=40)
    st.add_column("As of")
    st.add_column("Age", justify="right")
    st.add_column("Note", style="dim")
    for p in card.provenance:
        age = Text("—", style="dim") if p.age_days is None else Text(
            f"{p.age_days}d", style=(AMBER if p.age_days > 5 else ""))
        st.add_row(p.section, p.source, p.as_of or "—", age, p.note)
    src_panel = Panel(st, title="7 Where every number came from", title_align="left", border_style=GREY)
    return Group(verdict_panel, price_panel, peers_panel, opt_panel, ev_panel, fit_panel(card), src_panel)


def fit_panel(card: EvidenceCard) -> Panel:
    """6 Portfolio fit: what this adds to what you already hold (descriptive, never a gate)."""
    fit = card.fit
    g = Table.grid(padding=(0, 2))
    g.add_column(min_width=14)
    g.add_column()
    if fit is None:
        _kv(g, "Holdings", Text("not checked", style="dim"))
    else:
        by_src: Dict[str, List[str]] = {}
        for h in fit.holdings:
            label = h.ticker + (f" {h.weight * 100:.0f}%" if h.weight else "")
            by_src.setdefault(h.source, []).append(label)
        _kv(g, "You hold", Text(" · ".join(f"{src}: {', '.join(v)}" for src, v in sorted(by_src.items()))
                                 or "nothing recorded", style="" if by_src else "dim"))
        for w in fit.warnings:
            _kv(g, "Overlap", Text("⚠ " + w, style=AMBER))
        for n in fit.notes:
            _kv(g, "Note", n)
        if fit.correlations:
            _kv(g, "Co-movement", ", ".join(f"{tk} {r:+.2f}" for tk, r in fit.correlations[:5]))
        if fit.diversifiers:
            _kv(g, "Least correlated", Text(", ".join(f"{tk} {r:+.2f}" for tk, r in fit.diversifiers)
                                            + "  (sleeve-universe ETFs; information, not a recommendation)"))
        if not fit.warnings and by_src:
            _kv(g, "Verdict", Text("no concentration flag", style="dim"))
        _kv(g, "Method", Text(fit.method, style="dim"))
    return Panel(g, title="6 Portfolio fit", title_align="left", border_style=GREY)


def hub_overview(hub: DiscoveryHub) -> Group:
    return Group(breadcrumb(1), freshness_line(hub), market_panel(hub), board_panel(hub), ideas_overview(hub))


__all__ = [
    "LABEL_STYLE", "LABEL_MEANING", "STEP_TITLES", "chip", "signed", "legend", "breadcrumb",
    "market_panel", "board_table", "board_panel", "lens_header", "lens_table", "lens_panel",
    "ideas_overview", "null_ledger_panel", "tag_chips", "tag_legend", "freshness_line", "sector_members_panel", "card_renderable", "hub_overview",
    "fit_panel", "forward_panel",
]
