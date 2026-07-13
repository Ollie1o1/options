"""Pure HTML renderer over the morning-briefing sidecar, composed on the desk
kit. No I/O, no network: render(data) is deterministic so `--from sidecar`
rebuilds identical bytes.

Layout follows desk-dashboard convention: KPI cards on top (label → value →
delta), an auto-generated "what matters" callout, a dense two-column card grid
for the analytical zones, and secondary detail (portfolio, job table) folded
into <details> so it costs a click instead of screen space.
"""
import html as _html

from src.desk_kit import charts as CH
from src.desk_kit import shell
from src.desk_kit.theme import heat_inks

_ZONE_TITLES = {
    "health": "Automation Health", "market": "Market State",
    "vol": "Vol Intelligence", "macro_events": "Macro & Events",
    "signals": "Signals", "portfolio": "Portfolio",
    "gate": "Gate & Evidence", "notes": "Desk Notes",
}

# Morning-only styling on top of the kit sheet. The zone markup keeps its
# .good/.bad tone classes (aliases of the kit's .g/.b) because the takeaway
# and table builders write prose-level HTML, not kit components.
_CSS = """
h1{font-size:20px;color:var(--ink-strong);margin:0;
   font-family:ui-monospace,'SF Mono',Menlo,monospace;}
h2{font-size:10px;letter-spacing:.16em;text-transform:uppercase;
   color:var(--muted);border-bottom:1px solid var(--rule);font-weight:700;
   padding-bottom:5px;margin:0 0 10px;}
h3{font-size:11.5px;color:var(--ink-strong);margin:12px 0 5px;
   text-transform:uppercase;letter-spacing:.06em;}
.meta{color:var(--muted);font-size:12px;margin-top:3px;}
.good{color:var(--good);} .bad{color:var(--bad);} .warn{color:var(--warn);}
.muted{color:var(--muted);} .accent{color:var(--accent);}
.pill{display:inline-block;padding:0 8px;border-radius:9px;font-size:11px;
      border:1px solid var(--rule-hard);white-space:nowrap;}
.chip2{display:inline-block;padding:1px 8px;border-radius:3px;font-size:11px;
      margin-right:6px;border:1px solid var(--rule);}
.chip2.ok{color:var(--good);border-color:var(--chip-ok-bd);background:var(--chip-ok-bg);}
.chip2.stale{color:var(--warn);border-color:var(--chip-wn-bd);background:var(--chip-wn-bg);}
.chip2.crit{color:var(--bad);border-color:var(--chip-bad-bd);background:var(--chip-bad-bg);}
.placeholder{color:var(--muted);font-style:italic;padding:6px 0;}
.note{margin:2px 0;color:var(--muted);}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
      gap:10px;margin:14px 0 0;}
.kpis .kpi{background:var(--panel);border:1px solid var(--rule);
     border-radius:5px;padding:10px 12px;}
.kpi .l{font-size:9px;letter-spacing:.14em;text-transform:uppercase;
        color:var(--muted);}
.kpi .v{font-size:21px;font-weight:650;color:var(--ink-strong);margin:2px 0;}
.kpi .d{font-size:11.5px;color:var(--muted);}
.callout{background:var(--panel);border:1px solid var(--rule);
         border-left:3px solid var(--accent);
         border-radius:0 5px 5px 0;padding:10px 14px;margin:12px 0 0;}
.callout p{margin:3px 0;font-size:12.5px;}
.zone{margin-bottom:0;}
td.heat{background:var(--hl);border-radius:2px;padding-left:4px;}
[data-theme="dark"] td.heat{background:var(--hd);}
.bars text{font:10.5px ui-monospace,Menlo,monospace;fill:var(--muted);}
.bars .val{fill:var(--ink);font-variant-numeric:tabular-nums;}
"""


def _esc(v) -> str:
    return _html.escape("" if v is None else str(v))


def _num(v, fmt="{:,.2f}", dash="—"):
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return dash


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _sign_cls(v):
    f = _f(v)
    if f is None:
        return "muted"
    return "good" if f > 0 else ("bad" if f < 0 else "muted")


def _placeholder(pid) -> str:
    return (f"<div class='placeholder'>{_esc(_ZONE_TITLES.get(pid, pid))} "
            f"unavailable this morning — see Desk Notes.</div>")


def _heat_td(value, span, text) -> str:
    hl, hd = heat_inks(value, span)
    return (f"<td class='heat' style='--hl:{hl};--hd:{hd}'>"
            f"<span class='{_sign_cls(value)}'>{text}</span></td>")


def _table(headers, rows) -> str:
    """rows: list of cells; each cell is raw-HTML `<td>...` string OR
    (content, css_class) tuple rendered into a td."""
    th = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    trs = []
    for row in rows:
        tds = []
        for cell in row:
            if isinstance(cell, tuple):
                content, cls = cell
                tds.append(f"<td class='{cls}'>{content}</td>")
            else:
                tds.append(cell)
        trs.append(f"<tr>{''.join(tds)}</tr>")
    return f"<table><tr>{th}</tr>{''.join(trs)}</table>"


# ── SVG bar helpers (pure, deterministic) ────────────────────────────────────

def _svg_signed_bars(rows, w=430, bar_h=13, gap=5, label_w=56, val_w=62):
    """Horizontal signed bars: rows = [(label, value, display)]. Bars extend
    from a center axis; positive fills good, negative bad."""
    vals = [abs(v) for _, v, _ in rows if _f(v) is not None] or [1.0]
    span = max(vals) or 1.0
    plot_w = w - label_w - val_w
    cx = label_w + plot_w / 2.0
    h = len(rows) * (bar_h + gap) + 2
    out = [f"<svg class='bars' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"]
    out.append(f"<line x1='{cx:.1f}' y1='0' x2='{cx:.1f}' y2='{h}' "
               "stroke='var(--rule)' stroke-width='1'/>")
    y = 1
    for label, value, disp in rows:
        v = _f(value) or 0.0
        bw = abs(v) / span * (plot_w / 2.0 - 2)
        x = cx if v >= 0 else cx - bw
        fill = "var(--good)" if v > 0 else ("var(--bad)" if v < 0 else "var(--muted)")
        out.append(f"<text x='{label_w - 4}' y='{y + bar_h - 3}' "
                   f"text-anchor='end'>{_esc(label)}</text>")
        out.append(f"<rect x='{x:.1f}' y='{y}' width='{max(bw, 1):.1f}' "
                   f"height='{bar_h}' rx='2' fill='{fill}' opacity='0.85'/>")
        out.append(f"<text class='val' x='{w - 2}' y='{y + bar_h - 3}' "
                   f"text-anchor='end'>{_esc(disp)}</text>")
        y += bar_h + gap
    out.append("</svg>")
    return "".join(out)


def _svg_paired_bars(rows, w=430, bar_h=9, gap=11, label_w=56, val_w=88):
    """Paired comparison bars: rows = [(label, a, b, display)] where a is the
    primary (accent) and b the reference (muted) — e.g. IV vs RV."""
    vals = [x for _, a, b, _ in rows for x in (_f(a), _f(b)) if x is not None] or [1.0]
    span = max(vals) or 1.0
    plot_w = w - label_w - val_w
    row_h = bar_h * 2 + 3
    h = len(rows) * (row_h + gap) + 2
    out = [f"<svg class='bars' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"]
    y = 1
    for label, a, b, disp in rows:
        av, bv = _f(a) or 0.0, _f(b) or 0.0
        out.append(f"<text x='{label_w - 4}' y='{y + bar_h + 3}' "
                   f"text-anchor='end'>{_esc(label)}</text>")
        out.append(f"<rect x='{label_w}' y='{y}' width='{max(av / span * plot_w, 1):.1f}' "
                   f"height='{bar_h}' rx='2' fill='var(--accent)' opacity='0.9'/>")
        out.append(f"<rect x='{label_w}' y='{y + bar_h + 3}' "
                   f"width='{max(bv / span * plot_w, 1):.1f}' height='{bar_h}' rx='2' "
                   "fill='var(--muted)' opacity='0.55'/>")
        out.append(f"<text class='val' x='{w - 2}' y='{y + bar_h + 3}' "
                   f"text-anchor='end'>{_esc(disp)}</text>")
        y += row_h + gap
    out.append("</svg>")
    return "".join(out)


# ── "What matters this morning" ──────────────────────────────────────────────

def _takeaways(panels) -> list:
    """Auto-generated highlights, most actionable first. Pure over panels."""
    out = []
    p = panels.get("portfolio") or {}
    for w in (p.get("exits_due") or []):
        out.append(f"<b>Exit due:</b> <span class='warn'>{_esc(w)}</span>")
    h = panels.get("health") or {}
    if h.get("worst") not in (None, "OK"):
        stale = [j["name"] for j in h.get("jobs", [])
                 if j.get("severity") not in (None, "OK")]
        out.append(f"<b>Automation {_esc(h['worst'])}:</b> "
                   f"{_esc(', '.join(stale))} behind — data below may be stale.")
    v = panels.get("vol") or {}
    movers = v.get("movers") or []
    if movers:
        m = movers[0]
        out.append(f"<b>Biggest IV move:</b> {_esc(m.get('symbol'))} "
                   f"<span class='{_sign_cls(m.get('d_iv'))}'>"
                   f"{_num(m.get('d_iv'), '{:+.1%}')}</span> ATM vs prior snapshot.")
    vrp = sorted((r for r in (v.get("vrp") or []) if _f(r.get("vrp")) is not None),
                 key=lambda r: -abs(_f(r.get("vrp")) or 0.0))
    if vrp:
        r = vrp[0]
        out.append(f"<b>Widest VRP:</b> {_esc(r.get('symbol'))} "
                   f"{_num(r.get('vrp'), '{:+.1f}vp')} ({_esc(r.get('label'))}) — "
                   f"IV {_num(r.get('iv'), '{:.0%}')} vs RV {_num(r.get('rv'), '{:.0%}')}.")
    me = panels.get("macro_events") or {}
    cal = me.get("calendar") or []
    if cal:
        e = cal[0]
        out.append(f"<b>Next macro:</b> {_esc(e.get('name'))} on {_esc(e.get('date'))}.")
    for e in (me.get("earnings") or [])[:2]:
        out.append(f"<b>Earnings:</b> {_esc(e.get('sym'))} reports {_esc(e.get('date'))} "
                   "— IV crush risk on open premium.")
    s = panels.get("signals") or {}
    for i in (s.get("insider") or [])[:1]:
        out.append(f"<b>Insider:</b> {_esc(i.get('sym'))} {_esc(i.get('summary'))}.")
    g = panels.get("gate") or {}
    if g:
        out.append(f"<b>Gate:</b> {_esc(g.get('gate_decision', '?'))} at "
                   f"{_esc(g.get('cohort_n') or 0)}/{_esc(g.get('target_n') or 50)} "
                   "cohort trades — <b>real money stays OFF</b>.")
    return out


# ── KPI strip ────────────────────────────────────────────────────────────────

def _kpi(label, value, delta="", tone="") -> str:
    v_cls = f" {tone}" if tone else ""
    return (f"<div class='kpi'><div class='l'>{_esc(label)}</div>"
            f"<div class='v{v_cls}'>{value}</div>"
            f"<div class='d'>{delta}</div></div>")


def _kpi_strip(panels) -> str:
    m = panels.get("market") or {}
    r = m.get("regime") or {}
    cards = []
    vix = _f(r.get("vix"))
    vix_tone = "" if vix is None else ("good" if vix < 17 else
                                       "warn" if vix < 25 else "bad")
    cards.append(_kpi("VIX", _num(r.get("vix"), "{:.1f}"),
                      f"3M {_num(r.get('vix_3m'), '{:.1f}')} · "
                      f"{_esc(r.get('vix_term_structure', '?'))}", vix_tone))
    spy = next((ix for ix in m.get("indexes", []) if ix.get("sym") == "SPY"), None)
    if spy:
        cards.append(_kpi("SPY 1D",
                          f"<span class='{_sign_cls(spy.get('chg_1d_pct'))}'>"
                          f"{_num(spy.get('chg_1d_pct'), '{:+.2f}%')}</span>",
                          f"5d {_num(spy.get('chg_5d_pct'), '{:+.2f}%')} · "
                          f"last {_num(spy.get('last'))}"))
    cards.append(_kpi("POSTURE", _esc(r.get("posture", "?")),
                      f"PCR {_num(r.get('options_pcr'))}"))
    iv_prem = r.get("iv_premium")
    if iv_prem is not None:
        cards.append(_kpi("IV PREMIUM", _num(iv_prem, "{:+.0%}"),
                          f"SPY HV30 {_num(r.get('spy_hv_30'), '{:.1f}%')}",
                          "warn" if (_f(iv_prem) or 0) > 0.2 else ""))
    rates = m.get("rates") or {}
    cards.append(_kpi("10Y–3M", f"<span class='{_sign_cls(rates.get('slope'))}'>"
                                f"{_num(rates.get('slope'), '{:+.2f}pp')}</span>",
                      f"10Y {_num(rates.get('t10y'))}% · 3M {_num(rates.get('t3m'))}%"))
    g = panels.get("gate") or {}
    n, target = g.get("cohort_n") or 0, g.get("target_n") or 50
    cards.append(_kpi("GATE", f"{_esc(n)}/{_esc(target)}",
                      _esc(g.get("gate_decision", "?")),
                      "warn" if n < target else "good"))
    return "<div class='kpis'>" + "".join(cards) + "</div>"


# ── Zones ────────────────────────────────────────────────────────────────────

def _zone_health(p) -> str:
    worst = p.get("worst", "?")
    cls = {"OK": "ok", "STALE": "stale"}.get(worst, "crit")
    chips = []
    for j in p.get("jobs", []):
        jcls = {"OK": "ok", "STALE": "stale"}.get(j.get("severity"), "crit")
        chips.append(f"<span class='chip2 {jcls}'>{_esc(j.get('name'))}</span>")
    rows = [[(_esc(j.get("name")), ""), (_esc(j.get("cadence")), "muted"),
             (_esc(j.get("last_run") or "never"), "muted"),
             (_esc(j.get("stale_days")), ""),
             (_esc(j.get("severity")),
              {"OK": "good", "STALE": "warn"}.get(j.get("severity"), "bad"))]
            for j in p.get("jobs", [])]
    return (f"<p>Jobs <span class='pill {cls}'>{_esc(worst)}</span> &nbsp;"
            + "".join(chips) + "</p>"
            "<details><summary>Job detail</summary>"
            + _table(("job", "cadence", "last run", "stale (bd)", "status"), rows)
            + "</details>")


def _zone_market(p) -> str:
    r = p.get("regime") or {}
    parts = [f"<p class='muted'>{_esc(r.get('posture_rationale', ''))}</p>"]
    rows = []
    for ix in p.get("indexes", []):
        closes = [c for c in (ix.get("closes") or []) if c is not None]
        chart = (f"<span class='chart'>{CH.sparkline(closes)}</span>"
                 if len(closes) >= 2 else "")
        rows.append([(f"<b>{_esc(ix.get('sym'))}</b>", ""),
                     (_num(ix.get("last")), ""),
                     _heat_td(ix.get("chg_1d_pct"), 2.0,
                              _num(ix.get("chg_1d_pct"), "{:+.2f}%")),
                     _heat_td(ix.get("chg_5d_pct"), 4.0,
                              _num(ix.get("chg_5d_pct"), "{:+.2f}%")),
                     (chart, "")])
    if rows:
        parts.append(_table(("index", "last", "1d", "5d", "30d trend"), rows))
    return "".join(parts)


def _zone_vol(p) -> str:
    parts = []
    movers = p.get("movers") or []
    if movers:
        bars = [(m.get("symbol", "?"),
                 _f(m.get("d_iv")),
                 f"{_num(m.get('d_iv'), '{:+.1%}')} → {_num(m.get('iv'), '{:.0%}')}")
                for m in movers[:8]]
        parts.append("<h3>ΔATM-IV vs prior snapshot</h3>" + _svg_signed_bars(bars))
    vrp = p.get("vrp") or []
    if vrp:
        pairs = [(v.get("symbol", "?"), _f(v.get("iv")), _f(v.get("rv")),
                  f"{_num(v.get('vrp'), '{:+.1f}vp')} {v.get('label', '')}")
                 for v in vrp[:6]]
        parts.append("<h3>Implied (amber) vs realized (grey)</h3>"
                     + _svg_paired_bars(pairs))
        rich = [v for v in vrp if v.get("label") == "RICH"]
        cheap = [v for v in vrp if v.get("label") == "CHEAP"]
        if rich or cheap:
            bits = []
            if rich:
                bits.append("<b>sell-vol candidates:</b> "
                            + ", ".join(_esc(v.get("symbol")) for v in rich))
            if cheap:
                bits.append("<b>buy-vol candidates:</b> "
                            + ", ".join(_esc(v.get("symbol")) for v in cheap))
            parts.append("<p>" + " · ".join(bits) + "</p>")
    parts.append(f"<p class='muted'>Coverage: {_esc(p.get('n_cov'))} symbols · "
                 f"{_esc(p.get('crypto_note'))}</p>")
    return "".join(parts)


def _zone_macro(p) -> str:
    parts = []
    cal = p.get("calendar") or []
    if cal:
        rows = [[(_esc(e.get("date")), "muted"),
                 (f"<b>{_esc(e.get('name'))}</b>" if i == 0 else _esc(e.get("name")), "")]
                for i, e in enumerate(cal)]
        parts.append(_table(("date", "event"), rows))
    if p.get("pulse"):
        parts.append(f"<p>{_esc(p['pulse'])}</p>")
    for h in (p.get("headlines") or [])[:5]:
        parts.append(f"<p class='note'>• {_esc(h)}</p>")
    ern = p.get("earnings") or []
    if ern:
        rows = [[(f"<b>{_esc(e.get('sym'))}</b>", ""), (_esc(e.get("date")), "muted")]
                for e in ern]
        parts.append("<h3>Watchlist earnings (7d)</h3>" + _table(("symbol", "date"), rows))
    return "".join(parts) or "<p class='muted'>No macro events in window.</p>"


def _zone_signals(p) -> str:
    parts = []
    uoa = p.get("uoa") or []
    if uoa:
        rows = [[(f"<b>{_esc(u.get('symbol'))}</b>", ""), (_num(u.get("score")), ""),
                 _heat_td((_f(u.get("net_call_share")) or 0.5) - 0.5, 0.5,
                          _num(u.get("net_call_share"), "{:.0%}")),
                 (_esc(u.get("n_unusual")), "muted")] for u in uoa]
        parts.append("<h3>Unusual options activity (OI deltas)</h3>" +
                     _table(("symbol", "score", "call share", "unusual strikes"), rows))
    for i in p.get("insider") or []:
        parts.append(f"<p class='note'>• Insider <b>{_esc(i.get('sym'))}</b>: "
                     f"{_esc(i.get('summary'))}</p>")
    ol = p.get("outlook") or {}
    if ol.get("top") or ol.get("bottom"):
        parts.append("<h3>Sector outlook (1–3mo, relative)</h3>")
        for label, rows, cls in (("Top", ol.get("top") or [], "good"),
                                 ("Bottom", ol.get("bottom") or [], "bad")):
            if rows:
                names = ", ".join(_esc(r.get("ticker") or r.get("symbol")) for r in rows)
                parts.append(f"<p class='note'>{label}: "
                             f"<span class='{cls}'>{names}</span></p>")
        if ol.get("as_of"):
            parts.append(f"<p class='muted'>as of {_esc(ol['as_of'])}</p>")
    return "".join(parts) or "<p class='muted'>No active signals this morning.</p>"


def _zone_portfolio(p) -> str:
    """Collapsed by default: the book is secondary to the morning read."""
    n = p.get("n_open") or 0
    pnls = [_f(x.get("pnl_usd")) for x in p.get("positions", [])]
    pnls = [x for x in pnls if x is not None]
    tot = sum(pnls) if pnls else None
    summary = f"Portfolio — {n} open position(s)"
    if tot is not None:
        summary += (f" · P&amp;L <span class='{_sign_cls(tot)}'>"
                    f"{_num(tot, '{:+,.0f}')}</span>")
    n_exits = len(p.get("exits_due") or [])
    if n_exits:
        summary += f" · <span class='warn'>⚠ {n_exits} exit(s) due</span>"
    if not n:
        return "<details><summary>Portfolio — no open paper positions</summary></details>"
    rows = []
    for x in p.get("positions", []):
        pnl = x.get("pnl_pct", x.get("pnl_percent"))
        rows.append([(f"<b>{_esc(x.get('ticker'))}</b>", ""),
                     (_esc(x.get("strategy")), "muted"),
                     (_num(x.get("dte"), "{:.0f}"), ""),
                     _heat_td(pnl, 40.0, _num(pnl, "{:+.1f}%")),
                     (_num(x.get("delta")), "muted")])
    body = [_table(("ticker", "strategy", "DTE", "P&L", "Δ"), rows)]
    g = p.get("net_greeks") or {}
    gnum = {k: v for k, v in g.items() if isinstance(v, (int, float))}
    if gnum:
        body.append("<p class='muted'>Net book: " +
                    " · ".join(f"{_esc(k.replace('portfolio_', ''))} "
                               f"{_num(v, '{:+,.2f}')}"
                               for k, v in sorted(gnum.items())) + "</p>")
    for w in p.get("exits_due", []):
        body.append(f"<p class='warn'>⚠ {_esc(w)}</p>")
    for w in p.get("guard", []):
        body.append(f"<p class='warn'>⚠ {_esc(w)}</p>")
    return f"<details><summary>{summary}</summary>{''.join(body)}</details>"


def _zone_gate(p) -> str:
    n, target = p.get("cohort_n") or 0, p.get("target_n") or 50
    ic, pv = p.get("pooled_ic"), p.get("p_value")
    ic_read = "no demonstrated edge yet"
    if _f(ic) is not None and _f(pv) is not None:
        ic_read = ("statistically supported" if abs(_f(ic)) >= 0.05 and _f(pv) < 0.05
                   else "not statistically distinguishable from zero")
    return (f"<p>Gate: <span class='pill'>{_esc(p.get('gate_decision', '?'))}</span>"
            f" · cohort <b>{n}/{target}</b></p>{CH.meter(n, target)}"
            f"<p class='muted'>Walk-forward pooled IC "
            f"<b>{_num(ic, '{:+.2f}')}</b> (p={_num(pv)}, "
            f"n_oos={_esc(p.get('n_oos'))}) — {ic_read} · "
            f"as of {_esc(p.get('as_of') or '—')}</p>"
            f"<p><b>Real money is OFF</b> until the gate fires.</p>")


def _zone_notes(p) -> str:
    return "".join(f"<p class='note'>• {_esc(n)}</p>" for n in (p or []))


def _zone(pid, panel, fn, span=6, anchor=None) -> str:
    inner = _placeholder(pid) if panel is None else fn(panel)
    aid = f" id='{anchor}'" if anchor else ""
    return (f"<section class='zone card c{span}'{aid}>"
            f"<h2>{_esc(_ZONE_TITLES[pid])}</h2>{inner}</section>")


def _masthead(meta):
    chip = '<span class="chip">{}</span>'.format(_esc(meta.get("date")))
    kit_meta = shell.chipline([
        ("session", _esc(meta.get("session"))),
        ("generated", _esc(meta.get("generated_at"))),
    ])
    nav = shell.anchor_nav([("matters", "What matters"), ("market", "Market"),
                            ("vol", "Vol"), ("signals", "Signals"),
                            ("macro", "Macro"), ("gate", "Gate"),
                            ("book", "Portfolio")])
    return shell.masthead("MORNING", chip, meta_html=kit_meta, nav_html=nav,
                          where="briefings")


def render(data: dict) -> str:
    meta = data.get("meta", {})
    panels = data.get("panels", {})

    body = [f"<h1 style='margin-top:12px'>{_esc(meta.get('title', 'Morning Briefing'))}</h1>",
            f"<p class='meta'>{_esc(meta.get('date'))} · session: "
            f"<b>{_esc(meta.get('session'))}</b> · generated "
            f"{_esc(meta.get('generated_at'))}</p>"]

    # KPI strip + auto-highlights read the whole sidecar, not one panel.
    body.append(_kpi_strip(panels))
    takeaways = _takeaways(panels)
    if takeaways:
        body.append("<div class='callout' id='matters'>"
                    "<h2>What matters this morning</h2>"
                    + "".join(f"<p>{t}</p>" for t in takeaways) + "</div>")

    body.append("<div class='grid'>"
                + _zone("market", panels.get("market"), _zone_market,
                        anchor="market")
                + _zone("signals", panels.get("signals"), _zone_signals,
                        anchor="signals")
                + _zone("vol", panels.get("vol"), _zone_vol, anchor="vol")
                + _zone("macro_events", panels.get("macro_events"), _zone_macro,
                        anchor="macro")
                + _zone("gate", panels.get("gate"), _zone_gate, span=12,
                        anchor="gate")
                + "</div>")

    # Secondary detail: collapsed book, compact health, standing notes.
    pf = panels.get("portfolio")
    body.append("<div class='zone' id='book' style='margin-top:12px'>"
                "<h2>Portfolio</h2>"
                + (_placeholder("portfolio") if pf is None else _zone_portfolio(pf))
                + "</div>")
    body.append("<div class='grid'>"
                + _zone("health", panels.get("health"), _zone_health)
                + _zone("notes", panels.get("notes"), _zone_notes)
                + "</div>")

    body.append("<div class='foot'>"
                f"{_esc(meta.get('sidecar', ''))} · regenerate: "
                f"python -m src.morning --from {_esc(meta.get('sidecar', ''))}</div>")
    return shell.page(meta.get("title", "Morning Briefing"), _masthead(meta),
                      "".join(body), extra_css=_CSS)
