"""Pure HTML render over the research-desk sidecar, composed on the desk kit.
Zero network, zero data-fetching imports: render(json.load(sidecar)) must
reproduce the page."""
import html as _html

from src.desk_kit import charts as CH
from src.desk_kit import shell


def _esc(v):
    return _html.escape("" if v is None else str(v), quote=True)


def _num(v, fmt="{:,.2f}", dash="—"):
    try:
        f = float(v)
        if f != f:
            return dash
        return fmt.format(f)
    except (TypeError, ValueError):
        return dash


def _panel(data, pid):
    p = (data.get("panels") or {}).get(pid)
    return p if p else None


def _fail_reason(data, pid):
    for f in data.get("failures") or []:
        if f.startswith(pid + ":"):
            return f.split(":", 1)[1].strip()
    return "not collected"


def _ph(data, pid, label):
    return ('<div class="ph">{l} unavailable &mdash; {r}</div>'
            .format(l=_esc(label), r=_esc(_fail_reason(data, pid))))


_card = shell.card
_TONES = {"good": " g", "bad": " b", "warn": " w", "": ""}


def _kpi(label, value, sub="", tone=""):
    return shell.kpi(label, value, sub=sub, tone=tone, span=3)


def _chg_tone(v):
    try:
        return "good" if float(v) >= 0 else "bad"
    except (TypeError, ValueError):
        return ""


# ── Masthead + staleness banner ──────────────────────────────────────────────

def _stale_banner(data):
    h = _panel(data, "health")
    if not h or h.get("worst") in (None, "OK"):
        return ""
    stale = [j for j in h.get("jobs", [])
             if j.get("severity") not in (None, "OK")]
    if not stale:
        return ""
    items = ", ".join("{} ({}, {}d stale)".format(
        _esc(j.get("name")), _esc(j.get("last_run")), _esc(j.get("stale_days")))
        for j in stale)
    return ('<div class="stale">Automation staleness: {} &mdash; data below may '
            "be older than it looks.</div>".format(items))


_TAB_ORDER = (("market", "Market"), ("volatility", "Volatility"),
              ("macro", "Macro &amp; News"), ("ticker", "Ticker"))


def _tabs_present(data):
    out = []
    for tid, label in _TAB_ORDER:
        if tid == "ticker" and not _panel(data, "ticker"):
            continue
        out.append((tid, label))
    return out


def _masthead(data):
    m = data.get("meta", {})
    sym = m.get("symbol")
    chip = ('<span class="chip">{}</span>'.format(_esc(sym))) if sym else ""
    meta = shell.chipline([("generated", _esc(m.get("generated_at")))])
    nav = '<nav class="tabbar" data-tabgroup>' + "".join(
        '<button class="tabbtn" data-tab="{t}">{n}&nbsp;{l}</button>'.format(
            t=tid, n=i + 1, l=label)
        for i, (tid, label) in enumerate(_tabs_present(data))) + "</nav>"
    return shell.masthead("RESEARCH", chip, meta_html=meta, nav_html=nav,
                          where="research")


# ── Market tab ───────────────────────────────────────────────────────────────

def _vix_regime_label(reg):
    """Prefer the collected label; else derive with the repo's 15/25 cuts
    (mirrors data_fetching.determine_vix_regime defaults)."""
    if reg.get("vix_regime"):
        return str(reg["vix_regime"])
    vix = reg.get("vix")
    if not isinstance(vix, (int, float)):
        return "?"
    return "low" if vix < 15 else ("high" if vix > 25 else "normal")


def _market_kpis(data):
    mkt = _panel(data, "market") or {}
    reg = mkt.get("regime") or {}
    rates = mkt.get("rates") or {}
    vix = reg.get("vix")
    vix_sub = "{} · {}".format(_esc(_vix_regime_label(reg)),
                               _esc(reg.get("vix_term_structure", "?")))
    posture = _esc(str(reg.get("posture", "—")).replace("_", " "))
    slope = rates.get("slope")
    slope_tone = "bad" if (isinstance(slope, (int, float)) and slope < 0) else ""
    slope_sub = ("inverted curve" if slope_tone == "bad"
                 else "10Y {} · 3M {}".format(_num(rates.get("t10y"), "{:.2f}%"),
                                              _num(rates.get("t3m"), "{:.2f}%")))
    return ('<div class="grid">'
            + _kpi("VIX", _num(vix, "{:.1f}"), vix_sub)
            + _kpi("Put / Call", _num(reg.get("options_pcr"), "{:.2f}"),
                   "equity options")
            + _kpi("Posture", posture, _esc(reg.get("posture_rationale", "")))
            + _kpi("10Y − 3M", _num(slope, "{:+.2f}%"), slope_sub,
                   tone=slope_tone)
            + "</div>")


def _index_table(mkt):
    rows = mkt.get("indexes") or []
    if not rows:
        return '<div class="ph">no index data</div>'
    out = ["<table><tr><th></th><th>last</th><th>1d</th><th>5d</th></tr>"]
    for r in rows:
        out.append(
            '<tr><td class="m">{s}</td><td class="n">{l}</td>'
            '<td class="n {t1}">{c1}</td><td class="n {t5}">{c5}</td></tr>'.format(
                s=_esc(r.get("sym")), l=_num(r.get("last")),
                t1="g" if (r.get("chg_1d_pct") or 0) >= 0 else "b",
                c1=_num(r.get("chg_1d_pct"), "{:+.2f}%"),
                t5="g" if (r.get("chg_5d_pct") or 0) >= 0 else "b",
                c5=_num(r.get("chg_5d_pct"), "{:+.2f}%")))
    out.append("</table>")
    return "".join(out)


def _calendar_list(data):
    cal = _panel(data, "calendar")
    if not cal:
        return _ph(data, "calendar", "Macro calendar")
    return "".join('<div class="evt"><span class="m">{d}</span> {n}</div>'.format(
        d=_esc(e.get("date")), n=_esc(e.get("name"))) for e in cal)


def _tab_market(data):
    parts = [_market_kpis(data)]
    tape = _panel(data, "tape")
    grid = ['<div class="grid">']
    if tape:
        spy, vix = tape.get("spy") or {}, tape.get("vix") or {}
        spy_head = ('<div class="chead"><span class="m big">{0}</span>'
                    '<span class="m {t}">{c} 1d</span></div>').format(
            _num(spy.get("last")),
            t=_chg_tone(spy.get("chg_1d_pct"))[:1] or "mut",
            c=_num(spy.get("chg_1d_pct"), "{:+.2f}%"))
        grid.append(_card("SPY — 6 months",
                          spy_head + CH.area_chart(spy.get("closes"),
                                                   spy.get("dates"), "spy"),
                          span=8))
        # w=380: this card spans 4 of 12 columns, so a narrower viewBox keeps
        # axis text readable instead of scaling a 760-wide chart down by half.
        grid.append(_card("VIX — 6 months",
                          CH.area_chart(vix.get("closes"), vix.get("dates"),
                                        "vix", w=380, fmt="{:.1f}"), span=4))
    else:
        grid.append(_card("Index tape", _ph(data, "tape", "Index history"),
                          span=12))
    movers = _panel(data, "movers")
    if movers:
        rows = [(r.get("sym"), r.get("ret_5d_pct")) for r in movers]
        grid.append(_card("Movers — 5 day", CH.hbar_diverging(rows, unit="%"),
                          span=6))
    else:
        grid.append(_card("Movers — 5 day", _ph(data, "movers", "Movers"),
                          span=6))
    mkt = _panel(data, "market") or {}
    grid.append(_card("Index direction", _index_table(mkt), span=3))
    grid.append(_card("Macro calendar", _calendar_list(data), span=3))
    grid.append("</div>")
    return "".join(parts + grid)


# ── Volatility tab ───────────────────────────────────────────────────────────

def _vrp_chip(label):
    cls = {"RICH": "k-ok", "CHEAP": "k-bad"}.get(str(label).upper(), "k-warn")
    return '<span class="badge {c}">{l}</span>'.format(c=cls, l=_esc(label))


def _tab_vol(data):
    vol = _panel(data, "vol")
    grid = ['<div class="grid">']
    if not vol:
        grid.append(_card("Volatility intelligence",
                          _ph(data, "vol", "Vol archive"), span=12))
    else:
        movers = [(m.get("symbol"), (m.get("d_iv") or 0) * 100.0)
                  for m in vol.get("movers") or []
                  if m.get("d_iv") is not None]
        grid.append(_card("IV movers — day-over-day",
                          CH.hbar_diverging(movers, unit="vp") or
                          '<div class="ph">no movers today</div>', span=6))
        vrp_rows = vol.get("vrp") or []
        bars = CH.hbar_diverging([(r.get("symbol"), r.get("vrp"))
                                  for r in vrp_rows], unit="vp")
        table = ["<table><tr><th></th><th>IV</th><th>RV</th><th>VRP</th>"
                 "<th></th></tr>"]
        for r in vrp_rows:
            table.append(
                '<tr><td class="m">{s}</td><td class="n">{iv}</td>'
                '<td class="n">{rv}</td><td class="n {t}">{v}</td>'
                "<td>{chip}</td></tr>".format(
                    s=_esc(r.get("symbol")), iv=_num(r.get("iv"), "{:.0%}"),
                    rv=_num(r.get("rv"), "{:.0%}"),
                    t="g" if (r.get("vrp") or 0) >= 0 else "b",
                    v=_num(r.get("vrp"), "{:+.1f}vp"),
                    chip=_vrp_chip(r.get("label"))))
        table.append("</table>")
        grid.append(_card("Variance risk premium — implied vs realized",
                          bars + "".join(table), span=6))
        note = ('<div class="evt">{}</div><div class="evt mut">coverage: {} '
                "symbols in the chain archive</div>").format(
                    _esc(vol.get("crypto_note", "")), _esc(vol.get("n_cov", 0)))
        grid.append(_card("Notes", note, span=6))
    tape = _panel(data, "tape")
    if tape and tape.get("vix"):
        vix = tape["vix"]
        grid.append(_card("VIX — 6 months",
                          CH.area_chart(vix.get("closes"), vix.get("dates"),
                                        "vix2", w=560, fmt="{:.1f}"), span=6))
    grid.append("</div>")
    return "".join(grid)


# ── Macro & News tab ─────────────────────────────────────────────────────────

def _theme_rows(themes):
    out = []
    for t in themes or []:
        score = t.get("score")
        out.append(
            '<div class="evt"><span class="m {tone}">{sc}</span> '
            "<strong>{th}</strong> · {read} "
            '<span class="mut">({n} stories)</span></div>'.format(
                tone="g" if (score or 0) >= 0 else "b",
                sc=_num(score, "{:+.2f}"), th=_esc(t.get("theme")),
                read=_esc(t.get("read", "")), n=_esc(t.get("n", 0))))
    return "".join(out)


def _tab_macro(data):
    grid = ['<div class="grid">']
    pulse = _panel(data, "pulse")
    if pulse:
        head = ('<div class="chead"><span class="big m">{lean}</span>'
                '<span class="mut">confidence {conf} · {n} stories / '
                "{src} sources</span></div>"
                '<div class="lede">{hl}</div>'
                '<div class="flipbox"><strong>What would flip it:</strong> '
                "{flip}</div>").format(
                    lean=_esc(pulse.get("lean", "?")),
                    conf=_esc(pulse.get("confidence")),
                    n=_esc(pulse.get("n_items")),
                    src=_esc(pulse.get("n_sources")),
                    hl=_esc(pulse.get("headline", "")),
                    flip=_esc(pulse.get("what_would_flip", "")))
        grid.append(_card("Macro pulse (deterministic — no AI)",
                          head + _theme_rows(pulse.get("themes")), span=7))
    else:
        grid.append(_card("Macro pulse", _ph(data, "pulse", "Macro pulse"),
                          span=7))
    news = _panel(data, "news")
    if news:
        items = "".join(
            '<div class="evt"><a href="{u}" target="_blank" rel="noopener">{t}'
            '</a> <span class="mut">· {s}</span></div>'.format(
                u=_esc(i.get("url", "")), t=_esc(i.get("title", "")),
                s=_esc(i.get("source", ""))) for i in news.get("items") or [])
        line = ('<div class="mut" style="font-size:11px;margin-bottom:6px">{}'
                "</div>").format(_esc(news.get("line", "")))
        grid.append(_card("World headlines", line + items, span=5))
    else:
        grid.append(_card("World headlines", _ph(data, "news", "Headlines"),
                          span=5))
    sig = _panel(data, "signals") or {}
    uoa_rows = sig.get("uoa") or []
    if uoa_rows:
        t = ["<table><tr><th></th><th>score</th><th>call share</th>"
             "<th>n</th></tr>"]
        for r in uoa_rows:
            t.append('<tr><td class="m">{s}</td><td class="n">{sc}</td>'
                     '<td class="n">{cs}</td><td class="n">{n}</td></tr>'.format(
                         s=_esc(r.get("symbol")),
                         sc=_num(r.get("score"), "{:.1f}"),
                         cs=_num(r.get("net_call_share"), "{:.0%}"),
                         n=_esc(r.get("n_unusual"))))
        t.append("</table>")
        grid.append(_card("Unusual options activity", "".join(t), span=4))
    ins = sig.get("insider") or []
    if ins:
        body = "".join('<div class="evt"><span class="m">{s}</span> {t}</div>'
                       .format(s=_esc(r.get("sym")), t=_esc(r.get("summary")))
                       for r in ins)
        grid.append(_card("Insider clusters (EDGAR)", body, span=4))
    ol = sig.get("outlook") or {}
    if ol.get("top") or ol.get("bottom"):
        def _dir_tone(d):
            d = str(d or "").upper()
            if d in ("LONG", "BULLISH"):
                return "g"
            if d in ("SHORT", "BEARISH"):
                return "b"
            return "mut"
        body = "".join(
            '<div class="evt"><span class="m {c}">{d}</span> {t}</div>'.format(
                c=_dir_tone(r.get("direction")),
                d=_esc(r.get("direction")), t=_esc(r.get("ticker")))
            for r in (ol.get("top") or []) + (ol.get("bottom") or []))
        grid.append(_card("Sector outlook (1-3mo relative)", body, span=4))
    grid.append("</div>")
    return "".join(grid)


# ── Ticker deep-dive tab ─────────────────────────────────────────────────────

_VERDICT_CLS = {"BUY": "v-buy", "WAIT": "v-wait", "AVOID": "v-avoid",
                "NEUTRAL": "v-neutral"}


def _sig_row(s):
    v = s.get("value") or 0.0
    bar_w = min(50.0, abs(v) * 50.0)
    colour = "var(--good)" if v >= 0 else "var(--bad)"
    lab = "{:+.2f}".format(v) if s.get("directional", True) else "info"
    return ('<div class="sig"><div class="sighead"><span class="m">{n}</span>'
            '<span class="m" style="color:{c}">{lab}</span></div>'
            '<div class="sigbar"><span style="width:{w:.0f}%;background:{c};'
            '{side}"></span></div><div class="sigdet">{lb} · {d}</div></div>'
            .format(n=_esc(s.get("name")), c=colour, lab=lab, w=bar_w,
                    side=("margin-left:50%" if v >= 0 else
                          "margin-left:{:.0f}%".format(50 - bar_w)),
                    lb=_esc(s.get("label")), d=_esc(s.get("detail"))))


def _ticker_stats(t):
    st = t.get("state") or {}
    cells = [
        ("Price", _num(st.get("price"), "${:,.2f}"), ""),
        ("5d", _num((st.get("ret_5d") or 0) * 100 if st.get("ret_5d") is not None
                    else None, "{:+.1f}%"), _chg_tone(st.get("ret_5d"))),
        ("RSI", _num(st.get("rsi"), "{:.0f}"), ""),
        ("IV rank", _num(st.get("iv_rank"), "{:.0%}"), ""),
        ("Term", ("backwardated" if (t.get("term_spread") or 0) < 0
                  else "contango")
         if t.get("term_spread") is not None else "—",
         "bad" if (t.get("term_spread") or 0) < 0 else ""),
        ("Earnings", ("in {}d".format(st.get("days_to_earnings"))
                      if st.get("days_to_earnings") is not None else "—"),
         "warn" if (st.get("days_to_earnings") is not None
                    and 0 <= st["days_to_earnings"] <= 10) else ""),
        ("Bounce", ("{:.0%} (n={})".format(t["bounce"]["bounce_rate"],
                                           t["bounce"].get("n", 0))
                    if (t.get("bounce") or {}).get("bounce_rate") is not None
                    else "—"), ""),
    ]
    return shell.strip(cells)


def _related_tearsheets(t):
    rel = t.get("related_tearsheets") or []
    if not rel:
        return ""
    body = "".join(
        '<div class="evt"><a href="../tearsheets/{f}">{f}</a></div>'.format(
            f=_esc(f)) for f in rel)
    return _card("Tearsheets on file — {}".format(_esc(t.get("symbol"))),
                 body, span=12)


def _tab_ticker(data):
    t = _panel(data, "ticker")
    if not t:
        return _ph(data, "ticker", "Ticker briefing")
    v = t.get("verdict") or {}
    call = str(v.get("call", "NEUTRAL")).upper()
    banner = (
        '<div class="vrow"><span class="verdict {cls}">{call}</span>'
        '<span class="mut">confidence {conf} · composite {comp}</span>'
        "{note}{drivers}</div>").format(
            cls=_VERDICT_CLS.get(call, "v-neutral"), call=_esc(call),
            conf=_esc(v.get("confidence", "?")),
            comp=_num(v.get("composite"), "{:+.2f}"),
            note=('<span class="w">! {}</span>'.format(_esc(v["note"]))
                  if v.get("note") else ""),
            drivers="".join(
                '<span class="badge {k}">{g} {t} [{tag}]</span>'.format(
                    k="k-ok" if d.get("glyph") == "+" else
                      ("k-bad" if d.get("glyph") == "-" else "k-warn"),
                    g=_esc(d.get("glyph")), t=_esc(d.get("text")),
                    tag=_esc(d.get("tag"))) for d in v.get("drivers") or []))
    action = ('<div class="flipbox"><strong>What to do:</strong> {p}{s}</div>'
              .format(p=_esc(t.get("primary_action", "")),
                      s=(" <span class='mut'>Also: {}</span>".format(
                          _esc(t.get("secondary_action")))
                         if t.get("secondary_action") else "")))
    grid = ['<div class="grid">',
            '<div class="c12">' + banner + action + _ticker_stats(t) + "</div>"]
    chart = t.get("chart") or {}
    if chart.get("closes"):
        price = CH.price_chart(chart["closes"], chart.get("ma50"),
                               chart.get("ma200"), t.get("support"),
                               t.get("resist"), chart.get("dates"), "px")
        rsi = CH.rsi_strip(chart.get("rsi") or [])
        grid.append(_card("{} — 1 year, close / 50d / 200d".format(
            _esc(t.get("symbol"))), price + rsi, span=8))
    else:
        grid.append(_card("Price", '<div class="ph">price history unavailable'
                          "</div>", span=8))
    sigs = "".join(_sig_row(s) for s in t.get("signals") or [])
    grid.append(_card("Signals (reliability-weighted)",
                      sigs or '<div class="ph">no signals</div>', span=4))
    cone = CH.cone_chart(t.get("cone"))
    grid.append(_card("Realized-vol cone vs current",
                      cone or '<div class="ph">vol cone unavailable</div>',
                      span=6))
    term = CH.term_chart(t.get("term"))
    grid.append(_card("ATM IV term structure",
                      term or '<div class="ph">term structure unavailable</div>',
                      span=6))
    heads = t.get("headlines") or []
    if heads:
        grid.append(_card("Recent headlines", "".join(
            '<div class="evt">{}</div>'.format(_esc(h)) for h in heads),
            span=12))
    rel = _related_tearsheets(t)
    if rel:
        grid.append(rel)
    grid.append("</div>")
    return "".join(grid)


# ── Assembly ────────────────────────────────────────────────────────────────

_TAB_BUILDERS = {"market": _tab_market, "volatility": _tab_vol,
                 "macro": _tab_macro, "ticker": _tab_ticker}


def _footer(data):
    notes = _panel(data, "notes") or []
    m = data.get("meta", {})
    lines = "".join("<div>{}</div>".format(_esc(n)) for n in notes)
    regen = ("regenerate offline: python -m src.research --json "
             "reports/research/{}".format(m.get("sidecar", "")))
    return '<div class="foot">{}{}</div>'.format(
        lines, "<div>{}</div>".format(_esc(regen)))


_CSS = """
.vrow { display:flex; gap:12px; align-items:center; flex-wrap:wrap;
  margin-top:14px; }
"""


def render(data: dict) -> str:
    meta = data.get("meta", {})
    panes = []
    for tid, _label in _tabs_present(data):
        builder = _TAB_BUILDERS.get(tid)
        body = builder(data) if builder else ""
        panes.append('<div class="pane" id="pane-{t}">{b}</div>'.format(
            t=tid, b=body))
    body = _stale_banner(data) + "".join(panes) + _footer(data)
    return shell.page(meta.get("title", "Research Desk"), _masthead(data),
                      body, extra_css=_CSS)
