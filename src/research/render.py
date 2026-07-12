"""Pure HTML render over the research-desk sidecar. Zero network, zero
data-fetching imports: render(json.load(sidecar)) must reproduce the page."""
import html as _html

from src.research import charts as CH
from src.research import theme


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


def _card(title, body, span=6, extra_cls=""):
    return ('<section class="card c{s}{x}"><h5>{t}</h5>{b}</section>'
            .format(s=span, x=(" " + extra_cls if extra_cls else ""),
                    t=_esc(title), b=body))


_TONES = {"good": " g", "bad": " b", "warn": " w", "": ""}


def _kpi(label, value, sub="", tone=""):
    return ('<div class="card kpi c3"><div class="eye">{l}</div>'
            '<div class="kv{t}">{v}</div><div class="ks">{s}</div></div>'
            .format(l=_esc(label), t=_TONES.get(tone, ""), v=value, s=sub))


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


def _masthead(data):
    m = data.get("meta", {})
    sym = m.get("symbol")
    chip = ('<span class="chip">{}</span>'.format(_esc(sym))) if sym else ""
    return (
        '<header class="mast"><div class="mastrow">'
        '<div><span class="wordmark">RESEARCH DESK</span>{chip}</div>'
        '<div class="mastmeta"><span class="eye m">Generated {gen}</span>'
        '<button class="toggle" onclick="flipTheme()">'
        '<span id="tglabel">◑ Light</span></button></div></div>'
        '<nav class="tabbar">{tabs}</nav></header>'
    ).format(chip=chip, gen=_esc(m.get("generated_at")), tabs=_tabbar(data))


_TAB_ORDER = (("market", "Market"), ("volatility", "Volatility"),
              ("macro", "Macro &amp; News"), ("ticker", "Ticker"))


def _tabs_present(data):
    out = []
    for tid, label in _TAB_ORDER:
        if tid == "ticker" and not _panel(data, "ticker"):
            continue
        out.append((tid, label))
    return out


def _tabbar(data):
    return "".join(
        '<button class="tabbtn" data-tab="{t}">{n}&nbsp;{l}</button>'.format(
            t=tid, n=i + 1, l=label)
        for i, (tid, label) in enumerate(_tabs_present(data)))


# ── Market tab ───────────────────────────────────────────────────────────────

def _market_kpis(data):
    mkt = _panel(data, "market") or {}
    reg = mkt.get("regime") or {}
    rates = mkt.get("rates") or {}
    vix = reg.get("vix")
    vix_sub = "{} · {}".format(_esc(reg.get("vix_regime", "?")),
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
        grid.append(_card("VIX — 6 months",
                          CH.area_chart(vix.get("closes"), vix.get("dates"),
                                        "vix", fmt="{:.1f}"), span=4))
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
                                        "vix2", fmt="{:.1f}"), span=6))
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
        body = "".join(
            '<div class="evt"><span class="m {c}">{d}</span> {t}</div>'.format(
                c="g" if r.get("direction") == "LONG" else "b",
                d=_esc(r.get("direction")), t=_esc(r.get("ticker")))
            for r in (ol.get("top") or []) + (ol.get("bottom") or []))
        grid.append(_card("Sector outlook (1-3mo relative)", body, span=4))
    grid.append("</div>")
    return "".join(grid)


# ── Assembly ────────────────────────────────────────────────────────────────

_TAB_BUILDERS = {"market": _tab_market, "volatility": _tab_vol,
                 "macro": _tab_macro}


def _footer(data):
    notes = _panel(data, "notes") or []
    m = data.get("meta", {})
    lines = "".join("<div>{}</div>".format(_esc(n)) for n in notes)
    regen = ("regenerate offline: python -m src.research --json "
             "reports/research/{}".format(m.get("sidecar", "")))
    return '<div class="foot">{}{}</div>'.format(
        lines, "<div>{}</div>".format(_esc(regen)))


def render(data: dict) -> str:
    meta = data.get("meta", {})
    panes = []
    for tid, _label in _tabs_present(data):
        builder = _TAB_BUILDERS.get(tid)
        body = builder(data) if builder else ""
        panes.append('<div class="pane" id="pane-{t}">{b}</div>'.format(
            t=tid, b=body))
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>{title}</title><style>{tokens}{css}</style></head><body>"
        '{mast}<main class="desk">{stale}{panes}{foot}</main>'
        '<div class="tip" id="tip"></div>'
        "<script>{js}</script></body></html>"
    ).format(title=_esc(meta.get("title", "Research Desk")),
             tokens=theme.css_tokens(), css=_CSS, mast=_masthead(data),
             stale=_stale_banner(data), panes="".join(panes),
             foot=_footer(data), js=_JS)


_CSS = """
*, *::before, *::after { box-sizing:border-box; }
body { margin:0; background:var(--paper); color:var(--ink);
  font-family:"Iowan Old Style","Charter",Palatino,Georgia,serif;
  transition:background .18s ease,color .18s ease; }
.desk { max-width:1760px; margin:0 auto; padding:18px 28px 44px; }
.m { font-family:ui-monospace,"SF Mono",Menlo,monospace;
  font-variant-numeric:tabular-nums; }
.eye { font-family:ui-sans-serif,system-ui,sans-serif; font-size:9px;
  letter-spacing:.2em; text-transform:uppercase; color:var(--muted); }
h5 { font-family:ui-sans-serif,system-ui,sans-serif; font-size:9.5px;
  letter-spacing:.18em; text-transform:uppercase; margin:0 0 10px;
  font-weight:700; color:var(--muted); }
.g{color:var(--good)} .b{color:var(--bad)} .w{color:var(--warn)} .mut{color:var(--muted)}
.mast { position:sticky; top:0; z-index:20; background:var(--paper);
  border-bottom:1px solid var(--rule-hard); padding:12px 28px 0; }
.mastrow { display:flex; justify-content:space-between; align-items:center;
  max-width:1760px; margin:0 auto; }
.wordmark { font-family:ui-sans-serif,system-ui,sans-serif; font-weight:800;
  letter-spacing:.34em; font-size:13px; color:var(--ink-strong); }
.chip { font-family:ui-monospace,Menlo,monospace; font-size:12px; margin-left:14px;
  border:1px solid var(--accent); color:var(--accent); border-radius:3px;
  padding:2px 8px; letter-spacing:.08em; }
.mastmeta { display:flex; gap:14px; align-items:center; }
.toggle { cursor:pointer; font-family:ui-sans-serif,system-ui,sans-serif;
  font-size:10px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--muted); border:1px solid var(--rule); padding:4px 9px;
  border-radius:20px; background:transparent; }
.tabbar { display:flex; gap:2px; max-width:1760px; margin:10px auto 0; }
.tabbtn { cursor:pointer; font-family:ui-sans-serif,system-ui,sans-serif;
  font-size:10px; letter-spacing:.14em; text-transform:uppercase; font-weight:600;
  color:var(--muted); background:transparent; padding:9px 16px;
  border:1px solid transparent; border-bottom:none;
  border-radius:4px 4px 0 0; }
.tabbtn:hover { color:var(--ink); }
.tabbtn.on { color:var(--ink-strong); border-color:var(--rule);
  background:var(--panel); }
.stale { margin:16px 0 0; border-left:3px solid var(--warn);
  background:var(--chip-wn-bg); border-radius:0 3px 3px 0; padding:10px 14px;
  font-size:12.5px; }
.grid { display:grid; grid-template-columns:repeat(12,minmax(0,1fr));
  gap:16px; margin-top:16px; }
.c3{grid-column:span 3} .c4{grid-column:span 4} .c6{grid-column:span 6}
.c8{grid-column:span 8} .c12{grid-column:span 12}
@media (max-width:1100px){ .c3{grid-column:span 6} .c4{grid-column:span 6}
  .c8{grid-column:span 12} }
@media (max-width:760px){ .c3,.c4,.c6,.c8{grid-column:span 12} }
.card { background:var(--panel); border:1px solid var(--rule); border-radius:6px;
  padding:14px 16px; min-width:0; }
.kpi .kv { font-family:ui-monospace,Menlo,monospace;
  font-variant-numeric:tabular-nums; font-size:26px; margin-top:6px;
  color:var(--ink-strong); }
.kpi .kv.g{color:var(--good)} .kpi .kv.b{color:var(--bad)} .kpi .kv.w{color:var(--warn)}
.kpi .ks { font-size:11px; color:var(--muted); margin-top:2px; }
.chead { display:flex; gap:12px; align-items:baseline; margin-bottom:6px; }
.big { font-size:20px; color:var(--ink-strong); }
table { width:100%; border-collapse:collapse; font-size:12px; }
th { text-align:right; font-family:ui-sans-serif,system-ui,sans-serif;
  font-size:8.5px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--muted); font-weight:600; padding-bottom:5px; }
th:first-child { text-align:left; }
td { padding:3px 0; border-top:1px solid var(--grid); }
td.n { text-align:right; font-family:ui-monospace,Menlo,monospace;
  font-variant-numeric:tabular-nums; }
.evt { font-size:12.5px; padding:4px 0; border-top:1px solid var(--grid); }
.evt:first-child { border-top:none; }
.evt .m { color:var(--accent); margin-right:8px; }
.evt a { color:var(--ink-strong); text-decoration:none;
  border-bottom:1px solid var(--rule-hard); }
.evt a:hover { color:var(--accent); border-color:var(--accent); }
.lede { font-size:14px; line-height:1.5; margin:8px 0; max-width:70ch; }
.flipbox { margin:10px 0; font-size:12.5px; line-height:1.55;
  border-left:3px solid var(--accent); background:var(--paper);
  padding:8px 12px; border-radius:0 3px 3px 0; }
.badge { display:inline-block; font-family:ui-sans-serif,system-ui,sans-serif;
  font-size:8px; padding:1px 6px; border-radius:8px; border:1px solid; }
.k-bad{border-color:var(--chip-bad-bd);color:var(--bad);background:var(--chip-bad-bg)}
.k-ok{border-color:var(--chip-ok-bd);color:var(--good);background:var(--chip-ok-bg)}
.k-warn{border-color:var(--chip-wn-bd);color:var(--warn);background:var(--chip-wn-bg)}
.ph { border:1px dashed var(--rule); border-radius:3px; padding:8px 10px;
  color:var(--muted); font-family:ui-sans-serif,system-ui,sans-serif;
  font-size:11px; }
.foot { font-family:ui-monospace,Menlo,monospace; font-size:9.5px;
  color:var(--muted); margin-top:26px; border-top:1px solid var(--rule);
  padding-top:10px; line-height:1.7; }
.tip { position:fixed; display:none; z-index:50; pointer-events:none;
  background:var(--panel); border:1px solid var(--rule-hard); color:var(--ink-strong);
  font-family:ui-monospace,Menlo,monospace; font-size:11px; padding:5px 9px;
  border-radius:4px; box-shadow:0 4px 14px rgba(0,0,0,.25); }
.pane { animation:fadein .12s ease; }
@keyframes fadein { from{opacity:.4} to{opacity:1} }
@media print {
  .mast{position:static} .tabbar,.toggle{display:none}
  .pane{display:block !important; page-break-inside:avoid; margin-bottom:18px}
}
"""

_JS = """
(function () {
  var root = document.documentElement;
  function applyTheme(t) {
    root.setAttribute('data-theme', t);
    var el = document.getElementById('tglabel');
    if (el) el.textContent = t === 'dark' ? '\\u25d1 Light' : '\\u25d0 Dark';
  }
  var saved = null;
  try { saved = localStorage.getItem('desk-theme'); } catch (e) {}
  applyTheme(saved || 'dark');
  window.flipTheme = function () {
    var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    try { localStorage.setItem('desk-theme', next); } catch (e) {}
  };

  var tabs = Array.prototype.slice.call(document.querySelectorAll('.tabbtn'));
  function show(id) {
    tabs.forEach(function (b) { b.classList.toggle('on', b.dataset.tab === id); });
    Array.prototype.forEach.call(document.querySelectorAll('.pane'), function (p) {
      p.style.display = (p.id === 'pane-' + id) ? '' : 'none';
    });
    if (history.replaceState) history.replaceState(null, '', '#' + id);
  }
  tabs.forEach(function (b) {
    b.addEventListener('click', function () { show(b.dataset.tab); });
  });
  document.addEventListener('keydown', function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    var tag = (document.activeElement && document.activeElement.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    var i = parseInt(e.key, 10);
    if (i >= 1 && i <= tabs.length) show(tabs[i - 1].dataset.tab);
  });
  var want = (location.hash || '').slice(1);
  var ok = tabs.some(function (b) { return b.dataset.tab === want; });
  if (tabs.length) show(ok ? want : tabs[0].dataset.tab);

  var tip = document.getElementById('tip');
  Array.prototype.forEach.call(document.querySelectorAll('svg.xh'), function (svg) {
    var labels, ys;
    try {
      labels = JSON.parse(svg.dataset.labels || '[]');
      ys = JSON.parse(svg.dataset.ys || '[]');
    } catch (e) { return; }
    if (!labels.length) return;
    var x0 = parseFloat(svg.dataset.x0), step = parseFloat(svg.dataset.step);
    var line = svg.querySelector('.ch-line'), dot = svg.querySelector('.ch-dot');
    var vb = (svg.getAttribute('viewBox') || '0 0 760 200').split(/\\s+/);
    var vw = parseFloat(vb[2]);
    svg.addEventListener('mousemove', function (e) {
      var r = svg.getBoundingClientRect();
      var sx = (e.clientX - r.left) * (vw / r.width);
      var i = Math.round((sx - x0) / step);
      if (i < 0) i = 0;
      if (i >= labels.length) i = labels.length - 1;
      var cx = x0 + i * step;
      if (line) { line.setAttribute('x1', cx); line.setAttribute('x2', cx);
                  line.setAttribute('visibility', 'visible'); }
      if (dot && ys[i] != null) { dot.setAttribute('cx', cx);
        dot.setAttribute('cy', ys[i]); dot.setAttribute('visibility', 'visible'); }
      if (tip) {
        tip.textContent = labels[i];
        tip.style.display = 'block';
        var tx = e.clientX + 14, ty = e.clientY - 12;
        if (tx + tip.offsetWidth > window.innerWidth - 8)
          tx = e.clientX - tip.offsetWidth - 14;
        tip.style.left = tx + 'px'; tip.style.top = ty + 'px';
      }
    });
    svg.addEventListener('mouseleave', function () {
      if (line) line.setAttribute('visibility', 'hidden');
      if (dot) dot.setAttribute('visibility', 'hidden');
      if (tip) tip.style.display = 'none';
    });
  });
})();
"""
