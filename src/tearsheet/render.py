"""TearsheetData -> HTML. A pure function: no network, no DB, no clock.

Purity is load-bearing. It is what makes `--from <sidecar>.json` reproduce a
page byte-for-byte months later, and what lets the tests run without mocks.
"""
import html as _html
import math

from . import charts, theme

_ZONES = ("decision", "vol", "name", "narrative", "detail", "context")


def _esc(v) -> str:
    return _html.escape("" if v is None else str(v))


def _num(v, fmt="{:,.2f}", dash="—"):
    if v is None:
        return dash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return dash
    if not math.isfinite(f):
        return dash
    return fmt.format(f)


def _panel_ok(data, pid) -> bool:
    return data.get("panels", {}).get(pid, {}).get("status") == "ok"


def _placeholder(data, pid) -> str:
    p = data.get("panels", {}).get(pid, {})
    status = p.get("status", "unavailable")
    reason = _esc(p.get("reason", ""))
    word = "not fetched" if status == "not_fetched" else "unavailable"
    return ('<div class="ph">{w}{sep}{r}</div>'
            .format(w=word, sep=" — " if reason else "", r=reason))


def decide_verdict(net_ev, gross_ev, cost, waterfall):
    """(decision, reason). TAKE iff net_ev > 0; zero is SKIP.

    Never consults the quality score: its out-of-sample IC is ~0.03. When net EV
    is missing or non-finite (the HV-fallback path sets it to NaN) the verdict is
    INDETERMINATE rather than a silent fallback to a signal with no edge.
    """
    try:
        v = float(net_ev)
    except (TypeError, ValueError):
        v = None
    if v is None or not math.isfinite(v):
        return "INDETERMINATE", (
            "Net expected value is unavailable for this contract, so no verdict "
            "is offered. It is never inferred from the quality score.")
    if v > 0:
        return "TAKE", (
            "Net expected value is {} per contract after the {} round-trip cost."
            .format(_num(v, "${:+,.0f}"), _num(cost, "${:,.0f}")))
    negatives = [(l, x) for l, x in (waterfall or []) if x < 0]
    worst = min(negatives, key=lambda t: t[1])[0] if negatives else "transaction cost"
    return "SKIP", (
        "Net expected value is {} per contract. The gross edge of {} does not "
        "survive the {} round-trip cost; the largest single drag is {}."
        .format(_num(v, "${:+,.0f}"), _num(gross_ev, "${:+,.0f}"),
                _num(cost, "${:,.0f}"), worst))


_CSS = """
*, *::before, *::after { box-sizing: border-box; }
body { margin:0; padding:28px; background:var(--paper); color:var(--ink);
  font-family:"Iowan Old Style","Charter",Palatino,Georgia,serif;
  transition:background .18s ease,color .18s ease; }
.sheet { max-width:1040px; margin:0 auto; }
.m,.n { font-family:ui-monospace,"SF Mono",Menlo,monospace; font-variant-numeric:tabular-nums; }
.eye { font-family:ui-sans-serif,system-ui,sans-serif; font-size:8.5px; letter-spacing:.22em;
  text-transform:uppercase; color:var(--muted); }
h1 { font-size:26px; margin:3px 0 0; color:var(--ink-strong); font-weight:600; }
h5 { font-family:ui-sans-serif,system-ui,sans-serif; font-size:9.5px; letter-spacing:.18em;
  text-transform:uppercase; margin:0 0 8px; font-weight:700; }
.rule { height:2px; background:var(--rule-hard); margin:14px 0 10px; }
.thin { height:1px; background:var(--rule); margin:14px 0; }
.g{color:var(--good)} .b{color:var(--bad)} .w{color:var(--warn)} .mut{color:var(--muted)}
.verdict { display:inline-block; font-family:ui-sans-serif,system-ui,sans-serif; font-weight:700;
  letter-spacing:.16em; text-transform:uppercase; font-size:12px; color:var(--paper);
  padding:5px 12px; border-radius:3px; }
.v-take{background:var(--good)} .v-skip{background:var(--bad)} .v-ind{background:var(--warn)}
.lede { font-size:14.5px; line-height:1.5; margin-top:9px; max-width:64ch; }
.strip { display:grid; grid-template-columns:repeat(6,1fr); border-top:1px solid var(--rule-hard);
  border-bottom:1px solid var(--rule); margin-top:14px; }
.strip>div { padding:8px 10px; border-right:1px solid var(--rule); }
.strip>div:last-child{border-right:none}
.sv { font-family:ui-monospace,Menlo,monospace; font-variant-numeric:tabular-nums;
  font-size:16px; margin-top:2px; }
.cols { display:grid; grid-template-columns:1.15fr 1fr; gap:26px; }
@media (max-width:760px){ .cols,.strip{grid-template-columns:1fr} }
table { width:100%; border-collapse:collapse; font-size:12px; }
td { padding:3px 0; }
td.n { text-align:right; }
.heat { display:grid; gap:2px; font-size:10px; }
.heat div { padding:4px 2px; text-align:center; font-family:ui-monospace,Menlo,monospace;
  border-radius:2px; background:var(--hl); color:var(--ink); }
[data-theme="dark"] .heat div { background:var(--hd); }
.heat .rh { text-align:left; color:var(--muted); background:none !important; }
.badge { display:inline-block; font-family:ui-sans-serif,system-ui,sans-serif; font-size:8px;
  padding:1px 5px; border-radius:8px; border:1px solid; }
.k-bad{border-color:var(--chip-bad-bd);color:var(--bad);background:var(--chip-bad-bg)}
.k-ok{border-color:var(--chip-ok-bd);color:var(--good);background:var(--chip-ok-bg)}
.k-warn{border-color:var(--chip-wn-bd);color:var(--warn);background:var(--chip-wn-bg)}
.demote { opacity:.7; background:var(--panel); padding:16px 20px 6px; border-top:1px solid var(--rule);
  margin-top:16px; border-radius:4px; }
.ph { border:1px dashed var(--rule); border-radius:3px; padding:8px 10px; color:var(--muted);
  font-family:ui-sans-serif,system-ui,sans-serif; font-size:11px; }
.foot { font-family:ui-monospace,Menlo,monospace; font-size:9.5px; color:var(--muted);
  margin-top:18px; border-top:1px solid var(--rule); padding-top:8px; line-height:1.6; }
.wf { font-family:ui-monospace,Menlo,monospace; font-size:11px; }
.wf > div { margin:2px 0; }
.wfl { display:inline-block; width:96px; }
.wfbar { display:inline-block; height:9px; vertical-align:middle; border-radius:1px; margin-right:6px; }
.wftot { border-top:1px solid var(--rule); margin-top:4px; padding-top:4px; }
.toggle { cursor:pointer; font-family:ui-sans-serif,system-ui,sans-serif; font-size:10px;
  letter-spacing:.1em; text-transform:uppercase; color:var(--muted); border:1px solid var(--rule);
  padding:4px 9px; border-radius:20px; background:transparent; }
.cols3 { display:grid; grid-template-columns:repeat(3,1fr); gap:22px; }
@media (max-width:760px){ .cols3{grid-template-columns:1fr} }

/* Tabbed detail deck: radio + :checked, no JavaScript. */
.deck { margin-top:4px; }
.tabin { position:absolute; opacity:0; pointer-events:none; }
.tabbar { display:flex; gap:2px; border-bottom:1px solid var(--rule); margin-bottom:12px; }
.tablab { cursor:pointer; font-family:ui-sans-serif,system-ui,sans-serif; font-size:9.5px;
  letter-spacing:.14em; text-transform:uppercase; font-weight:600; color:var(--muted);
  padding:7px 12px; border:1px solid transparent; border-bottom:none; border-radius:3px 3px 0 0; }
.tablab:hover { color:var(--ink); }
.tabpane { display:none; }
#tab-greeks:checked ~ .tabbar label[for="tab-greeks"],
#tab-execution:checked ~ .tabbar label[for="tab-execution"],
#tab-chain:checked ~ .tabbar label[for="tab-chain"],
#tab-events:checked ~ .tabbar label[for="tab-events"],
#tab-raw:checked ~ .tabbar label[for="tab-raw"] {
  color:var(--ink-strong); border-color:var(--rule); background:var(--panel); }
#tab-greeks:checked ~ .tp-greeks,
#tab-execution:checked ~ .tp-execution,
#tab-chain:checked ~ .tp-chain,
#tab-events:checked ~ .tp-events,
#tab-raw:checked ~ .tp-raw { display:block; }
.raw { font-family:ui-monospace,Menlo,monospace; font-size:10px; line-height:1.45;
  background:var(--panel); border:1px solid var(--rule); border-radius:4px; padding:10px;
  max-height:420px; overflow:auto; white-space:pre-wrap; word-break:break-word; }

/* Paper never hides anything: printing expands every tab. */
@media print {
  .tabbar, .toggle { display:none; }
  .tabpane { display:block !important; page-break-inside:avoid; margin-bottom:14px; }
  .raw { max-height:none; }
}
"""

_JS = """
(function () {
  var root = document.documentElement;
  function apply(t) {
    root.setAttribute('data-theme', t);
    var el = document.getElementById('tglabel');
    if (el) el.textContent = t === 'dark' ? '\\u25d1 Light' : '\\u25d0 Dark';
  }
  var saved = null;
  try { saved = localStorage.getItem('tearsheet-theme'); } catch (e) {}
  var prefers = window.matchMedia
    && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  apply(saved || prefers);
  window.flipTheme = function () {
    var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    apply(next);
    try { localStorage.setItem('tearsheet-theme', next); } catch (e) {}
  };
})();
"""


def _masthead(m):
    return (
        '<div style="display:flex;justify-content:space-between;align-items:flex-start">'
        '<div><div class="eye">{mode} &middot; pick {rank} of {n} &middot; {typ} &middot; {dte} DTE</div>'
        '<h1>{tkr} {strike:g}{tl} &mdash; {exp}</h1></div>'
        '<div style="text-align:right">'
        '<button class="toggle" onclick="flipTheme()"><span id="tglabel">◐ Dark</span></button>'
        '<div class="eye" style="margin-top:6px">Generated {gen}</div>'
        '<div class="eye m">spot {spot} &middot; rfr {rfr} &middot; VIX {vix} ({reg})</div>'
        "</div></div>"
    ).format(mode=_esc(m.get("mode")), rank=_esc(m.get("rank")), n=_esc(m.get("n_picks")),
             typ=_esc(m.get("opt_type")), dte=_esc(m.get("dte")), tkr=_esc(m.get("ticker")),
             strike=float(m.get("strike") or 0), tl=_esc((m.get("opt_type") or "c")[0].upper()),
             exp=_esc(m.get("expiration")), gen=_esc(m.get("generated_at")),
             spot=_num(m.get("spot")), rfr=_num(m.get("rfr"), "{:.2%}"),
             vix=_num(m.get("vix"), "{:.1f}"), reg=_esc(m.get("vix_regime")))


def _verdict_block(data):
    v = data["verdict"]
    decision, reason = decide_verdict(v.get("net_ev"), v.get("gross_ev"),
                                      v.get("cost"), data.get("cost_waterfall"))
    cls = {"TAKE": "v-take", "SKIP": "v-skip"}.get(decision, "v-ind")
    s = data.get("stats", {})
    cells = (("Net EV", _num(v.get("net_ev"), "${:+,.0f}")),
             ("Gross EV", _num(v.get("gross_ev"), "${:+,.0f}")),
             ("RT cost", _num(v.get("cost"), "${:,.0f}")),
             ("POP", _num(s.get("pop"), "{:.0%}")),
             ("Max loss", _num(s.get("max_loss"), "${:,.0f}")),
             ("Breakeven", _num(s.get("breakeven"), "{:,.2f}")))
    strip = "".join('<div><div class="eye">{k}</div><div class="sv">{val}</div></div>'
                    .format(k=_esc(k), val=_esc(val)) for k, val in cells)
    return ('<div class="rule"></div>'
            '<span class="verdict {c}">{d}</span>'
            '<p class="lede">{r}</p>'
            '<div class="strip">{s}</div>').format(c=cls, d=_esc(decision),
                                                   r=_esc(reason), s=strip)


def _heat_grid(stress) -> str:
    """Spot x IV grid. Each cell carries BOTH inks; CSS picks one."""
    moves = stress.get("moves") or []
    rows = stress.get("rows") or []
    if not moves or not rows:
        return ""
    span = max((abs(p) for r in rows for p in r["pnls"]), default=1.0) or 1.0
    cols = "64px " + " ".join(["1fr"] * len(moves))
    out = ['<div class="heat" style="grid-template-columns:{}">'.format(cols)]
    out.append('<div class="rh"></div>')
    for m in moves:
        out.append('<div class="rh eye">{:+.0%}</div>'.format(float(m)))
    for r in rows:
        iv = float(r["iv"])
        label = "IV flat" if iv == 0 else "IV {:+.0f}pp".format(iv * 100)
        out.append('<div class="rh">{}</div>'.format(_esc(label)))
        for pnl in r["pnls"]:
            hl, hd = theme.heat_inks(pnl, span)
            out.append('<div class="hc" style="--hl:{hl};--hd:{hd}">{v}</div>'.format(
                hl=hl, hd=hd, v=_num(pnl, "{:+,.0f}")))
    out.append("</div>")
    return "".join(out)


def _rows(pairs) -> str:
    return "".join('<tr><td>{}</td><td class="n m">{}</td></tr>'.format(_esc(k), _esc(v))
                   for k, v in pairs)


def _zone_decision(data) -> str:
    g, l = data["greeks"], data["liquidity"]
    left = (
        '<div class="eye" style="margin-bottom:5px">Cost wall &mdash; where the edge goes</div>'
        + charts.waterfall_bars(data.get("cost_waterfall"))
        + '<div class="eye" style="margin:12px 0 4px">Greeks (per contract)</div><table>'
        + _rows((("Delta", _num(g.get("delta"))), ("Gamma", _num(g.get("gamma"), "{:.3f}")),
                 ("Vega", _num(g.get("vega"))), ("Theta", _num(g.get("theta")) + " /day")))
        + '</table><div class="eye" style="margin:12px 0 4px">Liquidity</div><table>'
        + _rows((("Spread", _num(l.get("spread_pct"), "{:.1%}")),
                 ("Open interest", _num(l.get("oi"), "{:,.0f}")),
                 ("Volume", _num(l.get("volume"), "{:,.0f}")),
                 ("Quote", _esc(l.get("quote_freshness")))))
        + "</table>")
    right = (
        '<div class="eye" style="margin-bottom:5px">Stress &mdash; P&amp;L across spot &times; IV</div>'
        + _heat_grid(data.get("stress", {}))
        + '<div class="eye" style="margin-top:5px">worst {}</div>'.format(
            _esc(data.get("stress", {}).get("worst", "n/a"))))
    return ('<div class="thin"></div><h5>I &middot; Decision-grade</h5>'
            '<div class="cols"><div>{}</div><div>{}</div></div>').format(left, right)


def _zone_vol(data) -> str:
    v = data["vol"]
    resid = v.get("svi_residual")
    if resid is None:
        rich = "no surface fit"
    elif float(resid) > 0:
        rich = "RICH +{:.2f}σ vs surface".format(float(resid))
    else:
        rich = "CHEAP {:.2f}σ vs surface".format(float(resid))
    left = ('<div class="eye" style="margin-bottom:4px">Vol cone</div>'
            + charts.vol_cone(v.get("cone"), v.get("iv"))
            + "<table>" + _rows((
                ("IV / HV30", "{} / {}".format(_num(v.get("iv"), "{:.1%}"),
                                               _num(v.get("hv"), "{:.1%}"))),
                ("VRP", _num(v.get("vrp"), "{:+.1%}")),
                ("IV rank", _num(v.get("iv_rank"), "{:.0%}")),
                ("vs SVI", rich))) + "</table>")
    term = v.get("term") or []
    if len(term) >= 2:
        term_html = charts.term_curve(term)
    else:
        # A heading over a blank chart reads as "no term structure". Say why.
        term_html = ('<div class="ph">no term curve — this scan surfaced only '
                     'one expiry for this name (need ≥2)</div>')
    right = ('<div class="eye" style="margin-bottom:4px">Term structure &amp; skew</div>'
             + term_html
             + "<table>" + _rows((
                 ("Skew 25Δ", "{}vp".format(_num(v.get("skew_vp"), "{:+.1f}"))),
                 ("Skew rank", _num(v.get("skew_rank"), "{:.0%}")),
                 ("Expected move", _num(v.get("expected_move"))),
                 ("Required move", _num(v.get("required_move"))))) + "</table>")
    return ('<div class="thin"></div><h5>II &middot; Vol complex</h5>'
            '<div class="cols"><div>{}</div><div>{}</div></div>').format(left, right)


_IC_EDGE_THRESHOLD = 0.05
_P_VALUE_THRESHOLD = 0.05


def _badge(text, kind) -> str:
    return '<span class="badge k-{k}">{t}</span>'.format(k=_esc(kind), t=_esc(text))


def _finite_or_none(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _has_scorer_edge(pooled_ic, p_value) -> bool:
    """An edge requires BOTH magnitude and significance.

    A large IC with p=0.48 is a coin flip wearing a decimal point. The project's
    own walk-forward reports an IC of +0.10 at p=0.48 on n=94 — reporting that as
    'has edge' would make this panel certify the very noise it exists to expose.
    """
    ic = _finite_or_none(pooled_ic)
    p = _finite_or_none(p_value)
    return (ic is not None and ic >= _IC_EDGE_THRESHOLD
            and p is not None and p < _P_VALUE_THRESHOLD)


def _ic_badge(pooled_ic, p_value=None):
    """(display_text, badge_html). Missing evidence fails closed.

    An absent track record is not a good track record. Neither is an
    insignificant one.
    """
    ic = _finite_or_none(pooled_ic)
    if ic is None:
        return "unknown", _badge("no edge", "bad")
    txt = _num(ic, "{:+.2f}")
    if ic < _IC_EDGE_THRESHOLD:
        return txt, _badge("no edge", "bad")
    if _has_scorer_edge(pooled_ic, p_value):
        return txt, _badge("has edge", "ok")
    return txt, _badge("underpowered", "warn")


def _fmt_uoa(uoa) -> str:
    """flow_summary() -> a sentence, not a repr of its dict."""
    if not isinstance(uoa, dict):
        return "n/a"
    share = uoa.get("net_call_share")
    lean = "n/a"
    if isinstance(share, (int, float)):
        lean = "call-led" if share > 0.55 else ("put-led" if share < 0.45 else "balanced")
    return "{} unusual · {} (calls {:+,.0f} / puts {:+,.0f} OI)".format(
        _num(uoa.get("n_unusual"), "{:,.0f}"), lean,
        float(uoa.get("call_oi_added") or 0), float(uoa.get("put_oi_added") or 0))


def _fmt_insider(ins) -> str:
    """cluster_score() -> a sentence, not a repr of its dict."""
    if not isinstance(ins, dict):
        return "n/a"
    return "{} · {} buyer(s) / {} buy(s) over {}d · bought {} vs sold {}".format(
        _esc(ins.get("label") or "no signal"),
        _num(ins.get("n_buyers"), "{:,.0f}"), _num(ins.get("n_buys"), "{:,.0f}"),
        _num(ins.get("window_days"), "{:,.0f}"),
        _num(ins.get("buy_value"), "${:,.0f}"), _num(ins.get("sell_value"), "${:,.0f}"))


def _fmt_history(h) -> str:
    """analog_stats() -> a sentence, not a repr of its dict."""
    if not isinstance(h, dict):
        return "no prior trades"
    n = h.get("n") or h.get("count")
    parts = []
    if n is not None:
        parts.append("{} comparable trade(s)".format(_num(n, "{:,.0f}")))
    for key, label, fmt in (("win_rate", "win rate", "{:.0%}"),
                            ("avg_pnl_pct", "avg", "{:+.1%}"),
                            ("median_pnl_pct", "median", "{:+.1%}")):
        if h.get(key) is not None:
            parts.append("{} {}".format(label, _num(h[key], fmt)))
    return " · ".join(parts) if parts else "no prior trades"


def _zone_name(data) -> str:
    n = data["name"]
    left = charts.price_with_bands(n.get("closes"), n.get("supports"), n.get("resistances"))
    sup = (n.get("supports") or [{}])
    res = (n.get("resistances") or [{}])
    left += "<table>" + _rows((
        ("RSI(14)", _num(n.get("rsi"), "{:.0f}")),
        ("5d return", _num(n.get("ret_5d"), "{:+.1%}")),
        ("Support", _num(sup[0].get("level"))),
        ("Resistance", _num(res[0].get("level"))))) + "</table>"
    right = ('<div class="eye" style="margin-bottom:4px">Flow &amp; positioning</div><table>'
             + _rows((("Put/call ratio", _num(n.get("pcr"))),
                      ("OI change (1d)", _num(n.get("oi_change"), "{:+,.0f}")),
                      ("Unusual activity", _fmt_uoa(n.get("uoa"))),
                      ("Max pain", _num(n.get("max_pain"))))) + "</table>")
    return ('<div class="thin"></div><h5>III &middot; The name &mdash; {t}</h5>'
            '<div class="cols"><div>{l}</div><div>{r}</div></div>').format(
                t=_esc(data.get("meta", {}).get("ticker")), l=left, r=right)


def _zone_evidence(data) -> str:
    e = data.get("evidence", {})
    ic_txt, ic_badge = _ic_badge(e.get("pooled_ic"), e.get("p_value"))
    gate = _esc(e.get("gate_decision") or "UNKNOWN")
    p_txt = _num(e.get("p_value"), "{:.3f}")
    rows = (
        '<tr><td>Scorer OOS IC</td><td class="n m">{}</td><td class="n">{}</td></tr>'.format(
            ic_txt, ic_badge),
        '<tr><td>p-value</td><td class="n m">{}</td><td class="n mut">{}</td></tr>'.format(
            p_txt, "significant" if _has_scorer_edge(e.get("pooled_ic"), e.get("p_value"))
            else "not significant"),
        '<tr><td>Walk-forward n</td><td class="n m">{}</td><td class="n mut">{}</td></tr>'.format(
            _num(e.get("n_oos"), "{:,.0f}"), _esc(e.get("as_of") or "n/a")),
        '<tr><td>Cohort gate</td><td class="n m">{} / 50</td><td class="n">{}</td></tr>'.format(
            _num(e.get("cohort_n"), "{:,.0f}"), _badge(gate.lower(), "warn")),
        '<tr><td>Cost model</td><td class="n m">round-trip</td><td class="n">{}</td></tr>'.format(
            _badge("validated", "ok")),
    )
    # Conditional, not hardcoded — and gated on significance, not magnitude alone.
    if _has_scorer_edge(e.get("pooled_ic"), e.get("p_value")):
        caption = ("The ranking that surfaced this pick has statistically significant "
                   "out-of-sample skill. The cost and Greeks arithmetic above is "
                   "independent of it.")
    else:
        caption = ("The ranking that surfaced this pick has no demonstrated "
                   "out-of-sample skill. The cost and Greeks arithmetic above does.")
    return ('<div class="eye" style="margin-bottom:4px">Model evidence</div>'
            "<table>{}</table>"
            '<div class="eye" style="margin-top:6px;line-height:1.5">{}</div>').format(
                "".join(rows), _esc(caption))


def _zone_narrative(data) -> str:
    """Zone IV. ALWAYS rendered, even when the narrative panel failed.

    The evidence panel lives in this zone's right column. If a narrative failure
    could take zone IV down with it, the page would silently drop the very panel
    that qualifies everything above it. Only the left column degrades.
    """
    if _panel_ok(data, "narrative"):
        nar = data.get("narrative", {})
        fit = nar.get("portfolio_fit") or []
        left = '<p class="lede">{}</p>'.format(_esc(nar.get("thesis")))
        caveat = nar.get("thesis_caveat")
        if caveat:
            # The thesis may lean on a signal this very page marks as no-edge.
            left += '<div class="eye" style="margin:-4px 0 8px">⚠ {}</div>'.format(_esc(caveat))
        if fit:
            counts = {}
            for f in fit:
                counts[f] = counts.get(f, 0) + 1
            fit_txt = "{} open on this name: {}".format(
                len(fit), ", ".join("{}×{}".format(v, k) for k, v in sorted(counts.items())))
        else:
            fit_txt = "no open positions on this name"
        left += "<table>"
        left += _rows((("Vehicle verdict", nar.get("vehicle") or "n/a"),
                       ("Portfolio fit", fit_txt),
                       ("Your history", _fmt_history(nar.get("history")))))
        left += "</table>"
    else:
        left = _placeholder(data, "narrative")
    return ('<div class="thin"></div><h5>IV &middot; Narrative &amp; provenance</h5>'
            '<div class="cols"><div>{l}</div><div>{r}</div></div>').format(
                l=left, r=_zone_evidence(data))


def _greek_rows(data):
    g = data.get("greeks_full") or {}
    first, second, dollar = g.get("first", {}), g.get("second", {}), g.get("dollar", {})
    return ("<table>" + _rows((
        ("Δ delta", _num(first.get("delta"), "{:+.4f}")),
        ("Γ gamma", _num(first.get("gamma"), "{:+.5f}")),
        ("ν vega", _num(first.get("vega"), "{:+.4f}")),
        ("Θ theta", _num(first.get("theta"), "{:+.4f}") + " /day"),
    )) + "</table>",
        "<table>" + _rows((
            ("ρ rho", _num(second.get("rho"), "{:+.4f}")),
            ("vanna", _num(second.get("vanna"), "{:+.5f}")),
            ("charm", _num(second.get("charm"), "{:+.5f}")),
        )) + "</table>",
        "<table>" + _rows((
            ("Vega $/1% IV", _num(dollar.get("vega_dollar"), "${:+,.2f}")),
            ("Γ/Θ ratio", _num(dollar.get("gamma_theta_ratio"), "{:,.3f}")),
            ("Θ burn rate", _num(dollar.get("theta_burn_rate"), "{:.2%}") + " /day"),
            ("|Δ|", _num(dollar.get("abs_delta"), "{:.4f}")),
        )) + "</table>")


def _tab_greeks(data) -> str:
    a, b, c = _greek_rows(data)
    return ('<div class="cols3">'
            '<div><div class="eye">First order</div>{}</div>'
            '<div><div class="eye">Second order</div>{}</div>'
            '<div><div class="eye">Dollar exposure</div>{}</div></div>').format(a, b, c)


def _tab_execution(data) -> str:
    """Order ticket + exit plan. The stress grid deliberately is NOT repeated
    here: it is decision-grade and stays in the always-visible zone I."""
    t = data.get("ticket") or {}
    left = ('<div class="eye" style="margin-bottom:4px">Order ticket</div><table>'
            + _rows((("Limit price", _num(t.get("entry_price"), "${:,.2f}")),
                     ("Breakeven", _num(t.get("breakeven"), "{:,.2f}")),
                     ("Max loss", _num(t.get("max_loss"), "${:,.0f}")),
                     ("Potential profit", _num(t.get("potential_profit"), "${:,.0f}")),
                     ("Risk / reward", _num(t.get("risk_reward_ratio"), "{:,.2f}x"))))
            + "</table>")
    right = ('<div class="eye" style="margin-bottom:4px">Exit plan</div><table>'
             + _rows((("Profit target", _num(t.get("profit_target"), "${:,.2f}")),
                      ("Stop loss", _num(t.get("stop_loss"), "${:,.2f}"))))
             + "</table>")
    if t.get("guidance"):
        right += '<div class="eye" style="margin-top:6px;line-height:1.5">{}</div>'.format(
            _esc(t["guidance"]))
    return '<div class="cols"><div>{}</div><div>{}</div></div>'.format(left, right)


def _tab_chain(data) -> str:
    q = data.get("quote") or {}
    m = data.get("meta") or {}
    left = "<table>" + _rows((
        ("Contract", "{} {:g}{} {}".format(m.get("ticker"), float(m.get("strike") or 0),
                                           str(m.get("opt_type", "c"))[0].upper(),
                                           m.get("expiration"))),
        ("Strategy", _esc(q.get("strategy_name") or "n/a")),
        ("DTE", _num(m.get("dte"), "{:,.0f}")),
        ("Premium", _num(q.get("premium"), "${:,.2f}")),
        ("Bid / Ask", "{} / {}".format(_num(q.get("bid"), "${:,.2f}"),
                                       _num(q.get("ask"), "${:,.2f}"))),
        ("Mid", _num(q.get("mid"), "${:,.2f}")),
        ("Spread", _num(q.get("spread_pct"), "{:.2%}")),
    )) + "</table>"
    right = "<table>" + _rows((
        ("Volume", _num(q.get("volume"), "{:,.0f}")),
        ("Open interest", _num(q.get("oi"), "{:,.0f}")),
        ("OI change", _num(q.get("oi_change"), "{:+,.0f}")),
        ("Liquidity", _esc(q.get("liquidity_flag") or "n/a")),
        ("Spread flag", _esc(q.get("spread_flag") or "n/a")),
        ("Quote", _esc(q.get("quote_freshness") or "n/a")),
        ("IV confidence", _esc(q.get("iv_confidence") or "n/a")),
        ("SVI fit quality", _num(q.get("iv_surface_confidence"), "{:.0%}")),
    )) + "</table>"
    third = "<table>" + _rows((
        ("Prob. of touch", _num(q.get("prob_touch"), "{:.0%}")),
        ("Risk / reward", _num(q.get("rr_ratio"), "{:,.2f}x")),
        ("Annualised return", _num(q.get("annualized_return"), "{:+.1%}")),
        ("Breakeven distance", _num(q.get("be_dist_pct"), "{:+.2%}")),
        ("Max-pain distance", _num(q.get("max_pain_dist_pct"), "{:+.2%}")),
        ("Gamma-pin distance", _num(q.get("gamma_pin_dist_pct"), "{:+.2%}")),
    )) + "</table>"
    return ('<div class="cols3"><div><div class="eye">Contract</div>{}</div>'
            '<div><div class="eye">Liquidity &amp; quote</div>{}</div>'
            '<div><div class="eye">Derived</div>{}</div></div>').format(left, right, third)


def _event_panel(data, pid, title, body_fn) -> str:
    if not _panel_ok(data, pid):
        return '<div><div class="eye">{}</div>{}</div>'.format(
            _esc(title), _placeholder(data, pid))
    return '<div><div class="eye">{}</div>{}</div>'.format(_esc(title), body_fn())


def _tab_events(data) -> str:
    """The slow tier. Collected under a 2.5s budget — and, until now, discarded."""
    ev = data.get("events") or {}

    def _earn():
        val = ev.get("earnings")
        return '<div class="lede">{}</div>'.format(
            _esc(val) if val else "no earnings date before expiry")

    def _ins():
        return '<div class="lede">{}</div>'.format(_fmt_insider(ev.get("insider")))

    def _news():
        items = ev.get("news")
        if isinstance(items, str):
            items = [x.strip() for x in items.split(";") if x.strip()]
        if not items:
            return '<div class="lede">no recent headlines</div>'
        return "<ul style='margin:4px 0;padding-left:16px;font-size:12.5px'>" + "".join(
            "<li>{}</li>".format(_esc(i)) for i in items[:5]) + "</ul>"

    return ('<div class="cols3">{}{}{}</div>').format(
        _event_panel(data, "earnings", "Earnings", _earn),
        _event_panel(data, "insider", "Insider (EDGAR, 90d)", _ins),
        _event_panel(data, "news", "News", _news))


def _tab_raw(data) -> str:
    import json as _json
    safe = {k: v for k, v in data.items() if k != "name"}   # drop the long close series
    blob = _json.dumps(safe, indent=2, sort_keys=True)
    return ('<div class="eye" style="margin-bottom:4px">Sidecar snapshot '
            "(price series omitted) &mdash; this is exactly what regenerates the page</div>"
            '<pre class="raw">{}</pre>').format(_esc(blob))


_TABS = (("greeks", "Greeks", _tab_greeks),
         ("execution", "Execution", _tab_execution),
         ("chain", "Chain &amp; quote", _tab_chain),
         ("events", "Events", _tab_events),
         ("raw", "Raw", _tab_raw))


def _zone_detail(data) -> str:
    """Tabbed depth. Pure CSS (radio + :checked) so it works offline with no JS.

    Nothing that QUALIFIES the trade lives here — the verdict, cost wall, model
    evidence and the no-edge zone all stay in the always-visible scroll. Only
    depth hides behind a tab. Print CSS expands every tab.
    """
    inputs, labels, panes = [], [], []
    for i, (key, label, fn) in enumerate(_TABS):
        checked = " checked" if i == 0 else ""
        inputs.append('<input type="radio" name="tsdeck" id="tab-{k}" class="tabin"{c}>'
                      .format(k=key, c=checked))
        labels.append('<label for="tab-{k}" class="tablab">{l}</label>'.format(k=key, l=label))
        try:
            body = fn(data)
        except Exception as exc:  # a bad tab must not kill the page
            body = '<div class="ph">unavailable — {}: {}</div>'.format(
                _esc(type(exc).__name__), _esc(exc))
        panes.append('<div class="tabpane tp-{k}">{b}</div>'.format(k=key, b=body))
    return ('<div class="thin"></div><h5>VI &middot; Detail</h5>'
            '<div class="deck">{i}<div class="tabbar">{l}</div>{p}</div>').format(
                i="".join(inputs), l="".join(labels), p="".join(panes))


def _zone_context(data) -> str:
    items = data.get("context") or []
    if not items:
        return ""
    rows = "".join(
        '<tr><td>{l}</td><td class="n m">{v}</td><td class="n">{b}</td></tr>'.format(
            l=_esc(i.get("label")), v=_esc(i.get("value")),
            b=_badge(i.get("badge", ""), i.get("badge_kind", "bad")))
        for i in items)
    return ('<div class="demote"><h5 class="mut">V &middot; Context &mdash; '
            "no demonstrated edge</h5><table>{}</table></div>").format(rows)


def render(data: dict) -> str:
    """The complete HTML document. Pure."""
    body = [_masthead(data.get("meta", {})), _verdict_block(data)]
    for zone in _ZONES:
        if zone in _ALWAYS_RENDERED:
            body.append(_BUILDERS[zone](data))     # degrades internally
        elif _panel_ok(data, zone) and zone in _BUILDERS:
            body.append(_BUILDERS[zone](data))
        elif not _panel_ok(data, zone):
            body.append('<div class="thin"></div>' + _placeholder(data, zone))
    meta = data.get("meta", {})
    body.append('<div class="foot">{}</div>'.format(_esc(
        "{} · config sha {} · regenerate: python -m src.tearsheet --from {}".format(
            meta.get("sidecar", ""), meta.get("config_sha", ""), meta.get("sidecar", "")))))
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>{title}</title><style>{tokens}{css}</style></head>"
        '<body><div class="sheet">{body}</div><script>{js}</script></body></html>'
    ).format(title=_esc("{} {} tearsheet".format(
        meta.get("ticker", ""), meta.get("expiration", ""))),
        tokens=theme.css_tokens(), css=_CSS, body="".join(body), js=_JS)


_BUILDERS = {
    "decision": _zone_decision,
    "vol": _zone_vol,
    "name": _zone_name,
    "narrative": _zone_narrative,   # embeds _zone_evidence beside the thesis
    "detail": _zone_detail,
    "context": _zone_context,
}

# Zones that render even when their panel failed, because they carry something
# the page must never lose. Zone IV holds the model-evidence panel; zone VI
# degrades per-tab rather than vanishing wholesale.
_ALWAYS_RENDERED = frozenset({"narrative", "detail"})
