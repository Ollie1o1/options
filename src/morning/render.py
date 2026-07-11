"""Pure HTML renderer over the morning-briefing sidecar. No I/O, no network:
render(data) is deterministic so `--from sidecar` rebuilds identical bytes."""
import html as _html

from src.tearsheet.theme import css_tokens
from src.tearsheet.render import _JS as _THEME_JS
from src.tearsheet.charts import line_chart

_ZONES = ("health", "market", "vol", "macro_events", "signals",
          "portfolio", "gate", "notes")

_ZONE_TITLES = {
    "health": "Automation Health", "market": "Market State",
    "vol": "Vol Intelligence", "macro_events": "Macro & Events",
    "signals": "Signals", "portfolio": "Portfolio",
    "gate": "Gate & Evidence", "notes": "Desk Notes",
}

_CSS = """
body{font:15px/1.5 -apple-system,'Helvetica Neue',sans-serif;margin:0;
     background:var(--paper);color:var(--ink);}
.wrap{max-width:920px;margin:0 auto;padding:28px 20px 60px;}
h1{font-size:22px;color:var(--ink-strong);margin:0;}
h2{font-size:13px;letter-spacing:.14em;text-transform:uppercase;
   color:var(--muted);border-bottom:1px solid var(--rule);
   padding-bottom:6px;margin:34px 0 12px;}
h3{font-size:13px;color:var(--ink-strong);margin:14px 0 6px;}
.meta{color:var(--muted);font-size:13px;margin-top:4px;}
table{border-collapse:collapse;width:100%;font-size:14px;}
th{color:var(--muted);font-weight:600;text-align:left;padding:4px 10px 4px 0;
   border-bottom:1px solid var(--rule);}
td{padding:4px 10px 4px 0;border-bottom:1px solid var(--grid);}
.good{color:var(--good);} .bad{color:var(--bad);} .warn{color:var(--warn);}
.muted{color:var(--muted);} .accent{color:var(--accent);}
.pill{display:inline-block;padding:1px 9px;border-radius:9px;font-size:12px;
      border:1px solid var(--rule-hard);}
.placeholder{color:var(--muted);font-style:italic;padding:8px 0;}
.note{margin:3px 0;color:var(--muted);}
.chart{margin:6px 0;}
.toggle{background:var(--panel);color:var(--ink);border:1px solid var(--rule-hard);
        border-radius:6px;padding:4px 10px;cursor:pointer;font-size:12px;}
.foot{margin-top:40px;color:var(--muted);font-size:12px;
      border-top:1px solid var(--rule);padding-top:10px;}
"""


def _esc(v) -> str:
    return _html.escape("" if v is None else str(v))


def _num(v, fmt="{:,.2f}", dash="—"):
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return dash


def _sign_cls(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "muted"
    return "good" if f > 0 else ("bad" if f < 0 else "muted")


def _placeholder(pid) -> str:
    return (f"<div class='placeholder'>{_esc(_ZONE_TITLES.get(pid, pid))} "
            f"unavailable this morning — see Desk Notes.</div>")


def _table(headers, rows) -> str:
    th = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    trs = []
    for row in rows:
        tds = "".join(f"<td class='{cls}'>{cell}</td>" for cell, cls in row)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><tr>{th}</tr>{''.join(trs)}</table>"


def _zone_health(p) -> str:
    worst = p.get("worst", "?")
    cls = {"OK": "good", "STALE": "warn"}.get(worst, "bad")
    rows = [[(_esc(j.get("name")), ""), (_esc(j.get("cadence")), "muted"),
             (_esc(j.get("last_run") or "never"), "muted"),
             (_esc(j.get("stale_days")), ""),
             (_esc(j.get("severity")),
              {"OK": "good", "STALE": "warn"}.get(j.get("severity"), "bad"))]
            for j in p.get("jobs", [])]
    head = f"<p>Worst job status: <span class='pill {cls}'>{_esc(worst)}</span></p>"
    return head + _table(("job", "cadence", "last run", "stale (bd)", "status"), rows)


def _zone_market(p) -> str:
    r = p.get("regime") or {}
    parts = [f"<p>VIX <b>{_num(r.get('vix'), '{:.1f}')}</b>"
             f" · term {_esc(r.get('vix_term_structure', '?'))}"
             f" · PCR {_num(r.get('options_pcr'))}"
             f" · posture <span class='pill'>{_esc(r.get('posture', '?'))}</span></p>",
             f"<p class='muted'>{_esc(r.get('posture_rationale', ''))}</p>"]
    rows = []
    for ix in p.get("indexes", []):
        closes = [c for c in (ix.get("closes") or []) if c is not None]
        chart = (f"<span class='chart'>{line_chart(closes, w=140, h=34)}</span>"
                 if len(closes) >= 2 else "")
        rows.append([(_esc(ix.get("sym")), ""), (_num(ix.get("last")), ""),
                     (_num(ix.get("chg_1d_pct"), "{:+.2f}%"), _sign_cls(ix.get("chg_1d_pct"))),
                     (_num(ix.get("chg_5d_pct"), "{:+.2f}%"), _sign_cls(ix.get("chg_5d_pct"))),
                     (chart, "")])
    if rows:
        parts.append(_table(("index", "last", "1d", "5d", "trend"), rows))
    rates = p.get("rates") or {}
    parts.append(f"<p>10Y {_num(rates.get('t10y'))}% · 3M {_num(rates.get('t3m'))}% · "
                 f"slope <span class='{_sign_cls(rates.get('slope'))}'>"
                 f"{_num(rates.get('slope'), '{:+.2f}')}pp</span></p>")
    return "".join(parts)


def _zone_vol(p) -> str:
    parts = []
    movers = p.get("movers") or []
    if movers:
        rows = [[(_esc(m.get("symbol")), ""),
                 (_num(m.get("iv"), "{:.1%}"), "muted"),
                 (_num(m.get("d_iv"), "{:+.1%}"), _sign_cls(m.get("d_iv")))]
                for m in movers]
        parts.append("<h3>IV movers (vs prior snapshot)</h3>" +
                     _table(("symbol", "ATM IV", "ΔIV"), rows))
    vrp = p.get("vrp") or []
    if vrp:
        rows = [[(_esc(v.get("symbol")), ""), (_num(v.get("iv"), "{:.1%}"), "muted"),
                 (_num(v.get("rv"), "{:.1%}"), "muted"),
                 (_num(v.get("vrp"), "{:+.1f}vp"), _sign_cls(v.get("vrp"))),
                 (_esc(v.get("label")), "accent")] for v in vrp]
        parts.append("<h3>Implied vs realized (VRP)</h3>" +
                     _table(("symbol", "IV", "RV", "VRP", "read"), rows))
    parts.append(f"<p class='muted'>Coverage: {_esc(p.get('n_cov'))} symbols · "
                 f"{_esc(p.get('crypto_note'))}</p>")
    return "".join(parts)


def _zone_macro(p) -> str:
    parts = []
    cal = p.get("calendar") or []
    if cal:
        rows = [[(_esc(e.get("date")), "muted"), (_esc(e.get("name")), "")] for e in cal]
        parts.append(_table(("date", "event"), rows))
    if p.get("pulse"):
        parts.append(f"<p>News pulse: {_esc(p['pulse'])}</p>")
    for h in (p.get("headlines") or [])[:5]:
        parts.append(f"<p class='note'>• {_esc(h)}</p>")
    ern = p.get("earnings") or []
    if ern:
        rows = [[(_esc(e.get("sym")), ""), (_esc(e.get("date")), "muted")] for e in ern]
        parts.append("<h3>Watchlist earnings (7d)</h3>" + _table(("symbol", "date"), rows))
    return "".join(parts) or "<p class='muted'>No macro events in window.</p>"


def _zone_signals(p) -> str:
    parts = []
    uoa = p.get("uoa") or []
    if uoa:
        rows = [[(_esc(u.get("symbol")), ""), (_num(u.get("score")), ""),
                 (_num(u.get("net_call_share"), "{:.0%}"), "muted"),
                 (_esc(u.get("n_unusual")), "muted")] for u in uoa]
        parts.append("<h3>Unusual options activity (OI deltas)</h3>" +
                     _table(("symbol", "score", "call share", "unusual strikes"), rows))
    for i in p.get("insider") or []:
        parts.append(f"<p class='note'>• Insider {_esc(i.get('sym'))}: "
                     f"{_esc(i.get('summary'))}</p>")
    ol = p.get("outlook") or {}
    if ol.get("top") or ol.get("bottom"):
        parts.append("<h3>Sector outlook (1–3mo, relative)</h3>")
        for label, rows in (("Top", ol.get("top") or []), ("Bottom", ol.get("bottom") or [])):
            if rows:
                names = ", ".join(_esc(r.get("ticker") or r.get("symbol")) for r in rows)
                parts.append(f"<p class='note'>{label}: {names}</p>")
        if ol.get("as_of"):
            parts.append(f"<p class='muted'>as of {_esc(ol['as_of'])}</p>")
    return "".join(parts) or "<p class='muted'>No active signals this morning.</p>"


def _zone_portfolio(p) -> str:
    if not p.get("n_open"):
        return "<p class='muted'>No open paper positions.</p>"
    rows = []
    for x in p.get("positions", []):
        pnl = x.get("pnl_pct", x.get("pnl_percent"))
        rows.append([(_esc(x.get("ticker")), ""), (_esc(x.get("strategy")), "muted"),
                     (_num(x.get("dte"), "{:.0f}"), ""),
                     (_num(pnl, "{:+.1f}%"), _sign_cls(pnl)),
                     (_num(x.get("delta")), "muted")])
    parts = [_table(("ticker", "strategy", "DTE", "P&L", "Δ"), rows)]
    g = p.get("net_greeks") or {}
    gnum = {k: v for k, v in g.items() if isinstance(v, (int, float))}
    if gnum:
        parts.append("<p class='muted'>Net book: " +
                     " · ".join(f"{_esc(k.replace('portfolio_', ''))} "
                                f"{_num(v, '{:+,.2f}')}"
                                for k, v in sorted(gnum.items())) + "</p>")
    for w in p.get("exits_due", []):
        parts.append(f"<p class='warn'>⚠ {_esc(w)}</p>")
    for w in p.get("guard", []):
        parts.append(f"<p class='warn'>⚠ {_esc(w)}</p>")
    return "".join(parts)


def _zone_gate(p) -> str:
    n, target = p.get("cohort_n") or 0, p.get("target_n") or 50
    pct = min(100, int(100 * n / target)) if target else 0
    bar = (f"<div style='background:var(--panel);border:1px solid var(--rule);"
           f"border-radius:4px;height:10px'><div style='width:{pct}%;height:100%;"
           f"background:var(--accent);border-radius:4px'></div></div>")
    return (f"<p>Gate: <span class='pill'>{_esc(p.get('gate_decision', '?'))}</span>"
            f" · cohort {n}/{target}</p>{bar}"
            f"<p class='muted'>Walk-forward pooled IC {_num(p.get('pooled_ic'), '{:+.2f}')}"
            f" (p={_num(p.get('p_value'))}, n_oos={_esc(p.get('n_oos'))})"
            f" · as of {_esc(p.get('as_of') or '—')}</p>"
            f"<p><b>Real money is OFF</b> until the gate fires.</p>")


def _zone_notes(p) -> str:
    return "".join(f"<p class='note'>• {_esc(n)}</p>" for n in (p or []))


_ZONE_FNS = {"health": _zone_health, "market": _zone_market, "vol": _zone_vol,
             "macro_events": _zone_macro, "signals": _zone_signals,
             "portfolio": _zone_portfolio, "gate": _zone_gate, "notes": _zone_notes}


def render(data: dict) -> str:
    meta = data.get("meta", {})
    body = ["<div style='display:flex;justify-content:space-between;"
            "align-items:flex-start'><div>"
            f"<h1>{_esc(meta.get('title', 'Morning Briefing'))}</h1>"
            f"<p class='meta'>{_esc(meta.get('date'))} · session: "
            f"{_esc(meta.get('session'))} · generated {_esc(meta.get('generated_at'))}</p>"
            "</div><button class='toggle' onclick='flipTheme()'>"
            "<span id='tglabel'>◐ Dark</span></button></div>"]
    panels = data.get("panels", {})
    for pid in _ZONES:
        body.append(f"<h2>{_esc(_ZONE_TITLES[pid])}</h2>")
        panel = panels.get(pid)
        if panel is None:
            body.append(_placeholder(pid))
        else:
            body.append(_ZONE_FNS[pid](panel))
    body.append("<div class='foot'>"
                f"{_esc(meta.get('sidecar', ''))} · regenerate: "
                f"python -m src.morning --from {_esc(meta.get('sidecar', ''))}</div>")
    return ("<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{_esc(meta.get('title', 'Morning Briefing'))}</title>"
            f"<style>{css_tokens()}{_CSS}</style></head>"
            "<body><div class=\"wrap\">" + "".join(body) + "</div>"
            f"<script>{_THEME_JS}</script></body></html>")
