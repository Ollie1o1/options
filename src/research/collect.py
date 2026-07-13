"""Gather the research-desk sidecar. Every panel is individually fail-safe:
one dead fetch degrades to a placeholder panel, never a dead page.

Reuses the morning-briefing panel fetchers where both pages ask the same
question; adds desk-only panels (tape, movers, macro pulse, linked news) and
the optional per-ticker deep dive.
"""
import threading
import time
from datetime import datetime

SCHEMA_VERSION = 1
PANEL_IDS = ("health", "market", "tape", "movers", "vol", "calendar",
             "pulse", "news", "signals", "ticker", "notes")

_STANDING_NOTES = [
    "Display-only research. Nothing here feeds scoring or execution.",
    "Real money is OFF until the forward-cohort gate (n>=50) fires.",
    "Quotes are 15+ min delayed free data; use for planning, not live execution.",
]

_CHART_SESSIONS = 180


def _safe(panel_id, fn, panels, failures):
    try:
        panels[panel_id] = fn()
    except Exception as exc:
        panels[panel_id] = None
        failures.append("{}: {}: {}".format(panel_id, type(exc).__name__, exc))


def _desk_notes(failures):
    notes = list(_STANDING_NOTES)
    for f in failures:
        notes.append("Panel unavailable — " + f)
    return notes


# ── Desk-only panels ─────────────────────────────────────────────────────────

def _hist_closes(sym, period):
    from src.regime_dashboard import _safe_hist
    series = _safe_hist(sym, period)
    if series is None or len(series) < 2:
        return None, None
    closes = [round(float(v), 4) for v in series.tolist()]
    try:
        dates = [d.strftime("%b %d") for d in series.index]
    except Exception:
        dates = ["" for _ in closes]
    return closes, dates


def _panel_tape(_hist_fn=None):
    """SPY + VIX 6-month close history for the headline charts."""
    hist = _hist_fn or (lambda sym: _hist_closes(sym, "6mo"))
    out = {}
    for key, sym in (("spy", "SPY"), ("vix", "^VIX")):
        closes, dates = hist(sym)
        if not closes:
            continue
        out[key] = {"closes": closes, "dates": dates, "last": closes[-1],
                    "chg_1d_pct": (closes[-1] / closes[-2] - 1.0) * 100.0}
    if not out:
        raise RuntimeError("no index history")
    return out


def _panel_movers(_movers_fn=None):
    if _movers_fn is None:
        from src.intel.market import gather_movers as _movers_fn
    return [{"sym": s, "ret_5d_pct": r * 100.0, "verdict": v}
            for s, r, v in _movers_fn()]


def _panel_calendar(_cal_fn=None):
    def _default():
        from src.worldnews.panel import next_events
        return next_events(limit=6)
    return [{"date": e.get("date"), "name": e.get("name") or "?"}
            for e in (_cal_fn or _default)()]


def _panel_pulse(_ctx_fn=None):
    # use_ai=False is load-bearing: the desk must never place an AI call
    # (macro-narrative AI gate). Do not thread use_ai=True through here.
    def _default():
        from src.macro_pulse.orchestrator import build_context
        return build_context(use_ai=False)
    ctx = (_ctx_fn or _default)()
    return {"pulse": ctx.pulse, "pulse_pctile": ctx.pulse_pctile,
            "lean": ctx.lean, "confidence": ctx.confidence,
            "n_items": ctx.n_items, "n_sources": ctx.n_sources,
            "bull_pct": ctx.bull_pct, "headline": ctx.headline,
            "what_would_flip": ctx.what_would_flip,
            "event_active": ctx.event_active, "event_name": ctx.event_name,
            "event_date": ctx.event_date,
            "themes": [{"theme": t.theme, "score": t.score, "n": t.n,
                        "read": t.read, "top_headline": t.top_headline}
                       for t in ctx.themes]}


def _panel_news(_fetch_fn=None, limit=10):
    from src.worldnews import panel, scoring, sources
    items = (_fetch_fn or sources.fetch_all)()
    if not items:
        raise RuntimeError("no headlines fetched")
    line = ""
    try:
        line = panel.pulse_line(scoring.aggregate(items))
    except Exception:
        pass
    return {"line": line, "n": len(items),
            "items": [{"title": i.get("title", ""), "source": i.get("source", ""),
                       "url": i.get("url", "")} for i in items[:limit]]}


# ── Ticker deep-dive panel ───────────────────────────────────────────────────

def _rsi_series(closes, period=14):
    """Wilder RSI over the whole series, aligned to closes[period:]."""
    closes = [float(c) for c in closes]
    if len(closes) <= period:
        return []
    ups = [max(0.0, closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    downs = [max(0.0, closes[i - 1] - closes[i]) for i in range(1, len(closes))]
    au = sum(ups[:period]) / period
    ad = sum(downs[:period]) / period
    out = []
    for i in range(period, len(ups) + 1):
        if i > period:
            au = (au * (period - 1) + ups[i - 1]) / period
            ad = (ad * (period - 1) + downs[i - 1]) / period
        out.append(100.0 if ad == 0 else 100.0 - 100.0 / (1.0 + au / ad))
    return out


def _ticker_chart(sym):
    closes, dates = _hist_closes(sym, "1y")
    if not closes or len(closes) < 30:
        raise RuntimeError("no price history")
    ma50 = [sum(closes[max(0, i - 49):i + 1]) / min(i + 1, 50)
            for i in range(len(closes))]
    ma200 = ([sum(closes[max(0, i - 199):i + 1]) / min(i + 1, 200)
              for i in range(len(closes))] if len(closes) >= 200 else [])
    rsi = _rsi_series(closes)
    n = _CHART_SESSIONS
    return {"closes": closes[-n:], "dates": dates[-n:], "ma50": ma50[-n:],
            "ma200": ma200[-n:], "rsi": rsi[-n:]}


def _default_cone(sym):
    from src.vol_analytics import compute_vol_cone
    return compute_vol_cone(sym)


def _default_term(sym):
    from src.vol_analytics import compute_iv_surface
    df = compute_iv_surface(sym)
    if df is None or getattr(df, "empty", True):
        return []
    out = []
    for _, r in df.iterrows():
        iv = r.get("atm_iv")
        if iv is not None and iv == iv:
            out.append([int(r["dte"]), float(iv)])
    return out


def _panel_ticker(sym, _gather_fn=None, _cone_fn=None, _surface_fn=None,
                  _chart_fn=None):
    if _gather_fn is None:
        from src.intel.briefing import gather as _gather_fn
    b = _gather_fn(sym)
    if not getattr(b, "ok", False):
        raise RuntimeError(getattr(b, "error", "") or "briefing failed")
    st = dict(b.state)
    support = st.pop("_nearest_support", None)
    resist = st.pop("_nearest_resist", None)
    headlines = st.pop("_headlines", []) or []
    bounce = st.pop("_bounce", {}) or {}
    term_spread = st.pop("_term_spread", None)
    v = b.verdict
    out = {"symbol": b.symbol, "name": b.name, "state": st,
           "support": support, "resist": resist, "bounce": bounce,
           "term_spread": term_spread, "headlines": headlines,
           "market_line": b.market_line,
           "primary_action": b.primary_action,
           "secondary_action": b.secondary_action,
           "signals": [{"name": s.name, "value": s.value, "label": s.label,
                        "detail": s.detail, "directional": s.directional}
                       for s in b.signals.values()],
           "verdict": None, "chart": None, "cone": [], "term": []}
    if v is not None:
        out["verdict"] = {"call": v.call, "confidence": v.confidence,
                          "composite": v.composite, "note": v.note,
                          "drivers": [{"glyph": d.glyph, "text": d.text,
                                       "tag": d.tag} for d in v.drivers]}
    try:
        out["chart"] = (_chart_fn or _ticker_chart)(sym)
    except Exception:
        out["chart"] = None
    try:
        cone = (_cone_fn or _default_cone)(sym) or {}
        out["cone"] = [dict(window=int(w),
                            **{k: float(cone[w][k]) for k in
                               ("p25", "median", "p75", "current", "pctile")})
                       for w in sorted(cone)]
    except Exception:
        out["cone"] = []
    try:
        out["term"] = (_surface_fn or _default_term)(sym) or []
    except Exception:
        out["term"] = []
    out["related_tearsheets"] = _related_tearsheets(sym)
    return out


def _related_tearsheets(sym, tearsheet_dir="reports/tearsheets", limit=6):
    """This symbol's tearsheets already on disk, newest first — filename scan
    only, so the ticker tab can deep-link into the pick-level pages."""
    import os
    try:
        files = [f for f in os.listdir(tearsheet_dir)
                 if f.startswith(str(sym) + "_") and f.endswith(".html")
                 and f != "latest.html"]
        files.sort(key=lambda f: os.path.getmtime(
            os.path.join(tearsheet_dir, f)), reverse=True)
        return files[:limit]
    except OSError:
        return []


# ── Assembly ─────────────────────────────────────────────────────────────────

def _default_fetchers(sym):
    from src.morning.collect import (_panel_health, _panel_market,
                                     _panel_signals, _panel_vol)
    fetchers = [("health", _panel_health), ("market", _panel_market),
                ("tape", _panel_tape), ("movers", _panel_movers),
                ("vol", _panel_vol), ("calendar", _panel_calendar),
                ("pulse", _panel_pulse), ("news", _panel_news),
                ("signals", _panel_signals)]
    if sym:
        fetchers.append(("ticker", lambda: _panel_ticker(sym)))
    return fetchers


def build(symbol=None, now=None, slow=True, budget_s=25.0, _fetchers=None) -> dict:
    sym = (symbol or "").upper().strip() or None
    ts = now if now is not None else datetime.now()
    base = "research_" + ts.strftime("%Y%m%d_%H%M") + ("_" + sym if sym else "")
    meta = {"schema": SCHEMA_VERSION,
            "generated_at": ts.strftime("%Y-%m-%d %H:%M"),
            "symbol": sym, "base": base, "sidecar": base + ".json",
            "title": "Research Desk — " + (sym or "Market")}
    fetchers = _fetchers if _fetchers is not None else _default_fetchers(sym)
    panels = {pid: None for pid in PANEL_IDS}
    failures = []

    if slow and fetchers:
        threads = []
        for pid, fn in fetchers:
            t = threading.Thread(target=_safe, args=(pid, fn, panels, failures),
                                 daemon=True)
            t.start()
            threads.append(t)
        deadline = time.monotonic() + budget_s
        for t in threads:
            t.join(max(0.0, deadline - time.monotonic()))
        for (pid, _), t in zip(fetchers, threads):
            if t.is_alive() and panels.get(pid) is None:
                failures.append("{}: TimeoutError: budget {:.0f}s exceeded"
                                .format(pid, budget_s))
    else:
        for pid, fn in fetchers:
            _safe(pid, fn, panels, failures)

    panels["notes"] = _desk_notes(failures)
    return {"meta": meta, "panels": panels, "failures": failures}
