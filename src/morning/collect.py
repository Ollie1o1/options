"""Gather the morning-briefing sidecar. Every panel is individually fail-safe:
one dead fetch degrades to a placeholder panel, never a dead page."""
import threading
import time
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    _EASTERN = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _EASTERN = None

SCHEMA_VERSION = 1
PANEL_IDS = ("health", "market", "vol", "macro_events", "signals",
             "portfolio", "gate", "notes")

# Static caveats that must appear on every page: the honesty footer.
_STANDING_NOTES = [
    "All signal panels are display-only; nothing here feeds scoring or execution.",
    "Real money is OFF until the forward-cohort gate (n>=50) fires.",
    "Quotes are 15+ min delayed free data; use for planning, not live execution.",
]


def _now_eastern(now=None):
    if now is not None:
        return now
    if _EASTERN is not None:
        return datetime.now(_EASTERN)
    return datetime.now()


def session_phase(now=None) -> str:
    """'pre-market' | 'open' | 'closed' by US/Eastern wall clock."""
    ts = _now_eastern(now)
    if ts.isoweekday() > 5:
        return "closed"
    hhmm = ts.hour * 100 + ts.minute
    if hhmm < 400:
        return "closed"
    if hhmm < 930:
        return "pre-market"
    return "open" if hhmm < 1600 else "closed"


def _safe(panel_id, fn, panels, failures, default=None):
    try:
        panels[panel_id] = fn()
    except Exception as exc:
        panels[panel_id] = default
        failures.append(f"{panel_id}: {type(exc).__name__}: {exc}")


def _desk_notes(panels, failures):
    notes = list(_STANDING_NOTES)
    for f in failures:
        notes.append(f"Panel unavailable this morning — {f}")
    return notes


def _fetch_health_report():
    from src.maintenance import load_state, DEFAULT_STATE_PATH
    from src.maintenance_health import compute_health
    return compute_health(load_state(DEFAULT_STATE_PATH), datetime.now())


def _panel_health(_report_fn=None) -> dict:
    report = (_report_fn or _fetch_health_report)()
    return {
        "worst": report.worst,
        "jobs": [{"name": j.name, "cadence": j.cadence, "last_run": j.last_run,
                  "stale_days": j.business_days_stale, "severity": j.severity}
                 for j in report.jobs],
    }


def _panel_gate(_evidence_fn=None) -> dict:
    from src.evidence import load_model_evidence, GATE_TARGET_N
    ev = dict((_evidence_fn or load_model_evidence)())
    ev["target_n"] = GATE_TARGET_N
    return ev


# One honest line on the crypto vol sleeve; static because the briefing must
# not fetch Deribit every morning just to restate a validated finding.
_CRYPTO_NOTE = ("Crypto: BTC short-vol carry is the one net-validated edge "
                "(carry, not timing); equity VRP is currently the opposite sign.")


def _fetch_vol_rows():
    from src.vol_intel.engine import build_rows
    return build_rows()


def _panel_vol(_rows_fn=None) -> dict:
    movers, vrp_rows = (_rows_fn or _fetch_vol_rows)()
    ranked = sorted((m for m in movers if m.get("d_iv") is not None),
                    key=lambda m: abs(m["d_iv"]), reverse=True)
    n_cov = len({r["symbol"] for r in vrp_rows})
    return {"movers": ranked[:8], "vrp": list(vrp_rows)[:8],
            "n_cov": n_cov, "crypto_note": _CRYPTO_NOTE}


_TAPE_SYMBOLS = ("SPY", "QQQ", "IWM")


def _index_rows_from_closes(closes_by_sym) -> dict:
    """{sym: {last, chg_1d_pct, chg_5d_pct, closes}} from raw close series."""
    out = {}
    for sym, closes in closes_by_sym.items():
        vals = [float(c) for c in (closes or []) if c is not None]
        if len(vals) < 2:
            continue
        last = vals[-1]
        row = {"sym": sym, "last": last,
               "chg_1d_pct": (last / vals[-2] - 1.0) * 100.0,
               "chg_5d_pct": (last / vals[-6] - 1.0) * 100.0 if len(vals) >= 6 else None,
               "closes": vals[-30:]}
        out[sym] = row
    return out


def _fetch_index_rows():
    # fetch_index_directions only returns {price, verdict, ...}; sparklines and
    # day changes need the raw closes, so reuse the same guarded fetcher.
    from src.regime_dashboard import _safe_hist
    closes_by_sym = {}
    for sym in _TAPE_SYMBOLS:
        series = _safe_hist(sym, "2mo")
        if series is not None and len(series) >= 2:
            closes_by_sym[sym] = series.tolist()
    return _index_rows_from_closes(closes_by_sym)


def _fetch_rates():
    from src.macro_rates import fetch_rates_snapshot
    snap = fetch_rates_snapshot()
    return {"t10y": getattr(snap, "dgs10", None),
            "t3m": getattr(snap, "dgs3mo", None)}


def _panel_market(_regime_fn=None, _dirs_fn=None, _rates_fn=None) -> dict:
    if _regime_fn is None:
        from src.regime_dashboard import fetch_market_regime
        _regime_fn = fetch_market_regime
    regime = _regime_fn()

    indexes = []
    try:
        dirs = (_dirs_fn or _fetch_index_rows)() or {}
        for sym in _TAPE_SYMBOLS:
            info = dirs.get(sym)
            if not isinstance(info, dict):
                continue
            indexes.append({
                "sym": sym,
                "last": info.get("last"),
                "chg_1d_pct": info.get("chg_1d_pct"),
                "chg_5d_pct": info.get("chg_5d_pct"),
                "closes": list(info.get("closes") or [])[-30:],
            })
    except Exception:
        indexes = []

    rates = {"t10y": None, "t3m": None, "slope": None}
    try:
        raw = (_rates_fn or _fetch_rates)()
        rates["t10y"], rates["t3m"] = raw.get("t10y"), raw.get("t3m")
        if rates["t10y"] is not None and rates["t3m"] is not None:
            rates["slope"] = rates["t10y"] - rates["t3m"]
    except Exception:
        pass

    return {"regime": regime, "indexes": indexes, "rates": rates}


def _fetch_calendar():
    from src.worldnews.panel import next_events
    return [{"date": e.get("date"), "name": e.get("name") or "?"}
            for e in next_events(limit=6)]


def _fetch_pulse():
    from src.worldnews import panel, scoring, sources
    items = sources.fetch_all()
    if not items:
        return None, []
    agg = scoring.aggregate(items)
    headlines = [i.get("title", "") for i in items[:5] if i.get("title")]
    return panel.pulse_line(agg), headlines


def _fetch_watchlist_earnings(limit=10, horizon_days=7):
    import json as _json
    from datetime import timedelta
    from src.watchlist import load_watchlist
    from src.earnings_provider import next_earnings_date, resolve_api_key
    try:
        with open("config.json") as f:
            cfg = _json.load(f)
    except Exception:
        cfg = None
    key = resolve_api_key(cfg)
    today = _now_eastern().date()
    out = []
    for sym in load_watchlist()[:limit]:
        try:
            dt = next_earnings_date(sym, api_key=key)
        except Exception:
            continue
        if dt is None:
            continue
        d = dt.date()
        if today <= d <= today + timedelta(days=horizon_days):
            out.append({"sym": sym, "date": d.strftime("%Y-%m-%d")})
    return out


_TIME_EXIT_DTE = 21  # mirror of the 21-DTE time-exit rule in exit enforcement

# Columns worth carrying into the sidecar when present (the paper DB is
# SELECT *, so be defensive about which exist).
_POSITION_COLS = ("ticker", "strategy", "type", "strike", "expiration", "dte",
                  "entry_price", "current_price", "pnl_usd", "pnl_pct",
                  "pnl_percent", "delta", "vega")


def _fetch_positions():
    from src.portfolio_risk import RiskAggregator
    df = RiskAggregator().get_open_positions_with_greeks()
    if df is None or getattr(df, "empty", True):
        return []
    df = df.copy()
    if "dte" not in df.columns and "T_years" in df.columns:
        df["dte"] = df["T_years"] * 365.0
    keep = [c for c in _POSITION_COLS if c in df.columns]
    rows = df[keep].head(20).to_dict("records")
    # sqlite/numpy scalars -> plain python so the sidecar stays JSON-clean
    out = []
    for r in rows:
        clean = {}
        for k, v in r.items():
            try:
                clean[k] = v.item()  # numpy scalar
            except AttributeError:
                clean[k] = v
        out.append(clean)
    return out


def _fetch_net_greeks():
    from src.portfolio_risk import RiskAggregator
    g = RiskAggregator().get_portfolio_greeks() or {}
    g.pop("positions_df", None)  # DataFrame is not sidecar material
    return {k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in g.items() if isinstance(v, (int, float, str))}


def _fetch_guard_lines():
    from src.formatting import _strip_ansi
    from src.portfolio_guard import format_guard_lines
    return [_strip_ansi(l) for l in format_guard_lines(_fetch_positions(),
                                                       "Portfolio")]


def _panel_portfolio(_positions_fn=None, _greeks_fn=None, _guard_fn=None) -> dict:
    positions = (_positions_fn or _fetch_positions)()
    exits_due = []
    for p in positions:
        try:
            dte = float(p.get("dte"))
        except (TypeError, ValueError):
            continue
        if dte <= _TIME_EXIT_DTE:
            exits_due.append(f"{p.get('ticker', '?')}: {dte:.0f} DTE <= "
                             f"{_TIME_EXIT_DTE} — time-exit window")
    greeks, guard = {}, []
    try:
        greeks = (_greeks_fn or _fetch_net_greeks)()
    except Exception:
        pass
    try:
        guard = list((_guard_fn or _fetch_guard_lines)())
    except Exception:
        pass
    return {"positions": positions, "net_greeks": greeks, "guard": guard,
            "exits_due": exits_due, "n_open": len(positions)}


def _fetch_uoa():
    from src.uoa import uoa_report
    return uoa_report()


def _fetch_insider(limit=6):
    from src.pick_context import insider_summary
    from src.watchlist import load_watchlist
    out = []
    for sym in load_watchlist()[:limit]:
        try:
            s = insider_summary(sym, fetch=True)
        except Exception:
            continue
        if not s:
            continue
        if isinstance(s, dict):
            label = s.get("label") or "?"
            if label in ("NONE", "?"):
                continue
            text = f"{label} (score {s.get('score', 0):.1f})"
        else:
            text = str(s)
        out.append({"sym": sym, "summary": text})
    return out


def _fetch_outlook():
    from src.outlook.display import load_outlook_cache, compute_outlook_cache
    cache = load_outlook_cache()
    if not cache:
        compute_outlook_cache()
        cache = load_outlook_cache()
    return cache


def _panel_signals(_uoa_fn=None, _insider_fn=None, _outlook_fn=None) -> dict:
    out = {"uoa": [], "insider": [],
           "outlook": {"top": [], "bottom": [], "as_of": None}}
    try:
        rep = (_uoa_fn or _fetch_uoa)() or {}
        for r in list(rep.get("rows") or [])[:8]:
            out["uoa"].append({"symbol": r.get("symbol"),
                               "score": r.get("score"),
                               "net_call_share": r.get("net_call_share"),
                               "n_unusual": len(r.get("unusual") or [])})
    except Exception:
        pass
    try:
        out["insider"] = (_insider_fn or _fetch_insider)()
    except Exception:
        pass
    try:
        cache = (_outlook_fn or _fetch_outlook)() or {}
        rows = list(cache.get("rows") or [])
        out["outlook"] = {"top": rows[:3],
                          "bottom": rows[-3:] if len(rows) > 3 else [],
                          "as_of": cache.get("as_of")}
    except Exception:
        pass
    return out


def _panel_macro_events(_calendar_fn=None, _pulse_fn=None, _earnings_fn=None) -> dict:
    out = {"calendar": [], "pulse": None, "headlines": [], "earnings": []}
    try:
        out["calendar"] = (_calendar_fn or _fetch_calendar)()
    except Exception:
        pass
    try:
        pulse, headlines = (_pulse_fn or _fetch_pulse)()
        out["pulse"], out["headlines"] = pulse, headlines
    except Exception:
        pass
    try:
        out["earnings"] = (_earnings_fn or _fetch_watchlist_earnings)()
    except Exception:
        pass
    return out


def _default_fetchers():
    # Each entry is (panel_id, zero-arg callable); grown panel by panel.
    return [("health", _panel_health), ("market", _panel_market),
            ("vol", _panel_vol), ("macro_events", _panel_macro_events),
            ("signals", _panel_signals), ("portfolio", _panel_portfolio),
            ("gate", _panel_gate)]


def build(now=None, slow=True, budget_s=20.0, _fetchers=None) -> dict:
    ts = _now_eastern(now)
    date_s = ts.strftime("%Y-%m-%d")
    meta = {
        "schema": SCHEMA_VERSION,
        "date": date_s,
        "generated_at": ts.strftime("%Y-%m-%d %H:%M %Z").strip(),
        "session": session_phase(ts),
        "sidecar": f"{date_s}.json",
        "title": f"Morning Briefing — {date_s}",
    }
    fetchers = _fetchers if _fetchers is not None else _default_fetchers()
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
                failures.append(f"{pid}: TimeoutError: budget {budget_s:.0f}s exceeded")
    else:
        for pid, fn in fetchers:
            _safe(pid, fn, panels, failures)

    panels["notes"] = _desk_notes(panels, failures)
    return {"meta": meta, "panels": panels, "failures": failures}
